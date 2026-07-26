import ast
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
VIDEO_ENV_PATH = (
    REPO_ROOT / "agent_system" / "environments" / "env_package" / "video_env.py"
)
VIDEO_ENV_SOURCE = VIDEO_ENV_PATH.read_text(encoding="utf-8")


def _load_temporal_reward_helpers():
    function_names = {
        "_spans_to_ndarray",
        "_merge_spans",
        "overlap_ratio",
        "_score_temporal_prediction",
        "_apply_answer_format_reward",
    }
    tree = ast.parse(VIDEO_ENV_SOURCE, filename=str(VIDEO_ENV_PATH))
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in function_names
    ]
    namespace = {"np": np}
    exec(
        compile(
            ast.Module(body=functions, type_ignores=[]),
            filename=str(VIDEO_ENV_PATH),
            mode="exec",
        ),
        namespace,
    )
    return namespace


HELPERS = _load_temporal_reward_helpers()
SPANS_TO_ARRAY = HELPERS["_spans_to_ndarray"]
SCORE_TEMPORAL_PREDICTION = HELPERS["_score_temporal_prediction"]
APPLY_FORMAT_REWARD = HELPERS["_apply_answer_format_reward"]


class TestVideoEnvTemporalReward(unittest.TestCase):
    def test_valid_temporal_prediction_keeps_existing_iou_behavior(self):
        prediction = SPANS_TO_ARRAY([[10, 20]])
        ground_truth = SPANS_TO_ARRAY([[10, 20]])

        reward, rejected = SCORE_TEMPORAL_PREDICTION(prediction, ground_truth)

        self.assertAlmostEqual(reward, 1.0)
        self.assertIs(rejected, False)

    def test_any_reversed_span_rejects_the_whole_prediction(self):
        ground_truth = SPANS_TO_ARRAY([[10, 20]])

        def unexpected_overlap(*_args, **_kwargs):
            raise AssertionError("rejected predictions must not reach overlap_ratio")

        for spans in ([[20, 10]], [[10, 20], [50, 40]]):
            with self.subTest(spans=spans):
                prediction = SPANS_TO_ARRAY(spans)
                with mock.patch.dict(
                    SCORE_TEMPORAL_PREDICTION.__globals__,
                    {"overlap_ratio": unexpected_overlap},
                ):
                    reward, rejected = SCORE_TEMPORAL_PREDICTION(
                        prediction,
                        ground_truth,
                    )

                self.assertEqual(reward, 0.0)
                self.assertIs(rejected, True)

    def test_rejected_prediction_never_receives_format_reward(self):
        prediction = SPANS_TO_ARRAY([[10, 20], [50, 40]])
        ground_truth = SPANS_TO_ARRAY([[10, 20]])
        reward, rejected = SCORE_TEMPORAL_PREDICTION(prediction, ground_truth)

        self.assertEqual(
            APPLY_FORMAT_REWARD(reward, False, suppress=rejected),
            0.0,
        )
        self.assertEqual(
            APPLY_FORMAT_REWARD(reward, True, suppress=rejected),
            0.0,
        )
        self.assertAlmostEqual(APPLY_FORMAT_REWARD(0.0, True), 0.1)

    def test_non_finite_spans_remain_format_errors(self):
        for bad_value in (np.nan, np.inf, -np.inf):
            with self.subTest(bad_value=bad_value):
                with self.assertRaisesRegex(ValueError, "NaN or Inf"):
                    SPANS_TO_ARRAY([[10, bad_value]])

    def test_video_env_answer_path_uses_strict_rejection_without_fail(self):
        self.assertIn(
            "iou, rejected_temporal_prediction = "
            "_score_temporal_prediction(p, g)",
            VIDEO_ENV_SOURCE,
        )
        self.assertIn(
            "suppress=rejected_temporal_prediction",
            VIDEO_ENV_SOURCE,
        )


if __name__ == "__main__":
    unittest.main()
