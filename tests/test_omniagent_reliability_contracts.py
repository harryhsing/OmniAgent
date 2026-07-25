# Copyright 2026 The verl team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ast
import shlex
import subprocess
import sys
import unittest
from pathlib import Path


_REPO_ROOT = Path(__file__).parents[1]
_TRAINING_SCRIPTS = (
    _REPO_ROOT / "examples/omniagent_train/train_GRPO.sh",
    _REPO_ROOT / "examples/omniagent_train/train_TAURA.sh",
)
_EVAL_SCRIPT = _REPO_ROOT / "examples/omniagent_eval/eval.sh"


def _contains_name(node: ast.AST, name: str) -> bool:
    return any(isinstance(child, ast.Name) and child.id == name for child in ast.walk(node))


def _load_will_only_validate():
    source_path = _REPO_ROOT / "verl/trainer/main_ppo.py"
    tree = ast.parse(source_path.read_text())
    function_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_will_only_validate"
    )
    namespace = {}
    module = ast.Module(body=[function_node], type_ignores=[])
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace["_will_only_validate"]


class TestOmniAgentReliabilityContracts(unittest.TestCase):
    def test_abs_bypass_validation_runs_before_ray_initialization(self):
        source = (_REPO_ROOT / "verl/trainer/main_ppo.py").read_text()
        tree = ast.parse(source)
        run_ppo = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "run_ppo"
        )
        validation_call = next(
            node
            for node in ast.walk(run_ppo)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "validate_abs_on_policy_rollout_compatibility"
        )
        ray_initialization = next(
            node
            for node in ast.walk(run_ppo)
            if isinstance(node, ast.If) and _contains_name(node.test, "ray")
        )
        validation_guard = next(
            node
            for node in ast.walk(run_ppo)
            if isinstance(node, ast.If)
            and any(child is validation_call for child in ast.walk(node))
        )
        self.assertTrue(_contains_name(validation_guard.test, "_will_only_validate"))
        self.assertLess(validation_call.lineno, ray_initialization.lineno)

    def test_abs_bypass_validation_matches_actor_update_control_flow(self):
        will_only_validate = _load_will_only_validate()
        cases = (
            (False, True, False),
            (True, True, True),
            (True, False, False),
        )

        for val_only, val_before_train, expected in cases:
            with self.subTest(
                val_only=val_only,
                val_before_train=val_before_train,
            ):
                config = type(
                    "Config",
                    (),
                    {
                        "trainer": {
                            "val_only": val_only,
                            "val_before_train": val_before_train,
                        },
                    },
                )()
                self.assertEqual(will_only_validate(config), expected)

    def test_bypass_entropy_state_is_initialized_before_branch_and_shared_gate(self):
        source = (_REPO_ROOT / "verl/trainer/ppo/ray_trainer.py").read_text()
        tree = ast.parse(source)
        fit = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "fit"
        )

        entropy_initializers = [
            node
            for node in ast.walk(fit)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "entropys" for target in node.targets)
            and isinstance(node.value, ast.Constant)
            and node.value.value is None
        ]
        bypass_branches = [
            node
            for node in ast.walk(fit)
            if isinstance(node, ast.If)
            and _contains_name(node.test, "bypass_recomputing_logprobs")
        ]
        shared_entropy_gates = [
            node
            for node in ast.walk(fit)
            if isinstance(node, ast.If)
            and _contains_name(node.test, "need_entropy")
            and _contains_name(node.test, "entropys")
        ]

        self.assertTrue(entropy_initializers)
        self.assertTrue(bypass_branches)
        self.assertTrue(shared_entropy_gates)
        self.assertLess(
            min(node.lineno for node in entropy_initializers),
            min(node.lineno for node in bypass_branches),
        )
        self.assertLess(
            min(node.lineno for node in entropy_initializers),
            min(node.lineno for node in shared_entropy_gates),
        )

    def test_launch_scripts_capture_python_status_before_cleanup(self):
        for script in (*_TRAINING_SCRIPTS, _EVAL_SCRIPT):
            with self.subTest(script=script.name):
                source = script.read_text()
                pipeline_end = source.rindex('2>&1 | tee -a "logs/${experiment_name}.log"')
                commands_after_pipeline = [
                    line.strip()
                    for line in source[pipeline_end:].splitlines()[1:]
                    if line.strip()
                ]

                self.assertEqual(commands_after_pipeline[0], "run_status=${PIPESTATUS[0]}")
                self.assertIn("ray stop --force || true", commands_after_pipeline)
                self.assertIn('exit "${run_status}"', commands_after_pipeline)

    def test_pipeline_status_round_trips_success_and_failure(self):
        python = shlex.quote(sys.executable)
        for expected_status in (0, 37):
            with self.subTest(expected_status=expected_status):
                result = subprocess.run(
                    [
                        "bash",
                        "-c",
                        (
                            f"{python} -c 'import sys; sys.exit({expected_status})' "
                            "2>&1 | tee /dev/null\n"
                            "run_status=${PIPESTATUS[0]}\n"
                            "true\n"
                            'exit "${run_status}"\n'
                        ),
                    ],
                    check=False,
                )
                self.assertEqual(result.returncode, expected_status)

    def test_eval_ignores_external_ppo_epochs(self):
        source = _EVAL_SCRIPT.read_text()
        self.assertIn("ppo_epochs=1", source)
        self.assertNotIn("${PPO_EPOCHS", source)


if __name__ == "__main__":
    unittest.main()
