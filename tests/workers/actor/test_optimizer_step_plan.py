# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import runpy
import unittest
from pathlib import Path

import torch


_MODULE_PATH = Path(__file__).parents[3] / "verl/workers/actor/optimizer_step_plan.py"
_MODULE = runpy.run_path(str(_MODULE_PATH))
build_optimizer_step_plan = _MODULE["build_optimizer_step_plan"]
is_abs_on_policy_enabled = _MODULE["is_abs_on_policy_enabled"]
validate_actor_batch_partition = _MODULE["validate_actor_batch_partition"]
validate_abs_on_policy_rollout_compatibility = _MODULE[
    "validate_abs_on_policy_rollout_compatibility"
]


class TestOptimizerStepPlan(unittest.TestCase):
    def test_abs_on_policy_true_values(self):
        for value in ("1", "True", "t", "YES", "y"):
            with self.subTest(value=value):
                self.assertTrue(is_abs_on_policy_enabled(value))

    def test_abs_on_policy_false_values(self):
        for value in (None, "", "0", "false", "no", "unexpected", " t ", " true "):
            with self.subTest(value=value):
                self.assertFalse(is_abs_on_policy_enabled(value))

    def test_abs_on_policy_allows_decoupled_rollout_mode(self):
        validate_abs_on_policy_rollout_compatibility(
            abs_on_policy=True,
            bypass_mode=False,
        )

    def test_regular_ppo_allows_rollout_bypass_mode(self):
        validate_abs_on_policy_rollout_compatibility(
            abs_on_policy=False,
            bypass_mode=True,
        )

    def test_abs_on_policy_rejects_rollout_bypass_mode(self):
        with self.assertRaisesRegex(
            ValueError,
            "ABS_ON_POLICY is incompatible with "
            "algorithm.rollout_correction.bypass_mode=True",
        ):
            validate_abs_on_policy_rollout_compatibility(
                abs_on_policy=True,
                bypass_mode=True,
            )

    def test_regular_ppo_steps_and_clears_each_mini_batch(self):
        plan = build_optimizer_step_plan(abs_on_policy=False, num_mini_batches=16, ppo_epochs=2)

        self.assertFalse(plan.zero_grad_before_update)
        self.assertTrue(plan.zero_grad_before_mini_batch)
        self.assertTrue(plan.step_after_mini_batch)
        self.assertFalse(plan.step_after_update)
        self.assertEqual(plan.outer_gradient_accumulation_steps, 1)

    def test_abs_on_policy_accumulates_one_normalized_full_batch_update(self):
        plan = build_optimizer_step_plan(abs_on_policy=True, num_mini_batches=16, ppo_epochs=1)

        self.assertTrue(plan.zero_grad_before_update)
        self.assertFalse(plan.zero_grad_before_mini_batch)
        self.assertFalse(plan.step_after_mini_batch)
        self.assertTrue(plan.step_after_update)
        self.assertEqual(plan.outer_gradient_accumulation_steps, 16)

    def test_optimizer_step_plan_rejects_empty_updates(self):
        for num_mini_batches, ppo_epochs in ((0, 1), (1, 0)):
            with self.subTest(num_mini_batches=num_mini_batches, ppo_epochs=ppo_epochs):
                with self.assertRaises(ValueError):
                    build_optimizer_step_plan(
                        abs_on_policy=True,
                        num_mini_batches=num_mini_batches,
                        ppo_epochs=ppo_epochs,
                    )

    def test_abs_on_policy_requires_one_ppo_epoch(self):
        with self.assertRaisesRegex(ValueError, "requires ppo_epochs == 1"):
            build_optimizer_step_plan(abs_on_policy=True, num_mini_batches=16, ppo_epochs=2)

    def test_actor_batch_partition_returns_exact_mini_batch_count(self):
        self.assertEqual(
            validate_actor_batch_partition(local_batch_size=4, ppo_mini_batch_size=1),
            4,
        )
        self.assertEqual(
            validate_actor_batch_partition(local_batch_size=4, ppo_mini_batch_size=4),
            1,
        )

    def test_actor_batch_partition_rejects_remainder(self):
        with self.assertRaisesRegex(ValueError, "must be divisible"):
            validate_actor_batch_partition(local_batch_size=5, ppo_mini_batch_size=2)

    def test_abs_on_policy_accumulates_the_mean_gradient(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=1.0)
        coefficients = (2.0, 4.0, 6.0, 8.0)
        plan = build_optimizer_step_plan(
            abs_on_policy=True,
            num_mini_batches=len(coefficients),
            ppo_epochs=1,
        )

        gradient_reset_count = 0
        optimizer_step_count = 0
        if plan.zero_grad_before_update:
            optimizer.zero_grad()
            gradient_reset_count += 1

        for coefficient in coefficients:
            if plan.zero_grad_before_mini_batch:
                optimizer.zero_grad()
                gradient_reset_count += 1
            loss = parameter * coefficient / plan.outer_gradient_accumulation_steps
            loss.backward()
            if plan.step_after_mini_batch:
                optimizer.step()
                optimizer_step_count += 1

        if plan.step_after_update:
            optimizer.step()
            optimizer_step_count += 1

        self.assertAlmostEqual(parameter.item(), -4.0)
        self.assertEqual(gradient_reset_count, 1)
        self.assertEqual(optimizer_step_count, 1)

    def test_regular_ppo_steps_once_per_mini_batch(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=1.0)
        coefficients = (2.0, 4.0, 6.0, 8.0)
        plan = build_optimizer_step_plan(
            abs_on_policy=False,
            num_mini_batches=len(coefficients),
            ppo_epochs=1,
        )

        gradient_reset_count = 0
        optimizer_step_count = 0
        for coefficient in coefficients:
            if plan.zero_grad_before_mini_batch:
                optimizer.zero_grad()
                gradient_reset_count += 1
            (parameter * coefficient).backward()
            if plan.step_after_mini_batch:
                optimizer.step()
                optimizer_step_count += 1

        self.assertAlmostEqual(parameter.item(), -19.0)
        self.assertEqual(gradient_reset_count, len(coefficients))
        self.assertEqual(optimizer_step_count, len(coefficients))


if __name__ == "__main__":
    unittest.main()
