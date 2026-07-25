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


_MODULE_PATH = Path(__file__).parents[3] / "verl/workers/actor/optimizer_step_plan.py"
_MODULE = runpy.run_path(str(_MODULE_PATH))
build_optimizer_step_plan = _MODULE["build_optimizer_step_plan"]
is_abs_on_policy_enabled = _MODULE["is_abs_on_policy_enabled"]
validate_actor_batch_partition = _MODULE["validate_actor_batch_partition"]


class TestOptimizerStepPlan(unittest.TestCase):
    def test_abs_on_policy_true_values(self):
        for value in ("1", "True", " t ", "YES", "y"):
            with self.subTest(value=value):
                self.assertTrue(is_abs_on_policy_enabled(value))

    def test_abs_on_policy_false_values(self):
        for value in (None, "", "0", "false", "no", "unexpected"):
            with self.subTest(value=value):
                self.assertFalse(is_abs_on_policy_enabled(value))

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


if __name__ == "__main__":
    unittest.main()
