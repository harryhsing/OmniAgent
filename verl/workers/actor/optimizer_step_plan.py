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

from dataclasses import dataclass


_TRUE_ENV_VALUES = frozenset({"1", "true", "t", "yes", "y"})


def is_abs_on_policy_enabled(value: str | None) -> bool:
    """Parse ABS_ON_POLICY using the same values accepted by common env parsers."""
    return value is not None and value.lower() in _TRUE_ENV_VALUES


def validate_abs_on_policy_rollout_compatibility(*, abs_on_policy: bool, bypass_mode: bool) -> None:
    """Reject rollout modes that require conflicting old-log-probability anchors."""
    if abs_on_policy and bypass_mode:
        raise ValueError(
            "ABS_ON_POLICY is incompatible with "
            "algorithm.rollout_correction.bypass_mode=True"
        )


@dataclass(frozen=True)
class OptimizerStepPlan:
    zero_grad_before_update: bool
    zero_grad_before_mini_batch: bool
    step_after_mini_batch: bool
    step_after_update: bool
    outer_gradient_accumulation_steps: int


def validate_actor_batch_partition(*, local_batch_size: int, ppo_mini_batch_size: int) -> int:
    """Validate actor mini-batch partitioning and return the number of mini-batches."""
    if local_batch_size <= 0:
        raise ValueError(f"local_batch_size must be positive, got {local_batch_size}")
    if ppo_mini_batch_size <= 0:
        raise ValueError(f"ppo_mini_batch_size must be positive, got {ppo_mini_batch_size}")
    if local_batch_size % ppo_mini_batch_size != 0:
        raise ValueError(
            "Local actor batch must be divisible by PPO mini-batch size: "
            f"local_batch_size={local_batch_size}, "
            f"ppo_mini_batch_size={ppo_mini_batch_size}"
        )
    return local_batch_size // ppo_mini_batch_size


def build_optimizer_step_plan(
    *,
    abs_on_policy: bool,
    num_mini_batches: int,
    ppo_epochs: int,
) -> OptimizerStepPlan:
    """Describe optimizer control flow for regular PPO or full-batch ABS updates."""
    if num_mini_batches <= 0:
        raise ValueError(f"num_mini_batches must be positive, got {num_mini_batches}")
    if ppo_epochs <= 0:
        raise ValueError(f"ppo_epochs must be positive, got {ppo_epochs}")

    if abs_on_policy:
        if ppo_epochs != 1:
            raise ValueError(f"ABS_ON_POLICY requires ppo_epochs == 1, got {ppo_epochs}")

        return OptimizerStepPlan(
            zero_grad_before_update=True,
            zero_grad_before_mini_batch=False,
            step_after_mini_batch=False,
            step_after_update=True,
            outer_gradient_accumulation_steps=num_mini_batches,
        )

    return OptimizerStepPlan(
        zero_grad_before_update=False,
        zero_grad_before_mini_batch=True,
        step_after_mini_batch=True,
        step_after_update=False,
        outer_gradient_accumulation_steps=1,
    )
