# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Redaction Env Environment Implementation.

A simple test environment that echoes back messages sent to it.
Perfect for testing HTTP server infrastructure.
"""

from uuid import uuid4
import random
from typing import Optional, Any

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import RedactionAction, RedactionObservation
    from ..graders import GraderDataset
    from ..reward import calculate_redaction_reward
except ImportError:
    from models import RedactionAction, RedactionObservation
    from graders import GraderDataset
    from reward import calculate_redaction_reward


class RedactionEnvironment(Environment):
    """
    A contextual PII redaction environment.

    This environment evaluates an agent's ability to redact PII from text
    based on context clues without over-redacting safe information.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize the redaction_env environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0
        self.dataset = GraderDataset()
        self.current_item = None

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> RedactionObservation:
        """
        Reset the environment and return a new text snippet to redact.
        """
        if seed is not None:
            random.seed(seed)
            
        ep_id = episode_id if episode_id is not None else str(uuid4())
        self._state = State(episode_id=ep_id, step_count=0)
        self._reset_count += 1
        
        # Pick a random difficulty tier for each episode
        difficulty = random.choice(["easy", "medium", "hard"])
        self.current_item = self.dataset.get_sample(difficulty)

        return RedactionObservation(
            tokens=self.current_item.tokens,
            is_public_record=self.current_item.is_public_record,
            context_info=self.current_item.context_info,
            done=False,
            reward=0.0,
        )

    def step(self, action: RedactionAction) -> RedactionObservation:  # type: ignore[override]
        """
        Evaluate the agent's redaction mask and return the reward.
        """
        self._state.step_count += 1
        
        if not self.current_item:
            # Failsafe if step called before reset
            return self.reset()

        predicted_mask = action.redact_mask
        
        # Calculate the reward based on the predicted mask vs ground truth
        reward = calculate_redaction_reward(
            predicted_mask=predicted_mask, 
            ground_truth_mask=self.current_item.ground_truth_mask
        )

        return RedactionObservation(
            tokens=self.current_item.tokens,
            is_public_record=self.current_item.is_public_record,
            context_info=self.current_item.context_info,
            done=True,  # Evaluation is a single-step episode
            reward=reward,
            metadata={
                "step": self._state.step_count,
                "ground_truth": self.current_item.ground_truth_mask,
                "predicted": predicted_mask
            },
        )

    @property
    def state(self) -> State:
        """
        Get the current environment state.

        Returns:
            Current State with episode_id and step_count
        """
        return self._state
