import numpy as np
from typing import List

def calculate_redaction_reward(
    predicted_mask: List[bool], 
    ground_truth_mask: List[bool],
    **kwargs
) -> float:
    """
    Calculates the reward strictly in the [0.0, 1.0] range based on accuracy.
    """
    if len(predicted_mask) != len(ground_truth_mask) or len(ground_truth_mask) == 0:
        return 0.0
        
    correct_tokens = sum(
        1 for pred, truth in zip(predicted_mask, ground_truth_mask) if pred == truth
    )
    
    # Return normalized score [0.0, 1.0]
    return float(correct_tokens) / len(ground_truth_mask)

def calculate_grpo_rewards(
    group_predicted_masks: List[List[bool]], 
    ground_truth_mask: List[bool],
    **kwargs
) -> List[float]:
    """
    Calculates Group Relative Policy Optimization (GRPO) rewards for a group of predictions.
    
    This function computes the raw redaction rewards for every rollout in the group, 
    and applies GRPO scoring (zero-mean, unit-variance normalization) so that the 
    agent learns relative to the group's average performance.
    """
    raw_rewards = [calculate_redaction_reward(mask, ground_truth_mask, **kwargs) for mask in group_predicted_masks]
    
    if len(raw_rewards) <= 1:
        # Cannot normalize a group of 1 or fewer
        return raw_rewards
        
    mean_reward = np.mean(raw_rewards)
    std_reward = np.std(raw_rewards)
    
    if std_reward < 1e-8:
        # All actions resulted in the same reward
        return [0.0 for _ in raw_rewards]
        
    grpo_rewards = [(r - mean_reward) / std_reward for r in raw_rewards]
    return grpo_rewards
