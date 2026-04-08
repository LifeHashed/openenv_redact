import numpy as np
from typing import List

def calculate_redaction_reward(
    predicted_mask: List[bool], 
    ground_truth_mask: List[bool],
    tp_reward: float = 1.0,
    fn_penalty: float = -5.0,
    fp_penalty: float = -1.0,
    tn_reward: float = 0.1
) -> float:
    """
    Calculates the reward for a given token-level redaction attempt.
    
    The reward structure is designed to strictly enforce privacy:
    - Failing to redact PII (False Negative) is heavily penalized.
    - Over-redacting safe text (False Positive) is penalized, but less severely.
    - Successfully redacting PII (True Positive) gives a positive reward.
    - Correctly keeping safe text (True Negative) gives a small positive reward.
    
    Args:
        predicted_mask: The boolean mask predicted by the agent.
        ground_truth_mask: The actual base truth boolean mask.
        tp_reward: Reward for correctly redacting a sensitive token.
        fn_penalty: Heavy penalty for missing a sensitive token (privacy violation).
        fp_penalty: Penalty for over-redacting (loss of utility).
        tn_reward: Small reward for correctly leaving safe text alone.
        
    Returns:
        A float representing the total reward for this action.
    """
    if len(predicted_mask) != len(ground_truth_mask):
        # Return a massive penalty if the mask size doesn't match the token length
        return -20.0
        
    total_reward = 0.0
    
    for pred, truth in zip(predicted_mask, ground_truth_mask):
        if pred and truth:
            # True Positive: Correctly redacted sensitive data
            total_reward += tp_reward
        elif not pred and truth:
            # False Negative: Missed PII! Very bad.
            total_reward += fn_penalty
        elif pred and not truth:
            # False Positive: Over-redacted safe text.
            total_reward += fp_penalty
        elif not pred and not truth:
            # True Negative: Correctly left safe text alone.
            total_reward += tn_reward
            
    return total_reward

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
