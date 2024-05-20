import numpy as np

def revise_state(s):
    # Original state dimensions
    original_state = s[:29]
    
    # Calculate the distance to the goal
    goal_distance = np.sqrt((s[27] - s[0])**2 + (s[28] - s[1])**2)
    
    # Calculate the stability metric (e.g., deviation from the upright orientation)
    stability_metric = 1 - abs(s[1]) - abs(s[2]) - abs(s[3]) - abs(s[4])
    
    # Calculate the coordination metric (e.g., variance of the angles and angular velocities)
    angles = s[5:13]
    angular_velocities = s[19:27]
    coordination_metric = np.var(angles) + np.var(angular_velocities)
    
    # Concatenate the new features to the original state
    updated_s = np.concatenate((original_state, [goal_distance, stability_metric, coordination_metric]))
    
    return updated_s

def intrinsic_reward(updated_s):
    # Extract the new features from the updated state
    goal_distance = updated_s[29]
    stability_metric = updated_s[30]
    coordination_metric = updated_s[31]
    
    # Define weights for the different components of the intrinsic reward
    weight_distance = -1.0  # Negative because we want to minimize distance
    weight_stability = 1.0  # Positive because we want to maximize stability
    weight_coordination = 1.0  # Positive because we want to maximize coordination
    
    # Calculate the intrinsic reward as a weighted sum of the new features
    intrinsic_reward = (weight_distance * goal_distance +
                        weight_stability * stability_metric +
                        weight_coordination * coordination_metric)
    
    return intrinsic_reward

