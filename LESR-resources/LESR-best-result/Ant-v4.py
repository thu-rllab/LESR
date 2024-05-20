import numpy as np

def revise_state(s):
    # Original state dimensions
    original_state_dim = s.shape[0]
    
    # Forward progress (x-coordinate velocity of the torso)
    forward_velocity = s[13]
    
    # Center of mass height (z-coordinate of the torso)
    com_height = s[0]
    
    # Torso orientation stability (variance of the torso orientation angles)
    orientation_stability = np.var(s[1:5])
    
    # Leg coordination (standard deviation of the angular velocities of the legs)
    leg_coordination = np.std(s[19:27])
    
    # Energy efficiency (sum of squared torques applied on the hinges, using angles as a proxy)
    energy_efficiency = np.sum(np.square(s[5:13]))
    
    # Leg clearance (minimum angle between the torso and the legs)
    leg_clearance = np.min(s[5:13])
    
    # Concatenate the new features to the original state
    updated_s = np.concatenate((s, [
        forward_velocity, com_height, orientation_stability, 
        leg_coordination, energy_efficiency, leg_clearance
    ]))
    
    return updated_s

def intrinsic_reward(updated_s):
    # Extract the additional features from the updated state
    forward_velocity = updated_s[27]
    com_height = updated_s[28]
    orientation_stability = updated_s[29]
    leg_coordination = updated_s[30]
    energy_efficiency = updated_s[31]
    leg_clearance = updated_s[32]
    
    # Define weights for each component of the reward
    velocity_weight = 2.0  # Encourage forward movement
    height_weight = 0.1  # Encourage a higher center of mass for better visibility and obstacle avoidance
    stability_weight = -0.5  # Penalize orientation instability
    coordination_weight = -0.1  # Penalize poor leg coordination
    energy_weight = -0.01  # Penalize low energy efficiency
    clearance_weight = 0.5  # Reward higher leg clearance
    
    # Calculate the intrinsic reward
    intrinsic_reward = (
        velocity_weight * forward_velocity +
        height_weight * com_height +
        stability_weight * orientation_stability +
        coordination_weight * leg_coordination +
        energy_weight * energy_efficiency +
        clearance_weight * leg_clearance
    )
    
    return intrinsic_reward

