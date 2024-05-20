import numpy as np

def revise_state(s):
    # Original state dimensions
    z_torso = s[0]
    angle_torso = s[1]
    angle_thigh = s[2]
    angle_leg = s[3]
    angle_foot = s[4]
    vel_x_torso = s[5]
    vel_z_torso = s[6]
    angular_vel_torso = s[7]
    angular_vel_thigh = s[8]
    angular_vel_leg = s[9]
    angular_vel_foot = s[10]
    
    # Calculate relative angles between body parts
    relative_angle_thigh_torso = angle_thigh - angle_torso
    relative_angle_leg_thigh = angle_leg - angle_thigh
    relative_angle_foot_leg = angle_foot - angle_leg
    
    # Calculate the overall velocity of the hopper
    overall_velocity = np.sqrt(vel_x_torso**2 + vel_z_torso**2)
    
    # Calculate a measure of stability (e.g., deviation from the vertical)
    stability_measure = np.abs(angle_torso)
    
    # Calculate energy efficiency (e.g., sum of absolute torques)
    energy_efficiency = np.abs(angular_vel_torso) + np.abs(angular_vel_thigh) + np.abs(angular_vel_leg) + np.abs(angular_vel_foot)
    
    # Calculate the forward distance covered (assuming initial x position is 0)
    forward_distance = vel_x_torso  # Assuming constant velocity and unit time step
    
    # Concatenate the new features to the original state
    updated_s = np.concatenate((s, [
        relative_angle_thigh_torso,
        relative_angle_leg_thigh,
        relative_angle_foot_leg,
        overall_velocity,
        stability_measure,
        energy_efficiency,
        forward_distance
    ]))
    
    return updated_s

def intrinsic_reward(updated_s):
    # Use the extra dimensions from the updated state for the intrinsic reward
    overall_velocity = updated_s[14]
    stability_measure = updated_s[15]
    energy_efficiency = updated_s[16]
    forward_distance = updated_s[17]
    
    # Design an intrinsic reward function that encourages forward movement, stability, and energy efficiency
    reward_forward_movement = overall_velocity  # Encourage forward movement
    reward_stability = -stability_measure  # Penalize instability
    reward_energy_efficiency = -0.1 * energy_efficiency  # Penalize high energy use
    
    # Combine the rewards into a single intrinsic reward value
    intrinsic_reward = reward_forward_movement + 2 * forward_distance + reward_stability + reward_energy_efficiency
    
    return intrinsic_reward

