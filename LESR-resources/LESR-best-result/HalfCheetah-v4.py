import numpy as np

def revise_state(s):
    # Retain the x-coordinate velocity of the front tip, which is highly correlated with reward
    forward_velocity_x = s[8]
    
    # Calculate the average angle of the joints, which can be a proxy for the overall body orientation
    average_joint_angle = np.mean(s[1:7])
    
    # Calculate the variance of the joint angles as a measure of stability
    joint_angle_variance = np.var(s[1:7])
    
    # Calculate the energy expenditure as the sum of squared torques
    energy_expenditure = np.sum(np.square(s[4:8]))
    
    # Calculate the forward momentum, which is a product of mass, velocity, and cosine of the orientation angle
    # Assuming a unit mass for simplicity
    forward_momentum = forward_velocity_x * np.cos(average_joint_angle)
    
    # Update the state with the new features
    updated_s = np.concatenate((s, [forward_velocity_x, average_joint_angle, joint_angle_variance, energy_expenditure, forward_momentum]))
    
    return updated_s

def intrinsic_reward(updated_s):
    # Define weights for different aspects of the reward
    forward_velocity_weight = 1.0
    energy_efficiency_weight = -0.001
    stability_weight = -0.01
    momentum_weight = 0.5  # Encourage maintaining forward momentum
    
    # Extract the relevant features from the updated state
    forward_velocity = updated_s[17]  # This is the forward_velocity_x feature
    joint_angle_variance = updated_s[19]
    energy_expenditure = updated_s[20]
    forward_momentum = updated_s[21]
    
    # Calculate the intrinsic reward components
    forward_velocity_reward = forward_velocity_weight * forward_velocity
    stability_penalty = stability_weight * joint_angle_variance
    energy_penalty = energy_efficiency_weight * energy_expenditure
    momentum_reward = momentum_weight * forward_momentum
    
    # Sum the components to get the total intrinsic reward
    intrinsic_reward = forward_velocity_reward + stability_penalty + energy_penalty + momentum_reward
    
    return intrinsic_reward

