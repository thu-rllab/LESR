import numpy as np

# Global variable to store the previous state for velocity calculation
previous_s = None

def revise_state(s):
    global previous_s
    updated_s = np.copy(s)
    
    # Calculate joint velocities and accelerations if previous state is available
    if previous_s is not None:
        velocities = s[:28] - previous_s[:28]
        accelerations = velocities - np.concatenate(([0], velocities[:-1]))
        updated_s = np.concatenate([updated_s, velocities, accelerations])
    else:
        # Initialize velocities and accelerations to zero if this is the first state
        velocities = np.zeros(28)
        accelerations = np.zeros(28)
        updated_s = np.concatenate([updated_s, velocities, accelerations])
    
    # Calculate the relative orientation between the palm and the door handle
    palm_to_handle_vector = s[35:38]
    palm_to_handle_distance = np.linalg.norm(palm_to_handle_vector)
    if palm_to_handle_distance > 0:
        palm_to_handle_orientation = palm_to_handle_vector / palm_to_handle_distance
    else:
        palm_to_handle_orientation = np.zeros(3)
    
    # Concatenate the new features to the original state
    updated_s = np.concatenate((updated_s, [palm_to_handle_distance], palm_to_handle_orientation))
    
    # Update previous state
    previous_s = np.copy(s)
    
    return updated_s

def intrinsic_reward(updated_s):
    # Extract the additional features from the updated state
    velocities = updated_s[39:67]  # Joint velocities
    accelerations = updated_s[67:95]  # Joint accelerations
    palm_to_handle_distance = updated_s[95]  # Distance from palm to handle
    palm_to_handle_orientation = updated_s[96:99]  # Orientation of palm to handle
    
    # Define weights for different aspects of the intrinsic reward
    weight_velocity = -0.1  # Penalize high velocities to encourage smooth movements
    weight_acceleration = -0.05  # Penalize high accelerations to encourage stable movements
    weight_distance = -1.0  # Penalize distance to handle to encourage reaching the goal
    weight_orientation = 1.0  # Reward proper orientation of palm to handle
    
    # Calculate the intrinsic reward components
    velocity_penalty = weight_velocity * np.sum(np.abs(velocities))
    acceleration_penalty = weight_acceleration * np.sum(np.abs(accelerations))
    distance_penalty = weight_distance * palm_to_handle_distance
    orientation_reward = weight_orientation * np.dot(palm_to_handle_orientation, np.array([0, 0, 1]))  # Assuming the handle is in the positive z direction
    
    # Sum the components to get the total intrinsic reward
    total_intrinsic_reward = velocity_penalty + acceleration_penalty + distance_penalty + orientation_reward
    
    return total_intrinsic_reward

