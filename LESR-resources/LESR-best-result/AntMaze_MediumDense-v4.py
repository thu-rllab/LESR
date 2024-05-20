import numpy as np

def revise_state(s):
    # Extract the original state components
    torso_pos = s[0:3]  # x, y, z position of the torso
    torso_orient = s[1:5]  # orientation of the torso (quaternion or Euler angles)
    joint_angles = s[5:13]  # angles of the joints
    torso_vel = s[13:16]  # velocity of the torso
    joint_velocities = s[19:27]  # angular velocities of the joints
    goal_pos = s[27:29]  # x, y position of the goal
    
    # Calculate additional features
    # Distance to the goal (Euclidean distance in the x-y plane)
    distance_to_goal = np.linalg.norm(torso_pos[:2] - goal_pos)
    
    # Orientation towards the goal
    goal_direction = np.arctan2(goal_pos[1] - torso_pos[1], goal_pos[0] - torso_pos[0])
    torso_direction = np.arctan2(np.sin(torso_orient[2]), np.cos(torso_orient[2]))  # Assuming z-orientation is the forward direction
    orientation_to_goal = np.cos(torso_direction - goal_direction)
    
    # Coordination measure (e.g., standard deviation of joint angles)
    coordination = np.std(joint_angles)
    
    # Leg movement smoothness (e.g., variance of joint angular velocities)
    smoothness = np.var(joint_velocities)
    
    # Concatenate the new features to the original state
    updated_s = np.concatenate((s, [distance_to_goal, orientation_to_goal, coordination, smoothness]))
    
    return updated_s

def intrinsic_reward(updated_s):
    # Extract the additional features from the updated state
    distance_to_goal = updated_s[29]
    orientation_to_goal = updated_s[30]
    coordination = updated_s[31]
    smoothness = updated_s[32]
    
    # Define weights for the different components of the intrinsic reward
    weight_distance = -0.5  # Negative because we want to minimize distance
    weight_orientation = 0.3  # Positive because we want to face the goal
    weight_coordination = 0.2  # Positive because we want to maximize coordination
    weight_smoothness = 0.1  # Positive because we want smooth leg movements
    
    # Calculate the intrinsic reward
    intrinsic_reward_value = (
        weight_distance * distance_to_goal +
        weight_orientation * orientation_to_goal +
        weight_coordination * coordination +
        weight_smoothness * smoothness
    )
    
    # Prioritize coordination learning by adding a bonus if coordination is below a threshold
    coordination_threshold = 0.5
    if coordination < coordination_threshold:
        intrinsic_reward_value += 5.0 * (coordination_threshold - coordination)
    
    return intrinsic_reward_value

