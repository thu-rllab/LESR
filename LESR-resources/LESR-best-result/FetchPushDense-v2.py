import numpy as np

def revise_state(s):
    # Original state dimensions
    end_effector_pos = s[:3]  # x, y, z positions of the end effector
    block_pos = s[3:6]        # x, y, z positions of the block
    goal_block_pos = s[25:28]  # x, y, z positions of the final goal for the block

    # Calculate distances
    distance_to_goal = np.linalg.norm(block_pos - goal_block_pos)
    distance_gripper_to_block = np.linalg.norm(end_effector_pos - block_pos)

    # Calculate angles
    vector_gripper_to_block = block_pos - end_effector_pos
    vector_block_to_goal = goal_block_pos - block_pos
    angle_gripper_block_goal = np.arccos(
        np.dot(vector_gripper_to_block, vector_block_to_goal) /
        (np.linalg.norm(vector_gripper_to_block) * np.linalg.norm(vector_block_to_goal) + 1e-8)
    )

    # Calculate velocities
    gripper_velocity = s[20:23]
    block_velocity = s[14:17]
    relative_velocity = gripper_velocity - block_velocity

    # Calculate stability metric
    block_angular_velocity = s[17:20]
    stability_metric = 1.0 / (1.0 + np.linalg.norm(block_angular_velocity))

    # Concatenate the new features to the original state
    updated_s = np.concatenate((
        s,
        [distance_to_goal, distance_gripper_to_block, angle_gripper_block_goal, stability_metric],
        relative_velocity
    ))

    return updated_s

def intrinsic_reward(updated_s):
    # Extract the additional features from the updated state
    distance_to_goal = updated_s[28]
    distance_gripper_to_block = updated_s[29]
    angle_gripper_block_goal = updated_s[30]
    stability_metric = updated_s[31]
    relative_velocity = updated_s[32:35]

    # Define weights for the different components of the reward
    w_distance_to_goal = -1.0
    w_distance_gripper_to_block = -0.5
    w_angle = -0.5
    w_stability = 1.0
    w_relative_velocity = 0.1

    # Calculate the components of the reward
    reward_distance_to_goal = w_distance_to_goal * distance_to_goal
    reward_distance_gripper_to_block = w_distance_gripper_to_block * distance_gripper_to_block
    reward_angle = w_angle * angle_gripper_block_goal
    reward_stability = w_stability * stability_metric
    reward_relative_velocity = w_relative_velocity * np.linalg.norm(relative_velocity)

    # Combine the reward components
    intrinsic_reward = (
        reward_distance_to_goal +
        reward_distance_gripper_to_block +
        reward_angle +
        reward_stability +
        reward_relative_velocity
    )

    return intrinsic_reward

