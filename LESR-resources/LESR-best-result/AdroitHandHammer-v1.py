import numpy as np

def revise_state(s):
    # Calculate additional features to enhance state representation
    
    # Distance between hammer's center of mass and the nail
    hammer_nail_distance = np.linalg.norm(s[36:39] - s[42:45])
    
    # Alignment of the hammer with the nail (cosine of the angle between them)
    hammer_orientation = s[39:42]
    nail_direction = np.array([0, 0, -1])  # Assuming the nail should be driven straight down
    alignment_cosine = np.dot(hammer_orientation, nail_direction) / (np.linalg.norm(hammer_orientation) * np.linalg.norm(nail_direction))
    
    # Coordination of the fingers (standard deviation of finger joint angles)
    finger_joint_angles = s[4:26]
    finger_coordination = np.std(finger_joint_angles)
    
    # Coordination of the arm (standard deviation of arm joint angles)
    arm_joint_angles = s[0:4]
    arm_coordination = np.std(arm_joint_angles)
    
    # Velocity of the hammer head towards the nail
    hammer_velocity = s[27:30]
    velocity_towards_nail = np.dot(hammer_velocity, nail_direction)
    
    # Concatenate the new features to the original state
    updated_s = np.concatenate((s, [hammer_nail_distance, alignment_cosine, finger_coordination, arm_coordination, velocity_towards_nail]))
    
    return updated_s

def intrinsic_reward(updated_s):
    # Define weights for different components of the intrinsic reward
    weight_distance = -1.0  # Negative reward for greater distances
    weight_alignment = 2.0  # Positive reward for better alignment
    weight_finger_coordination = 1.0  # Positive reward for better finger coordination
    weight_arm_coordination = 1.0  # Positive reward for better arm coordination
    weight_velocity_towards_nail = 1.5  # Positive reward for velocity towards the nail
    
    # Calculate the intrinsic reward components
    reward_distance = weight_distance * updated_s[46]
    reward_alignment = weight_alignment * updated_s[47]
    reward_finger_coordination = weight_finger_coordination * updated_s[48]
    reward_arm_coordination = weight_arm_coordination * updated_s[49]
    reward_velocity_towards_nail = weight_velocity_towards_nail * updated_s[50]
    
    # Sum up the components to get the total intrinsic reward
    intrinsic_reward_value = (reward_distance + reward_alignment +
                              reward_finger_coordination + reward_arm_coordination +
                              reward_velocity_towards_nail)
    
    # Encourage coordination learning before reaching the goal
    if updated_s[26] < 0.1:  # Assuming the nail starts with 0 insertion
        intrinsic_reward_value += 2.0 * (reward_finger_coordination + reward_arm_coordination)
    else:
        # Once the nail starts being driven in, balance between coordination and progress
        intrinsic_reward_value += 0.5 * (reward_finger_coordination + reward_arm_coordination)
    
    # Penalize for excessive force
    force_exerted = updated_s[45]
    friction_threshold = 15  # Maximum force before slipping occurs
    if force_exerted > friction_threshold:
        intrinsic_reward_value -= (force_exerted - friction_threshold) ** 2  # Quadratic penalty for excessive force
    
    return intrinsic_reward_value

