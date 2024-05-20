import numpy as np

def revise_state(s):
    # Original state dimensions
    z_torso = s[0]
    torso_angle = s[1]
    joint_angles = s[2:8]  # All joint angles
    velocities = s[8:]  # All velocities
    
    # Calculate forward progress as the x-velocity of the torso
    forward_progress = s[8]
    
    # Calculate the torso's orientation with respect to the vertical axis
    torso_uprightness = np.cos(torso_angle)
    
    # Calculate the relative angles between adjacent body parts for symmetry
    relative_angles = np.diff(joint_angles)
    
    # Calculate the total kinetic energy (simplified, assuming unit mass)
    kinetic_energy = 0.5 * np.sum(velocities ** 2)
    
    # Calculate the potential energy (simplified, assuming unit mass and gravity)
    potential_energy = z_torso
    
    # Calculate the energy efficiency
    energy_efficiency = forward_progress / (kinetic_energy + 1e-6)  # Adding a small constant to avoid division by zero
    
    # Concatenate the original state with the new features
    updated_s = np.concatenate((s, [forward_progress, torso_uprightness, kinetic_energy, potential_energy, energy_efficiency] + list(relative_angles)))
    
    return updated_s

def intrinsic_reward(updated_s):
    # Extract the additional features from the updated state
    forward_progress = updated_s[17]
    torso_uprightness = updated_s[18]
    kinetic_energy = updated_s[19]
    potential_energy = updated_s[20]
    energy_efficiency = updated_s[21]
    
    # Design an intrinsic reward function that encourages forward progress and energy efficiency
    reward_forward_progress = forward_progress
    reward_energy_efficiency = energy_efficiency
    
    # Encourage the agent to maintain an upright torso
    reward_uprightness = torso_uprightness
    
    # Penalize excessive kinetic energy to encourage energy efficiency
    penalty_kinetic_energy = -0.01 * kinetic_energy
    
    # Combine the rewards and penalties into a single intrinsic reward value
    intrinsic_reward = reward_forward_progress + reward_energy_efficiency + reward_uprightness + penalty_kinetic_energy
    
    return intrinsic_reward

