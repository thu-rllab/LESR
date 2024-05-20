import numpy as np
import torch
import gymnasium as gym
import argparse
import os

import utils
import TD3

import importlib
import sys 

def import_and_reload_module(module_name):
    if module_name in sys.modules: 
        del sys.modules[module_name] 
    imported_module = importlib.import_module(module_name) 
    return imported_module

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
### gymnasium-robotics ###
def eval_policy(policy, env_name, seed, eval_episodes=10):
	if 'antmaze' in args.env.lower():
		eval_env = gym.make(args.env, continuing_task = False, reset_target = False)
	else:
		eval_env = gym.make(args.env) 

	avg_reward = 0.
	if 'antmaze' in args.env.lower() or 'fetch' in args.env.lower() or 'adroit' in args.env.lower():
		eval_episodes = 50
		success_rate = 0
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		while not done:
			if type(state) == tuple: state = state[0]
			if type(state) == type({}): 
				state = np.concatenate([state['observation'].reshape([1, -1]), state['desired_goal'].reshape([1, -1])], axis=1)
			
			action = policy.select_action(revise_state(np.array(state).flatten()).reshape([1, -1]))
			state, reward, terminated, truncated, info = eval_env.step(action)

			if 'fetch' in args.env.lower() and abs(reward) < 0.05:
				terminated = True
			
			if 'adroithanddoor' in args.env.lower() and abs(reward) > 0.9 * 20:
				terminated = True
			
			if 'adroithandhammer' in args.env.lower() and abs(reward) > 0.9 * 25:
				terminated = True 
			
			if 'antmaze' in args.env.lower() or 'fetch' in args.env.lower() or 'adroit' in args.env.lower():
				if terminated: success_rate += 1

			done = truncated or terminated

			avg_reward += reward
	
	if 'antmaze' in args.env.lower() or 'fetch' in args.env.lower() or 'adroit' in args.env.lower():
		success_rate /= eval_episodes 

	avg_reward /= eval_episodes

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")

	if 'antmaze' in args.env.lower() or 'fetch' in args.env.lower() or 'adroit' in args.env.lower():
		return success_rate
	return avg_reward
### gymnasium-robotics ###

def cal_lipschitz(state_change, reward_change):
	lipschitz = np.zeros([state_dim, ])
	for ii in range(state_dim):
		cur_index = np.argsort(state_change[ii])

		cur_s, cur_r = state_change[ii].copy(), reward_change.squeeze().copy()
		cur_s, cur_r = cur_s[cur_index], cur_r[cur_index]

		cur_lipschitz = np.abs( (cur_r[:-1] - cur_r[1:]) ) / (np.abs( (cur_s[:-1] - cur_s[1:]) ) + 1e-2)
		
		lipschitz[ii] = cur_lipschitz.max()
	
	return lipschitz

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--policy", default="TD3")                  # Policy name (TD3, DDPG or OurDDPG)
	parser.add_argument("--env", default="HalfCheetah-v4")          # OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--start_timesteps", default=25e3, type=int)# Time steps initial random policy is used
	parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
	parser.add_argument("--expl_noise", default=0.1, type=float)    # Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99, type=float)     # Discount factor
	parser.add_argument("--tau", default=0.005, type=float)         # Target network update rate
	parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
	parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
	parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name

	parser.add_argument("--revise_path", default="", type=str)      # path of the revise function
	parser.add_argument("--version", default="", type=str)          # path of the revise function

	parser.add_argument("--sid_result_path", default="", type=str)  # path of the revise function
	parser.add_argument("--corr_result_path", default="", type=str)  # path of the revise function
	parser.add_argument("--corr_tau", default=0.005, type=float)    # Corr Update rate
	parser.add_argument("--intrinsic_w", default=0.001, type=float) # intrinsic_reward_w 
	parser.add_argument("--eval", default=0, type=int) # intrinsic_reward_w 

	args = parser.parse_args()

	args.save_model = True

	file_name = f"{args.version}_seed{args.seed}"

	if not os.path.exists("./results"):
		os.makedirs("./results")

	if args.save_model and not os.path.exists("./models"):
		os.makedirs("./models")

	### gymnasium-robotics ###
	if 'antmaze' in args.env.lower():
		env = gym.make(args.env, continuing_task = False, reset_target = False)
	else: 
		env = gym.make(args.env)
	### gymnasium-robotics ###

	# Set seeds
	env.action_space.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	
	#####################################################################################################
	########################### set state dim and function##########################################
	if args.eval == 1:
		args.revise_path = f'LESR-resources.run-v{args.version}-{args.env}.best_result.v{args.version}-best-{args.env}'
		print('#' * 20)
		print('Evaluate Stage:', args.revise_path)
		print('#' * 20)
	### gym - robotics ###
	if type(env.observation_space) == gym.spaces.Dict:
		src_state_dim = env.observation_space['observation'].shape[0] + env.observation_space['desired_goal'].shape[0]
	else: src_state_dim = env.observation_space.shape[0]
	### gym - robotics ###
	action_dim = env.action_space.shape[0] 
	max_action = float(env.action_space.high[0])
	revise_state = import_and_reload_module(args.revise_path).revise_state
	intrinsic_reward = import_and_reload_module(args.revise_path).intrinsic_reward

	test_state = np.zeros([src_state_dim, ])
	state_dim = revise_state(test_state).shape[0]
	#####################################################################################################
	#####################################################################################################
	
	print("-----------------------------------------------")
	print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}, State Dim: {state_dim}")
	print("-----------------------------------------------")

	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": args.discount,
		"tau": args.tau,
	}

	# Initialize policy
	if args.policy == "TD3":
		# Target policy smoothing is scaled wrt the action scale
		kwargs["policy_noise"] = args.policy_noise * max_action
		kwargs["noise_clip"] = args.noise_clip * max_action
		kwargs["policy_freq"] = args.policy_freq
		policy = TD3.TD3(**kwargs) 

	if args.load_model != "":
		policy_file = file_name if args.load_model == "default" else args.load_model
		policy.load(f"./models/{policy_file}")

	replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
	
	# Evaluate untrained policy
	### gymnasium-robotics ###
	t = 0
	### gymnasium-robotics ###
	evaluations = [eval_policy(policy, args.env, args.seed)]
	evaluations_steps = [[evaluations[-1], 0]]
	intrinsic_ratio = []

	### init correlation ###
	soft_state_correlation = np.zeros([state_dim, ])

	state, done = env.reset(), False

	if type(state) == tuple: state = state[0]
	### gymnasium-robotics ###
	if type(state) == type({}): 
		state = np.concatenate([state['observation'].reshape([1, -1]), state['desired_goal'].reshape([1, -1])], axis=1)
	### gymnasium-robotics ###

	episode_reward, episode_intrinsic_reward = 0, 0
	episode_timesteps = 0
	episode_num = 0
	episode_all_state, episode_all_reward = [], []

	for t in range(int(args.max_timesteps)):
		
		episode_timesteps += 1

		# Select action randomly or according to policy
		if t < args.start_timesteps:
			action = env.action_space.sample()
		else:
			action = (
				policy.select_action(revise_state(np.array(state).flatten()).reshape([1, -1]))
				+ np.random.normal(0, max_action * args.expl_noise, size=action_dim)
			).clip(-max_action, max_action)

		# Perform action
		next_state, reward, terminated, truncated, info = env.step(action)

		### save state and reward ###
		episode_all_state.append(revise_state(np.array(state).flatten()).reshape([-1, 1]))
		episode_all_reward.append(reward)

		if type(next_state) == tuple: next_state = next_state[0]
		### gymnasium-robotics ###
		if type(next_state) == type({}): 
			next_state = np.concatenate([next_state['observation'].reshape([1, -1]), next_state['desired_goal'].reshape([1, -1])], axis=1)
		### gymnasium-robotics ###
		done = truncated or terminated

		done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

		# Store data in replay buffer
		intrinsic_r = args.intrinsic_w * intrinsic_reward(revise_state(np.array(state).flatten()))
		replay_buffer.add(revise_state(np.array(state).flatten()).reshape([1, -1]), action, revise_state(np.array(next_state).flatten()).reshape([1, -1]), reward + intrinsic_r, done_bool)

		state = next_state
		episode_reward += reward
		episode_intrinsic_reward += intrinsic_r

		# Train agent after collecting sufficient data
		if t >= args.start_timesteps:
			policy.train(replay_buffer, args.batch_size)

		if done:  
			# +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
			print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
			# Reset environment
			state, done = env.reset(), False

			if type(state) == tuple: state = state[0]
			### gymnasium-robotics ###
			if type(state) == type({}): 
				state = np.concatenate([state['observation'].reshape([1, -1]), state['desired_goal'].reshape([1, -1])], axis=1)
			### gymnasium-robotics ###

			### caculate correlationship ###
			state_change = np.concatenate(episode_all_state, axis = 1)
			reward_change = np.array(episode_all_reward).reshape([1, -1])

			# state_correlation = np.corrcoef(reward_change, state_change)[0, 1:]

			state_lipschitz_constant_correlation = cal_lipschitz(state_change, reward_change)
			assert state_lipschitz_constant_correlation.shape == soft_state_correlation.shape, state_lipschitz_constant_correlation.shape

			state_lipschitz_constant_correlation[np.isnan(state_lipschitz_constant_correlation)] = 0
			### soft update ###
			soft_state_correlation = args.corr_tau * state_lipschitz_constant_correlation + (1 - args.corr_tau) * soft_state_correlation

			intrinsic_ratio.append(abs(episode_intrinsic_reward) / (abs(episode_reward) + 1e-5))

			episode_reward, episode_intrinsic_reward = 0, 0
			episode_timesteps = 0
			episode_num += 1 

		# Evaluate episode
		if (t + 1) % args.eval_freq == 0:
			evaluations.append(eval_policy(policy, args.env, args.seed))
			evaluations_steps.append([evaluations[-1], t])

			np.save(f"./results/{file_name}", evaluations)
			if args.save_model: policy.save(f"./models/{file_name}") 

	### all over, then save datas to result ###
	evaluations_steps = np.array(evaluations_steps)
	np.save(args.sid_result_path, evaluations_steps)

	if args.eval == 0:
		np.save(args.corr_result_path, soft_state_correlation)
