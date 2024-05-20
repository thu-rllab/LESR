import pandas as pd 
import argparse
import numpy as np
import requests
import json
import traceback
import gymnasium as gym
import os 
import importlib
import sys 
import argparse
import os
import libtmux
import time
import random
import psutil
import getpass
import pynvml
import shutil 
import openai

def import_and_reload_module(module_name):
    if module_name in sys.modules: 
        del sys.modules[module_name] 
    imported_module = importlib.import_module(module_name) 
    return imported_module

def init_prompt():
    cur_env = args.env.split('-')[0].lower() 
    obs_file = pd.read_excel(args.observation_path, header=None, sheet_name=cur_env) 
    content = list(obs_file.iloc[:, 1])
    unit = list(obs_file.iloc[:, -2])
    detail_content = ''
    for ii in range(len(content)):
        detail_content += '- `s[{}]`: '.format(ii) + content[ii] + f' , the unit is {unit[ii]}.' '\n'

    task_description = obs_file.iloc[0, -1]
    total_dim = len(content)

    if 'antmaze' in args.env.lower() or 'fetch' in args.env.lower() or 'adroit' in args.env.lower() or 'handmanipulate' in args.env.lower() or 'pointmaze' in args.env.lower():
        additional_prompt = 'Most Importantly: As for this task, the agent should firstly learn how to coordinate various parts of the body itself before it finally reach the goal. Like a baby should learn how to walk before finally walking to the goal. Therefore, when you design state representation and reward, you should cosider how to make the agent learn to coordinate various parts of the body as well as how to finally reach the goal.\n'
    else: additional_prompt = ''

    init_prompt_template = f"""
Revise the state representation for a reinforcement learning agent. 
=========================================================
The agent’s task description is:
{task_description}
=========================================================

The current state is represented by a {total_dim}-dimensional Python NumPy array, denoted as `s`.

Details of each dimension in the state `s` are as follows:
{detail_content}
You should design a task-related state representation based on the the source {total_dim} dim to better for reinforcement training, using the detailed information mentioned above to do some caculations, and feel free to do complex caculations, and then concat them to the source state. 

Besides, we want you to design an intrinsic reward function based on the revise_state python function.

That is to say, we will:
1. use your revise_state python function to get an updated state: updated_s = revise_state(s)
2. use your intrinsic reward function to get an intrinsic reward for the task: r = intrinsic_reward(updated_s)
3. to better design the intrinsic_reward, we recommond you use some source dim in the updated_s, which is between updated_s[0] and updated_s[{total_dim - 1}] 
4. however, you must use the extra dim in your given revise_state python function, which is between updated_s[{total_dim}] and the end of updated_s
{additional_prompt}
Your task is to create two Python functions, named `revise_state`, which takes the current state `s` as input and returns an updated state representation, and named `intrinsic_reward`, which takes the updated state `updated_s` as input and returns an intrinsic reward. The functions should be executable and ready for integration into a reinforcement learning environment.

The goal is to better for reinforcement training. Lets think step by step. Below is an illustrative example of the expected output:

```python
import numpy as np
def revise_state(s):
    # Your state revision implementation goes here
    return updated_s
def intrinsic_reward(updated_s):
    # Your intrinsic reward code implementation goes here
    return intrinsic_reward
```
"""
    return init_prompt_template, detail_content

def find_window_and_execute_command(command):
    window_can_use=False
    idx = 5 ### start check from window 4 ? ### 
    while not window_can_use:
        idx += 1
        if idx > 60:
            print("Window index too large, exit.")
            exit(0)
        # print("Try window index", idx)
        try:
            w = session.new_window(attach = False, window_index = idx)
            time.sleep(2)
            # print("Create new window index", idx)
        except:
            w = session.find_where({'window_index':str(idx)})
            # print("Select existed windows index", idx)
        try:
            pm = w.list_panes()[0]
            lastLine = pm.cmd('capture-pane', '-p').stdout[-1]
            if args.user_name+'@' in lastLine and lastLine.endswith("$"):
                window_can_use = True
                # print('Window available')
            else:
                # print('Window occupied')
                pass 
        except:
            continue

    ### put the command into window to run ###
    print("=======================================")
    print(f"Current Train:{train} Current Window:{idx}")
    print("=======================================")
    
    current_file_path = os.path.abspath(__file__)
    current_file_path = current_file_path[:current_file_path.rindex('/')]
    pm.send_keys(f'cd {current_file_path}', enter=True)
    pm.send_keys(command, enter=True)

def get_cot_prompt(every_code, every_score, max_id, every_factor, every_dim, state_dim):

    s_feedback = ''
    for ii in range(len(every_code)):
        s_feedback += f'========== State Revise and Intrinsic Reward Code -- {ii + 1} ==========\n'
        s_feedback += every_code[ii] + '\n'

        policy_performance = 'Final Policy Performance'
        if 'antmaze' in args.env.lower() or 'fetch' in args.env.lower() or 'adroit' in args.env.lower():
            policy_performance = 'Final Policy Success Rate'
        s_feedback += f'========== State Revise and Intrinsic Reward Code -- {ii + 1}\'s {policy_performance}: {round(every_score[ii], 2)} ==========\n'
        try:
            s_feedback += f'In this State Revise Code {ii + 1}, the source state dim is from s[0] to s[{state_dim - 1}], the Lipchitz constant between them and the reward are(Note: The Lipschitz constant is always greater than or equal to 0, and a lower Lipschitz constant implies better smoothness.):\n'
            
            cur_dim_corr = ''
            for kk in range(0, state_dim):
                cur_dim_corr += f's[{kk}] lipschitz constant with reward = {round(every_factor[ii][kk], 2)}\n'

            s_feedback += cur_dim_corr + '\n'

            if every_dim[ii] - state_dim > 0:
                s_feedback += f'In this State Revise Code {ii + 1}, you give {every_dim[ii] - state_dim} extra dim from s[{state_dim}] to s[{every_dim[ii] - 1}], the lipschitz constant between them and the reward are:\n'
                cur_dim_corr = ''
                for kk in range(state_dim, every_dim[ii]):
                    cur_dim_corr += f's[{kk}] lipschitz constant with reward = {round(every_factor[ii][kk], 2)}\n'
        except: pass 
        
        s_feedback += cur_dim_corr + '\n======================================================================\n\n'

    cot_prompt = f"""
We have successfully trained Reinforcement Learning (RL) policy using {args.sample_count} different state revision codes and intrinsic reward function codes sampled by you, and each pair of code is associated with the training of a policy relatively.

Throughout every state revision code's training process, we monitored:
1. The final policy performance(accumulated reward).
2. Most importantly, every state revise dim's Lipschitz constant with the reward. That is to say, you can see which state revise dim is more related to the reward and which dim can contribute to enhancing the continuity of the reward function mapping. Lower Lipchitz constant means better continuity and smoother of the mapping. Note: Lower Lipchitz constant is better.

Here are the results:
{s_feedback}

You should analyze the results mentioned above and give suggestions about how to imporve the performace of the "state revision code".

Here are some tips for how to analyze the results:
(a) if you find a state revision code's performance is very low, then you should analyze to figure out why it fail
(b) if you find some dims' are more related to the final performance, you should analyze to figure out what makes it success
(c) you should also analyze how to imporve the performace of the "state revision code" and "intrinsic reward code" later

Lets think step by step. Your solution should aim to improve the overall performance of the RL policy.
"""
    return cot_prompt, s_feedback

def get_next_iteration_prompt(all_it_func_results, all_it_cot_suggestions):
    cur_env = args.env.split('-')[0].lower() 
    obs_file = pd.read_excel(args.observation_path, header=None, sheet_name=cur_env)

    content = list(obs_file.iloc[:, 1])
    unit = list(obs_file.iloc[:, -2])
    detail_content = ''
    for ii in range(len(content)):
        detail_content += '- `s[{}]`: '.format(ii) + content[ii] + f' , the unit is {unit[ii]}.' '\n'

    task_description = obs_file.iloc[0, -1]
    
    total_dim = len(content)
 
    if 'antmaze' in args.env.lower() or 'fetch' in args.env.lower() or 'adroit' in args.env.lower():
        additional_prompt = 'Most Importantly: As for this task, the agent should firstly learn how to coordinate various parts of the body itself before it finally reach the goal. Like a baby should learn how to walk before finally walking to the goal. Therefore, when you design state representation and reward, you should cosider how to make the agent learn to coordinate various parts of the body as well as how to finally reach the goal.\n'
    else: additional_prompt = ''

    former_histoy = ''
    for ii in range(len(all_it_func_results)):
        former_histoy += f'\n\n\nFormer Iteration:{ii + 1}\n'
        former_histoy += all_it_func_results[ii]
        former_histoy += f'\n\nFrom Former Iteration:{ii + 1}, we have some suggestions for you:\n{all_it_cot_suggestions[ii]}'

    init_prompt_template = f"""
Revise the state representation for a reinforcement learning agent. 
=========================================================
The agent’s task description is:
{task_description}
=========================================================

The current state is represented by a {total_dim}-dimensional Python NumPy array, denoted as `s`.

Details of each dimension in the state `s` are as follows:
{detail_content}
You should design a task-related state representation based on the source {total_dim} dim to better for reinforcement training, using the detailed information mentioned above to do some caculations, and feel free to do complex caculations, and then concat them to the source state. 

For this problem, we have some history experience for you, here are some state revision codes we have tried in the former iterations:
{former_histoy}

Based on the former suggestions. We are seeking an improved state revision code and an improved intrinsic reward code that can enhance the model's performance on the task. The state revised code should incorporate calculations, and the results should be concatenated to the original state.

Besides, We are seeking an improved intrinsic reward code.

That is to say, we will:
1. use your revise_state python function to get an updated state: updated_s = revise_state(s)
2. use your intrinsic reward function to get an intrinsic reward for the task: r = intrinsic_reward(updated_s)
3. to better design the intrinsic_reward, we recommond you use some source dim in the updated_s, which is between updated_s[0] and updated_s[{total_dim - 1}] 
4. however, you must use the extra dim in your given revise_state python function, which is between updated_s[{total_dim}] and the end of updated_s
{additional_prompt}
Your task is to create two Python functions, named `revise_state`, which takes the current state `s` as input and returns an updated state representation, and named `intrinsic_reward`, which takes the updated state `updated_s` as input and returns an intrinsic reward. The functions should be executable and ready for integration into a reinforcement learning environment.

The goal is to better for reinforcement training. Lets think step by step. Below is an illustrative example of the expected output:

```python
import numpy as np
def revise_state(s):
    # Your state revision implementation goes here
    return updated_s
def intrinsic_reward(updated_s):
    # Your intrinsic reward code implementation goes here
    return intrinsic_reward
```
"""
    return init_prompt_template

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="HalfCheetah-v4", type=str)                                            # OpenAI gym environment name 
    parser.add_argument("--max_timesteps", default=1e6, type=int)                                            # OpenAI gym environment name 
    parser.add_argument("--max_evaluate_timesteps", default=1e6, type=int)                                            # OpenAI gym environment name 
    parser.add_argument("--dir", default="LESR-resources/", type=str)                                            # observation description dir
    parser.add_argument("--observation_path", default="LESR-resources/mujoco_observation_space.xlsx", type=str)  # observation description path
    parser.add_argument("--sample_count", default=2, type=int)                                                  # how many samples every iteration
    parser.add_argument("--evaluate_count", default=5, type=int)                                                # how many seeds to evaluate
    parser.add_argument("--train_seed_count", default=1, type=int)                                              # how many seeds to training
    parser.add_argument("--iteration", default=5, type=int)                                                     # how many iteration count
    parser.add_argument("--model", default="gpt-4-1106-preview", type=str)                                      # which llm should use
    parser.add_argument("--openai_key", default="", type=str)                                                   # openai key
    parser.add_argument("--temperature", default=1.0, type=float)                                               # init sampling temperature
    parser.add_argument("--cuda", default=0, type=int)                                                          # which gpu training on
    parser.add_argument('--session_name', action='store', default='0', type=str, help="Session name for run.")
    parser.add_argument('--user_name', default=getpass.getuser(), type=str, help="User's name for window available check.")
    parser.add_argument("--v", default=1, type=int)                                                             # version 
    parser.add_argument("--intrinsic_w", default=0.001, type=float)                                               # init sampling temperature

    args = parser.parse_args()

    gpt_key = args.openai_key
    assert gpt_key != '', 'You should pass an OpenAI Key to get access to GPT.'
    openai.api_key = gpt_key
    temperature = args.temperature

    print('-' * 20)
    print('model: ', args.model)
    print('-' * 20)
    
    ### find_current_observation_space ###
    env = gym.make(args.env)
    ### gym - robotics ###
    if type(env.observation_space) == gym.spaces.Dict:
        state_dim = env.observation_space['observation'].shape[0] + env.observation_space['desired_goal'].shape[0]
    else: state_dim = env.observation_space.shape[0]
    ### gym - robotics ###
    test_state = np.random.randn(state_dim, )
    
    ### set initial propmt ### 
    prompt, source_state_description = init_prompt()
    dialogs = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    ### check the output dir ###
    code_dir = os.path.join(args.dir, f'run-v{args.v}-' + args.env)
    if os.path.exists(code_dir):
        shutil.rmtree(code_dir)
    os.makedirs(code_dir)
    
    ### check the output training result dir ###
    sid_result_dir = os.path.join(code_dir, 'result')
    os.makedirs(sid_result_dir)

    ### init best result dir ###
    best_result_dir = os.path.join(code_dir, 'best_result')
    os.makedirs(best_result_dir)

    ### init tmux server ###
    server = libtmux.Server()
    session = server.find_where({"session_name": args.session_name})
    if not session:
        session = server.new_session(session_name=args.session_name)
    assert session

    ### init every max score ###
    every_max_score = np.zeros([args.iteration, ])
    every_max_score_id = np.zeros([args.iteration, ])
    
    all_it_func_results, all_it_cot_suggestions = [], []
    ### begin iteration ### 
    for it in range(args.iteration):

        """
            begin sample state revise function 
        """
        valid_sample_count = 0
        revise_lib_path_buffer, revise_code_buffer, revise_dim_buffer = [], [], []
        assitant_reply, assitant_reward_reply = [0] * args.sample_count, [0] * args.sample_count

        trying_count = 0
        while valid_sample_count != args.sample_count:
            trying_count += 1
            if trying_count == 50:
                print('...Trying Too Much...')
                exit()

            print("---------------------------------------")
            print(f"Current Iteration:{it} Current Sample Trying:{valid_sample_count + 1} Trying Count:{trying_count}")
            print("---------------------------------------")

            try:
                ### get response and parse output into python code ###
                params = {'model':args.model, 'messages': dialogs, 'temperature':temperature }
                completion = openai.ChatCompletion.create(**params)
                sid_code = completion['choices'][0]['message']['content']

                assitant_reply[valid_sample_count] = sid_code
                ### find where is 'return' ###
                ret_id = sid_code.rindex('return')
                while ret_id < len(sid_code):
                    if sid_code[ret_id] == '\n': break
                    ret_id += 1
                sid_code = sid_code[sid_code.index('import numpy as np'):ret_id + 1].replace('`', '') + '\n'

                ### save it to code dir ###
                cur_code_path = os.path.join(code_dir, f'it_{it}_sample_{valid_sample_count}.py')
                with open(cur_code_path, 'w') as fp:
                    fp.write(sid_code)
                    fp.close()
                
                ### test whether it can be executed ###
                cur_module = import_and_reload_module(cur_code_path[:-3].replace('/', '.'))
                cur_revise_state_v11 = cur_module.revise_state(test_state)
                cur_revise_dim = cur_revise_state_v11.shape[0]

                cur_intrinsic_reward = cur_module.intrinsic_reward(cur_revise_state_v11)
                assert cur_intrinsic_reward >= -100.0 and cur_intrinsic_reward <= 100.0, cur_intrinsic_reward

                ### !!!!!!!!!!!!! must last append !!!!!!!!!!!!!!!   ###
                ### save revise function and new state dim to buffer ###
                revise_lib_path_buffer.append(cur_code_path[:-3].replace('/', '.'))
                revise_dim_buffer.append(cur_revise_dim)
                revise_code_buffer.append(sid_code)
                
                ### valid function ###
                valid_sample_count += 1
                
            except requests.Timeout:
                print("...request timeout...")
            except Exception as e:
                traceback.print_exc()
        
        """
            begin training using the functions
        """
        for train in range(args.sample_count):
            for seed_try in range(args.train_seed_count): ### every n seeds !!! ###
                cur_version = f'v{args.v}-{args.env}-it{it}-train{train}'
                cur_sid_result_path = f'{sid_result_dir}/it{it}_train{train}_s{seed_try}.npy'
                cur_corr_result_path = f'{sid_result_dir}/it{it}_train{train}_corr_s{seed_try}.npy'
                cur_seed = random.randint(0, 100000)
                cur_training_command = f'CUDA_VISIBLE_DEVICES={args.cuda} python lesr_train.py --env {args.env} --revise_path {revise_lib_path_buffer[train]} --version {cur_version} --sid_result_path {cur_sid_result_path} --corr_result_path {cur_corr_result_path} --seed {cur_seed} --max_timesteps {int(args.max_timesteps)} --intrinsic_w {args.intrinsic_w}'
                
                ### find a tmux window and put it into it, then training ###
                find_window_and_execute_command(cur_training_command)
        
        """
            wait until all training ends, we should get:
            all training datas: such as reward scores +++ every dim's lipschitz constant factor
        """
        while True:
            finish_count, all_results, all_results_corr = 0, [[] for _ in range(args.sample_count)], [[] for _ in range(args.sample_count)]
            for train in range(args.sample_count):
                for seed_try in range(args.train_seed_count):
                    cur_sid_result_path = f'{sid_result_dir}/it{it}_train{train}_s{seed_try}.npy'
                    cur_corr_result_path = f'{sid_result_dir}/it{it}_train{train}_corr_s{seed_try}.npy'
                    if os.path.exists(cur_corr_result_path):
                        all_results[train].append(np.load(cur_sid_result_path))
                        all_results_corr[train].append(np.abs(np.load(cur_corr_result_path)))
                        finish_count += 1
            if finish_count == args.sample_count * args.train_seed_count: break
            else:
                print(f'.......Wait Until Finished, Now it = {it}, now finished = {finish_count}.........')
            time.sleep(5)
        
        """
            get mean over all seeds
        """
        results, results_corr = [], []
        for train in range(args.sample_count):
            results.append(sum(all_results[train]) / len(all_results[train]))
            results_corr.append(sum(all_results_corr[train]) / len(all_results_corr[train]))

        """
            find which training is the best
        """
        max_score, max_score_id, every_score = -1e6, -1, []
        for train in range(args.sample_count):
            every_score.append(results[train][-10:, 0].mean())
            if results[train][-10:, 0].mean() > max_score:
                max_score = results[train][-10:, 0].mean()
                max_score_id = train
        every_max_score[it] = max_score
        every_max_score_id[it] = max_score_id
        np.save(sid_result_dir + f'/all_res_{args.env}_v{args.v}.npy', every_max_score)
        np.save(sid_result_dir + f'/all_res_{args.env}_v{args.v}_id.npy', every_max_score_id)

        """
            save the best sample !!! best sample over all iterations
        """
        if it == args.iteration - 1:
            best_it = every_max_score.argmax()
            best_id = int(every_max_score_id[best_it])
            best_sample_path = os.path.join(code_dir, f'it_{best_it}_sample_{best_id}.py')
            shutil.copy(best_sample_path, os.path.join(best_result_dir, f'v{args.v}-best-{args.env}.py'))
        
        """
            COT get analysis first, then get the final answer
        """
        cot_prompt, cur_it_func_results = get_cot_prompt(revise_code_buffer, every_score, max_score_id, results_corr, revise_dim_buffer, state_dim)
        all_it_func_results.append(cur_it_func_results)
        dialogs += [
        {
            "role": "assistant",
            "content": assitant_reply[max_score_id],
        },
        {
            "role": "user",
            "content": cot_prompt,
        }]
        ### get the cot first response and concat ###
        while True:
            try:
                    ### get response and parse output into python code ###
                    params = {'model':args.model, 'messages': dialogs, 'temperature':temperature }
                    completion = openai.ChatCompletion.create(**params)
                    cur_it_cot_suggestions = completion['choices'][0]['message']['content']

                    all_it_cot_suggestions.append(cur_it_cot_suggestions)
                    dialogs += [
                        {
                            "role": "assistant",
                            "content": cur_it_cot_suggestions,
                        }]
                    break
            except requests.Timeout:
                    print("...request timeout...")
            except Exception as e:
                traceback.print_exc()
        
        """
            save the dialogs
        """
        with open(f'{code_dir}/dialogs_it{it}.txt', 'w') as fp:
            s_dialogs = ''
            for dialog in dialogs:
                cur_role, cur_content = dialog['role'], dialog['content']
                s_dialogs += '*' * 50 + '\n'
                s_dialogs += '*' * 20 + f'role:{cur_role}' + '*' * 20 + '\n'
                s_dialogs += '*' * 50 + '\n'
                s_dialogs += f'{cur_content}' + '\n\n'
            fp.write(s_dialogs)
            fp.close()
        
        ### get the final revise description and concat ###
        """
            get new feedback prompt, and get new dialog
        """
        next_iteration_prompt = get_next_iteration_prompt(all_it_func_results, all_it_cot_suggestions)
        dialogs = [
        {
            "role": "user",
            "content": next_iteration_prompt,
        }]


    """
        after training, then evaluate
    """
    for train in range(args.evaluate_count):
        cur_seed = random.randint(0, 100000)
        cur_sid_result_path = f'{sid_result_dir}/evaluate_seed{train}.npy'

        cur_training_command = f'CUDA_VISIBLE_DEVICES={args.cuda} python lesr_train.py --env {args.env} --eval 1 --version {args.v} --seed {cur_seed}  --sid_result_path {cur_sid_result_path} --max_timesteps {int(args.max_evaluate_timesteps)} --intrinsic_w {args.intrinsic_w}'
        find_window_and_execute_command(cur_training_command)
    
    """
        wait until all evaluate ends
    """
    while True:
        finish_count = 0 
        evaluete_results = []
        for train in range(args.evaluate_count):
            cur_sid_result_path = f'{sid_result_dir}/evaluate_seed{train}.npy'
            if os.path.exists(cur_sid_result_path):
                evaluete_results.append(np.load(cur_sid_result_path))
                finish_count += 1
        if finish_count == args.evaluate_count: break
        else:
            print(f'.......Wait Until Evaluating Finished, now finished = {finish_count}.........')
        time.sleep(5)