import argparse
import datetime
import os

parallel_procs = "1"
os.environ["OMP_NUM_THREADS"] = parallel_procs
os.environ["MKL_NUM_THREADS"] = parallel_procs

import gym
import CustomGymEnvs
import numpy as np
import itertools
import torch
from Graph_SAC.sac import SAC
# from torch.utils.tensorboard import SummaryWriter
from Graph_SAC.replay_memory import ReplayMemory
from utils import state_2_graph, state_2_graphbatch, save_object
import pandas as pd
import os
from pathlib import Path

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="FetchReachEnv-v0",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--exp-type', default="standard",
                    help='Type of the experiment like normal or abnormal')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_episodes', type=int, default=10000, metavar='N',
                    help='maximum number of steps (default: 10000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('-dsf', '--data_save_freq', type=int, default=100, metavar='N',
                    help='Save checkpoint every msf episodes')
parser.add_argument('-ef', '--evaluation_freq', type=int, default=10, metavar='N',
                    help='Evaluate the policy every ef episodes')
parser.add_argument('--aggregation', default="avg",
                    help='Aggregation type in nodes and globals (default: average)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')

args = parser.parse_args()

exp_path = Path(os.path.abspath(__file__)).parent.parent.parent
exp_path = os.path.join(exp_path, 'Data', args.env_name, args.exp_type, f'seed{args.seed}')

if not os.path.exists(exp_path):
    os.makedirs(exp_path)

save_object(args, os.path.join(exp_path, 'parameters.pkl'))
# Environment
# env = NormalizedActions(gym.make(args.env_name))
env = gym.make(args.env_name)
env.seed(args.seed)
env.action_space.seed(args.seed)

num_nodes = env.observation_space['node_features'].shape[0]
num_edges = env.observation_space['edge_features'].shape[0]
num_node_features = env.observation_space['node_features'].shape[1]
num_edge_features = env.observation_space['edge_features'].shape[1]
num_global_features = env.observation_space['global_features'].shape[0]

torch.manual_seed(args.seed)
np.random.seed(args.seed)

print('num_node_features', num_node_features)
print('num_edge_features', num_edge_features)
print('num_global_features', num_global_features)

# Agent
agent = SAC(num_node_features, num_edge_features, num_global_features, env.action_space, relevance=False, args=args)

# Tesnorboard
# writer = SummaryWriter(
#     'runs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
#                                   args.policy, "autotune" if args.automatic_entropy_tuning else ""))

# Memory
memory = ReplayMemory(args.replay_size, args.seed)

# Training Loop
total_numsteps = 0
updates = 0

device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')

# losses = np.empty([0, 6])
# train_reward = np.empty([0, 5])
# eval_reward = np.empty([0, 4])

max_episode_steps = env.spec.max_episode_steps
losses = np.zeros([args.updates_per_step * max_episode_steps * args.num_episodes, 6])
train_reward = np.zeros([args.num_episodes, 5])
eval_reward = np.zeros([int(args.num_episodes / args.evaluation_freq), 4])


def save_data():
    loss_df = pd.DataFrame(losses,
                           columns=['num_updates', 'critic_1_loss', 'critic_2_loss', 'policy_loss', 'ent_loss',
                                    'alpha'])
    train_reward_df = pd.DataFrame(train_reward, columns=['num_episodes', 'num_steps', 'num_updates', 'episode_steps',
                                                          'train_reward'])
    eval_reward_df = pd.DataFrame(eval_reward, columns=['num_episodes', 'num_steps', 'num_updates', 'eval_reward'])

    loss_df.to_csv(os.path.join(exp_path, 'loss.csv'), index=False)
    train_reward_df.to_csv(os.path.join(exp_path, 'train.csv'), index=False)
    eval_reward_df.to_csv(os.path.join(exp_path, 'eval.csv'), index=False)


for i_episode in range(args.num_episodes):
    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()

    while not done:
        # print('#' * 50)
        # print(state['observation']['node_features'])
        if args.start_steps > total_numsteps:
            action = env.action_space.sample()  # Sample random action
        else:
            action = agent.select_action(state_2_graphbatch(state).to(device))  # Sample action from policy

        if len(memory) > args.batch_size:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory,
                                                                                                     args.batch_size,
                                                                                                     updates)
                # losses = np.append(losses, [[updates, critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha]],
                #                    axis=0)
                losses[updates] = np.array([updates, critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha])
                updates += 1

        next_state, reward, done, _ = env.step(action)  # Step
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = 1 if episode_steps == env._max_episode_steps else float(not done)

        memory.push(state_2_graph(state), action, reward, state_2_graph(next_state),
                    mask)  # Append transition to memory

        state = next_state

    # train_reward = np.append(train_reward, [[i_episode, total_numsteps, updates, episode_steps, episode_reward]],
    #                          axis=0)

    train_reward[i_episode] = np.array([i_episode, total_numsteps, updates, episode_steps, episode_reward])

    # writer.add_scalar('reward/train', episode_reward, i_episode)
    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps,
                                                                                  episode_steps,
                                                                                  round(episode_reward, 2)))

    # save checkpoint
    if (i_episode + 1) % args.data_save_freq == 0:
        print('Saving ...')
        agent.save_checkpoint(exp_path)
        save_data()

    if (i_episode + 1) % args.evaluation_freq == 0 and args.eval is True:
        avg_reward = 0.
        episodes = 10
        for _ in range(episodes):
            state = state_2_graphbatch(env.reset()).to(device)
            episode_reward = 0
            done = False
            while not done:
                action = agent.select_action(state, evaluate=True)
                next_state, reward, done, _ = env.step(action)
                next_state = state_2_graphbatch(next_state).to(device)
                episode_reward += reward

                state = next_state
            avg_reward += episode_reward
        avg_reward /= episodes

        # writer.add_scalar('avg_reward/test', avg_reward, i_episode)
        # eval_reward = np.append(eval_reward, [[i_episode, total_numsteps, updates, avg_reward]], axis=0)
        eval_reward[int(i_episode / args.evaluation_freq)] = np.array([i_episode, total_numsteps, updates, avg_reward])
        print("----------------------------------------")
        print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
        print("----------------------------------------")

env.close()
