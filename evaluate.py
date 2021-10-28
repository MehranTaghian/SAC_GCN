import argparse
import datetime
import gym
import CustomGymEnvs
import numpy as np
import itertools
import torch
from SAC.sac import SAC
from torch.utils.tensorboard import SummaryWriter
from SAC.replay_memory import ReplayMemory
from utils import state_2_graph, state_2_graphbatch

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="HalfCheetah-v2",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
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
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--hidden_action_size', type=int, default=32, metavar='N',
                    help='hidden size for action layer in Q-function (default: 32)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('-msf', '--model_save_freq', type=int, default=100, metavar='N',
                    help='Save checkpoint every msf episodes')
parser.add_argument('-ef', '--evaluation_freq', type=int, default=10, metavar='N',
                    help='Evaluate the policy every ef episodes')
parser.add_argument('-chp', '--checkpoint_path',
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')

parser.add_argument('--aggregation', default="avg",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
args = parser.parse_args()

# Environment
# env = NormalizedActions(gym.make(args.env_name))
env = gym.make(args.env_name)
env.seed(args.seed)
env.action_space.seed(args.seed)

num_nodes = env.observation_space['observation_nodes'].shape[0]
num_edges = env.observation_space['observation_edges'].shape[0]
num_node_features = env.observation_space['observation_nodes'].shape[1]
num_edge_features = env.observation_space['observation_edges'].shape[1]
num_global_features = env.observation_space['achieved_goal'].shape[0]

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Agent
agent = SAC(num_node_features, num_edge_features, num_global_features, env.action_space, args)

agent.load_checkpoint(args.checkpoint_path, evaluate=True)

# Tesnorboard
writer = SummaryWriter(
    'runs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
                                  args.policy, "autotune" if args.automatic_entropy_tuning else ""))

device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')

for i_episode in itertools.count(1):
    avg_reward = 0.
    episodes = 10
    for _ in range(episodes):
        state = state_2_graphbatch(env.reset()).to(device)
        env.render()
        episode_reward = 0
        done = False
        while not done:
            action = agent.select_action(state, evaluate=True)
            next_state, reward, done, _ = env.step(action)
            next_state = state_2_graphbatch(next_state).to(device)
            episode_reward += reward
            state = next_state
            env.render()
        avg_reward += episode_reward
    avg_reward /= episodes

    writer.add_scalar('avg_reward/test', avg_reward, i_episode)

    print("----------------------------------------")
    print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
    print("----------------------------------------")

env.close()
