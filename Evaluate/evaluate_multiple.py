import argparse
import os

import gym
import matplotlib
import numpy as np
import torch
from tqdm import tqdm
import CustomGymEnvs
from pathlib import Path
from SAC.sac import SAC
# from torch.utils.tensorboard import SummaryWriter
from utils import state_2_graphbatch

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="FetchReachEnv-v0",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--exp-type', default="standard",
                    help='Type of the experiment like normal or abnormal')
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
parser.add_argument('--aggregation', default="avg",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
args = parser.parse_args()

exp_path = os.path.join(Path(__file__).parent.parent, 'Data', args.env_name, args.exp_type)
experiment_seed = os.listdir(exp_path)
experiment_seed = [d for d in experiment_seed if os.path.isdir(os.path.join(exp_path, d))]

# Environment
# env = NormalizedActions(gym.make(args.env_name))
env = gym.make(args.env_name)
num_nodes = env.observation_space['node_features'].shape[0]
num_edges = env.observation_space['edge_features'].shape[0]
num_node_features = env.observation_space['node_features'].shape[1]
num_edge_features = env.observation_space['edge_features'].shape[1]
num_global_features = env.observation_space['global_features'].shape[0]

device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')

episodes = 1

edge_list = env.robot_graph.edge_list
node_list = env.robot_graph.node_list

rel_freq_edge = {}
rel_freq_node = {}
rel_freq_global = {}
fig, ax = plt.subplots(figsize=[18, 12])
width = 1 / len(experiment_seed)
step = - np.floor(len(experiment_seed) / 2)

for s in range(len(experiment_seed)):
    env.seed(s)
    env.action_space.seed(s)
    torch.manual_seed(s)
    np.random.seed(s)
    # Agent
    checkpoint_path = os.path.join(exp_path, f'seed{s}', 'model')
    agent = SAC(num_node_features, num_edge_features, num_global_features, env.action_space, False, args)
    agent_relevance = SAC(num_node_features, num_edge_features, num_global_features, env.action_space, True, args)

    agent.load_checkpoint(checkpoint_path, evaluate=True)
    agent_relevance.load_checkpoint(checkpoint_path, evaluate=True)

    rel_freq_edge[s] = {}
    rel_freq_node[s] = {}
    rel_freq_global[s] = 0

    for j in edge_list:
        if j is not None:
            rel_freq_edge[s][j.attrib['name']] = 0
    for n in node_list:
        if 'name' in n.attrib:
            rel_freq_node[s][n.attrib['name']] = 0

    num_samples = 0
    avg_reward = 0.
    for _ in tqdm(range(episodes)):
        state = env.reset()
        episode_reward = 0
        done = False
        while not done:
            state = state_2_graphbatch(state).requires_grad_().to(device)
            graph_out = agent_relevance.policy.graph_net(state)
            out = agent_relevance.policy.mean_linear(graph_out).global_features
            out.backward(out)
            node_rel = state.node_features.grad.sum(dim=1)
            edge_rel = state.edge_features.grad.sum(dim=1)
            rel_freq_global[s] += state.global_features.grad.sum(dim=1)
            joint_ids = torch.argsort(edge_rel)
            body_ids = torch.argsort(node_rel)
            for id in joint_ids:
                if edge_list[id] is not None:
                    rel_freq_edge[s][edge_list[id].attrib['name']] += edge_rel[id]

            for id in body_ids:
                if 'name' in node_list[id].attrib:
                    rel_freq_node[s][node_list[id].attrib['name']] += node_rel[id]
            num_samples += 1
            action = agent.select_action(state, evaluate=True)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state
        avg_reward += episode_reward
    avg_reward /= episodes

    # writer.add_scalar('avg_reward/test', avg_reward, i_episode)

    print("----------------------------------------")
    print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
    print("----------------------------------------")

    x = [2 * x + step * width for x in range(len(rel_freq_edge[s]) + len(rel_freq_node[s]) + 1)]
    ax.bar(x,
           np.array(list(rel_freq_edge[s].values()) + list(rel_freq_node[s].values()) + [rel_freq_global[s]]) / num_samples,
           width,
           label=f'seed{s}')
    step += 1

fig_name = os.path.join(exp_path, 'LRP_result.jpg')
x = [2 * x for x in range(len(rel_freq_edge[0]) + len(rel_freq_node[0]) + 1)]
keys = [' '.join(k.split(':')[1].split('_')) for k in rel_freq_edge[0].keys()]
keys += [(' '.join(k.split(':')[1].split('_')) if 'robot0' in k else k) for k in rel_freq_node[0].keys()]
keys += ['global features']
ax.set_xticks(x, keys, rotation=90)
ax.legend()
plt.savefig(fig_name, dpi=300)
plt.show()
plt.close()
# print(rel_freq_node)
# plt.figure(figsize=[12, 15])
# plt.bar(range(len(rel_freq_node)), np.array(list(rel_freq_node.values())) / num_samples, align='center')
# plt.xticks(range(len(rel_freq_node)), list(rel_freq_node.keys()), rotation=90)
# plt.show()

env.close()
