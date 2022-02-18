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

plt.rcParams['font.size'] = '20'

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

episodes = 10

edge_list = env.robot_graph.edge_list
node_list = env.robot_graph.node_list

rel_freq_edge = {}
rel_freq_node = {}
rel_freq_global = {}
avg_rel_freq_node = {}
avg_rel_freq_edge = {}


def set_ax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD')
    ax.tick_params(bottom=False, left=False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color='#EEEEEE')
    ax.xaxis.grid(False)


figure_width = 35
figure_height = 16
label_rotation = 0

fig, ax = plt.subplots(figsize=[figure_width, figure_height])
fig2, ax2 = plt.subplots(figsize=[figure_width, figure_height])
fig3, ax3 = plt.subplots(figsize=[figure_width, figure_height])

set_ax(ax)
set_ax(ax2)
set_ax(ax3)

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

    total_relevance = np.array(list(rel_freq_edge[s].values()) +
                               list(rel_freq_node[s].values()) +
                               [rel_freq_global[s]]) / num_samples

    avg_rel_freq_node[s] = np.array(list(rel_freq_node[s].values())) / num_samples
    avg_rel_freq_edge[s] = np.array(list(rel_freq_edge[s].values())) / num_samples

    ax.bar(x, total_relevance, width, label=f'seed{s}')

    x2 = [2 * x + step * width for x in range(len(rel_freq_node[s]))]
    ax2.bar(x2, avg_rel_freq_node[s], width, label=f'seed{s}')

    x3 = [2 * x + step * width for x in range(len(rel_freq_edge[s]))]
    ax3.bar(x3, avg_rel_freq_edge[s], width, label=f'seed{s}')

    step += 1


def process_keys(keys):
    final_keys = []
    for key in keys:
        if 'robot0' in key:
            sep_key = key.split(':')[1].split('_')
        else:
            sep_key = key.split('_')
        final_key = ''
        for sk in sep_key:
            if len(sk) == 1:
                final_key += sk + '-'
            else:
                final_key += sk + '\n'
        final_keys.append(final_key)
    return final_keys


fig_name = os.path.join(exp_path, 'LRP_result_total.jpg')
x = [2 * x for x in range(len(rel_freq_edge[0]) + len(rel_freq_node[0]) + 1)]
# keys = ['\n'.join(k.split(':')[1].split('_')) for k in rel_freq_edge[0].keys()]
# keys += [('\n'.join(k.split(':')[1].split('_')) if 'robot0' in k else k) for k in rel_freq_node[0].keys()]
# keys += ['global\nfeatures']
keys = process_keys(rel_freq_edge[0].keys())
keys += process_keys(rel_freq_node[0].keys())
keys += ['global\nfeatures']

ax.set_xticks(x, keys, rotation=label_rotation)
ax.legend()
ax.set_xlabel("Name of the graph's nodes, edges, and global features")
ax.set_ylabel("LRP score")
ax.set_title("LRP score for each part of the input graph's nodes, edges and global features")
fig.savefig(fig_name, dpi=300)

# Node features plot
fig_name = os.path.join(exp_path, 'LRP_result_nodes.jpg')
x = [2 * x for x in range(len(rel_freq_node[0]))]
# keys = [('\n'.join(k.split(':')[1].split('_')) if 'robot0' in k else k) for k in rel_freq_node[0].keys()]
keys = process_keys(rel_freq_node[0].keys())
ax2.set_xticks(x, keys, rotation=label_rotation)
ax2.legend()
ax2.set_xlabel("Name of the graph's nodes (robot's links)")
ax2.set_ylabel("LRP score")
ax2.set_title("LRP score for each part of the input graph's nodes")
fig2.savefig(fig_name, dpi=300)

# Edge features plot
fig_name = os.path.join(exp_path, 'LRP_result_edges.jpg')
x = [2 * x for x in range(len(rel_freq_edge[0]))]
# keys = ['\n'.join(k.split(':')[1].split('_')) for k in rel_freq_edge[0].keys()]
keys = process_keys(rel_freq_edge[0].keys())
ax3.set_xticks(x, keys, rotation=label_rotation)
ax3.legend()
ax3.set_xlabel("Name of the graph's edges (robot's joints)")
ax3.set_ylabel("LRP score")
ax3.set_title("LRP score for each part of the input graph's edges")
fig3.savefig(fig_name, dpi=300)

node_relevances = np.zeros([len(experiment_seed), len(rel_freq_node[0])])
edge_relevances = np.zeros([len(experiment_seed), len(rel_freq_edge[0])])

# average relevances over seeds
for s in range(len(experiment_seed)):
    node_relevances[s, :] = avg_rel_freq_node[s]
    edge_relevances[s, :] = avg_rel_freq_edge[s]

fig_name = os.path.join(exp_path, 'LRP_result_nodes_avg.jpg')
fig4, ax4 = plt.subplots(figsize=[figure_width, figure_height])
set_ax(ax4)
x = [x for x in range(len(rel_freq_node[0]))]
# keys = [('\n'.join(k.split(':')[1].split('_')) if 'robot0' in k else k) for k in rel_freq_node[0].keys()]
keys = process_keys(rel_freq_node[0].keys())
ax4.bar(x, np.mean(node_relevances, axis=0))
ax4.set_xticks(x, keys, rotation=label_rotation)
ax4.set_xlabel("Name of the graph's nodes (robot's links)")
ax4.set_ylabel(f"Average LRP score across {len(experiment_seed)} seeds")
ax4.set_title("Average LRP score for each part of the input graph's nodes")
fig4.savefig(fig_name, dpi=300)

fig_name = os.path.join(exp_path, 'LRP_result_edges_avg.jpg')
fig5, ax5 = plt.subplots(figsize=[figure_width, figure_height])
set_ax(ax5)
x = [x for x in range(len(rel_freq_edge[0]))]
keys = process_keys(rel_freq_edge[0].keys())
ax5.bar(x, np.mean(edge_relevances, axis=0))
ax5.set_xticks(x, keys, rotation=label_rotation)
ax5.set_xlabel("Name of the graph's edges (robot's joints)")
ax5.set_ylabel(f"Average LRP score across {len(experiment_seed)} seeds")
ax5.set_title("Average LRP score for each part of the input graph's edges")
fig5.savefig(fig_name, dpi=300)

env.close()
