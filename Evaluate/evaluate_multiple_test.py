import argparse
import os

import gym
import matplotlib
import numpy as np
import torch
from tqdm import tqdm
import CustomGymEnvs
from pathlib import Path
from Graph_SAC.sac import SAC
# from torch.utils.tensorboard import SummaryWriter
from utils import state_2_graphbatch

# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = '20'

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="FetchReachEnvGraph-v0",
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
parser.add_argument('--hidden_action_size', type=int, default=256, metavar='N',
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
experiment_seed = experiment_seed[:2]

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
rel_score_edge = {}
rel_freq_global = {}
avg_rel_freq_edge = {}

for joint_list in edge_list.values():
    if len(joint_list) > 0:
        if len(joint_list) == 1:
            joint_name = joint_list[0].attrib['name']
        else:
            joint_name = '\n'.join([j.attrib['name'] for j in joint_list])

        for i in range(env.action_space.shape[0]):
            if i not in rel_freq_edge.keys():
                rel_freq_edge[i] = {}
                rel_score_edge[i] = {}

            rel_freq_edge[i][joint_name] = {}
            rel_score_edge[i][joint_name] = {}
            for s in range(len(experiment_seed)):
                rel_freq_edge[i][joint_name][s] = 0
                rel_score_edge[i][joint_name][s] = []

for s in range(len(experiment_seed)):
    avg_rel_freq_edge[s] = {}
    rel_freq_global[s] = {}
    for i in range(env.action_space.shape[0]):
        rel_freq_global[i][s] = 0


def set_ax(ax, n_rows, n_cols):
    for i in range(n_rows):
        for j in range(n_cols):
            ax[i, j].spines['top'].set_visible(False)
            ax[i, j].spines['right'].set_visible(False)
            ax[i, j].spines['left'].set_visible(False)
            ax[i, j].spines['bottom'].set_color('#DDDDDD')
            ax[i, j].tick_params(bottom=False, left=False)
            ax[i, j].set_axisbelow(True)
            ax[i, j].yaxis.grid(True, color='#EEEEEE')
            ax[i, j].xaxis.grid(False)


figure_width = 35
figure_height = 16
label_rotation = 0

fig_num_cols = int(np.ceil(np.sqrt(env.action_space.shape[0])))
fig_rows, fig_cols = (int(env.action_space.shape[0] / fig_num_cols)
                      if env.action_space.shape[0] % fig_num_cols == 0
                      else int(env.action_space.shape[0] / fig_num_cols)), fig_num_cols

fig_total, ax_total = plt.subplots(fig_rows, fig_cols, figsize=[figure_width, figure_height])
fig_edge, ax_edge = plt.subplots(fig_rows, fig_cols, figsize=[figure_width, figure_height])
fig_edge_avg, ax_edge_avg = plt.subplots(fig_rows, fig_cols, figsize=[figure_width, figure_height])

set_ax(ax_total, fig_rows, fig_cols)
set_ax(ax_edge, fig_rows, fig_cols)
set_ax(ax_edge_avg, fig_rows, fig_cols)

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

    rel_score_edge_seed = {}
    for joint_list in edge_list.values():
        if len(joint_list) > 0:
            if len(joint_list) == 1:
                joint_name = joint_list[0].attrib['name']
            else:
                joint_name = '\n'.join([j.attrib['name'] for j in joint_list])

            for i in range(env.action_space.shape[0]):
                if i not in rel_score_edge_seed.keys():
                    rel_score_edge_seed[i] = {}
                rel_score_edge_seed[i][joint_name] = []

    num_samples = 0
    avg_reward = 0.

    for i in tqdm(range(episodes)):
        state = env.reset()
        episode_reward = 0
        done = False
        while not done:
            for n in range(env.action_space.shape[0]):
                batch_state = state_2_graphbatch(state).requires_grad_().to(device)
                out = agent_relevance.policy.graph_net(batch_state)
                out = agent_relevance.policy.mean_linear(out)[0]
                global_relevance = torch.zeros_like(out.global_features)
                global_relevance[n] = out.global_features[n]
                batch_state.zero_grad_()
                out.global_features.backward(global_relevance)

                # node_rel = state.node_features.grad.sum(dim=1)
                edge_rel = batch_state.edge_features.grad.sum(dim=1)
                global_rel = batch_state.global_features.grad.sum(dim=1)
                rel_freq_global[n][s] += global_rel

                for e, edge in enumerate(edge_list.values()):
                    if len(edge) > 0:
                        if len(edge) == 1:
                            joint_name = edge[0].attrib['name']
                        else:
                            joint_name = '\n'.join([j.attrib['name'] for j in edge])

                        rel_freq_edge[n][joint_name][s] += edge_rel[e]
                        if len(rel_score_edge_seed[n][joint_name]) - 1 < i:
                            rel_score_edge_seed[n][joint_name].append([])
                        rel_score_edge_seed[n][joint_name][i].append(edge_rel[e])

            num_samples += 1
            action = agent.select_action(state_2_graphbatch(state), evaluate=True)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state
        avg_reward += episode_reward
    avg_reward /= episodes

    # Calculating average relevance scores across episodes for one seed:
    for i in range(env.action_space.shape[0]):
        for k in rel_score_edge[s][i].keys():
            scores = np.array(rel_score_edge_seed[i][k])
            rel_score_edge[s][i][k].append(np.mean(scores, axis=0))

    # writer.add_scalar('avg_reward/test', avg_reward, i_episode)

    print("----------------------------------------")
    print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
    print("----------------------------------------")

    x_total = [2 * x + step * width for x in range(len(rel_freq_edge[0][s]) + 1)]
    x_edge = [2 * x + step * width for x in range(len(rel_freq_edge[0][s]))]

    for i in range(env.action_space.shape[0]):
        total_relevance = np.array(list(rel_freq_edge[i][s].values()) +
                                   [rel_freq_global[s][i]]) / num_samples
        avg_rel_freq_edge[s][i] = np.array(list(rel_freq_edge[s][i].values())) / num_samples

        ax_row, ax_col = int(i / fig_num_cols), int(i % fig_num_cols)

        ax_total[ax_row, ax_col].bar(x_total, total_relevance, width, label=f'seed{s}')
        ax_edge[ax_row, ax_col].bar(x_edge, avg_rel_freq_edge[s][i], width, label=f'seed{s}')

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


# fig_name = os.path.join(exp_path, 'LRP_result_total.jpg')
# x_total = [2 * x for x in range(len(rel_freq_edge[0][0]) + 1)]
# keys = process_keys(rel_freq_edge[0][0].keys())
# keys += ['global\nfeatures']
#
# for i, j in zip(range(fig_rows), range(fig_cols)):
#     ax_total[i, j].set_xticks(x_total, keys, rotation=label_rotation)
#     ax_total[i, j].legend()
#     ax_total[i, j].set_xlabel("Name of the graph's nodes, edges, and global features")
#     ax_total[i, j].set_ylabel("LRP score")
#     ax_total[i, j].set_title("LRP score for each part of the input graph's nodes, edges and global features")
# fig_total.savefig(fig_name, dpi=300)

# Edge features plot
fig_name = os.path.join(exp_path, 'LRP_result_edges.jpg')
x_total = [2 * x for x in range(len(rel_freq_edge[0][0]))]
# keys = ['\n'.join(k.split(':')[1].split('_')) for k in rel_freq_edge[0].keys()]
keys = process_keys(rel_freq_edge[0][0].keys())

for i, j in zip(range(fig_rows), range(fig_cols)):
    ax_edge[i, j].set_xticks(x_total, keys, rotation=label_rotation)
    ax_edge[i, j].legend()
    ax_edge[i, j].set_xlabel("Name of the graph's edges (robot's joints)")
    ax_edge[i, j].set_ylabel("LRP score")
    ax_edge[i, j].set_title("LRP score for each part of the input graph's edges")
fig_edge.savefig(fig_name, dpi=300)

fig_name = os.path.join(exp_path, 'LRP_result_edges_avg.jpg')
for i in range(env.action_space.shape[0]):
    edge_relevances = np.zeros([len(experiment_seed), len(rel_freq_edge[0][0])])
    # average relevances over seeds
    for s in range(len(experiment_seed)):
        edge_relevances[s, :] = avg_rel_freq_edge[s][i]

    x_total = [x for x in range(len(rel_freq_edge[0][0]))]
    keys = process_keys(rel_freq_edge[0][0].keys())
    ax_row, ax_col = int(i / fig_num_cols), int(i % fig_num_cols)
    ax_edge_avg[ax_row, ax_col].bar(x_total, np.mean(edge_relevances, axis=0))
    ax_edge_avg[ax_row, ax_col].set_xticks(x_total, keys, rotation=label_rotation)
    ax_edge_avg[ax_row, ax_col].set_xlabel("Name of the graph's edges (robot's joints)")
    ax_edge_avg[ax_row, ax_col].set_ylabel(f"Average LRP score across {len(experiment_seed)} seeds")
    ax_edge_avg[ax_row, ax_col].set_title(f"Action {i}")
    ax_edge_avg[ax_row, ax_col].set_title("Average LRP score for each part of the input graph's edges")
fig_edge_avg.savefig(fig_name, dpi=300)

for k in rel_score_edge[0][0].keys():
    fig_name = os.path.join(exp_path, f'LRP_score_{k}.jpg')
    fig_episodic, ax_episodic = plt.subplots(fig_rows, fig_cols, figsize=[figure_width, figure_height])
    for i in range(env.action_space.shape[0]):
        scores = np.array(rel_score_edge[:][i][k])
        average_score = np.mean(scores, axis=0)
        std_score = np.std(scores, axis=0) / np.sqrt(scores.shape[0])
        x = np.linspace(1, average_score.shape[0], average_score.shape[0])

        ax_row, ax_col = int(i / fig_num_cols), int(i % fig_num_cols)
        ax_episodic[ax_row, ax_col].plot(x, average_score, label=k)
        ax_episodic[ax_row, ax_col].fill_between(x, average_score - 2.26 * std_score, average_score + 2.26 * std_score,
                                                 alpha=0.2)

        ax_episodic[ax_row, ax_col].set_xlabel("Time steps in an episode")
        ax_episodic[ax_row, ax_col].set_ylabel(f"Average LRP score across {len(experiment_seed)} seeds")

        ax_episodic[ax_row, ax_col].set_title("Average LRP score for each part of the input graph's edges at each step")
        ax_episodic[ax_row, ax_col].legend()
    fig_episodic.savefig(fig_name, dpi=300)
    fig_episodic.close()

env.close()
