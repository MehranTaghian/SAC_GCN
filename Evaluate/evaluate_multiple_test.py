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

num_episodes = 1

edge_list = env.robot_graph.edge_list
node_list = env.robot_graph.node_list


# TODO edit this function to be inline
def process_joint_name(joint_name):
    if 'robot0' in joint_name:
        sep_key = joint_name.split(':')[1].split('_')
    else:
        sep_key = joint_name.split('_')
    final_key = ''
    for sk in sep_key:
        if len(sk) == 1:
            final_key += sk + '-'
        else:
            final_key += sk + ' '
    return final_key


joint_names = []
for joint_list in edge_list.values():
    if len(joint_list) > 0:
        joint_names.append(
            process_joint_name(joint_list[0].attrib['name'])
            if len(joint_list) == 1
            else '\n'.join([process_joint_name(j.attrib['name']) for j in joint_list])
        )

# rel_edge_sum[joint_index, action_index, seed] = [sum of relevance scores within an episode]
# rel_edge_episode[joint_index, action_index, seed, num_episodes, episode_steps] =
#                 [relevance score for each time-step of episode]
# rel_global_sum[action_index, seed] = [sum of relevance scores for global feature within an episode]
rel_edge_sum = np.zeros([len(joint_names), env.action_space.shape[0], len(experiment_seed)])
rel_edge_episode = np.zeros(
    [len(joint_names),
     env.action_space.shape[0],
     len(experiment_seed),
     num_episodes,
     env.spec.max_episode_steps])

rel_global_sum = np.zeros([env.action_space.shape[0], len(experiment_seed)])


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

    num_samples = 0
    avg_reward = 0.

    for episode in tqdm(range(num_episodes)):
        state = env.reset()
        episode_reward = 0
        done = False
        step = 0
        while not done:
            for action_index in range(env.action_space.shape[0]):
                batch_state = state_2_graphbatch(state).requires_grad_().to(device)
                out = agent_relevance.policy.graph_net(batch_state)
                out = agent_relevance.policy.mean_linear(out)[0]
                global_relevance = torch.zeros_like(out.global_features)
                global_relevance[action_index] = out.global_features[action_index]
                batch_state.zero_grad_()
                out.global_features.backward(global_relevance)

                edge_rel = batch_state.edge_features.grad.sum(dim=1)
                global_rel = batch_state.global_features.grad.sum(dim=1)
                rel_global_sum[action_index, s] += global_rel

                for joint_index, joint_name in enumerate(joint_names):
                    rel_edge_sum[joint_index, action_index, s] += edge_rel[joint_index]
                    rel_edge_episode[joint_index, action_index, s, episode, step] = edge_rel[joint_index]
            step += 1
            num_samples += 1
            action = agent.select_action(state_2_graphbatch(state), evaluate=True)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state
        avg_reward += episode_reward
    avg_reward /= num_episodes

    # Calculating average relevance scores across episodes for one seed, also average of total relevance:
    rel_edge_sum[:, :, s] /= num_samples

    print("----------------------------------------")
    print("Test Episodes: {}, Avg. Reward: {}".format(num_episodes, round(avg_reward, 2)))
    print("----------------------------------------")

    step += 1

env.close()

figure_width = 35
figure_height = 16
label_rotation = 0

fig_edge_avg, ax_edge_avg = plt.subplots(figsize=[figure_width, figure_height])
fig_name = os.path.join(exp_path, 'edge_relevance_heatmap.jpg')

action_indices = [a for a in range(env.action_space.shape[0])]
edge_relevances = np.zeros([len(joint_names), len(action_indices)])
# average relevances over seeds
for joint_index in range(len(joint_names)):
    for action_index in action_indices:
        edge_relevances[joint_index, action_index] = np.mean(rel_edge_sum[joint_index, action_index, :])

edge_relevances = edge_relevances / np.abs(edge_relevances).max(axis=0)

im = ax_edge_avg.imshow(edge_relevances)
cbar = ax_edge_avg.figure.colorbar(im, ax=ax_edge_avg)
cbar.ax.set_ylabel('Avg relevance score across seeds', rotation=-90, va="bottom")
ax_edge_avg.set_xticks(np.arange(len(action_indices)), labels=action_indices)
ax_edge_avg.set_yticks(np.arange(len(joint_names)), labels=[j for j in reversed(joint_names)])
ax_edge_avg.set_xlabel("Action index")
ax_edge_avg.set_ylabel(f"Joints' names")
ax_edge_avg.set_title("Average relevance score of each action given to each joint")
fig_edge_avg.savefig(fig_name, dpi=300)

################ PLOT EPISODIC

fig_num_cols = int(np.ceil(np.sqrt(env.action_space.shape[0])))
fig_rows, fig_cols = (int(env.action_space.shape[0] / fig_num_cols)
                      if env.action_space.shape[0] % fig_num_cols == 0
                      else int(env.action_space.shape[0] / fig_num_cols)), fig_num_cols
fig_name = os.path.join(exp_path, f'LRP_score_episodic.jpg')
fig_episodic, ax_episodic = plt.subplots(fig_rows, fig_cols, figsize=[figure_width, figure_height])

for action_index in range(env.action_space.shape[0]):
    scores = rel_edge_episode[:, action_index, :, :, :]
    avg_episode_scores = np.mean(scores, axis=2)
    avg_score = np.mean(avg_episode_scores, axis=1)
    std_score = np.std(avg_episode_scores, axis=1) / np.sqrt(avg_episode_scores.shape[0])
    x = np.linspace(1, avg_score.shape[1], avg_score.shape[1])

    ax_row, ax_col = int(action_index / fig_num_cols), int(action_index % fig_num_cols)
    for joint_index, joint_name in enumerate(joint_names):
        ax_episodic[ax_row, ax_col].plot(x, avg_score[joint_index], label=joint_name)
        ax_episodic[ax_row, ax_col].fill_between(x, avg_score[joint_index] - 2.26 * std_score[joint_index],
                                                 avg_score[joint_index] + 2.26 * std_score[joint_index],
                                                 alpha=0.2)

    ax_episodic[ax_row, ax_col].set_xlabel("Time steps in an episode")
    ax_episodic[ax_row, ax_col].set_ylabel(f"Average LRP score across {len(experiment_seed)} seeds")

    ax_episodic[ax_row, ax_col].set_title("Average LRP score for each part of the input graph's edges at each step")
    ax_episodic[ax_row, ax_col].legend()

fig_episodic.savefig(fig_name, dpi=300)
