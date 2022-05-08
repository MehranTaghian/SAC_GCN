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
from utils import state_2_graphbatch

# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('tableau-colorblind10')

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
    separated = joint_name.split(':')[1].split('_') if 'robot0' in joint_name else joint_name.split('_')
    final_key = ''
    for sk in separated:
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

action_indices = [a for a in range(env.action_space.shape[0])]

# edge_relevance[joint_index, action_index, seed, num_episodes, episode_steps] =
#                 [relevance score for each time-step of episode]
# global_relevance[action_index, seed] = [sum of relevance scores for global feature within an episode]
edge_relevance = np.zeros(
    [len(joint_names),
     env.action_space.shape[0],
     len(experiment_seed),
     num_episodes,
     env.spec.max_episode_steps])

global_relevance = np.zeros([env.action_space.shape[0], len(experiment_seed)])


def calculate_relevance():
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
                    output_relevance = torch.zeros_like(out.global_features)
                    output_relevance[action_index] = out.global_features[action_index]
                    batch_state.zero_grad_()
                    out.global_features.backward(output_relevance)

                    edge_rel = batch_state.edge_features.grad.sum(dim=1)
                    global_rel = batch_state.global_features.grad.sum(dim=1)
                    global_relevance[action_index, s] += global_rel

                    for joint_index, joint_name in enumerate(joint_names):
                        edge_relevance[joint_index, action_index, s, episode, step] = edge_rel[joint_index]
                step += 1
                action = agent.select_action(state_2_graphbatch(state), evaluate=True)
                next_state, reward, done, _ = env.step(action)
                episode_reward += reward
                state = next_state
            avg_reward += episode_reward
        avg_reward /= num_episodes

        print("----------------------------------------")
        print("Test Episodes: {}, Avg. Reward: {}".format(num_episodes, round(avg_reward, 2)))
        print("----------------------------------------")

    env.close()


def plot_joint_action_heatmap(fig, ax, data, title, file_name):
    im = ax.imshow(data)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Avg relevance score across seeds', rotation=-90, va="bottom")
    ax.set_xticks(np.arange(len(action_indices)), labels=action_indices)
    ax.set_yticks(np.arange(len(joint_names)), labels=[j for j in reversed(joint_names)])
    ax.set_xlabel("Action index")
    ax.set_ylabel(f"Joints' names")
    ax.set_title(title)
    fig.savefig(file_name, dpi=300)


def plot_joint_action_timestep_curve(ax, data, title, palette=sns.color_palette('colorblind'), label_y_axis=True):
    avg_episodes = np.mean(data, axis=2)
    avg_seeds = np.mean(avg_episodes, axis=1)
    std_seeds = np.std(avg_episodes, axis=1) / np.sqrt(avg_episodes.shape[0])
    x = np.linspace(1, avg_seeds.shape[1], avg_seeds.shape[1])
    label_list = []
    for joint_index, joint_name in enumerate(joint_names):
        ax.plot(x, avg_seeds[joint_index], label=joint_name, color=palette[joint_index])
        ax.fill_between(x, avg_seeds[joint_index] - 2.26 * std_seeds[joint_index],
                        avg_seeds[joint_index] + 2.26 * std_seeds[joint_index],
                        alpha=0.2, color=palette[joint_index])

    ax.set_xlabel("Time steps")
    if label_y_axis:
        ax.set_ylabel(f"Average LRP score across {len(experiment_seed)} seeds")
    ax.set_title(title)
    ax.legend()

    return label_list


if __name__ == '__main__':
    figure_width = 35
    figure_height = 16

    calculate_relevance()

    fig_edge_avg, ax_edge_avg = plt.subplots(figsize=[figure_width, figure_height])
    fig_name = os.path.join(exp_path, 'edge_relevance_heatmap.jpg')
    # average across steps in an episode, across episodes, then across seeds
    avg_edge_rel = edge_relevance.mean(axis=4).mean(axis=3).mean(axis=2)
    plot_joint_action_heatmap(fig_edge_avg,
                              ax_edge_avg,
                              avg_edge_rel,
                              "Avg actions' relevance scores given to joints",
                              fig_name)

    fig_num_cols = int(np.ceil(np.sqrt(env.action_space.shape[0])))
    fig_rows, fig_cols = (int(env.action_space.shape[0] / fig_num_cols)
                          if env.action_space.shape[0] % fig_num_cols == 0
                          else int(env.action_space.shape[0] / fig_num_cols)), fig_num_cols
    fig_episodic, ax_episodic = plt.subplots(fig_rows, fig_cols, figsize=[figure_width, figure_height])
    fig_name = os.path.join(exp_path, f'LRP_score_episodic.jpg')
    # labels = []
    for action_index in range(env.action_space.shape[0]):
        scores = edge_relevance[:, action_index, :, :, :]
        row, col = int(action_index / fig_num_cols), int(action_index % fig_num_cols)
        labels = plot_joint_action_timestep_curve(ax_episodic[row, col],
                                                   scores,
                                                   "Average LRP score for each part of "
                                                   "the input graph's edges at each step",
                                                   label_y_axis=(col == 0))

    fig_episodic.legend(labels,  # The line objects
                        labels=joint_names,  # The labels for each line
                        loc="center right",  # Position of legend
                        borderaxespad=0.1,  # Small spacing around legend box
                        title="Joints"  # Title for the legend
                        )

    # Adjust the scaling factor to fit your legend text completely outside the plot
    # (smaller value results in more space being made for the legend)
    plt.subplots_adjust(right=0.85)

    fig_episodic.savefig(fig_name, dpi=300)
