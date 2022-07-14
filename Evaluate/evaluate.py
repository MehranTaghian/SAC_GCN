import argparse
import os
import gym
import numpy as np
import torch
from tqdm import tqdm
from CustomGymEnvs import MujocoGraphNormalWrapper, FetchReachGraphWrapper
from pathlib import Path
from Graph_SAC.sac import SAC
from utils import state_2_graphbatch, load_object, save_object
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('tableau-colorblind10')

plt.rcParams['font.size'] = '20'

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="FetchReachDense-v1",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--exp-type', default="standard",
                    help='Type of the experiment like normal or abnormal')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--num-episodes', type=int, default=10, metavar='N',
                    help='Number of episodes for evaluation')

args = parser.parse_args()
env_name = args.env_name
seed = args.seed
exp_path = os.path.join(Path(__file__).parent.parent, 'Data', args.env_name, args.exp_type)
args = load_object(os.path.join(exp_path, 'seed0', 'parameters.pkl'))
args.env_name = env_name
args.seed = seed

experiment_seed = os.listdir(exp_path)
experiment_seed = [d for d in experiment_seed if os.path.isdir(os.path.join(exp_path, d))]
# experiment_seed = experiment_seed[:2]
if args.seed < len(experiment_seed):
    experiment_seed = [f'seed{args.seed}']
    exp_path = os.path.join(exp_path, experiment_seed[0])

# Environment
if 'FetchReach' in args.env_name:
    env = FetchReachGraphWrapper(gym.make(args.env_name))
else:
    env = MujocoGraphNormalWrapper(gym.make(args.env_name))

num_node_features = env.observation_space['node_features'].shape[1]
num_edge_features = env.observation_space['edge_features'].shape[1]
num_global_features = env.observation_space['global_features'].shape[0]

# device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
device = torch.device('cpu')
args.cuda = False

num_episodes = 20

edge_list = env.robot_graph.edge_list
node_list = env.robot_graph.node_list

render = False


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
joint_indices = []
for edge_id, joint_list in enumerate(edge_list.values()):
    if len(joint_list) > 0:
        joint_names.append(
            process_joint_name(joint_list[0].attrib['name'])
            if len(joint_list) == 1
            else '\n'.join([process_joint_name(j.attrib['name']) for j in joint_list])
        )
        joint_indices.append(edge_id)

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
    for s, seed in enumerate(experiment_seed):
        seed = int(seed[-1])
        env.seed(seed)
        env.action_space.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        # Agent
        checkpoint_path = os.path.join(exp_path, f'seed{seed}', 'model') \
            if len(experiment_seed) > 1 \
            else os.path.join(exp_path, 'model')
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
                    edge_relevance[:, action_index, s, episode, step] = edge_rel[joint_indices]
                step += 1
                action = agent.select_action(state_2_graphbatch(state), evaluate=True)
                next_state, reward, done, _ = env.step(action)
                episode_reward += reward
                state = next_state
                if render:
                    env.render()
            avg_reward += episode_reward
        avg_reward /= num_episodes

        print("----------------------------------------")
        print("Test Episodes: {}, Avg. Reward: {}".format(num_episodes, round(avg_reward, 2)))
        print("----------------------------------------")

    env.close()


def plot_joint_action_heatmap(data, width, height, title, file_name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(width, height), gridspec_kw={'width_ratios': (30, 1)})
    sns.heatmap(data, ax=ax1, cbar=False, cmap="YlGn", linewidth=1, vmin=np.min(data), vmax=np.max(data))
    ax1.set_xticks(np.arange(len(joint_names)) + 0.5, labels=[j for j in joint_names], rotation=45)
    ax1.set_title(title, fontsize=20, pad=40)
    ax1.set_ylabel("Action index")
    ax1.set_xlabel(f"Joints' names")

    ax3 = ax1.twiny()
    ax3.set_xlim([0, ax1.get_xlim()[1]])
    ax3.set_xticks(ax1.get_xticks())
    ax3.set_xticklabels(np.round(data.mean(axis=0), 2))

    plt.colorbar(plt.cm.ScalarMappable(cmap="YlGn", norm=plt.Normalize(vmin=np.min(data), vmax=np.max(data))), cax=ax2)
    ax2.yaxis.set_ticks_position('left')
    ax2.set_ylabel('Avg relevance score across seeds')

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax1.text(j + 0.5, i + 0.5, round(data[i, j], 3), color="black")

    fig.savefig(file_name, dpi=300)


def plot_joint_action_timestep_curve(ax, data, title, palette=sns.color_palette('colorblind'), label_y_axis=True):
    avg_episodes = np.mean(data, axis=2)
    avg_seeds = np.mean(avg_episodes, axis=1)
    std_seeds = np.std(avg_episodes, axis=1) / np.sqrt(avg_episodes.shape[0])
    x = np.linspace(1, avg_seeds.shape[1], avg_seeds.shape[1])
    for joint_index, joint_name in enumerate(joint_names):
        ax.plot(x, avg_seeds[joint_index], label=joint_name, color=palette[joint_index])
        ax.fill_between(x, avg_seeds[joint_index] - 2.26 * std_seeds[joint_index],
                        avg_seeds[joint_index] + 2.26 * std_seeds[joint_index],
                        alpha=0.2, color=palette[joint_index])

    ax.set_xlabel("Time steps")
    if label_y_axis:
        ax.set_ylabel(f"Average LRP score across {len(experiment_seed)} seeds")
    ax.set_title(title)
    return ax.get_legend_handles_labels()


if __name__ == '__main__':
    figure_width = 35
    figure_height = 16

    calculate_relevance()
    save_object(edge_relevance, os.path.join(exp_path, 'edge_relevance.pkl'))

    fig_name = os.path.join(exp_path, 'edge_relevance_heatmap.jpg')
    # average across steps in an episode, across episodes, then across seeds
    avg_edge_rel = edge_relevance.mean(axis=4).mean(axis=3).mean(axis=2)
    avg_edge_rel /= np.max(np.abs(avg_edge_rel), axis=0)
    plot_joint_action_heatmap(np.abs(avg_edge_rel.T),
                              figure_width,
                              figure_height,
                              "Avg actions' relevance scores given to joints",
                              fig_name)

    # fig_num_cols = int(np.ceil(np.sqrt(env.action_space.shape[0])))
    # fig_rows, fig_cols = (int(env.action_space.shape[0] / fig_num_cols)
    #                       if env.action_space.shape[0] % fig_num_cols == 0
    #                       else int(env.action_space.shape[0] / fig_num_cols)), fig_num_cols
    # fig_episodic, ax_episodic = plt.subplots(fig_rows, fig_cols, figsize=[figure_width, figure_height])
    # fig_name = os.path.join(exp_path, f'LRP_score_episodic.jpg')
    # handles, labels = None, None
    # for action_index in range(env.action_space.shape[0]):
    #     scores = edge_relevance[:, action_index, :, :, :]
    #     row, col = int(action_index / fig_num_cols), int(action_index % fig_num_cols)
    #     handles, labels = plot_joint_action_timestep_curve(ax_episodic[row, col],
    #                                                        scores,
    #                                                        f'Relevance scores for action {action_index}',
    #                                                        label_y_axis=(col == 0))
    #
    # fig_episodic.legend(handles,  # The line objects
    #                     labels,  # The labels for each line
    #                     loc="center right",  # Position of legend
    #                     borderaxespad=0.1,  # Small spacing around legend box
    #                     title="Joints"  # Title for the legend
    #                     )
    #
    # fig_episodic.suptitle("Average actions' relevance scores given to each joint at each step")
    # # Adjust the scaling factor to fit your legend text completely outside the plot
    # # (smaller value results in more space being made for the legend)
    # plt.subplots_adjust(right=0.85)
    # fig_episodic.savefig(fig_name, dpi=300)
