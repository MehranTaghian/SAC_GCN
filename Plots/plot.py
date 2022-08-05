import seaborn as sns
from matplotlib import pyplot as plt
import gym
from CustomGymEnvs import FetchReachGraphWrapper, MujocoGraphNormalWrapper
import pandas as pd
import numpy as np
import argparse
import pathlib
import os
from scipy.stats import ttest_ind
import matplotlib.style as style
from pathlib import Path
from utils import load_object

parser = argparse.ArgumentParser(description="Draw results of the experiments inside a directory")

parser.add_argument('--env-name', default="FetchReach-v2",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--percentage', default=1, type=int,
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')

args = parser.parse_args()

X_AXIS = ['num_episodes', 'num_time_steps', 'num_updates', 'num_samples']
X_AXIS_TO_LABEL = {'num_time_steps': 'Time step',
                   'num_updates': 'Number of updates',
                   'num_samples': 'Number of samples',
                   'num_episodes': 'Number of episodes'}


def draw(env_exp_types, colors, title_curves, title_ttest):
    exp_type_eval_results = {}
    for type in env_exp_types:
        exp_type_path = os.path.join(exp_path, type)
        experiment_seed = os.listdir(exp_type_path)
        experiment_seed = [d for d in experiment_seed if os.path.isdir(os.path.join(exp_type_path, d))]
        num_seeds = len(experiment_seed)
        first = True
        eval_average_returns = None
        data_eval = None
        for seed in range(len(experiment_seed)):
            data_eval = pd.read_csv(os.path.join(exp_type_path, experiment_seed[seed], 'eval.csv'))
            data_eval = data_eval.loc[~(data_eval == 0).all(axis=1)]
            num_data_points_eval = int(len(data_eval) / args.percentage)

            if first:
                eval_average_returns = np.zeros([num_seeds, num_data_points_eval])
                first = False
            eval_average_returns[seed] = data_eval['eval_reward'][:num_data_points_eval]
        eval_average = np.mean(eval_average_returns, axis=0)
        eval_standard_error = np.std(eval_average_returns, axis=0) / np.sqrt(eval_average_returns.shape[0])

        eval_x = {
            'num_episodes': np.array(data_eval['num_episodes'][:num_data_points_eval]),
            'num_time_steps': np.array(data_eval['num_steps'][:num_data_points_eval]),
            'num_updates': np.array(data_eval['num_updates'][:num_data_points_eval]),
            'num_samples': np.array(data_eval['num_episodes'][:num_data_points_eval])}

        exp_type_eval_results[type] = (eval_x, eval_average, eval_standard_error)

    for x in X_AXIS:
        plot_learning_curve(exp_type_eval_results, x, 'Average Return', title_curves, colors)

    plot_significancy_test(exp_type_eval_results, title_ttest)


def plot_learning_curve(experiment_results, x_label, y_label, title, colors):
    width = 15
    height = 12
    sns.set_theme()
    sns.set(font_scale=2)
    fig, ax = plt.subplots(figsize=[width, height])
    for type in experiment_results.keys():
        x, average, standard_error = experiment_results[type]
        ax.plot(x[x_label], average, label=type, linewidth=3, color=colors[type])
        ax.fill_between(x[x_label], average - 2.26 * standard_error, average + 2.26 * standard_error,
                        color=colors[type],
                        alpha=0.2)
    ax.set_xlabel(X_AXIS_TO_LABEL[x_label])
    ax.set_ylabel(y_label)
    legs = fig.legend(ncol=3, fancybox=True, loc='lower center', bbox_to_anchor=(0.6, 0.1))
    for leg in legs.legendHandles:
        leg.set_linewidth(10.0)
    ax.set_title(title)
    # ax.spines.right.set_visible(False)
    # ax.spines.top.set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(exp_path, x_label + '.jpg'))
    sns.reset_orig()


def plot_significancy_test(exp_results, title):
    p_values = np.zeros([len(exp_results), len(exp_results)])

    for i, type1 in enumerate(exp_results.keys()):
        for j, type2 in enumerate(exp_results.keys()):
            p_values[i, j] = ttest_ind(exp_results[type1][1], exp_results[type2][1]).pvalue

    labels = [' '.join(j.split('_')[:-1]).strip() if len(j.split('_')) > 1 else j for j in exp_results.keys()]
    plot_t_test_heatmap(p_values, labels, title)


def plot_t_test_heatmap(data, labels, title):
    width = 10
    height = 10
    style.use('tableau-colorblind10')
    # style.use('seaborn-colorblind')
    plt.rcParams.update({'font.size': 14})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(width, height), gridspec_kw={'width_ratios': (30, 1)})
    mask = np.triu(np.ones_like(data))
    sns.heatmap(data, ax=ax1, cbar=False, cmap="cividis", linewidth=1, vmin=np.min(data), vmax=np.max(data),
                annot=True,
                fmt='.2f',
                mask=mask)
    ax1.set_xticks(np.arange(len(labels) - 1) + 0.5, labels=list(labels)[:-1], rotation=45)
    ax1.set_yticks(np.arange(len(labels) - 1) + 1.5, labels=list(labels)[1:], rotation=45)
    ax1.set_title(title, pad=40)
    ax1.set_ylabel("Joint name")
    ax1.set_xlabel("Joint name")

    plt.colorbar(plt.cm.ScalarMappable(cmap="YlGnBu", norm=plt.Normalize(vmin=np.min(data), vmax=np.max(data))),
                 cax=ax2)
    ax2.yaxis.set_ticks_position('left')
    ax2.set_ylabel('P values (P < 0.05 means they are statistically significantly different)')
    fig.tight_layout()
    fig.savefig(os.path.join(exp_path, 't-test.jpg'), dpi=300)


def plot_action_importance(action_rel, action_labels, pallet):
    colors = [pallet[l] for l in action_labels]
    sns.set_theme()
    sns.set(font_scale=2)
    plt.figure(figsize=(10, 8))
    plt.bar(action_labels, action_rel, color=colors)
    plt.xlabel('Actions (Torque applied to each joint)')
    plt.ylabel('Action importance score')
    plt.xticks(rotation=45)
    plt.title(f'Action importance - {env_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(exp_path, 'action_importance.jpg'), dpi=300)
    sns.reset_orig()


def plot_joint_importance(joint_rel, joint_labels, pallet):
    colors = [pallet[l] for l in joint_labels]
    joint_labels = [j if 'joint' not in j else j.split('joint')[0].strip() for j in joint_labels]
    sns.set_theme()
    sns.set(font_scale=2)
    plt.figure(figsize=(10, 8))
    plt.bar(joint_labels, joint_rel, color=colors)
    plt.xlabel('Joint name')
    plt.ylabel('Relevance score')
    plt.xticks(rotation=45)
    plt.title(f'Joint importance in the observation - {env_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(exp_path, 'joint_importance.jpg'), dpi=300)
    sns.reset_orig()


def process_joint_name(joint_name):
    separated = joint_name.split(':')[1].split('_') if 'robot0' in joint_name else joint_name.split('_')
    final_key = ''
    for sk in separated:
        if len(sk) == 1:
            final_key += sk + '-'
        else:
            final_key += sk + ' '
    return final_key.strip()


def get_joint_labels(edge_list):
    joint_labels = []
    for joint_list in edge_list.values():
        if len(joint_list) > 0:
            joint_labels.append(
                process_joint_name(joint_list[0].attrib['name'])
                if len(joint_list) == 1
                else '\n'.join([process_joint_name(j.attrib['name']) for j in joint_list])
            )

    if env_name == 'FetchReach-v2':
        joint_labels.remove('l-gripper finger joint')
        joint_labels.remove('r-gripper finger joint')

    return joint_labels, action_labels


if __name__ == "__main__":
    exp_path = os.path.join(pathlib.Path(__file__).parent.parent, 'Data', args.env_name)
    env_exp_types = os.listdir(exp_path)
    if 'graph' in env_exp_types:
        env_exp_types.remove('graph')
    env_exp_types = [d for d in env_exp_types if os.path.isdir(os.path.join(exp_path, d))]

    # pallet = plt.cm.cividis(np.linspace(0, 1, len(experiment_results.keys())))
    # pallet = plt.cm.tab20b(np.linspace(0, 1, len(experiment_results.keys())))
    pallet = sns.color_palette('colorblind', n_colors=len(env_exp_types))

    colors = {}
    for i, type in enumerate(env_exp_types):
        colors[type] = pallet[i]

    title_curves = f'Average return of the model on {args.env_name} after occluding joints'
    title_ttest = f'Statistical T-test of learning curves - {args.env_name} occluded joints'
    draw(env_exp_types, colors, title_curves, title_ttest)

    # ---------------------- BROKEN JOINTS------------------------------
    env_name = args.env_name.split('-')
    env_name[0] += 'Broken'
    env_name = '-'.join(env_name)
    exp_path = os.path.join(pathlib.Path(__file__).parent.parent, 'Data', env_name)
    env_exp_types = os.listdir(exp_path)
    if 'graph' in env_exp_types:
        env_exp_types.remove('graph')
    env_exp_types = [d for d in env_exp_types if os.path.isdir(os.path.join(exp_path, d))]

    title_curves = f'Average return of the model on {args.env_name} after blocking joints'
    title_ttest = f'Statistical T-test of learning curves - {args.env_name} blocked joints'
    draw(env_exp_types, colors, title_curves, title_ttest)

    # ---------------------- Importance plots --------------------------
    exp_path = os.path.join(Path(os.getcwd()), 'Data', args.env_name, 'graph')
    edge_rel_path = os.path.join(exp_path, 'edge_relevance.pkl')
    global_rel_path = os.path.join(exp_path, 'global_relevance.pkl')
    edge_relevance = load_object(edge_rel_path)
    global_relevance = load_object(global_rel_path)

    # Remove l-gripper-finger-joint and r-gripper-finger-joint
    if env_name == 'FetchReach-v2':
        edge_relevance = np.delete(edge_relevance, 7, axis=0)
        edge_relevance = np.delete(edge_relevance, 7, axis=0)

    # Environment
    if 'FetchReach' in env_name:
        env = FetchReachGraphWrapper(gym.make(env_name))
    else:
        env = MujocoGraphNormalWrapper(env_name)

    edge_list = env.robot_graph.edge_list

    avg_edge_rel = edge_relevance.mean(axis=4).mean(axis=3).mean(axis=2)
    avg_edge_rel /= np.max(np.abs(avg_edge_rel), axis=0)

    action_rel = np.abs(avg_edge_rel.mean(axis=0))
    action_rel /= np.max(action_rel)

    joint_rel = np.abs(action_rel.dot(avg_edge_rel.T))
    joint_rel /= np.max(joint_rel)

    joint_labels = get_joint_labels(edge_list)
    plot_joint_importance(joint_rel, joint_labels, colors)

    action_labels = [j for j in joint_labels if 'root' not in j]
    plot_action_importance(action_rel, action_labels, colors)
