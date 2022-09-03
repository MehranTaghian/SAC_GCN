import seaborn as sns
from matplotlib import pyplot as plt
import gym
from CustomGymEnvs import FetchReachGraphWrapper, MujocoGraphNormalWrapper
import pandas as pd
import numpy as np
import argparse
import os
from scipy.stats import ttest_ind
from pathlib import Path
from utils import load_object

parser = argparse.ArgumentParser(description="Draw results of the experiments inside a directory")

parser.add_argument('--env-name', default="FetchReach-v2",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--percentage', default=1, type=int,
                    help='The percentage of time-steps for learning curve')
parser.add_argument('--window-size', default=40, type=int,
                    help='How many final episodes should be considered as performance')

args = parser.parse_args()

params = {
    'font.size': 16,
    'legend.fontsize': 'x-large',
    'axes.labelsize': 'x-large',
    'axes.titlesize': 'x-large',
    'xtick.labelsize': 'x-large',
    'ytick.labelsize': 'x-large'
}
plt.rcParams.update(params)

root_path = os.path.join(Path(__file__).parent.parent)
result_path = os.path.join(root_path, 'Result')

if not os.path.exists(result_path):
    os.makedirs(result_path)

patterns = ["/", "\\", "|", "-", "+", "x", "o", "O", ".", "*"]


def eval(exp_types, exp_path):
    exp_type_eval_results = {}
    for type in exp_types:
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

        eval_x = np.array(data_eval['num_episodes'][:num_data_points_eval])
        exp_type_eval_results[type] = (eval_x, eval_average, eval_standard_error)

    return exp_type_eval_results


# def plot_learning_curve(ax, experiment_results, y_label, title, colors):
#     for type in experiment_results.keys():
#         x, average, standard_error = experiment_results[type]
#         entity_name = ' '.join(type.split('_'))
#         ax.plot(x, average, label=entity_name, linewidth=3, color=colors[entity_name])
#         ax.fill_between(x, average - 2.26 * standard_error, average + 2.26 * standard_error,
#                         color=colors[entity_name],
#                         alpha=0.2)
#     ax.set_ylabel(y_label)
#     ax.set_title(title)

def plot_performance_bar(ax, experiment_results, y_label, title, colors, labels):
    for i, type in enumerate(labels):
        key = '_'.join(type.split(' '))
        _, average, _ = experiment_results[key]
        if type != 'standard':
            ax.bar(i, np.mean(average[:-args.window_size]), color=colors[type], label=type)

    _, average, _ = experiment_results['standard']
    ax.axhline(y=np.mean(average[:-args.window_size]),
               # xmin=0, xmax=len(experiment_results.keys()),
               label='standard',
               linestyle='--',
               linewidth=3)

    ax.set_ylabel(y_label)
    ax.set_title(title)


def plot_action_importance(ax, action_rel, action_labels, pallet):
    colors = [pallet[l] for l in action_labels]
    action_labels = [a if 'joint' not in a else a.split('joint')[0].strip() for a in action_labels]
    ax.bar(action_labels, action_rel, color=colors)
    ax.set_ylabel('Importance score')
    ax.set_title(f'Joint importance in the action space')


def plot_joint_importance(ax, joint_rel, joint_labels, pallet):
    colors = [pallet[l] for l in joint_labels]
    joint_labels = [j if 'joint' not in j else j.split('joint')[0].strip() for j in joint_labels]
    ax.bar(joint_labels, joint_rel, color=colors)
    ax.set_ylabel('Importance score')
    ax.set_title(f'Entity importance in the observation space')


def significancy_test(exp_results):
    p_values = np.zeros([len(exp_results), len(exp_results)])

    for i, type1 in enumerate(exp_results.keys()):
        for j, type2 in enumerate(exp_results.keys()):
            p_values[i, j] = ttest_ind(exp_results[type1][1], exp_results[type2][1]).pvalue

    return p_values


def plot_t_test_heatmap(ax1, data, labels, cbar=True):
    mask = np.triu(np.ones_like(data))
    ax1.set_facecolor((0, 0, 0, 0))
    labels = [l.split('joint')[0] if 'joint' in l else l for l in labels]
    sns.heatmap(data, ax=ax1, cmap="cividis", linewidth=0.8, vmin=np.min(data), vmax=np.max(data),
                annot=True,
                fmt='.2f',
                annot_kws={"size": 15},
                mask=mask,
                cbar=cbar,
                cbar_kws={'shrink': 0.75})
    ax1.set_xticks(np.arange(len(labels) - 1) + 0.5, labels=list(labels)[:-1], rotation=90)
    ax1.set_yticks(np.arange(len(labels) - 1) + 1.5, labels=list(labels)[1:], rotation=0)
    ax1.set_ylabel("Learning Curve Label")


def process_joint_name(joint_name):
    separated = joint_name.split(':')[1].split('_') if 'robot0' in joint_name else joint_name.split('_')
    final_key = ''
    for sk in separated:
        if len(sk) == 1:
            final_key += sk + '-'
        else:
            final_key += sk + ' '
    return final_key.strip()


def get_labels(env_name, edge_list):
    entity_names = []
    for joint_list in edge_list.values():
        if len(joint_list) > 0:
            if len(joint_list) == 1:
                joint_label = process_joint_name(joint_list[0].attrib['name'])
            elif len(joint_list) > 1 and 'root' in joint_list[0].attrib['name']:
                joint_label = 'torso'
            else:
                joint_label = '\n'.join([process_joint_name(j.attrib['name']) for j in joint_list])
            entity_names.append(joint_label)

    if env_name == 'FetchReach-v2':
        entity_names = ['goal'] + entity_names
        entity_names.remove('l-gripper finger joint')
        entity_names.remove('r-gripper finger joint')

    action_labels = [j for j in entity_names if j not in ['torso', 'goal']]

    return entity_names, action_labels


def get_lrp(env_name):
    exp_path = os.path.join(root_path, 'Data', env_name, 'graph')
    obj_path = os.path.join(exp_path, 'edge_relevance.pkl')
    edge_relevance = load_object(obj_path)

    global_relevance = None
    if env_name == 'FetchReach-v2':
        obj_path = os.path.join(exp_path, 'global_relevance.pkl')
        global_relevance = load_object(obj_path)
        # remove l_gripper_finger_joint and r_gripper_finger_joint
        edge_relevance = np.delete(edge_relevance, 7, axis=0)
        edge_relevance = np.delete(edge_relevance, 7, axis=0)

    avg_relevance = edge_relevance.sum(axis=4).sum(axis=3).sum(axis=2)

    if global_relevance is not None:
        avg_global_rel = global_relevance.sum(axis=3).sum(axis=2).sum(axis=1)
        avg_relevance = np.concatenate((avg_global_rel[np.newaxis, :], avg_relevance))

    # -------------- Normalization ---------------------
    avg_relevance /= np.max(np.abs(avg_relevance), axis=0)

    return edge_relevance, global_relevance, avg_relevance


if __name__ == "__main__":
    sns.set_theme()
    sns.set(font_scale=1.75)
    width, height = 27, 16
    fig, ax = plt.subplots(2, 4, figsize=(width, height), gridspec_kw={'width_ratios': (7, 7, 12, 1)})

    # ---------------------- Importance plots --------------------------
    edge_relevance, global_relevance, avg_relevance = get_lrp(args.env_name)

    if 'FetchReach' in args.env_name:
        env = FetchReachGraphWrapper(gym.make(args.env_name))
    else:
        env = MujocoGraphNormalWrapper(args.env_name)
    edge_list = env.robot_graph.edge_list
    entity_names, action_labels = get_labels(args.env_name, edge_list)

    # pallet = plt.cm.cividis(np.linspace(0, 1, len(experiment_results.keys())))
    # pallet = plt.cm.tab20b(np.linspace(0, 1, len(experiment_results.keys())))
    pallet = sns.color_palette('colorblind', n_colors=len(entity_names))
    colors = {}
    for i, type in enumerate(entity_names):
        colors[type] = pallet[i]

    joint_importance = np.abs(avg_relevance.mean(axis=1))
    joint_importance /= np.max(joint_importance)

    action_importance = edge_relevance.sum(axis=4).sum(axis=3).sum(axis=2).sum(axis=0)
    if global_relevance is not None:
        action_importance += global_relevance.sum(axis=3).sum(axis=2).sum(axis=1)
    action_importance = np.abs(action_importance)
    action_importance /= np.max(action_importance)

    plot_joint_importance(ax[0, 0], joint_importance, entity_names, colors)
    plot_action_importance(ax[1, 0], action_importance, action_labels, colors)
    # ------------------------- Performance Occluded ---------------------------------
    exp_path = os.path.join(root_path, 'Data', args.env_name)
    exp_types = os.listdir(exp_path)
    if 'graph' in exp_types:
        exp_types.remove('graph')
    exp_types = [d for d in exp_types if os.path.isdir(os.path.join(exp_path, d))]
    title_curves = f'Average return after occluding joints'
    eval_results = eval(exp_types, exp_path)

    # plot_learning_curve(ax[0, 1], eval_results, 'Average Return', title_curves, colors)
    plot_performance_bar(ax[0, 1], eval_results, 'Average Return', title_curves, colors, entity_names)
    _, labels = ax[0, 1].get_legend_handles_labels()
    p_values = significancy_test(eval_results)
    plot_t_test_heatmap(ax[0, 2], p_values, labels, cbar=True)

    # ---------------------- Performance Blocked ------------------------------
    env_name = args.env_name.split('-')
    env_name[0] += 'Broken'
    env_name = '-'.join(env_name)
    exp_path = os.path.join(root_path, 'Data', env_name)
    exp_types = os.listdir(exp_path)
    if 'graph' in exp_types:
        exp_types.remove('graph')
    exp_types = [d for d in exp_types if os.path.isdir(os.path.join(exp_path, d))]

    title_curves = f'Average return after blocking joints'
    eval_results = eval(exp_types, exp_path)

    # plot_learning_curve(ax[1, 1], eval_results, 'Average Return', title_curves, colors)
    plot_performance_bar(ax[1, 1], eval_results, 'Average Return', title_curves, colors, action_labels)
    ax[1, 1].set_xlabel("Number of Episodes")
    ax[0, 1].get_shared_x_axes().join(ax[0, 1], ax[1, 1])
    ax[0, 1].set_xticklabels([])
    _, labels = ax[1, 1].get_legend_handles_labels()
    p_values = significancy_test(eval_results)
    plot_t_test_heatmap(ax[1, 2], p_values, labels, cbar=True)

    ax[0, 2].set_title('Learning curve statistical T-test')
    ax[1, 2].set_xlabel("Learning Curve Label")

    # -----------------------------------------------------------------------------------

    for i in range(ax.shape[0]):
        ax[i, -1].axis('off')
    handles, labels = ax[0, 1].get_legend_handles_labels()
    leg = fig.legend(handles, labels,
                     # bbox_to_anchor=(1.25, 0.5),
                     loc='center right',
                     borderaxespad=1,
                     title="Entity names")

    ax[0, 0].set_xticklabels([])
    ax[1, 0].set_xticklabels([])
    ax[1, 1].set_xticklabels([])
    ax[1, 0].set_xlabel('Entities')

    # Adjust the scaling factor to fit your legend text completely outside the plot
    # (smaller value results in more space being made for the legend)

    for i, l in enumerate(leg.get_lines()):
        if leg.texts[i].get_text() != 'standard':
            l.set_linewidth(6)

    fig.suptitle(f"Evaluating importance scores for {args.env_name} environment")
    fig.tight_layout()
    fig.savefig(os.path.join(result_path, f'{args.env_name}.jpg'), dpi=300)
