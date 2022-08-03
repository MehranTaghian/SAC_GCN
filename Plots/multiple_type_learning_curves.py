import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import argparse
import pathlib
import os
from scipy.stats import ttest_ind
import matplotlib.style as style

# style.use('seaborn-colorblind')
style.use('tableau-colorblind10')

parser = argparse.ArgumentParser(description="Draw results of the experiments inside a directory")

parser.add_argument('--env-name', default="FetchReachEnvGraph-v0",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--percentage', default=1, type=int,
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')

args = parser.parse_args()

X_AXIS = ['num_episodes', 'num_time_steps', 'num_updates', 'num_samples']
X_AXIS_TO_LABEL = {'num_time_steps': 'Time step',
                   'num_updates': 'Number of updates',
                   'num_samples': 'Number of samples',
                   'num_episodes': 'Number of episodes'}

exp_path = os.path.join(pathlib.Path(__file__).parent.parent, 'Data', args.env_name)


def draw():
    env_exp_types = os.listdir(exp_path)
    if 'graph' in env_exp_types:
        env_exp_types.remove('graph')
    env_exp_types = [d for d in env_exp_types if os.path.isdir(os.path.join(exp_path, d))]
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
        plot_learning_curve(exp_type_eval_results, x, 'Average Return',
                            f'Average return of the model on {args.env_name} in different modes')

    plot_significancy_test(exp_type_eval_results)


def plot_learning_curve(experiment_results, x_label, y_label, title):
    width = 15
    height = 12
    sns.set_theme()
    sns.set(font_scale=2)
    fig, ax = plt.subplots(figsize=[width, height])
    colors = plt.cm.cividis(np.linspace(0, 1, len(experiment_results.keys())))
    # colors = plt.cm.tab20b(np.linspace(0, 1, len(experiment_results.keys())))
    # colors = sns.color_palette('colorblind', n_colors=len(experiment_results.keys()))
    for type, color in zip(experiment_results.keys(), colors):
        x, average, standard_error = experiment_results[type]
        ax.plot(x[x_label], average, label=type, color=color, linewidth=3)
        ax.fill_between(x[x_label], average - 2.26 * standard_error, average + 2.26 * standard_error,
                         color=color,
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

def plot_significancy_test(exp_results):
    p_values = np.zeros([len(exp_results), len(exp_results)])

    for i, type1 in enumerate(exp_results.keys()):
        for j, type2 in enumerate(exp_results.keys()):
            p_values[i, j] = ttest_ind(exp_results[type1][1], exp_results[type2][1]).pvalue

    labels = [' '.join(j.split('_')[:-1]).strip() if len(j.split('_')) > 1 else j for j in exp_results.keys()]
    plot_t_test_heatmap(p_values, labels)


def plot_t_test_heatmap(data, labels):
    width = 10
    height = 10
    plt.rcParams.update({'font.size': 14})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(width, height), gridspec_kw={'width_ratios': (30, 1)})
    mask = np.triu(np.ones_like(data))
    sns.heatmap(data, ax=ax1, cbar=False, cmap="cividis", linewidth=1, vmin=np.min(data), vmax=np.max(data),
                annot=True,
                fmt='.2f',
                mask=mask)
    ax1.set_xticks(np.arange(len(labels) - 1) + 0.5, labels=list(labels)[:-1], rotation=45)
    ax1.set_yticks(np.arange(len(labels) - 1) + 1.5, labels=list(labels)[1:], rotation=45)
    ax1.set_title(
        "Statistical T-Test of learning curves",
        fontsize=20, pad=40)
    ax1.set_ylabel("Occluded joint name")
    ax1.set_xlabel(f"Occluded joint name")

    plt.colorbar(plt.cm.ScalarMappable(cmap="YlGnBu", norm=plt.Normalize(vmin=np.min(data), vmax=np.max(data))),
                 cax=ax2)
    ax2.yaxis.set_ticks_position('left')
    ax2.set_ylabel('P values (P < 0.05 means they are statistically significantly different)')

    fig.savefig(os.path.join(exp_path, 't-test.jpg'), dpi=300)


if __name__ == "__main__":
    draw()
