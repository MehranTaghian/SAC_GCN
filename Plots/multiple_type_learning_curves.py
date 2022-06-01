import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import argparse
import pathlib
import os, shutil, pickle
from matplotlib.patches import ConnectionPatch
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Draw results of the experiments inside a directory")

parser.add_argument('--env-name', default="FetchReachEnvGraph-v0",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')

args = parser.parse_args()

X_AXIS = ['num_time_steps', 'num_updates', 'num_samples']
X_AXIS_TO_LABEL = {'num_time_steps': 'Time step',
                   'num_updates': 'Number of updates',
                   'num_samples': 'Number of samples'}

exp_path = os.path.join(pathlib.Path(__file__).parent.parent, 'Data', args.env_name)


def draw():
    env_exp_types = os.listdir(exp_path)
    env_exp_types = [d for d in env_exp_types if os.path.isdir(os.path.join(exp_path, d))]
    exp_type_train_results = {}
    exp_type_eval_results = {}
    for type in env_exp_types:
        exp_type_path = os.path.join(exp_path, type)
        experiment_seed = os.listdir(exp_type_path)
        experiment_seed = [d for d in experiment_seed if os.path.isdir(os.path.join(exp_type_path, d))]
        num_seeds = len(experiment_seed)
        first = True
        train_average_returns = None
        eval_average_returns = None
        data_train = None
        data_eval = None
        for seed in range(len(experiment_seed)):
            data_train = pd.read_csv(os.path.join(exp_type_path, experiment_seed[seed], 'train.csv'))
            data_train = data_train[(data_train.T != 0).any()]
            data_eval = pd.read_csv(os.path.join(exp_type_path, experiment_seed[seed], 'eval.csv'))
            data_eval = data_eval[(data_eval.T != 0).any()]

            if first:
                train_average_returns = np.zeros([num_seeds, len(data_train)])
                eval_average_returns = np.zeros([num_seeds, len(data_eval)])
                first = False
            train_average_returns[seed] = data_train['train_reward']
            eval_average_returns[seed] = data_eval['eval_reward']

        train_average = np.mean(train_average_returns, axis=0)
        train_standard_error = np.std(train_average_returns, axis=0) / np.sqrt(train_average_returns.shape[0])
        eval_average = np.mean(eval_average_returns, axis=0)
        eval_standard_error = np.std(eval_average_returns, axis=0) / np.sqrt(eval_average_returns.shape[0])

        train_x = {'num_time_steps': np.array(data_train['num_steps']),
                   'num_updates': np.array(data_train['num_updates']),
                   'num_samples': np.array(data_train['num_episodes'])}

        eval_x = {'num_time_steps': np.array(data_eval['num_steps']),
                  'num_updates': np.array(data_eval['num_updates']),
                  'num_samples': np.array(data_eval['num_episodes'])}

        exp_type_train_results[type] = (train_x, train_average, train_standard_error)
        exp_type_eval_results[type] = (eval_x, eval_average, eval_standard_error)

    for x in X_AXIS:
        single_plot(exp_type_train_results, x, 'Average Return',
                    f'Average return of the model on {args.env_name} in different mode')
        single_plot(exp_type_eval_results, x, 'Average Return',
                    f'Average return of the model on {args.env_name} in different modes')


def single_plot(experiment_results, x_label, y_label, title):
    plt.figure(figsize=[12, 9])
    for type in experiment_results.keys():
        x, average, standard_error = experiment_results[type]
        plt.plot(x[x_label], average, label=type)
        plt.fill_between(x[x_label], average - 2.26 * standard_error, average + 2.26 * standard_error, alpha=0.2)
    plt.xlabel(X_AXIS_TO_LABEL[x_label])
    plt.ylabel(y_label)
    plt.legend()
    plt.title(title)
    plt.savefig(os.path.join(exp_path, x_label + '.jpg'))


if __name__ == "__main__":
    draw()
