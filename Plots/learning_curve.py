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

parser.add_argument('--env-name', default="FetchReachEnv-v0",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--exp-type', default="standard",
                    help='Type of the experiment like normal or abnormal')

args = parser.parse_args()

X_AXIS = ['num_time_steps', 'num_updates', 'num_samples']
X_AXIS_TO_LABEL = {'num_time_steps': 'Time step',
                   'num_updates': 'Number of updates',
                   'num_samples': 'Number of samples'}

exp_path = os.path.join('..', 'Data', args.env_name, args.exp_type)


def draw():
    experiment_seed = os.listdir(exp_path)
    num_seeds = len(experiment_seed)
    first = True
    train_average_returns = None
    eval_average_returns = None
    data_train = None
    data_eval = None
    for seed in range(len(experiment_seed)):
        data_train = pd.read_csv(os.path.join(exp_path, experiment_seed[seed], 'train.csv'))
        data_eval = pd.read_csv(os.path.join(exp_path, experiment_seed[seed], 'eval.csv'))
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

    for x in X_AXIS:
        single_plot(train_average, train_standard_error, train_x[x], x, 'Average Return',
                    f'Average return of the model on {args.env_name} in the {args.exp_type} mode')
        single_plot(eval_average, eval_standard_error, eval_x[x], x, 'Average Return',
                    f'Average return of the model on {args.env_name} in the {args.exp_type} mode')


def single_plot(average, standard_error, x, x_label, y_label, title):
    plt.figure(figsize=[12, 9])
    plt.plot(x, average)
    plt.fill_between(x, average - 2.26 * standard_error, average + 2.26 * standard_error, alpha=0.2)
    plt.xlabel(X_AXIS_TO_LABEL[x_label])
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig(os.path.join(exp_path, x_label + '.jpg'))


if __name__ == "__main__":
    draw()
