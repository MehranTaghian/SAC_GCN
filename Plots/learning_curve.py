from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import argparse
import pathlib
import os

parser = argparse.ArgumentParser(description="Draw results of the experiments inside a directory")

parser.add_argument('--env-name', default="FetchReachEnvGraph-v0",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--exp-type', default="standard",
                    help='Type of the experiment like normal or abnormal')

args = parser.parse_args()

X_AXIS = ['num_time_steps', 'num_updates', 'num_samples']
X_AXIS_TO_LABEL = {'num_time_steps': 'Time step',
                   'num_updates': 'Number of updates',
                   'num_samples': 'Number of samples'}

exp_path = os.path.join(pathlib.Path(__file__).parent.parent, 'Data', args.env_name)


def draw():
    env_exp_types = os.listdir(exp_path)
    if 'graph' in env_exp_types:
        env_exp_types.remove('graph')
    env_exp_types = [d for d in env_exp_types if os.path.isdir(os.path.join(exp_path, d))]
    for exp_type in env_exp_types:
        exp_type_path = os.path.join(exp_path, exp_type)
        experiment_seed = os.listdir(exp_type_path)
        experiment_seed = [d for d in experiment_seed if os.path.isdir(os.path.join(exp_type_path, d))]
        plt.figure(figsize=[15, 12])
        for seed in range(len(experiment_seed)):
            data_train = pd.read_csv(os.path.join(exp_type_path, experiment_seed[seed], 'train.csv'))
            data_eval = pd.read_csv(os.path.join(exp_type_path, experiment_seed[seed], 'eval.csv'))
            plt.plot(data_eval['eval_reward'], label=experiment_seed[seed])

        plt.xlabel('Number of episodes')
        plt.ylabel('Average evaluation reward')
        plt.title('Learning curve')
        plt.legend()
        plt.savefig(os.path.join(exp_type_path, 'learning_curve.jpg'))
        plt.close()


if __name__ == "__main__":
    draw()
