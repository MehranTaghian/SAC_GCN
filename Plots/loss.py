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
parser.add_argument('--exp-type', default="standard",
                    help='Type of the experiment like normal or abnormal')

args = parser.parse_args()
exp_path = os.path.join(pathlib.Path(__file__).parent.parent, 'Data', args.env_name, args.exp_type)


def draw():
    experiment_seed = os.listdir(exp_path)
    experiment_seed = [d for d in experiment_seed if os.path.isdir(os.path.join(exp_path, d))]
    for seed in range(len(experiment_seed)):
        path = os.path.join(exp_path, experiment_seed[seed])
        data = pd.read_csv(os.path.join(path, 'loss.csv'))
        single_plot(data['num_updates'], data['critic_1_loss'], 'critic_1_loss', os.path.join(path, 'critic1_loss.jpg'))
        single_plot(data['num_updates'], data['critic_2_loss'], 'critic_2_loss', os.path.join(path, 'critic2_loss.jpg'))
        single_plot(data['num_updates'], data['policy_loss'], 'policy_loss', os.path.join(path, 'policy_loss.jpg'))
        single_plot(data['num_updates'], data['ent_loss'], 'ent_loss', os.path.join(path, 'ent_loss.jpg'))
        single_plot(data['num_updates'], data['alpha'], 'alpha', os.path.join(path, 'alpha.jpg'))


def single_plot(x, y, title, path):
    plt.figure(figsize=[12, 9])
    plt.plot(x, y)
    plt.title(title)
    plt.savefig(path, dpi=300)
    plt.close()


if __name__ == "__main__":
    draw()
