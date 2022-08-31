import os
import gym
import numpy as np
import torch
from tqdm import tqdm
from CustomGymEnvs import MujocoGraphNormalWrapper, FetchReachGraphWrapper
from pathlib import Path
from Graph_SAC.sac import SAC
from utils import state_2_graphbatch, load_object
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.style as style
import os

envs = ['FetchReach-v2', 'Walker2d-v2']

root_path = Path(os.path.abspath(__file__)).parent.parent
result_path = os.path.join(root_path, 'Result')

if not os.path.exists(result_path):
    os.makedirs(result_path)


def plot_heatmap(data, ax, entity_names, action_labels, env_name):
    sns.heatmap(data, ax=ax, cbar=False,
                cmap="cividis",
                linewidth=1,
                vmin=np.min(data),
                vmax=np.max(data),
                annot=True,
                fmt='.2f')
    ax.set_xticks(np.arange(len(entity_names)) + 0.5, labels=[j for j in entity_names], rotation=45)
    ax.set_yticks(np.arange(len(action_labels)) + 0.5, labels=action_labels, rotation=45)
    ax.set_title(f"{env_name}")
    # ax.set_title(f"Avg action-joint relevance score - {env_name}")
    # ax.set_xlabel(f"Joint name")


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

    entity_names = [j if 'joint' not in j else j.split('joint')[0].strip() for j in entity_names]
    action_labels = [j for j in entity_names if j not in ['torso', 'goal']]

    return entity_names, action_labels

if __name__ == "__main__":
    width, height = 25, 12
    plt.rcParams.update({'font.size': 20})
    fig, ax = plt.subplots(1, 3, figsize=(width, height), gridspec_kw={'width_ratios': (12.25, 12.25, 0.5)})

    for count, env_name in enumerate(envs):
        if 'FetchReach' in env_name:
            env = FetchReachGraphWrapper(gym.make(env_name))
        else:
            env = MujocoGraphNormalWrapper(env_name)

        edge_list = env.robot_graph.edge_list
        entity_names, action_labels = get_labels(env_name, edge_list)
        edge_relevance, global_relevance, avg_relevance = get_lrp(env_name)
        plot_heatmap(np.abs(avg_relevance.T), ax[count], entity_names, action_labels, env_name)

    plt.colorbar(plt.cm.ScalarMappable(
        cmap="cividis",
        norm=plt.Normalize(vmin=0.0, vmax=1.0)),
        cax=ax[2])
    ax[-1].yaxis.set_ticks_position('left')
    ax[-1].set_ylabel('Avg relevance score across seeds')

    fig.text(0, 0.5, 'Actions (Torque applied to each joint)', va='center', rotation='vertical')
    fig.text(0.5, 0.04, 'Entity in the observation graph', ha='center')
    fig.suptitle(f"Averager entity-action relevance score")
    fig.tight_layout()
    fig.savefig(os.path.join(result_path, 'LRP_result.jpg'), dpi=300)
