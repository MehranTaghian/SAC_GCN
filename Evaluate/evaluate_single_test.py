import argparse
import datetime
import gym
import CustomGymEnvs
import numpy as np
import itertools
import torch
from Graph_SAC.sac import SAC
# from torch.utils.tensorboard import SummaryWriter
from utils import state_2_graph, state_2_graphbatch
import torchgraphs as tg
import matplotlib
from tqdm import tqdm
import os
from pathlib import Path

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

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
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
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
parser.add_argument('-chp', '--checkpoint_path',
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')

parser.add_argument('--aggregation', default="avg",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
args = parser.parse_args()

exp_path = os.path.join(Path(__file__).parent.parent, 'Data', args.env_name, args.exp_type, f'seed{args.seed}')

# Environment
# env = NormalizedActions(gym.make(args.env_name))
env = gym.make(args.env_name)
env.seed(args.seed)
env.action_space.seed(args.seed)

num_nodes = env.observation_space['node_features'].shape[0]
num_edges = env.observation_space['edge_features'].shape[0]
num_node_features = env.observation_space['node_features'].shape[1]
num_edge_features = env.observation_space['edge_features'].shape[1]
num_global_features = env.observation_space['global_features'].shape[0]

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Agent
agent = SAC(num_node_features, num_edge_features, num_global_features, env.action_space, False, args)
agent_relevance = SAC(num_node_features, num_edge_features, num_global_features, env.action_space, True, args)

checkpoint_path = os.path.join(exp_path, 'model')
agent.load_checkpoint(checkpoint_path, evaluate=True)
agent_relevance.load_checkpoint(checkpoint_path, evaluate=True)

device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
render = False

num_samples = 0
edge_list = env.robot_graph.edge_list
node_list = env.robot_graph.node_list
rel_freq_edge = {}
rel_score_edge = {}
for joint_list in edge_list.values():
    if len(joint_list) > 0:
        if len(joint_list) == 1:
            joint_name = joint_list[0].attrib['name']
        else:
            joint_name = '\n'.join([j.attrib['name'] for j in joint_list])

        for i in range(env.action_space.shape[0]):
            if i not in rel_freq_edge.keys():
                rel_freq_edge[i] = {}
                rel_score_edge[i] = {}
            rel_freq_edge[i][joint_name] = 0
            rel_score_edge[i][joint_name] = []

rel_freq_global = {}
# rel_freq_global = 0
for i in range(env.action_space.shape[0]):
    rel_freq_global[i] = 0

rel_freq_node = {}
for n in node_list:
    if 'name' in n.attrib:
        rel_freq_node[n.attrib['name']] = 0

# for i_episode in itertools.count(1):
avg_reward = 0.
episodes = 20
for i in tqdm(range(episodes)):
    state = env.reset()
    if render:
        env.render()
    episode_reward = 0
    done = False
    episode_step = 0
    while not done:
        # iterate over actions
        for n in range(env.action_space.shape[0]):
            batch_state = state_2_graphbatch(state).requires_grad_().to(device)
            out = agent_relevance.policy.graph_net(batch_state)
            out = agent_relevance.policy.mean_linear(out)[0]
            global_relevance = torch.zeros_like(out.global_features)
            global_relevance[n] = out.global_features[n]
            batch_state.zero_grad_()
            out.global_features.backward(global_relevance)

            # node_rel = state.node_features.grad.sum(dim=1)
            edge_rel = batch_state.edge_features.grad.sum(dim=1)
            global_rel = batch_state.global_features.grad.sum(dim=1)
            rel_freq_global[n] += global_rel

            for e, edge in enumerate(edge_list.values()):
                if len(edge) > 0:
                    if len(edge) == 1:
                        joint_name = edge[0].attrib['name']
                    else:
                        joint_name = '\n'.join([j.attrib['name'] for j in edge])

                    rel_freq_edge[n][joint_name] += edge_rel[e]
                    if len(rel_score_edge[n][joint_name]) - 1 < i:
                        rel_score_edge[n][joint_name].append([])
                    rel_score_edge[n][joint_name][i].append(edge_rel[e])

        # state = state_2_graphbatch(state).requires_grad_().to(device)
        # out = agent_relevance.policy.graph_net(state)
        # out = agent_relevance.policy.mean_linear(out)[0]
        # global_relevance = torch.ones_like(out.global_features) * (out.global_features != 0).float()
        # state.zero_grad_()
        # out.global_features.backward(global_relevance)

        # for id in body_ids:
        #     if 'name' in node_list[id].attrib:
        #         rel_freq_node[node_list[id].attrib['name']] += node_rel[id]

        num_samples += 1

        action = agent.select_action(state_2_graphbatch(state), evaluate=True)
        next_state, reward, done, _ = env.step(action)
        episode_reward += reward
        state = next_state
        if render:
            env.render()

        episode_step += 1
    avg_reward += episode_reward

for i in range(env.action_space.shape[0]):
    plt.figure(figsize=[12, 15])
    for k in rel_score_edge[i].keys():
        scores = np.array(rel_score_edge[i][k])
        average_score = np.mean(scores, axis=0)
        std_score = np.std(scores, axis=0) / np.sqrt(scores.shape[0])
        x = np.linspace(1, len(average_score), len(average_score))
        plt.plot(x, average_score, label=k)
        plt.fill_between(x, average_score - 2.26 * std_score, average_score + 2.26 * std_score, alpha=0.2)

    plt.legend()
    plt.show()

avg_reward /= episodes

# writer.add_scalar('avg_reward/test', avg_reward, i_episode)

print("----------------------------------------")
print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
print("----------------------------------------")

for i in range(env.action_space.shape[0]):
    plt.figure(figsize=[12, 15])
    plt.bar(range(len(rel_freq_edge[i])), np.array(list(rel_freq_edge[i].values())) / num_samples, align='center')
    plt.xticks(range(len(rel_freq_edge[i])), list(rel_freq_edge[i].keys()), rotation=90)
    plt.show()

# print(rel_freq_node)
# plt.figure(figsize=[12, 15])
# plt.bar(range(len(rel_freq_node)), np.array(list(rel_freq_node.values())) / num_samples, align='center')
# plt.xticks(range(len(rel_freq_node)), list(rel_freq_node.keys()), rotation=90)
# plt.show()

env.close()
