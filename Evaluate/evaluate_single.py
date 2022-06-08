import argparse
import gym
from CustomGymEnvs import MujocoGraphWrapper, FetchReachGraphWrapper
import numpy as np
import torch
from Graph_SAC.sac import SAC
from utils import state_2_graphbatch, load_object
import matplotlib
from tqdm import tqdm
import os
from pathlib import Path

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="FetchReachDense-v1",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--exp-type', default="standard",
                    help='Type of the experiment like normal or abnormal')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
args = parser.parse_args()

env_name = args.env_name
exp_path = os.path.join(Path(__file__).parent.parent, 'Data', args.env_name, args.exp_type, f'seed{args.seed}')
args = load_object(os.path.join(exp_path, 'parameters.pkl'))
args.env_name = env_name


# Environment
if 'FetchReach' in args.env_name:
    env = FetchReachGraphWrapper(gym.make(args.env_name))
else:
    env = MujocoGraphWrapper(gym.make(args.env_name))

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

# Tesnorboard
# writer = SummaryWriter(
#     'runs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
#                                   args.policy, "autotune" if args.automatic_entropy_tuning else ""))

device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
render = True

num_samples = 0
edge_list = env.robot_graph.edge_list
node_list = env.robot_graph.node_list
rel_freq_edge = {}
rel_score_edge = {}
for j in edge_list:
    if j is not None:
        rel_freq_edge[j.attrib['name']] = 0
        rel_score_edge[j.attrib['name']] = []

rel_freq_node = {}
for n in node_list:
    if 'name' in n.attrib:
        rel_freq_node[n.attrib['name']] = 0

rel_freq_global = 0
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
        state = state_2_graphbatch(state).requires_grad_().to(device)
        graph_out = agent_relevance.policy.graph_net(state)
        out = agent_relevance.policy.mean_linear(graph_out).global_features
        state.zero_grad_()
        out.backward(out)
        node_rel = state.node_features.grad.sum(dim=1)
        edge_rel = state.edge_features.grad.sum(dim=1)
        global_rel = state.global_features.grad.sum(dim=1)
        rel_freq_global += global_rel
        joint_ids = torch.argsort(edge_rel)
        body_ids = torch.argsort(node_rel)
        for id in joint_ids:
            if edge_list[id] is not None:
                # print(edge_list[id].attrib['name'], edge_rel[id])
                rel_freq_edge[edge_list[id].attrib['name']] += edge_rel[id]
                if len(rel_score_edge[edge_list[id].attrib['name']]) - 1 < i:
                    rel_score_edge[edge_list[id].attrib['name']].append([])
                rel_score_edge[edge_list[id].attrib['name']][i].append(edge_rel[id])

        for id in body_ids:
            if 'name' in node_list[id].attrib:
                rel_freq_node[node_list[id].attrib['name']] += node_rel[id]
        num_samples += 1
        action = agent.select_action(state, evaluate=True)
        next_state, reward, done, _ = env.step(action)
        episode_reward += reward
        state = next_state
        if render:
            env.render()

        episode_step += 1
    avg_reward += episode_reward

plt.figure(figsize=[12, 15])
for k in rel_score_edge.keys():
    scores = np.array(rel_score_edge[k])
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

print(rel_freq_global / num_samples)
print(rel_freq_edge)
plt.figure(figsize=[12, 15])
plt.bar(range(len(rel_freq_edge)), np.array(list(rel_freq_edge.values())) / num_samples, align='center')
plt.xticks(range(len(rel_freq_edge)), list(rel_freq_edge.keys()), rotation=90)
plt.show()

# print(rel_freq_node)
# plt.figure(figsize=[12, 15])
# plt.bar(range(len(rel_freq_node)), np.array(list(rel_freq_node.values())) / num_samples, align='center')
# plt.xticks(range(len(rel_freq_node)), list(rel_freq_node.keys()), rotation=90)
# plt.show()

env.close()
