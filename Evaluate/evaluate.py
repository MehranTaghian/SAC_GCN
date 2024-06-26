import argparse
import os
import gym
import numpy as np
import torch
from tqdm import tqdm
from CustomGymEnvs import MujocoGraphNormalWrapper, FetchReachGraphWrapper
from pathlib import Path
from Graph_SAC.sac import SAC
from utils import state_2_graphbatch, load_object, save_object

parser = argparse.ArgumentParser(description='Evaluation using LRP Args')
parser.add_argument('--env-name', default="FetchReach-v2",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--exp-type', default="standard",
                    help='Type of the experiment like standard, graph, etc.')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--num-episodes', type=int, default=10, metavar='N',
                    help='Number of episodes for evaluation')
parser.add_argument('--time-step', type=int, default=10999, metavar='N',
                    help='Which time-step of the saved model should be used')

args = parser.parse_args()
env_name = args.env_name
seed = args.seed
time_step = args.time_step
exp_path = os.path.join(Path(__file__).parent.parent, 'Data', args.env_name, args.exp_type)
args = load_object(os.path.join(exp_path, 'seed0', 'parameters.pkl'))
args.env_name = env_name
args.seed = seed
args.time_step = time_step

experiment_seed = os.listdir(exp_path)
experiment_seed = [d for d in experiment_seed if os.path.isdir(os.path.join(exp_path, d))]
# experiment_seed = experiment_seed[:2]
if args.seed < len(experiment_seed):
    experiment_seed = [f'seed{args.seed}']
    exp_path = os.path.join(exp_path, experiment_seed[0])

# Environment
if 'FetchReach' in args.env_name:
    env = FetchReachGraphWrapper(gym.make(args.env_name))
else:
    env = MujocoGraphNormalWrapper(args.env_name)

num_node_features = env.observation_space['node_features'].shape[1]
num_edge_features = env.observation_space['edge_features'].shape[1]
num_global_features = env.observation_space['global_features'].shape[0]

device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
device = torch.device('cpu')
args.cuda = False

num_episodes = 20

edge_list = env.robot_graph.edge_list
node_list = env.robot_graph.node_list

render = False


def process_joint_name(joint_name):
    separated = joint_name.split(':')[1].split('_') if 'robot0' in joint_name else joint_name.split('_')
    final_key = ''
    for sk in separated:
        if len(sk) == 1:
            final_key += sk + '-'
        else:
            final_key += sk + ' '
    return final_key


joint_names = []
joint_indices = []
for edge_id, joint_list in enumerate(edge_list.values()):
    if len(joint_list) > 0:
        joint_names.append(
            process_joint_name(joint_list[0].attrib['name'])
            if len(joint_list) == 1
            else '\n'.join([process_joint_name(j.attrib['name']) for j in joint_list])
        )
        joint_indices.append(edge_id)

action_indices = [a for a in range(env.action_space.shape[0])]
action_labels = [j for j in joint_names if 'torso' not in j]

# edge_relevance[joint_index, action_index, seed, num_episodes, episode_steps] =
#                 [relevance score for each time-step of episode]
# global_relevance[action_index, seed] = [sum of relevance scores for global feature within an episode]
action_edge_relevance = np.zeros(
    [len(joint_names),
     env.action_space.shape[0],
     len(experiment_seed),
     num_episodes,
     env.spec.max_episode_steps])

global_relevance = np.zeros([env.action_space.shape[0],
                             len(experiment_seed),
                             num_episodes,
                             env.spec.max_episode_steps])


def calculate_relevance():
    for s, seed in enumerate(experiment_seed):
        seed = int(seed[-1])
        env.seed(seed)
        env.action_space.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        # Agent
        if os.path.isdir(os.path.join(exp_path, f'seed{seed}', 'model')):
            checkpoint_path = os.path.join(exp_path, f'seed{seed}', 'model', f'{args.time_step}.pt') \
                if len(experiment_seed) > 1 \
                else os.path.join(exp_path, 'model', f'{args.time_step}.pt')
        else:
            checkpoint_path = os.path.join(exp_path, f'seed{seed}', 'model') \
                if len(experiment_seed) > 1 \
                else os.path.join(exp_path, 'model')

        agent = SAC(num_node_features, num_edge_features, num_global_features, env.action_space, False, args)
        agent_relevance = SAC(num_node_features, num_edge_features, num_global_features, env.action_space, True, args)

        agent.load_checkpoint(checkpoint_path, evaluate=True)
        agent_relevance.load_checkpoint(checkpoint_path, evaluate=True)

        avg_reward = 0.

        for episode in tqdm(range(num_episodes)):
            state = env.reset()
            episode_reward = 0
            done = False
            step = 0
            while not done:
                action_edge_rel_calc(agent_relevance, episode, s, state, step)
                step += 1
                action = agent.select_action(state_2_graphbatch(state), evaluate=True)
                next_state, reward, done, _ = env.step(action)
                episode_reward += reward
                state = next_state
                if render:
                    env.render()
            avg_reward += episode_reward
        avg_reward /= num_episodes

        print("----------------------------------------")
        print("Test Episodes: {}, Avg. Reward: {}".format(num_episodes, round(avg_reward, 2)))
        print("----------------------------------------")

    env.close()


def action_edge_rel_calc(agent_relevance, episode, s, state, step):
    for action_index in range(env.action_space.shape[0]):
        batch_state = state_2_graphbatch(state).requires_grad_().to(device)
        out = agent_relevance.policy.graph_net(batch_state)
        out = agent_relevance.policy.mean_linear(out)[0]
        output_relevance = torch.zeros_like(out.global_features)
        output_relevance[action_index] = out.global_features[action_index]
        batch_state.zero_grad_()
        out.global_features.backward(output_relevance)

        edge_rel = batch_state.edge_features.grad.sum(dim=1)
        global_rel = batch_state.global_features.grad.sum(dim=1)
        global_relevance[action_index, s, episode, step] = global_rel
        action_edge_relevance[:, action_index, s, episode, step] = edge_rel[joint_indices]


if __name__ == '__main__':
    calculate_relevance()
    save_object(action_edge_relevance, os.path.join(exp_path, 'edge_relevance.pkl'))
    save_object(global_relevance, os.path.join(exp_path, 'global_relevance.pkl'))
