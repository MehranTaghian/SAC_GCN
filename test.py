import gym
import numpy as np
import CustomGymEnvs
from CustomGymEnvs import FetchReachWrapper
import os
from pathlib import Path

# joint_list = ['robot0:shoulder_lift_joint',
#               'robot0:elbow_flex_joint',
#               'robot0:wrist_flex_joint']
#
# joint_list = [
#     'robot0:shoulder_pan_joint',
#     'robot0:shoulder_lift_joint',
#     'robot0:upperarm_roll_joint',
#     'robot0:elbow_flex_joint',
#     'robot0:forearm_roll_joint',
#     'robot0:wrist_flex_joint',
#     'robot0:wrist_roll_joint']
# env = FetchReachWrapper(gym.make("FetchReachEnv-v0"), joint_list)

# env = gym.make("FetchReachEnvGraph-v7")
env = gym.make("AntEnvGraph-v0")
# env = gym.make("FetchReachEnv-v4")
# env = gym.make("Ant-v2")
# env = gym.make("HalfCheetahEnvGraph-v0")
# env = gym.make("FetchPickAndPlaceEnv-v0")

obs = env.reset()
print(obs['edge_features'])
print(obs['global_features'].shape)

# print(env.observation_space.shape)
# print(env.action_space.shape)

# print(env.sim.model.joint_names)
# print()

# print(env.action_space.shape[0])
# print(env.observation_space)

# print([j.attrib['name'] for j in env.joint_list])
# print(len(env.joint_list))
# print(env.sim.data.qpos)
# print(env.sim.model.get_xml())
# print(env.robot_graph.edge_features)
# g = env.robot_graph
# while True:
#     action = env.action_space.sample()
#     # print(action.shape)
#     # print(obs['global_features'])
#     # print(env.sim.data.qpos)
#     # print(env.sim.data.qvel)
#     # print(env.sim.data.cfrc_ext.flat)
#
#     # edge_id_list = []
#     # for e in g.edge_list:
#     #     if e is not None:
#     #         edge_id_list.append(int(env.sim.model.joint_name2id(e.attrib['name'])))
#     #
#     # for id in sorted(edge_id_list):
#     #     name = env.sim.model.joint_id2name(id)
#     #     # print(id, name, g.edge_features[id, :])
#     #     print(id, name, env.sim.data.get_joint_qpos(env.sim.model.joint_id2name(id)))
#
#     # print(action)
#     # action = np.array([0, 0, 0, 0, 0, 0, 0, 0])
#     # action[:4] = [0, 0, 0, 0]
#     # print(action)
#     obs, _, _, _ = env.step(action)
#     # print('node_features', obs['observation']['node_features'])
#     # print('edge_features', obs['observation']['edge_features'])
#     # print(obs)
#     env.render()
