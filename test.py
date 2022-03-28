import gym
import numpy as np
import CustomGymEnvs

env = gym.make("FetchReachEnv-v3")
# env = gym.make("AntEnv-v0")
# env = gym.make("FetchReach-v1")
# env = gym.make("Ant-v2")

obs = env.reset()
print(obs)
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
while True:
    action = env.action_space.sample()
    # print(action.shape)
    # print(obs['global_features'])
    # print(env.sim.data.qpos)
    # print(env.sim.data.qvel)
    # print(env.sim.data.cfrc_ext.flat)

    # edge_id_list = []
    # for e in g.edge_list:
    #     if e is not None:
    #         edge_id_list.append(int(env.sim.model.joint_name2id(e.attrib['name'])))
    #
    # for id in sorted(edge_id_list):
    #     name = env.sim.model.joint_id2name(id)
    #     # print(id, name, g.edge_features[id, :])
    #     print(id, name, env.sim.data.get_joint_qpos(env.sim.model.joint_id2name(id)))

    # print(action)
    # action = np.array([0, 0, 0, 0, 0, 0, 0, 0])
    # action[:4] = [0, 0, 0, 0]
    # print(action)
    obs, _, _, _ = env.step(action)
    # print('node_features', obs['observation']['node_features'])
    # print('edge_features', obs['observation']['edge_features'])
    print(obs)
    env.render()
