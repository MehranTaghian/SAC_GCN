import gym
import CustomGymEnvs
from pathlib import Path

# env = gym.make('FetchReachEnvGraph-v0')
env = gym.make('AntEnvGraph-v0')
# env = gym.make('HalfCheetahEnvGraph-v0')

# home = str(Path.home())
g = env.robot_graph
# print(len(g.node_list))


# for n in g.node_list:
#     print(n.attrib['name'])

# print('g.node_features.shape', g.node_features.shape)
# print('g.edge_features.shape', g.edge_features.shape)

# print(g.edges_from)
# print(g.edges_to)

# node_id_list = []
# for n in range(g.node_features.shape[0]):
#     # node_id_list.append(env.sim.model.body_name2id(n.attrib['name']))
#     print(g.node_list[n].attrib['name'], g.node_features[n])
#     # print(env.get_body_com(g.node_list[n].attrib['name']))
# for id in sorted(node_id_list):
#     print(id, env.sim.model.body_id2name(id))
# print('#' * 100)


# for e, f in zip(g.edge_list.keys(), g.extract_edge_features()):
#     print(e, f)

print(g.extract_edge_features())


# edge_id_list = []
# for e in g.edge_list:
#     if e is not None:
#         edge_id_list.append(int(env.sim.model.joint_name2id(e.attrib['name'])))
#
# for id in sorted(edge_id_list):
#     name = env.sim.model.joint_id2name(id)
#     print(id, name, g.edge_features[id, :])
#     # print(name, env.sim.data.get_joint_qpos(name))

# print(env.sim.data.get_body_xpos("robot0:forearm_roll_link"))
# print(env.sim.model.body_name2id("robot0:forearm_roll_link"))
# id = env.sim.model.body_name2id("robot0:forearm_roll_link")
# print(env.sim.model.body_mass[id])
