import gym
import numpy as np
import CustomGymEnvs

env = gym.make("FetchReachEnv-v0")
# env = gym.make("Ant-v2")

env.reset()

print(env.action_space.shape[0])

print([j.attrib['name'] for j in env.joint_list])
print(len(env.joint_list))
while True:
    action = env.action_space.sample()
    # action = np.array([0, 0, 0, 0, 0, 0, 0])
    # action[:4] = [0, 0, 0, 0]
    # print(action)
    obs, _, _, _ = env.step(action)
    # print('node_features', obs['observation']['node_features'])
    # print('edge_features', obs['observation']['edge_features'])
    env.render()
