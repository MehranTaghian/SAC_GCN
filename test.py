import gym

import CustomGymEnvs

env = gym.make("FetchReachEnv-v0")
# env = gym.make("Ant-v2")

env.reset()

# print(env.sim.data.qpos)
print(env.sim.model.nmocap)
print(env.sim.model.njnt)
while True:
    action = env.action_space.sample()
    obs, _, _, _ = env.step(action)
    # print('node_features', obs['observation']['node_features'])
    # print('edge_features', obs['observation']['edge_features'])
    env.render()
