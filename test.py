import gym

import CustomGymEnvs

env = gym.make("FetchReachEnv-v0")
# env = gym.make("Ant-v2")

env.reset()

# print(env.sim.data.qpos)

while True:
    action = env.action_space.sample()
    env.step(action)
    env.render()