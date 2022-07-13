import gym
import numpy as np
import CustomGymEnvs
from CustomGymEnvs import FetchReachWrapper, MujocoWrapper, FetchReachGraphWrapper, MujocoGraphWrapper, \
    FetchReachBrokenWrapper, Walker2dBrokenWrapper
from RobotGraphModel import ModelParser
import os
from pathlib import Path

# env = gym.make("ShoulderPanBroken-v0")
# env = gym.make("ShoulderLiftBroken-v0")
# env = gym.make("WristFlexBroken-v0")
# env = gym.make("WristRollBroken-v0")
# env = gym.make("UpperarmRoll-v0")
# env = FetchReachBrokenWrapper('shoulder_pan_joint')
# env = FetchReachBrokenWrapper('shoulder_lift_joint')
# env = FetchReachBrokenWrapper('wrist_flex_joint')

# env = Walker2dBrokenWrapper('foot')
# env = Walker2dBrokenWrapper('foot_left')
# env = Walker2dBrokenWrapper('leg')
# env = Walker2dBrokenWrapper('leg_left')
# env = Walker2dBrokenWrapper('thigh')
env = Walker2dBrokenWrapper('thigh_left')

env.reset()

while True:
    env.step(env.action_space.sample())
    env.render()

# env = FetchReachWrapper(gym.make("FetchReachDense-v1"), 'standard')
# env = FetchReachGraphWrapper(gym.make("FetchReachDense-v1"))
# print(env.observation_space)
# env = MujocoWrapper(gym.make("HalfCheetahEnv-v0"))
# env2 = MujocoWrapper(gym.make("HalfCheetahEnv-v0"), 'bfoot')
# env2 = MujocoWrapper(gym.make("HalfCheetahEnv-v0"), 'fthigh')

# env = MujocoWrapper(gym.make("Walker2d-v2"), occluded_joint='foot_left_joint')
# env2 = MujocoWrapper(gym.make("Walker2d-v2"), occluded_joint='leg_left_joint')
# env3 = MujocoWrapper(gym.make("Walker2d-v2"), occluded_joint='thigh_left_joint')
# env4 = MujocoWrapper(gym.make("Walker2d-v2"), occluded_joint='standard')

# env = MujocoGraphWrapper(gym.make('Ant-v2'))
# env = MujocoGraphWrapper(gym.make('Hopper-v2'))

# env2 = MujocoWrapper(gym.make("Walker2d-v2"))
# env2 = MujocoWrapper(gym.make("Walker2d-v2"))
