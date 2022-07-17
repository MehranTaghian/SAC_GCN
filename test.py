import gym
import numpy as np
import CustomGymEnvs
from CustomGymEnvs import FetchReachBaseWrapper, MujocoBaseWrapper, FetchReachGraphWrapper, MujocoGraphNormalWrapper, \
    FetchReachBrokenWrapper, Walker2dBrokenWrapper, HopperBrokenWrapper, Walker2dBrokenGraphWrapper, \
    HopperBrokenGraphWrapper, HalfCheetahBrokenWrapper
from RobotGraphModel import ModelParser
import os
from pathlib import Path

# env = FetchReachBrokenWrapper('shoulder_pan_joint')
# env = FetchReachBrokenWrapper('shoulder_lift_joint')
# env = FetchReachBrokenWrapper('wrist_flex_joint')
# env = FetchReachBrokenWrapper('wrist_roll_joint')
# env = FetchReachBrokenWrapper('elbow_flex_joint')
# env = FetchReachBrokenWrapper('forearm_roll_joint')
# env = FetchReachBrokenWrapper('upperarm_roll_joint')

# env = Walker2dBrokenWrapper('foot_joint')
# env = Walker2dBrokenWrapper('foot_left_joint')
# env = Walker2dBrokenWrapper('leg_joint')
# env = Walker2dBrokenWrapper('leg_left_joint')
# env = Walker2dBrokenWrapper('thigh_joint')
# env = Walker2dBrokenWrapper('thigh_left_joint')

# env = Walker2dBrokenGraphWrapper('foot_joint')
# env = Walker2dBrokenGraphWrapper('foot_left_joint')
# env = Walker2dBrokenGraphWrapper('leg_joint')
# env = Walker2dBrokenGraphWrapper('leg_left_joint')
# env = Walker2dBrokenGraphWrapper('thigh_joint')
# env = Walker2dBrokenGraphWrapper('thigh_left_joint')

# env = HopperBrokenGraphWrapper('thigh_joint')
# env = HopperBrokenGraphWrapper('foot_joint')
# env = HopperBrokenGraphWrapper('leg_joint')

env = HalfCheetahBrokenWrapper('ffoot')
# env = HalfCheetahBrokenWrapper('fshin')
# env = HalfCheetahBrokenWrapper('fthigh')
# env = HalfCheetahBrokenWrapper('bfoot')
# env = HalfCheetahBrokenWrapper('bshin')
# env = HalfCheetahBrokenWrapper('bthigh')

env.reset()
while True:
    env.step(env.action_space.sample())
    env.render()

# env = FetchReachWrapper(gym.make("FetchReachDense-v1"), 'standard')
# env = FetchReachGraphWrapper(gym.make("FetchReachDense-v1"))
# print(env.observation_space)
# env = MujocoWrapper(gym.make("HalfCheetah-v2"), 'root')
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
