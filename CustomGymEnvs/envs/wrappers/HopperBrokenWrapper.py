import gym
from .MujocoBaseWrapper import MujocoBaseWrapper
from CustomGymEnvs.faulty_envs import HopperBrokenJoints

class HopperBrokenWrapper(MujocoBaseWrapper):
    def __init__(self, env_type):
        if env_type == 'foot_joint':
            env = gym.make('HopperFootBroken-v0')
        elif env_type == 'leg_joint':
            env = gym.make('HopperLegBroken-v0')
        elif env_type == 'thigh_joint':
            env = gym.make('HopperThighBroken-v0')
        else:
            raise Exception("Joint not found!")

        super().__init__(env)
