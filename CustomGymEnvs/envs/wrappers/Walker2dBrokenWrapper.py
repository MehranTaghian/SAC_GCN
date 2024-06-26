import gym
from .MujocoBaseWrapper import MujocoBaseWrapper
from CustomGymEnvs.faulty_envs import Walker2dBrokenJoints


class Walker2dBrokenWrapper(MujocoBaseWrapper):
    def __init__(self, env_type):
        if env_type == 'foot_joint':
            env = gym.make('Walker2dFootBroken-v0')
        elif env_type == 'foot_left_joint':
            env = gym.make('Walker2dFootLeftBroken-v0')
        elif env_type == 'leg_joint':
            env = gym.make('Walker2dLegBroken-v0')
        elif env_type == 'leg_left_joint':
            env = gym.make('Walker2dLegLeftBroken-v0')
        elif env_type == 'thigh_joint':
            env = gym.make('Walker2dThighBroken-v0')
        elif env_type == 'thigh_left_joint':
            env = gym.make('Walker2dThighLeftBroken-v0')
        else:
            raise Exception("Joint not found!")

        super().__init__(env)
