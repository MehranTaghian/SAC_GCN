import gym
from .MujocoBaseWrapper import MujocoBaseWrapper
from CustomGymEnvs.faulty_envs import HalfCheetahBrokenJoints


class HalfCheetahBrokenWrapper(MujocoBaseWrapper):
    def __init__(self, env_type):
        if env_type == 'bfoot':
            env = gym.make('HalfCheetahBFootBroken-v0')
        elif env_type == 'bshin':
            env = gym.make('HalfCheetahBShinBroken-v0')
        elif env_type == 'bthigh':
            env = gym.make('HalfCheetahBThighBroken-v0')
        elif env_type == 'ffoot':
            env = gym.make('HalfCheetahFFootBroken-v0')
        elif env_type == 'fshin':
            env = gym.make('HalfCheetahFShinBroken-v0')
        elif env_type == 'fthigh':
            env = gym.make('HalfCheetahFThighBroken-v0')
        else:
            raise Exception("Joint not found!")

        super().__init__(env)
