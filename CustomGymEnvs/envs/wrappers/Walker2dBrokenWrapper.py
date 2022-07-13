import gym
from gym import spaces, register
import numpy as np
from RobotGraphModel import ModelParser

register(
    id="Walker2dFootBroken-v0",
    entry_point="CustomGymEnvs.envs.Walker2dBrokenJoints.walker2d:Walker2dEnv",
    max_episode_steps=200,
    kwargs={
        'malfunction_joint': 'foot'
    }
)
register(
    id="Walker2dFootLeftBroken-v0",
    entry_point="CustomGymEnvs.envs.Walker2dBrokenJoints.walker2d:Walker2dEnv",
    max_episode_steps=200,
    kwargs={
        'malfunction_joint': 'foot_left'
    }
)
register(
    id="Walker2dLegBroken-v0",
    entry_point="CustomGymEnvs.envs.Walker2dBrokenJoints.walker2d:Walker2dEnv",
    max_episode_steps=200,
    kwargs={
        'malfunction_joint': 'leg'
    }
)
register(
    id="Walker2dLegLeftBroken-v0",
    entry_point="CustomGymEnvs.envs.Walker2dBrokenJoints.walker2d:Walker2dEnv",
    max_episode_steps=200,
    kwargs={
        'malfunction_joint': 'leg_left'
    }
)
register(
    id="Walker2dThighBroken-v0",
    entry_point="CustomGymEnvs.envs.Walker2dBrokenJoints.walker2d:Walker2dEnv",
    max_episode_steps=200,
    kwargs={
        'malfunction_joint': 'thigh'
    }
)
register(
    id="Walker2dThighLeftBroken-v0",
    entry_point="CustomGymEnvs.envs.Walker2dBrokenJoints.walker2d:Walker2dEnv",
    max_episode_steps=200,
    kwargs={
        'malfunction_joint': 'thigh_left'
    }
)


class Walker2dBrokenWrapper(gym.ObservationWrapper):
    def __init__(self, env_type):
        if env_type == 'foot_joint':
            self.env = gym.make('Walker2dFootBroken-v0')
        elif env_type == 'foot_left_joint':
            self.env = gym.make('Walker2dFootLeftBroken-v0')
        elif env_type == 'leg_joint':
            self.env = gym.make('Walker2dLegBroken-v0')
        elif env_type == 'leg_left_joint':
            self.env = gym.make('Walker2dLegLeftBroken-v0')
        elif env_type == 'thigh_joint':
            self.env = gym.make('Walker2dThighBroken-v0')
        elif env_type == 'thigh_left_joint':
            self.env = gym.make('Walker2dThighLeftBroken-v0')

        super().__init__(self.env)
        parser = ModelParser(self.env.sim.model.get_xml())
        self.joint_list = [j.attrib['name'] for j in parser.joints]
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(2 * len(self.joint_list),), dtype='float32')

        # env.spec.max_episode_steps = 200
        # env._max_episode_steps = 200
        # self._max_episode_steps = 200

    def observation(self, obs):
        joint_features = []
        for j in self.joint_list:
            joint_features.append(self.sim.data.get_joint_qpos(j).copy())
            joint_features.append(self.sim.data.get_joint_qvel(j).copy())

        return np.array(joint_features)
