import gym
from gym import spaces, register
import numpy as np
from RobotGraphModel import ModelParser

register(
    id="HopperFootBroken-v0",
    entry_point="CustomGymEnvs.envs.HopperBrokenJoints.hopper:HopperEnv",
    max_episode_steps=200,
    kwargs={
        'malfunction_joint': 'foot'
    }
)
register(
    id="HopperLegBroken-v0",
    entry_point="CustomGymEnvs.envs.HopperBrokenJoints.hopper:HopperEnv",
    max_episode_steps=200,
    kwargs={
        'malfunction_joint': 'leg'
    }
)
register(
    id="HopperThighBroken-v0",
    entry_point="CustomGymEnvs.envs.HopperBrokenJoints.hopper:HopperEnv",
    max_episode_steps=200,
    kwargs={
        'malfunction_joint': 'thigh'
    }
)


class HopperBrokenWrapper(gym.ObservationWrapper):
    def __init__(self, env_type):
        if env_type == 'foot_joint':
            self.env = gym.make('HopperFootBroken-v0')
        elif env_type == 'leg_joint':
            self.env = gym.make('HopperLegBroken-v0')
        elif env_type == 'thigh_joint':
            self.env = gym.make('HopperThighBroken-v0')
        else:
            raise Exception("Joint not found!")

        super().__init__(self.env)
        parser = ModelParser(self.env.sim.model.get_xml())
        self.joint_list = [j.attrib['name'] for j in parser.joints]
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(2 * len(self.joint_list),), dtype='float32')

        self.env.spec.max_episode_steps = 200
        self.env._max_episode_steps = 200
        self._max_episode_steps = 200

    def observation(self, obs):
        joint_features = []
        for j in self.joint_list:
            joint_features.append(self.sim.data.get_joint_qpos(j).copy())
            joint_features.append(self.sim.data.get_joint_qvel(j).copy())

        return np.array(joint_features)
