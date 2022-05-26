import gym
from gym import spaces
import numpy as np
from RobotGraphModel import ModelParser


class HalfCheetahWrapper(gym.ObservationWrapper):
    def __init__(self, env, occluded_joint=None):
        super().__init__(env)
        assert 'HalfCheetahEnv' in env.unwrapped.spec.id, 'Environment must be HalfCheetah'
        self.env = env

        parser = ModelParser(env.sim.model.get_xml(), "HalfCheetahEnv-v0")
        self.joint_list = [j.attrib['name'] for j in parser.joints]
        if occluded_joint is not None:
            self.joint_list.remove(occluded_joint)

        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(2 * len(self.joint_list),), dtype='float32')
        self._max_episode_steps = env._max_episode_steps

    def observation(self, obs):
        joint_features = []
        for j in self.joint_list:
            joint_features.append(self.sim.data.get_joint_qpos(j).copy())
            joint_features.append(self.sim.data.get_joint_qvel(j).copy())

        return np.array(joint_features)
