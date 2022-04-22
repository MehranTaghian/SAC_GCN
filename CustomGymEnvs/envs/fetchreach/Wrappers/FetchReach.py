import gym
from gym import spaces
import numpy as np


class FetchReachWrapper(gym.ObservationWrapper):
    def __init__(self, env, joint_list):
        super().__init__(env)
        assert 'FetchReachEnv' in env.unwrapped.spec.id, 'Environment must be FetchReachEnv-v0'
        self.env = env
        self.joint_list = joint_list
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(len(self.joint_list) * 2 + 3,), dtype='float32')
        self._max_episode_steps = env._max_episode_steps

    def observation(self, obs):
        joint_features = []
        for j in self.joint_list:
            joint_features.append(self.sim.data.get_joint_qpos(j).copy())
            joint_features.append(self.sim.data.get_joint_qvel(j).copy())

        joint_features = np.array(joint_features)
        return np.concatenate([joint_features, obs['desired_goal']])
