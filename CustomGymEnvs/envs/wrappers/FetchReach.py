import gym
from gym import spaces
import numpy as np
from RobotGraphModel import ModelParser


class FetchReachWrapper(gym.ObservationWrapper):
    def __init__(self, env, occluded_joint):
        super().__init__(env)
        assert 'FetchReach' in env.unwrapped.spec.id, 'Environment must be FetchReach'
        self.env = env

        parser = ModelParser(env.sim.model.get_xml(), env.unwrapped.spec.id)
        self.joint_list = [j.attrib['name'] for j in parser.joints]
        if occluded_joint is not None:
            if occluded_joint in self.joint_list:
                self.joint_list.remove(occluded_joint)
            else:
                raise Exception('Occluded joint is not in the list of joints')

        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(len(self.joint_list) * 2 + 3,), dtype='float32')
        self._max_episode_steps = env._max_episode_steps

    def observation(self, obs):
        joint_features = []
        for j in self.joint_list:
            joint_features.append(self.sim.data.get_joint_qpos(j).copy())
            joint_features.append(self.sim.data.get_joint_qvel(j).copy())

        joint_features = np.array(joint_features)
        return np.concatenate([joint_features, obs['desired_goal']])
