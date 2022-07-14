import gym
from gym import spaces
import numpy as np
from RobotGraphModel import FetchReachGraph


class FetchReachGraphWrapper(gym.ObservationWrapper):
    def __init__(self, env, weld_joints=None):
        super().__init__(env)
        if weld_joints is None:
            weld_joints = ['robot0:torso_lift_joint',
                           'robot0:head_pan_joint',
                           'robot0:head_tilt_joint',
                           'robot0:slide0',
                           'robot0:slide1',
                           'robot0:slide2']
        assert 'FetchReach' in env.unwrapped.spec.id, 'Environment must be FetchReach'
        self.env = env
        self.robot_graph = FetchReachGraph(self.env.sim, weld_joints=weld_joints)

        obs = self.robot_graph.get_graph_obs()
        self.observation_space = spaces.Dict(dict(
            observation=spaces.Box(-np.inf, np.inf, shape=obs['edge_features'].shape,
                                   dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=env.observation_space['achieved_goal'].shape,
                                     dtype='float32'),
            desired_goal=spaces.Box(-np.inf, np.inf, shape=env.observation_space['desired_goal'].shape,
                                    dtype='float32'),
            node_features=spaces.Box(-np.inf, np.inf, shape=obs['node_features'].shape,
                                     dtype='float32'),
            edge_features=spaces.Box(-np.inf, np.inf, shape=obs['edge_features'].shape,
                                     dtype='float32'),
            global_features=spaces.Box(-np.inf, np.inf, shape=env.observation_space['desired_goal'].shape,
                                       dtype='float32'),
            edges_from=spaces.Box(-np.inf, np.inf, shape=obs['edges_from'].shape,
                                  dtype='float32'),
            edges_to=spaces.Box(-np.inf, np.inf, shape=obs['edges_to'].shape,
                                dtype='float32'),
        ))
        self._max_episode_steps = env._max_episode_steps

    def observation(self, obs):
        new_obs = self.robot_graph.get_graph_obs()
        new_obs['global_features'] = obs['desired_goal'].copy()
        new_obs['observation'] = None
        new_obs['achieved_goal'] = obs['achieved_goal'].copy()
        new_obs['desired_goal'] = obs['desired_goal'].copy()
        return new_obs
