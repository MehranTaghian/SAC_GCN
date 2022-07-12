import gym
from gym import spaces
import numpy as np
from RobotGraphModel import AntGraph, Walker2dGraph, HalfCheetahGraph, HopperGraph


class MujocoGraphWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        env_name = env.spec.id
        self.robot_graph = None
        if env_name == 'Ant-v2':
            self.robot_graph = AntGraph(self.env.sim, env_name=env_name)
        elif env_name == 'Walker2d-v2':
            self.robot_graph = Walker2dGraph(self.env.sim, env_name=env_name)
        elif env_name == 'HalfCheetah-v2':
            self.robot_graph = HalfCheetahGraph(self.env.sim, env_name=env_name)
        elif env_name == 'Hopper-v2':
            self.robot_graph = HopperGraph(self.env.sim, env_name=env_name)
        else:
            raise Exception("Environment not found! Consider using version 2 of the Mujoco environments.")

        obs = self.robot_graph.get_graph_obs()
        obs['global_features'] = np.empty([0])
        self.observation_space = spaces.Dict(dict(
            node_features=spaces.Box(-np.inf, np.inf, shape=obs['node_features'].shape,
                                     dtype='float32'),
            edge_features=spaces.Box(-np.inf, np.inf, shape=obs['edge_features'].shape,
                                     dtype='float32'),
            global_features=spaces.Box(-np.inf, np.inf, shape=obs['global_features'].shape,
                                       dtype='float32'),
            edges_from=spaces.Box(-np.inf, np.inf, shape=obs['edges_from'].shape,
                                  dtype='float32'),
            edges_to=spaces.Box(-np.inf, np.inf, shape=obs['edges_to'].shape,
                                dtype='float32'),
        ))

        env.spec.max_episode_steps = 200
        env._max_episode_steps = 200
        self._max_episode_steps = 200

    def observation(self, obs):
        obs = self.robot_graph.get_graph_obs()
        obs['global_features'] = np.empty([0])
        return obs
