import gym
from RobotGraphModel import AntGraph, Walker2dGraph, HalfCheetahGraph, HopperGraph
from .MujocoGraphBase import MujocoGraphBaseWrapper


class MujocoGraphNormalWrapper(MujocoGraphBaseWrapper):
    def __init__(self, env_name):
        if env_name == 'Ant-v2':
            env = gym.make(env_name)
            robot_graph = AntGraph(self.env.sim)
        elif env_name == 'Walker2d-v2':
            env = gym.make(env_name)
            robot_graph = Walker2dGraph(self.env.sim)
        elif env_name == 'HalfCheetah-v2':
            env = gym.make(env_name)
            robot_graph = HalfCheetahGraph(self.env.sim)
        elif env_name == 'Hopper-v2':
            env = gym.make(env_name)
            robot_graph = HopperGraph(self.env.sim)
        else:
            raise Exception("Environment not found! Consider using version 2 of the Mujoco environments.")
        super().__init__(env, robot_graph)
