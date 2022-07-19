import gym
from RobotGraphModel import AntGraph, Walker2dGraph, HalfCheetahGraph, HopperGraph
from .MujocoGraphBase import MujocoGraphBaseWrapper


class MujocoGraphNormalWrapper(MujocoGraphBaseWrapper):
    def __init__(self, env_name):
        if env_name == 'Ant-v2':
            env = gym.make(env_name)
            robot_graph = AntGraph(env.sim)
        elif env_name == 'Walker2d-v2':
            env = gym.make(env_name)
            robot_graph = Walker2dGraph(env.sim)
        elif env_name == 'HalfCheetah-v2':
            env = gym.make(env_name)
            robot_graph = HalfCheetahGraph(env.sim)
        elif env_name == 'Hopper-v2':
            env = gym.make(env_name)
            robot_graph = HopperGraph(env.sim)
        else:
            raise Exception("Environment not found! Consider using version 2 of the Mujoco environments.")
        print(robot_graph.edge_list.values())
        super().__init__(env, robot_graph)
