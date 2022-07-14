import gym
from RobotGraphModel import HopperGraph
from .MujocoGraphBase import MujocoGraphBaseWrapper
from CustomGymEnvs.faulty_envs import HopperBrokenJoints


class HopperBrokenGraphWrapper(MujocoGraphBaseWrapper):
    def __init__(self, env_type):
        if env_type == 'foot_joint':
            env = gym.make('HopperFootBroken-v0')
            robot_graph = HopperGraph(env.sim)
        elif env_type == 'leg_joint':
            env = gym.make('HopperLegBroken-v0')
            robot_graph = HopperGraph(env.sim)
        elif env_type == 'thigh_joint':
            env = gym.make('HopperThighBroken-v0')
            robot_graph = HopperGraph(env.sim)
        else:
            raise Exception("Joint not found!")
        super().__init__(env, robot_graph)
