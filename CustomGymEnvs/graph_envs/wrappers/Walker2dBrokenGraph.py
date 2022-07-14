import gym
from RobotGraphModel import Walker2dGraph
from .MujocoGraphBase import MujocoGraphBaseWrapper
from CustomGymEnvs.faulty_envs import Walker2dBrokenJoints


class Walker2dBrokenGraphWrapper(MujocoGraphBaseWrapper):
    def __init__(self, env_type):
        if env_type == 'foot_joint':
            env = gym.make('Walker2dFootBroken-v0')
            robot_graph = Walker2dGraph(env.sim)
        elif env_type == 'foot_left_joint':
            env = gym.make('Walker2dFootLeftBroken-v0')
            robot_graph = Walker2dGraph(env.sim)
        elif env_type == 'leg_joint':
            env = gym.make('Walker2dLegBroken-v0')
            robot_graph = Walker2dGraph(env.sim)
        elif env_type == 'leg_left_joint':
            env = gym.make('Walker2dLegLeftBroken-v0')
            robot_graph = Walker2dGraph(env.sim)
        elif env_type == 'thigh_joint':
            env = gym.make('Walker2dThighBroken-v0')
            robot_graph = Walker2dGraph(env.sim)
        elif env_type == 'thigh_left_joint':
            env = gym.make('Walker2dThighLeftBroken-v0')
            robot_graph = Walker2dGraph(env.sim)
        else:
            raise Exception("Joint not found!")
        super().__init__(env, robot_graph)
