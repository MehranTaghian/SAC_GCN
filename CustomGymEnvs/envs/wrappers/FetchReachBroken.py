import gym
from .FetchReachBaseWrapper import FetchReachBaseWrapper
from CustomGymEnvs.faulty_envs import FetchReachBrokenJoints


class FetchReachBrokenWrapper(FetchReachBaseWrapper):
    def __init__(self, env_type):
        if env_type == 'shoulder_pan_joint':
            env = gym.make('ShoulderPanBroken-v0')
        elif env_type == 'shoulder_lift_joint':
            env = gym.make('ShoulderLiftBroken-v0')
        elif env_type == 'upperarm_roll_joint':
            env = gym.make('UpperarmRollBroken-v0')
        elif env_type == 'forearm_roll_joint':
            env = gym.make('ForearmRollBroken-v0')
        elif env_type == 'wrist_flex_joint':
            env = gym.make('WristFlexBroken-v0')
        elif env_type == 'wrist_roll_joint':
            env = gym.make('WristRollBroken-v0')
        elif env_type == 'elbow_flex_joint':
            env = gym.make('ElbowFlexBroken-v0')
        else:
            raise Exception('Broken Fetch Reach environment not found')
        super().__init__(env)
