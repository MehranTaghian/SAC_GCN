import gym
import numpy as np
from gym import spaces
from RobotGraphModel import ModelParser
from gym.envs.registration import register

register(
    id="ShoulderPanBroken-v0",
    entry_point="CustomGymEnvs.envs.FetchReachBrokenJoints.shoulder_pan.reach:FetchReachEnv",
    max_episode_steps=50,
)
register(
    id="ShoulderLiftBroken-v0",
    entry_point="CustomGymEnvs.envs.FetchReachBrokenJoints.shoulder_lift.reach:FetchReachEnv",
    max_episode_steps=50,
)
register(
    id="UpperarmRollBroken-v0",
    entry_point="CustomGymEnvs.envs.FetchReachBrokenJoints.upperarm_roll.reach:FetchReachEnv",
    max_episode_steps=50,
)
register(
    id="ForearmRollBroken-v0",
    entry_point="CustomGymEnvs.envs.FetchReachBrokenJoints.forearm_roll.reach:FetchReachEnv",
    max_episode_steps=50,
)
register(
    id="ElbowFlexBroken-v0",
    entry_point="CustomGymEnvs.envs.FetchReachBrokenJoints.elbow_flex.reach:FetchReachEnv",
    max_episode_steps=50,
)
register(
    id="WristFlexBroken-v0",
    entry_point="CustomGymEnvs.envs.FetchReachBrokenJoints.wrist_flex.reach:FetchReachEnv",
    max_episode_steps=50,
)
register(
    id="WristRollBroken-v0",
    entry_point="CustomGymEnvs.envs.FetchReachBrokenJoints.wrist_roll.reach:FetchReachEnv",
    max_episode_steps=50,
)


class FetchReachBrokenWrapper(gym.ObservationWrapper):
    def __init__(self, env_type):
        if env_type == 'shoulder_pan_joint':
            self.env = gym.make('ShoulderPanBroken-v0')
        elif env_type == 'shoulder_lift_joint':
            self.env = gym.make('ShoulderLiftBroken-v0')
        elif env_type == 'upperarm_roll_joint':
            self.env = gym.make('UpperarmRollBroken-v0')
        elif env_type == 'forearm_roll_joint':
            self.env = gym.make('ForearmRollBroken-v0')
        elif env_type == 'wrist_flex_joint':
            self.env = gym.make('WristFlexBroken-v0')
        elif env_type == 'wrist_roll_joint':
            self.env = gym.make('WristRollBroken-v0')
        elif env_type == 'elbow_flex_joint':
            self.env = gym.make('ElbowFlexBroken-v0')
        else:
            raise Exception('Broken Fetch Reach environment not found')
        super().__init__(self.env)

        parser = ModelParser(self.env.sim.model.get_xml())
        self.joint_list = [j.attrib['name'] for j in parser.joints]
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(len(self.joint_list) * 2 + 3,), dtype='float32')
        self._max_episode_steps = self.env._max_episode_steps

    def observation(self, obs):
        joint_features = []
        for j in self.joint_list:
            joint_features.append(self.sim.data.get_joint_qpos(j).copy())
            joint_features.append(self.sim.data.get_joint_qvel(j).copy())

        joint_features = np.array(joint_features)
        return np.concatenate([joint_features, obs['desired_goal']])
