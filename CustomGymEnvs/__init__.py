from gym.envs.registration import register
from CustomGymEnvs.envs.wrappers.FetchReach import FetchReachWrapper
from CustomGymEnvs.envs.wrappers.MujocoWrapper import MujocoWrapper
from CustomGymEnvs.graph_envs.wrappers.MujocoGraph import MujocoGraphWrapper
from CustomGymEnvs.graph_envs.wrappers.FetchReachGraph import FetchReachGraphWrapper

# FetchReach - Graph

register(
    id="FetchReachEnvGraph-v0",
    entry_point="CustomGymEnvs.graph_envs.fetchreach.CustomFetchReach.fetch.reach:FetchReachEnv",
    max_episode_steps=50,
)

register(
    id="FetchReachEnvGraph-v1",
    entry_point="CustomGymEnvs.graph_envs.fetchreach.FetchReach_BrokenShoulderLift.fetch.reach:FetchReachEnv",
    max_episode_steps=50,
)

register(
    id="FetchReachEnvGraph-v2",
    entry_point="CustomGymEnvs.graph_envs.fetchreach.FetchReach_ElbowFlexNoisyMovement.fetch.reach:FetchReachEnv",
    max_episode_steps=50,
)

register(
    id="FetchReachEnvGraph-v3",
    entry_point="CustomGymEnvs.graph_envs.fetchreach.FetchReach_WristFlexNoisyMovement.fetch.reach:FetchReachEnv",
    max_episode_steps=50,
)

register(
    id="FetchReachEnvGraph-v4",
    entry_point="CustomGymEnvs.graph_envs.fetchreach.FetchReach_ShoulderLiftNoisyMovement.fetch.reach:FetchReachEnv",
    max_episode_steps=50,
)

register(
    id="FetchReachEnvGraph-v5",
    entry_point="CustomGymEnvs.graph_envs.fetchreach.FetchReach_JointVel.fetch.reach:FetchReachEnv",
    max_episode_steps=50,
)

register(
    id="FetchReachEnvGraph-v6",
    entry_point="CustomGymEnvs.graph_envs.fetchreach.FetchReach_Occluded.fetch.reach:FetchReachEnv",
    max_episode_steps=50,
)

register(
    id="FetchReachEnvGraph-v7",
    entry_point="CustomGymEnvs.graph_envs.fetchreach.FetchReach_ImportantJointOnly.fetch.reach:FetchReachEnv",
    max_episode_steps=50,
)
