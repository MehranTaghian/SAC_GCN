from gym.envs.registration import register
from .envs.fetchreach.Wrappers.FetchReach import FetchReachWrapper

# Ant - Graph
register(
    id="AntEnvGraph-v0",
    entry_point="CustomGymEnvs.graph_envs.ant.AntEnv_v0_Normal:AntEnvV0",
    max_episode_steps=1000,
)

# HalfCheetah - Graph
register(
    id="HalfCheetahEnvGraph-v0",
    entry_point="CustomGymEnvs.graph_envs.halfcheetah.HalfCheetahEnv_v0.half_cheetah:HalfCheetahEnvV0",
    max_episode_steps=1000,
)

# FetchReach - Graph

register(
    id="FetchReachEnvGraph-v0",
    entry_point="CustomGymEnvs.graph_envs.fetchreach.sh.CustomFetchReach.fetch.reach:FetchReachEnv",
    max_episode_steps=50,
)

register(
    id="FetchReachEnvGraph-v1",
    entry_point="CustomGymEnvs.graph_envs.fetchreach.sh.FetchReach_BrokenShoulderLift.fetch.reach:FetchReachEnv",
    max_episode_steps=50,
)

register(
    id="FetchReachEnvGraph-v2",
    entry_point="CustomGymEnvs.graph_envs.fetchreach.sh.FetchReach_ElbowFlexNoisyMovement.fetch.reach:FetchReachEnv",
    max_episode_steps=50,
)

register(
    id="FetchReachEnvGraph-v3",
    entry_point="CustomGymEnvs.graph_envs.fetchreach.sh.FetchReach_WristFlexNoisyMovement.fetch.reach:FetchReachEnv",
    max_episode_steps=50,
)

register(
    id="FetchReachEnvGraph-v4",
    entry_point="CustomGymEnvs.graph_envs.fetchreach.sh.FetchReach_ShoulderLiftNoisyMovement.fetch.reach:FetchReachEnv",
    max_episode_steps=50,
)

register(
    id="FetchReachEnvGraph-v5",
    entry_point="CustomGymEnvs.graph_envs.fetchreach.sh.FetchReach_JointVel.fetch.reach:FetchReachEnv",
    max_episode_steps=50,
)

register(
    id="FetchReachEnvGraph-v6",
    entry_point="CustomGymEnvs.graph_envs.fetchreach.sh.FetchReach_Occluded.fetch.reach:FetchReachEnv",
    max_episode_steps=50,
)

# FetchReach

register(
    id="FetchReachEnv-v0",
    entry_point="CustomGymEnvs.envs.fetchreach.sh.CustomFetchReach.fetch.reach:FetchReachEnv",
    max_episode_steps=50,
)

