from gym.envs.registration import register
from CustomGymEnvs.envs.Wrappers.FetchReach import FetchReachWrapper
from CustomGymEnvs.envs.Wrappers.MujocoWrapper import MujocoWrapper

# Ant - Graph
register(
    id="AntEnvGraph-v0",
    entry_point="CustomGymEnvs.graph_envs.ant.AntEnv_v0_Normal:AntEnv",
    max_episode_steps=200,
)

# Ant

register(
    id="AntEnv-v0",
    entry_point="CustomGymEnvs.envs.ant.AntEnv_v0_Normal.ant:AntEnv",
    max_episode_steps=200,
)

# HalfCheetah - Graph

register(
    id="HalfCheetahEnvGraph-v0",
    entry_point="CustomGymEnvs.graph_envs.halfcheetah.HalfCheetahEnv_v0.half_cheetah:HalfCheetahEnv",
    max_episode_steps=200,
)

# HalfCheetah

register(
    id="HalfCheetahEnv-v0",
    entry_point="CustomGymEnvs.envs.halfcheetah.HalfCheetahEnv_v0_Normal.half_cheetah:HalfCheetahEnv",
    max_episode_steps=200,
)

# walker2d

# walker2d - graph
register(
    id="Walker2dEnvGraph-v0",
    entry_point="CustomGymEnvs.graph_envs.walker2d.Walker2dEnv_v0.walker2d:Walker2dEnv",
    max_episode_steps=200,
)

# Hopper

# Hopper - graph
register(
    id="HopperEnvGraph-v0",
    entry_point="CustomGymEnvs.graph_envs.hopper.HopperEnv_v0.hopper:HopperEnv",
    max_episode_steps=200,
)

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

# FetchReach

register(
    id="FetchReachEnv-v0",
    entry_point="CustomGymEnvs.envs.fetchreach.CustomFetchReach.fetch.reach:FetchReachEnv",
    max_episode_steps=50,
)
