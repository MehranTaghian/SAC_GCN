from gym.envs.registration import register

register(
    id="AntEnv-v0",
    entry_point="CustomGymEnvs.envs.ant.AntEnv_v0_Normal:AntEnvV0",
    max_episode_steps=1000,
)

# HalfCheetah
register(
    id="HalfCheetahEnv-v0",
    entry_point="CustomGymEnvs.envs.halfcheetah.HalfCheetahEnv_v0.half_cheetah:HalfCheetahEnvV0",
    max_episode_steps=1000,
)

# FetchReach

register(
    id="FetchReachEnv-v0",
    entry_point="CustomGymEnvs.envs.fetchreach.CustomFetchPickAndPlace.fetch.reach:FetchReachEnv",
    max_episode_steps=50,
)

register(
    id="FetchReachEnv-v1",
    entry_point="CustomGymEnvs.envs.fetchreach.FetchReach_BrokenShoulderLift.fetch.reach:FetchReachEnv",
    max_episode_steps=50,
)

register(
    id="FetchReachEnv-v2",
    entry_point="CustomGymEnvs.envs.fetchreach.FetchReach_ElbowFlexNoisyMovement.fetch.reach:FetchReachEnv",
    max_episode_steps=50,
)

register(
    id="FetchReachEnv-v3",
    entry_point="CustomGymEnvs.envs.fetchreach.FetchReach_WristFlexNoisyMovement.fetch.reach:FetchReachEnv",
    max_episode_steps=50,
)

register(
    id="FetchReachEnv-v4",
    entry_point="CustomGymEnvs.envs.fetchreach.FetchReach_ShoulderLiftNoisyMovement.fetch.reach:FetchReachEnv",
    max_episode_steps=50,
)

# FetchPickAndPlace

register(
    id="FetchPickAndPlaceEnv-v0",
    entry_point="CustomGymEnvs.envs.fetchpickandplace.CustomFetchPickAndPlace."
                "fetch.pick_and_place:FetchPickAndPlaceEnv",
    max_episode_steps=50,
)
