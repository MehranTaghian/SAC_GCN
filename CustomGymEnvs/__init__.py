from gym.envs.registration import register

register(
    id="AntEnv-v0",
    entry_point="CustomGymEnvs.envs.ant.AntEnv_v0_Normal:AntEnvV0",
    max_episode_steps=1000,
)

register(
    id="AntEnv-v1",
    entry_point="CustomGymEnvs.envs.ant.AntEnv_v1_BrokenSeveredLimb:AntEnvV1",
    max_episode_steps=1000,
)

register(
    id="AntEnv-v2",
    entry_point="CustomGymEnvs.envs.ant.AntEnv_v2_Hip4ROM:AntEnvV2",
    max_episode_steps=1000,
)

register(
    id="AntEnv-v3",
    entry_point="CustomGymEnvs.envs.ant.AntEnv_v3_Ankle4ROM:AntEnvV3",
    max_episode_steps=1000,
)

register(
    id="AntEnv-v4",
    entry_point="CustomGymEnvs.envs.ant.AntEnv_v4_BrokenUnseveredLimb:AntEnvV4",
    max_episode_steps=1000,
)


# FetchReach

register(
    id="FetchReachEnv-v0",
    entry_point="CustomGymEnvs.envs.fetchreach.FetchReachEnv_v0_Normal.fetch.reach:FetchReachEnv",
    max_episode_steps=50,
)