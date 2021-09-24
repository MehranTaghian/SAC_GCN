from gym.envs.registration import register

register(
    id="AntEnv-v0",
    entry_point="custom_gym_envs.envs.ant.AntEnv_v0_Normal:AntEnvV0",
    max_episode_steps=1000,
)

register(
    id="AntEnv-v1",
    entry_point="custom_gym_envs.envs.ant.AntEnv_v1_BrokenSeveredLimb:AntEnvV1",
    max_episode_steps=1000,
)

register(
    id="AntEnv-v2",
    entry_point="custom_gym_envs.envs.ant.AntEnv_v2_Hip4ROM:AntEnvV2",
    max_episode_steps=1000,
)

register(
    id="AntEnv-v3",
    entry_point="custom_gym_envs.envs.ant.AntEnv_v3_Ankle4ROM:AntEnvV3",
    max_episode_steps=1000,
)

register(
    id="AntEnv-v4",
    entry_point="custom_gym_envs.envs.ant.AntEnv_v4_BrokenUnseveredLimb:AntEnvV4",
    max_episode_steps=1000,
)


# FetchReach

register(
    id="FetchReachEnv-v0",
    entry_point="custom_gym_envs.envs.fetchreach.FetchReachEnv_v0_Normal.fetch.reach:FetchReachEnv",
    max_episode_steps=50,
)
register(
    id="FetchReachEnv-v1",
    entry_point="custom_gym_envs.envs.fetchreach.FetchReachEnv_v1_BrokenShoulderLiftJoint.fetch.reach:FetchReachEnv",
    max_episode_steps=50,
)
register(
    id="FetchReachEnv-v2",
    entry_point="custom_gym_envs.envs.fetchreach.FetchReachEnv_v2_BrokenElbowFlexJoint.fetch.reach:FetchReachEnv",
    max_episode_steps=50,
)
register(
    id="FetchReachEnv-v3",
    entry_point="custom_gym_envs.envs.fetchreach.FetchReachEnv_v3_BrokenWristFlexJoint.fetch.reach:FetchReachEnv",
    max_episode_steps=50,
)
register(
    id="FetchReachEnv-v4",
    entry_point="custom_gym_envs.envs.fetchreach.FetchReachEnv_v4_BrokenShoulderLiftSensor.fetch.reach:FetchReachEnv",
    max_episode_steps=50,
)
register(
    id="FetchReachEnv-v5",
    entry_point="custom_gym_envs.envs.fetchreach.FetchReachEnv_v5_BrokenShoulderLiftSensor.fetch.reach:FetchReachEnv",
    max_episode_steps=50,
)
register(
    id="FetchReachEnv-v6",
    entry_point="custom_gym_envs.envs.fetchreach.FetchReachEnv_v6_ElbowFlexNoisyMovement.fetch.reach:FetchReachEnv",
    max_episode_steps=50,
)
