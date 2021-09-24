from gym.envs.registration import register

register(
    id="AntEnv-v0",
    entry_point="custom_gym_envs.envs.ant.AntEnv_v0_Normal:AntEnvV0",
    max_episode_steps=1000,
)

register(
    id="AntEnv-v1",
    entry_point="custom_gym_envs.envs.ant.AntEnv_v1_BrokenSeveredLeg:AntEnvV1",
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
    entry_point="custom_gym_envs.envs.ant.AntEnv_v4_BrokenUnseveredLeg:AntEnvV4",
    max_episode_steps=1000,
)


# FetchReach

for goal_elimination in [True, False]:
    suffix = "GE" if goal_elimination else ""
    kwargs = {
        "reward_type": "dense",
        "goal_elimination": goal_elimination,
    }
    register(
        id="FetchReachEnv{}-v0".format(suffix),
        entry_point="custom_gym_envs.envs.fetchreach.FetchReachEnv_v0_Normal.fetch.reach:FetchReachEnv",
        kwargs=kwargs,
        max_episode_steps=50,
    )
    register(
        id="FetchReachEnv{}-v1".format(suffix),
        entry_point="custom_gym_envs.envs.fetchreach.FetchReachEnv_v1_BrokenShoulderLiftJoint.fetch.reach:FetchReachEnv",
        kwargs=kwargs,
        max_episode_steps=50,
    )
    register(
        id="FetchReachEnv{}-v2".format(suffix),
        entry_point="custom_gym_envs.envs.fetchreach.FetchReachEnv_v2_BrokenElbowFlexJoint.fetch.reach:FetchReachEnv",
        kwargs=kwargs,
        max_episode_steps=50,
    )
    register(
        id="FetchReachEnv{}-v3".format(suffix),
        entry_point="custom_gym_envs.envs.fetchreach.FetchReachEnv_v3_BrokenWristFlexJoint.fetch.reach:FetchReachEnv",
        kwargs=kwargs,
        max_episode_steps=50,
    )
    register(
        id="FetchReachEnv{}-v4".format(suffix),
        entry_point="custom_gym_envs.envs.fetchreach.FetchReachEnv_v4_BrokenShoulderLiftSensor.fetch.reach:FetchReachEnv",
        kwargs=kwargs,
        max_episode_steps=50,
    )
    register(
        id="FetchReachEnv{}-v5".format(suffix),
        entry_point="custom_gym_envs.envs.fetchreach.FetchReachEnv_v5_BrokenShoulderLiftSensor.fetch.reach:FetchReachEnv",
        kwargs=kwargs,
        max_episode_steps=50,
    )
    register(
        id="FetchReachEnv{}-v6".format(suffix),
        entry_point="custom_gym_envs.envs.fetchreach.FetchReachEnv_v6_ElbowFlexNoisyMovement.fetch.reach:FetchReachEnv",
        kwargs=kwargs,
        max_episode_steps=50,
    )
