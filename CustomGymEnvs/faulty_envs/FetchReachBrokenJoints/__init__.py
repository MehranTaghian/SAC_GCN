from gym import register

kwargs = {
    'reward_type': 'dense',
}

register(
    id="ShoulderPanBroken-v0",
    entry_point="CustomGymEnvs.faulty_envs.FetchReachBrokenJoints.shoulder_pan.reach:FetchReachEnv",
    max_episode_steps=50,
    kwargs=kwargs
)
register(
    id="ShoulderLiftBroken-v0",
    entry_point="CustomGymEnvs.faulty_envs.FetchReachBrokenJoints.shoulder_lift.reach:FetchReachEnv",
    max_episode_steps=50,
    kwargs=kwargs
)
register(
    id="UpperarmRollBroken-v0",
    entry_point="CustomGymEnvs.faulty_envs.FetchReachBrokenJoints.upperarm_roll.reach:FetchReachEnv",
    max_episode_steps=50,
    kwargs=kwargs
)
register(
    id="ForearmRollBroken-v0",
    entry_point="CustomGymEnvs.faulty_envs.FetchReachBrokenJoints.forearm_roll.reach:FetchReachEnv",
    max_episode_steps=50,
    kwargs=kwargs
)
register(
    id="ElbowFlexBroken-v0",
    entry_point="CustomGymEnvs.faulty_envs.FetchReachBrokenJoints.elbow_flex.reach:FetchReachEnv",
    max_episode_steps=50,
    kwargs=kwargs
)
register(
    id="WristFlexBroken-v0",
    entry_point="CustomGymEnvs.faulty_envs.FetchReachBrokenJoints.wrist_flex.reach:FetchReachEnv",
    max_episode_steps=50,
    kwargs=kwargs
)
register(
    id="WristRollBroken-v0",
    entry_point="CustomGymEnvs.faulty_envs.FetchReachBrokenJoints.wrist_roll.reach:FetchReachEnv",
    max_episode_steps=50,
    kwargs=kwargs
)
