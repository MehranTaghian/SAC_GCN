from gym import register

kwargs = {
    'reward_type': 'dense',
}

register(
    id="FetchReach-v2",
    entry_point="CustomGymEnvs.changed_envs.FetchReach_v2.fetch.reach:FetchReachEnv",
    max_episode_steps=50,
    kwargs=kwargs
)
