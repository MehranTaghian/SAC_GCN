from gym.envs.registration import register

register(
    id="AntEnv-v0",
    entry_point="CustomGymEnvs.envs.ant.AntEnv_v0_Normal:AntEnvV0",
    max_episode_steps=1000,
)

# FetchReach

register(
    id="FetchReachEnv-v0",
    entry_point="CustomGymEnvs.envs.fetchreach.CustomFetchReach.fetch.reach:FetchReachEnv",
    max_episode_steps=50,
)