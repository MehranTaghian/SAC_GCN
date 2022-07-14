from gym import register

register(
    id="HopperFootBroken-v0",
    entry_point="CustomGymEnvs.faulty_envs.HopperBrokenJoints.hopper:HopperEnv",
    max_episode_steps=200,
    kwargs={
        'malfunction_joint': 'foot'
    }
)
register(
    id="HopperLegBroken-v0",
    entry_point="CustomGymEnvs.faulty_envs.HopperBrokenJoints.hopper:HopperEnv",
    max_episode_steps=200,
    kwargs={
        'malfunction_joint': 'leg'
    }
)
register(
    id="HopperThighBroken-v0",
    entry_point="CustomGymEnvs.faulty_envs.HopperBrokenJoints.hopper:HopperEnv",
    max_episode_steps=200,
    kwargs={
        'malfunction_joint': 'thigh'
    }
)
