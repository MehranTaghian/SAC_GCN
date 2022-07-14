from gym import register

register(
    id="Walker2dFootBroken-v0",
    entry_point="CustomGymEnvs.faulty_envs.Walker2dBrokenJoints.walker2d:Walker2dEnv",
    max_episode_steps=200,
    kwargs={
        'malfunction_joint': 'foot'
    }
)
register(
    id="Walker2dFootLeftBroken-v0",
    entry_point="CustomGymEnvs.faulty_envs.Walker2dBrokenJoints.walker2d:Walker2dEnv",
    max_episode_steps=200,
    kwargs={
        'malfunction_joint': 'foot_left'
    }
)
register(
    id="Walker2dLegBroken-v0",
    entry_point="CustomGymEnvs.faulty_envs.Walker2dBrokenJoints.walker2d:Walker2dEnv",
    max_episode_steps=200,
    kwargs={
        'malfunction_joint': 'leg'
    }
)
register(
    id="Walker2dLegLeftBroken-v0",
    entry_point="CustomGymEnvs.faulty_envs.Walker2dBrokenJoints.walker2d:Walker2dEnv",
    max_episode_steps=200,
    kwargs={
        'malfunction_joint': 'leg_left'
    }
)
register(
    id="Walker2dThighBroken-v0",
    entry_point="CustomGymEnvs.faulty_envs.Walker2dBrokenJoints.walker2d:Walker2dEnv",
    max_episode_steps=200,
    kwargs={
        'malfunction_joint': 'thigh'
    }
)
register(
    id="Walker2dThighLeftBroken-v0",
    entry_point="CustomGymEnvs.faulty_envs.Walker2dBrokenJoints.walker2d:Walker2dEnv",
    max_episode_steps=200,
    kwargs={
        'malfunction_joint': 'thigh_left'
    }
)
