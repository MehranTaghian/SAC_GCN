from gym import register

register(
    id="HalfCheetahBFootBroken-v0",
    entry_point="CustomGymEnvs.faulty_envs.HalfCheetahBrokenJoints.half_cheetah:HalfCheetahEnv",
    max_episode_steps=200,
    kwargs={
        'malfunction_joint': 'bfoot'
    }
)
register(
    id="HalfCheetahBShinBroken-v0",
    entry_point="CustomGymEnvs.faulty_envs.HalfCheetahBrokenJoints.half_cheetah:HalfCheetahEnv",
    max_episode_steps=200,
    kwargs={
        'malfunction_joint': 'bshin'
    }
)
register(
    id="HalfCheetahBThighBroken-v0",
    entry_point="CustomGymEnvs.faulty_envs.HalfCheetahBrokenJoints.half_cheetah:HalfCheetahEnv",
    max_episode_steps=200,
    kwargs={
        'malfunction_joint': 'bthigh'
    }
)
register(
    id="HalfCheetahFFootBroken-v0",
    entry_point="CustomGymEnvs.faulty_envs.HalfCheetahBrokenJoints.half_cheetah:HalfCheetahEnv",
    max_episode_steps=200,
    kwargs={
        'malfunction_joint': 'ffoot'
    }
)
register(
    id="HalfCheetahFShinBroken-v0",
    entry_point="CustomGymEnvs.faulty_envs.HalfCheetahBrokenJoints.half_cheetah:HalfCheetahEnv",
    max_episode_steps=200,
    kwargs={
        'malfunction_joint': 'fshin'
    }
)
register(
    id="HalfCheetahFThighBroken-v0",
    entry_point="CustomGymEnvs.faulty_envs.HalfCheetahBrokenJoints.half_cheetah:HalfCheetahEnv",
    max_episode_steps=200,
    kwargs={
        'malfunction_joint': 'fthigh'
    }
)
