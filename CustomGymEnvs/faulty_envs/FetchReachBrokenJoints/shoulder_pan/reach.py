import os
from gym import utils
from CustomGymEnvs.faulty_envs.FetchReachBrokenJoints import fetch_env

# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('shoulder_pan', 'reach.xml')


class FetchReachEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
            'robot0:slide0': 0.4049,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'robot0:torso_lift_joint': - 2.1080440849725328e-05,
            'robot0:head_pan_joint': 1.8044805716579981e-10,
            'robot0:head_tilt_joint': 0.06002881058785685,
            'robot0:shoulder_pan_joint': 0.009675803961848417,
            'robot0:shoulder_lift_joint': -0.8282310869438312,
            'robot0:upperarm_roll_joint': -0.0030562595712368703,
            'robot0:elbow_flex_joint': 1.4439797539850172,
            'robot0:forearm_roll_joint': 0.0025342393749823184,
            'robot0:wrist_flex_joint': 0.9550999960912975,
            'robot0:wrist_roll_joint': 0.005960935925724559
        }
        fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=False, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)
