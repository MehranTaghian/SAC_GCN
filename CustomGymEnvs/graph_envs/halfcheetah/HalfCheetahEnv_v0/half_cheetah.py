import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import os
from pathlib import Path
from gym import spaces
from RobotGraphModel import HalfCheetahGraph


class HalfCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.filepath = os.path.join(Path(__file__).parent, 'half_cheetah.xml')
        self.robot_graph = None
        mujoco_env.MujocoEnv.__init__(self, self.filepath, 5)
        utils.EzPickle.__init__(self)

        # MODIFICATION
        obs = self._get_obs()
        self.observation_space = spaces.Dict(dict(
            node_features=spaces.Box(-np.inf, np.inf, shape=obs['node_features'].shape,
                                     dtype='float32'),
            edge_features=spaces.Box(-np.inf, np.inf, shape=obs['edge_features'].shape,
                                     dtype='float32'),
            global_features=spaces.Box(-np.inf, np.inf, shape=obs['global_features'].shape,
                                       dtype='float32'),
            edges_from=spaces.Box(-np.inf, np.inf, shape=obs['edges_from'].shape,
                                  dtype='float32'),
            edges_to=spaces.Box(-np.inf, np.inf, shape=obs['edges_to'].shape,
                                dtype='float32'),
        ))
        # END MODIFICATION

    def step(self, action):
        # modification here
        # old_value = self.sim.data.get_joint_qpos("bshin").copy()
        # end of modification

        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]

        # modification here
        # new_value = self.sim.data.get_joint_qpos("bshin").copy()
        # delta = new_value - old_value
        # # slippery_value = old_value + delta * self.np_random.normal(0, 0.1)
        # slippery_value = new_value + (delta != 0) * 0.1
        # self.sim.data.set_joint_qpos("bshin", slippery_value)
        # self.sim.forward()
        # end of modification

        reward_ctrl = -0.1 * np.square(action).sum()
        reward_run = (xposafter - xposbefore) / self.dt
        reward = reward_ctrl + reward_run
        done = False

        # MODIFICATION
        if self.robot_graph is None:
            self.robot_graph = HalfCheetahGraph(self.sim, env_name='HalfCheetahEnv_v0_Normal')
        # END MODIFICATION
        ob = self._get_obs()

        return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        obs = self.robot_graph.get_graph_obs()
        obs['global_features'] = np.empty([0])
        # obs['global_features'] = np.concatenate([self.sim.data.qpos.flat[1:], self.sim.data.qvel.flat])
        original_obs = np.concatenate([self.sim.data.qpos.flat[1:], self.sim.data.qvel.flat])
        # print('Original observation', original_obs)
        # print(obs['edge_features'])
        return obs

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            low=-0.1, high=0.1, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
