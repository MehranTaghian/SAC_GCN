import numpy as np
import os

from gym import utils, spaces
from gym.envs.mujoco import mujoco_env
from pathlib import Path
from RobotGraphModel import Walker2dGraph


class Walker2dEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.filepath = os.path.join(Path(__file__).parent, "walker2d.xml")
        self.robot_graph = None
        mujoco_env.MujocoEnv.__init__(self, self.filepath, 4)
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

    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        done = not (height > 0.8 and height < 2.0 and ang > -1.0 and ang < 1.0)

        # MODIFICATION
        if self.robot_graph is None:
            self.robot_graph = Walker2dGraph(self.sim, env_name='Walker2dEnv_v0')
        # END MODIFICATION
        ob = self._get_obs()
        return ob, reward, done, {}

    def _get_obs(self):
        # qpos = self.sim.data.qpos
        # qvel = self.sim.data.qvel
        # return np.concatenate([qpos[1:], np.clip(qvel, -10, 10)]).ravel()
        obs = self.robot_graph.get_graph_obs()
        obs['global_features'] = np.empty([0])
        return obs

    def reset_model(self):
        self.set_state(
            self.init_qpos
            + self.np_random.uniform(low=-0.005, high=0.005, size=self.model.nq),
            self.init_qvel
            + self.np_random.uniform(low=-0.005, high=0.005, size=self.model.nv),
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20
