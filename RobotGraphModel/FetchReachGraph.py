from RobotGraphModel.RobotGraph import RobotGraph
import numpy as np


class FetchReachGraph(RobotGraph):
    def __init__(self, sim, env_name, weld_joints=None):
        super(FetchReachGraph, self).__init__(sim, env_name, weld_joints)

    def extract_node_features(self):
        return np.zeros([len(self.node_list), 0])

    def extract_edge_features(self):
        feature_list = []
        for edge in self.edge_list.values():
            if len(edge) > 0:
                e = edge[0]
                jnt_qpos = self.sim.data.get_joint_qpos(e.attrib['name'])
                jnt_qvel = self.sim.data.get_joint_qvel(e.attrib['name'])
                edge_feature = np.array([jnt_qpos.copy(), jnt_qvel.copy()])
            else:
                edge_feature = np.array([0, 0])

            feature_list.append(edge_feature)

        return np.array(feature_list)
