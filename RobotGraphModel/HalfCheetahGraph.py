from RobotGraphModel.RobotGraph import RobotGraph
import numpy as np


class HalfCheetahGraph(RobotGraph):
    def __init__(self, sim, env_name, weld_joints=None):
        super(HalfCheetahGraph, self).__init__(sim, env_name, weld_joints)

    def extract_node_features(self):
        return np.zeros([len(self.node_list), 0])

    def extract_edge_features(self):
        feature_list = []
        for edge in self.edge_list:
            edge_feature = np.zeros(6)
            if 'root' not in edge.attrib['name']:
                jnt_qpos = self.sim.data.get_joint_qpos(edge.attrib['name'])
                jnt_qvel = self.sim.data.get_joint_qvel(edge.attrib['name'])
                edge_feature[:2] = [jnt_qpos.copy(), jnt_qvel.copy()]

            feature_list.append(edge_feature)

        return np.array(feature_list)
