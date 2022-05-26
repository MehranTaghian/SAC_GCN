from RobotGraphModel.RobotGraph import RobotGraph
import numpy as np


class Walker2dGraph(RobotGraph):
    def __init__(self, sim, env_name, weld_joints=None):
        super(Walker2dGraph, self).__init__(sim, env_name, weld_joints)

    def extract_node_features(self):
        return np.zeros([len(self.node_list), 0])

    def extract_edge_features(self):
        feature_list = []
        for edge in self.edge_list.values():
            if len(edge) > 0:
                edge_feature = np.array(
                    [x for e in edge for x in [self.sim.data.get_joint_qpos(e.attrib['name']).copy(),
                                               np.clip(self.sim.data.get_joint_qvel(e.attrib['name']).copy(), -10,
                                                       10)]])
            else:  # Welded edges
                edge_feature = np.zeros(2)

            feature_list.append(edge_feature)

        feature_matrix = np.zeros([len(feature_list), len(max(feature_list, key=lambda x: len(x)))])
        for i, j in enumerate(feature_list):
            feature_matrix[i][:len(j)] = j

        return feature_matrix
