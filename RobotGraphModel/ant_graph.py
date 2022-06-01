from RobotGraphModel.robot_graph import RobotGraph
import numpy as np


class AntGraph(RobotGraph):
    def __init__(self, sim, env_name, weld_joints=None):
        super(AntGraph, self).__init__(sim, env_name, weld_joints)

    def extract_node_features(self):
        return np.zeros([len(self.node_list), 0])

    def extract_edge_features(self):
        feature_list = []
        for edge in self.edge_list.values():
            if len(edge) > 0:
                if edge[0].attrib['name'] != 'root':
                    edge_feature = np.array(
                        [self.sim.data.get_joint_qpos(edge[0].attrib['name']).copy(),
                         self.sim.data.get_joint_qvel(edge[0].attrib['name']).copy()])
                else:
                    edge_feature = np.concatenate(
                        [self.sim.data.get_joint_qpos(edge[0].attrib['name']).copy(),
                         self.sim.data.get_joint_qvel(edge[0].attrib['name']).copy()])

            else:  # Welded edges
                edge_feature = np.zeros(2)
            feature_list.append(edge_feature)

        feature_matrix = np.zeros([len(feature_list), len(max(feature_list, key=lambda x: len(x)))])
        for i, j in enumerate(feature_list):
            feature_matrix[i][:len(j)] = j
        return feature_matrix