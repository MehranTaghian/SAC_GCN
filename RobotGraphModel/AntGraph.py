from RobotGraphModel.RobotGraph import RobotGraph
import numpy as np


class AntGraph(RobotGraph):
    def __init__(self, sim, env_name, weld_joints=None):
        super(AntGraph, self).__init__(sim, env_name, weld_joints)

    def extract_node_features(self):
        return np.zeros([len(self.node_list), 0])

    def extract_edge_features(self):
        feature_list = []
        for edge in self.edge_list.values():
            if edge is not None:
                # if there was only one joint between two nodes
                if not isinstance(edge, list):
                    jnt_qpos = self.sim.data.get_joint_qpos(edge.attrib['name'])
                    jnt_qvel = self.sim.data.get_joint_qvel(edge.attrib['name'])
                    edge_feature = np.array([jnt_qpos.copy(), jnt_qvel.copy()])
                # if two nodes were connected with more than one joint
                else:
                    edge_feature = []
                    for e in edge:
                        jnt_qpos = self.sim.data.get_joint_qpos(e.attrib['name'])
                        jnt_qvel = self.sim.data.get_joint_qvel(e.attrib['name'])
                        edge_feature += [jnt_qpos, jnt_qvel]
            else:
                edge_feature = np.zeros([0, 0])

            feature_list.append(edge_feature)

        feature_matrix = np.zeros([len(feature_list), len(max(feature_list, key=lambda x: len(x)))])
        for i, j in enumerate(feature_list):
            feature_matrix[i][:len(j)] = j

        return feature_matrix
