from RobotGraphModel.RobotGraph import RobotGraph
import numpy as np


class AntGraph(RobotGraph):
    def __init__(self, sim, env_name, weld_joints=None):
        super(AntGraph, self).__init__(sim, env_name, weld_joints)

    def extract_node_features(self):
        feature_list = []
        len_features = None
        for node in self.node_list:
            # body_xpos = self.sim.data.get_body_xpos(node.attrib['name'])
            # body_xquat = self.sim.data.get_body_xquat(node.attrib['name'])
            # body_xvelp = self.sim.data.get_body_xvelp(node.attrib['name'])
            # body_xvelr = self.sim.data.get_body_xvelr(node.attrib['name'])
            #
            # if (len(body_xpos.shape) > 0 or
            #         len(body_xquat.shape) > 0 or
            #         len(body_xvelp.shape) > 0 or
            #         len(body_xvelr.shape) > 0):
            #     node_feature = np.concatenate([
            #         body_xpos.copy(),
            #         body_xquat.copy(),
            #         # body_xvelp.copy(),
            #         # body_xvelr.copy()
            #     ])
            # else:
            #     node_feature = np.concatenate([
            #         # jnt_ranges.copy(),
            #         # jnt_axis.copy(),
            #         # jnt_xaxis.copy(),
            #         # jnt_xanchor.copy(),
            #         [body_xpos.copy()],
            #         [body_xquat.copy()],
            #         # [body_xvelp.copy()],
            #         # [body_xvelr.copy()]
            #     ])

            node_feature = np.empty([0])
            # find the feature vector with maximum length of dimension
            if (len_features is not None and node_feature.shape[0] > len_features) or len_features is None:
                len_features = node_feature.shape[0]

            feature_list.append(node_feature)

        self.node_features = np.zeros([len(feature_list), len_features])
        for i in range(len(feature_list)):
            self.node_features[i, :feature_list[i].shape[0]] = feature_list[i]

    def extract_edge_features(self):
        # axis is a static constant feature which does not change during the runtime. Thus we remove it and use
        # dynamic features instead.
        feature_list = []
        len_features = None
        for edge in self.edge_list:
            if edge is not None:
                jnt_qpos = self.sim.data.get_joint_qpos(edge.attrib['name'])
                jnt_qvel = self.sim.data.get_joint_qvel(edge.attrib['name'])

                if len(jnt_qpos.shape) > 0 or len(jnt_qvel.shape) > 0:
                    edge_feature = np.concatenate([jnt_qpos.copy(), jnt_qvel.copy()])
                else:
                    edge_feature = np.concatenate([
                        [jnt_qpos.copy()],
                        [jnt_qvel.copy()]])

                # find the feature vector with maximum length of dimension
                if (len_features is not None and edge_feature.shape[0] > len_features) or len_features is None:
                    len_features = edge_feature.shape[0]

            else:
                edge_feature = None

            # modification
            # edge_feature = np.empty([0])
            # end modification

            feature_list.append(edge_feature)

        self.edge_features = np.zeros([len(feature_list), len_features])
        for i in range(len(feature_list)):
            if feature_list[i] is not None:
                self.edge_features[i, :feature_list[i].shape[0]] = feature_list[i]
