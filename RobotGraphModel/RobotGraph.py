from ModelParser import ModelParser
import numpy as np


class RobotGraph:
    def __init__(self, env, model_path):
        """
        Based on the definition in the MuJoCo documentation:
        This element creates a joint. As explained in Kinematic tree, a joint creates motion degrees of freedom
        between the body where it is defined and the body's parent.
        Using this definition, we parsed the XML file and generate the graph of nodes (bodies) and edges (joints).
        ATTENTION: the graph might be a directed graph (from one point, it is attached to a body, and from the other
        point, another body is attached to it)

        For node and edge features, I used the following link to a complete reference of the bodies and joints:
        http://www.mujoco.org/book/XMLreference.html

        Node features:
            Each node is a body which contains inertial features along with geometric features. For a complete reference
            of what inertial and geom tags show, refer to the link above.
            Node features consist of diaginertia, mass, pos, quat which are all attributes of the <inertial> tag
            inside the body. Also they include the attributes of the body tag itself.

        Edge features:
            Edges are joints, therefore edge features are selected among <joint> attributes. These include axis, range,
            armature, damping, frictionloss, stiffness, etc. Some of these attributes are set to default values in
            <default> tag (e.g. for fetchreach, the default values are in the shared.xml file).



        :param model_path: path to the robot's xml file
        """
        self.env = env
        self.parser = ModelParser(model_path)
        self.node_list = set()
        self.edge_list = set()
        self.edges_from = []
        self.edges_to = []
        self.node_features = None
        self.edge_features = None
        self.generate_graph()

    def generate_graph(self):
        """
        The graph contains nodes and edges. Nodes are bodies which can have various attributes like axis, quat, etc.
        Edges are joints which attach separate bodies together and can have multiple features including the range of
        motion, axis, etc.
        :return:
        """
        self.fill_node_edge_lists()
        self.generate_adjacency_matrix()
        self.extract_node_features()
        self.extract_edge_features()

    def generate_adjacency_matrix(self):
        # DEBUG
        # for i in range(len(self.node_list)):
        #     print(i, self.node_list[i].attrib)
        # END DEBUG

        for n1, n2 in self.parser.joints_connections.values():
            edge_from = self.node_list.index(n1)
            edge_to = self.node_list.index(n2)
            self.edges_from.append(edge_from)
            self.edges_to.append(edge_to)

    def fill_node_edge_lists(self):
        for joint, parents in self.parser.joints_connections.items():
            node1, node2 = parents
            self.node_list.add(node1)
            self.node_list.add(node2)
            self.edge_list.add(joint)
            # DEBUG
            # print('node1: ' + node1.attrib['name'],
            #       'node2: ' + node2.attrib['name'],
            #       'joint: ' + joint.attrib['name'])
            # END DEBUG

        self.node_list = list(self.node_list)
        self.edge_list = list(self.edge_list)

    def extract_node_features(self):
        """
        For extracting node features which are body features, we use accessor functions to access each body feature
        by name. This data can be accessed through PyMjModel and PyMjData in env.sim.model and env.sim.data. These are
        static features like mass, inertia, etc, in addition to dynamic features like xvelp, xvelr, xpos, and xquat that
        change during the run time and by doing forward kinematics.

        The body features according to the http://www.mujoco.org/book/APIreference.html are as follows:

        mass: mass of the body [1]
        pos: position offset rel. to parent body [3]
        quat: orientation offset rel. to parent body [4]
        xpos: Cartesian position of body frame (the value is stored in this variable after
                doing forward kinematics) [3]
        xquat: Cartesian orientation of body frame (the value is stored in this variable after
                doing forward kinematics) [4]
        ipos: local position of center of mass [3]
        iquat: local orientation of center of mass [4]
        inertia: diagonal inertia in ipos/iquat frame [3]
        xvelp: positional velocity [3]
        xvelr: rotational velocity [3]

        """
        mask = [env.sim.model.body_name2id(body_name.attrib['name']) for body_name in self.node_list]

        bodies_mass = env.sim.model.body_mass[mask, np.newaxis]
        bodies_inertia = env.sim.model.body_inertia[mask, :]
        bodies_pos = env.sim.model.body_pos[mask, :]
        bodies_quat = env.sim.model.body_quat[mask, :]
        bodies_xpos = env.sim.data.body_xpos[mask, :]
        bodies_xquat = env.sim.data.body_xquat[mask, :]
        bodies_xvelp = env.sim.data.body_xvelp[mask, :]
        bodies_xvelr = env.sim.data.body_xvelr[mask, :]
        bodies_ipos = env.sim.model.body_ipos[mask, :]
        bodies_iquat = env.sim.model.body_iquat[mask, :]

        # DEBUG
        # print(bodies_mass.shape)
        # print(bodies_inertia.shape)
        # print(bodies_pos.shape)
        # print(bodies_quat.shape)
        # print(bodies_xpos.shape)
        # print(bodies_xquat.shape)
        # print(bodies_xvelp.shape)
        # print(bodies_xvelr.shape)
        # print(bodies_ipos.shape)
        # print(bodies_iquat.shape)
        # END DEBUG
        self.node_features = np.concatenate(
            [bodies_mass, bodies_inertia, bodies_pos, bodies_quat, bodies_xpos, bodies_xquat,
             bodies_xvelp, bodies_xvelr, bodies_ipos, bodies_iquat], axis=1)

    def extract_edge_features(self):
        """
        For extracting edge features which are joint features, we use accessor functions to access each joint feature
        by name. This data can be accessed through PyMjModel and PyMjData in env.sim.model and env.sim.data. These are
        static features like range of freedom, , etc.

        TODO: write the detailed documentation of each feature
        Returns:

        """
        mask = [env.sim.model.joint_name2id(joint_name.attrib['name']) for joint_name in self.edge_list]

        jnt_ranges = self.env.sim.model.jnt_range[mask, :]
        jnt_axis = self.env.sim.model.jnt_axis[mask, :]
        jnt_xaxis = self.env.sim.data.xaxis[mask, :]
        jnt_xanchor = self.env.sim.data.xanchor[mask]
        jnt_qpos = self.env.sim.data.qpos[mask]
        jnt_qvel = self.env.sim.data.qvel[mask]

        print(jnt_ranges.shape)
        print(jnt_axis.shape)
        print(jnt_xaxis.shape)
        print(jnt_xanchor.shape)
        print(jnt_qpos.shape)
        print(jnt_qvel.shape)


if __name__ == '__main__':
    import gym
    from pathlib import Path

    env = gym.make('FetchReachEnv-v0')
    home = str(Path.home())
    g = RobotGraph(env,
                   home + '/Documents/SAC_GCN/CustomGymEnvs/envs/fetchreach/FetchReachEnv_v0_Normal/assets/fetch/')
    # print(env.sim.data.get_body_xpos("robot0:forearm_roll_link"))
    # print(env.sim.model.body_name2id("robot0:forearm_roll_link"))
    # id = env.sim.model.body_name2id("robot0:forearm_roll_link")
    # print(env.sim.model.body_mass[id])
