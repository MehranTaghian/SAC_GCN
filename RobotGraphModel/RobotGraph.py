from RobotGraphModel.ModelParser import ModelParser
import numpy as np
import matplotlib.pyplot as plt


class RobotGraph:
    def __init__(self, sim, model_path, weld_joints, plot_log=False):
        """
        Based on the definition in the MuJoCo documentation:
        This element creates a joint. As explained in Kinematic tree, a joint creates motion degrees of freedom
        between the body where it is defined and the body's parent.
        Using this definition, we parsed the XML file and generate the graph of nodes (bodies) and edges (joints or
        welded). As mentioned in the documentation, the link between two bodies is specified with parent and
        child relationships in the xml file. If a joint is defined in the child body, then there is a motion
        between the parent and child. If there is no joint defined, then the child is welded to the parent. We should
        consider a specific feature vector for welded links. One way is to set it to all zero vector.
        ATTENTION: the graph might be a directed graph (from one point, it is attached to a body, and from the
        other point, another body is attached to it)
        There is also a weld_joints argument to this class which shows the list of joints in the original robot that
        we want them not to move. For example, in the FetchReach environment, we don't want the list of joints:
        ['robot0:torso_lift_joint', 'robot0:head_pan_joint', 'robot0:head_tilt_joint', 'robot0:shoulder_pan_joint']
        to move. Therefore, when we are adding joints into the edge_list, we replace their object with None
        in the fill_node_edge_lists function.

        For node and edge features, I used the following link to a complete reference of the bodies and joints:
        http://www.mujoco.org/book/XMLreference.html

        Node features:
            Each node is a body which contains inertial features along with geometric features. For a complete reference
            of what inertial and geom tags show, refer to the link above.
            Node features consist of diaginertia, mass, pos, quat which are all attributes of the <inertial> tag
            inside the body. Also they include the attributes of the body tag itself.
            The node_features list contains the features of edges in the order the came in the node_list.

        Edge features:
            Edges are joints, therefore edge features are selected among <joint> attributes. These include axis, range,
            armature, damping, frictionloss, stiffness, etc. Some of these attributes are set to default values in
            <default> tag (e.g. for fetchreach, the default values are in the shared.xml file).
            The edge_features list contains the features of edges in the order the came in the edge_list. (also
            in the order they came in edge_from and edge_to)

        TODO: add gripper as a node which is a leaf node attached to a body. And also check the
         gripper coordinate with the robot0:r_gripper_finger_joint and robot0:l_gripper_finger_joint.
         This part is mandatory because our reward function depends on that and the agent should have
         the position of the gripper as part of its observation.

        :@param sim simulator of the environment
        :@param model_path path to the robot's xml file
        :@param weld_joints weld some specific joints in order to prevent them from moving
        :@param save_log if true, a log of the change in node and edge features would be saved.
        """
        self.weld_joints = weld_joints
        self.sim = sim
        self.parser = ModelParser(model_path)
        self.node_list = set()
        self.edge_list = []
        # edges_from and edges_to are based on the index of the node in the node_list, not the joint_id or body_id
        self.edges_from = []
        self.edges_to = []
        self.node_features = None
        self.edge_features = None
        self.plot_log = plot_log
        if self.plot_log:
            self.plot_itr = 100
            self.plot_itr_counter = 0
            self.log = {
                'node': {
                    'mass': [],
                    'pos': [],
                    'quat': [],
                    'xpos': [],
                    'xquat': [],
                    'ipos': [],
                    'iquat': [],
                    'inertia': [],
                    'xvelp': [],
                    'xvelr': []},
                'edge': {
                    'ranges': [],
                    'axis': [],
                    'xaxis': [],
                    'xanchor': [],
                    'qpos': [],
                    'qvel': []}
            }

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

    def get_graph_obs(self):
        """

        :return:
        """
        self.extract_node_features()
        self.extract_edge_features()

        if self.plot_log:
            self.plot_itr_counter += 1
            if self.plot_itr_counter == self.plot_itr:
                self.log_plot()

        return {'node_features': self.node_features.copy(),
                'edge_features': self.edge_features.copy(),
                'edges_from': self.edges_from,
                'edges_to': self.edges_to}

    def generate_adjacency_matrix(self):
        # DEBUG
        # for i in range(len(self.node_list)):
        #     print(i, self.node_list[i].attrib)
        # END DEBUG

        for n1, n2, _ in self.parser.connections:
            edge_from = self.node_list.index(n1)
            edge_to = self.node_list.index(n2)
            self.edges_from.append(edge_from)
            self.edges_to.append(edge_to)

        self.edges_from = np.array(self.edges_from)
        self.edges_to = np.array(self.edges_to)

    def fill_node_edge_lists(self):
        for node1, node2, joint in self.parser.connections:
            self.node_list.add(node1)
            self.node_list.add(node2)
            joint = joint if (joint is not None and joint.attrib['name'] not in self.weld_joints) else None
            self.edge_list.append(joint)
            # DEBUG
            # print('node1: ' + node1.attrib['name'],
            #       'node2: ' + node2.attrib['name'],
            #       'joint: ' + joint.attrib['name'])
            # END DEBUG

        self.node_list = list(self.node_list)

    def extract_node_features(self):
        """
        For extracting node features which are body features, we use accessor functions to access each body feature
        by name. This data can be accessed through PyMjModel and PyMjData in env.sim.model and env.sim.data. These are
        static features like mass, inertia, etc, in addition to dynamic features like xvelp, xvelr, xpos, and xquat that
        change during the run time and by doing forward kinematics.

        The body features according to the http://www.mujoco.org/book/APIreference.html are as follows:

        mass [1]: mass of the body
        pos [3]: position offset rel. to parent body
        quat [4]: orientation offset rel. to parent body
        xpos [3]: Cartesian position of body frame (the value is stored in this variable after
                doing forward kinematics)
        xquat [4]: Cartesian orientation of body frame (the value is stored in this variable after
                doing forward kinematics)
        ipos [3]: local position of center of mass
        iquat [4]: local orientation of center of mass
        inertia [3]: diagonal inertia in ipos/iquat frame
        xvelp [3]: positional velocity
        xvelr [3]: rotational velocity

        """
        mask = [self.sim.model.body_name2id(body_name.attrib['name']) for body_name in self.node_list]

        bodies_mass = self.sim.model.body_mass[mask, np.newaxis]
        # The following features were constant. Instead of using static features for pos and quat, we use dynamic
        # features xpos and xquat which change during the runtime. You can see them in the log
        bodies_inertia = self.sim.model.body_inertia[mask, :]
        bodies_pos = self.sim.model.body_pos[mask, :]
        bodies_quat = self.sim.model.body_quat[mask, :]
        bodies_ipos = self.sim.model.body_ipos[mask, :]
        bodies_iquat = self.sim.model.body_iquat[mask, :]
        bodies_xpos = self.sim.data.body_xpos[mask, :]
        bodies_xquat = self.sim.data.body_xquat[mask, :]
        bodies_xvelp = self.sim.data.body_xvelp[mask, :]
        bodies_xvelr = self.sim.data.body_xvelr[mask, :]

        if self.plot_log:
            self.log['node']['mass'].append(bodies_mass.copy())
            self.log['node']['inertia'].append(bodies_inertia.copy())
            self.log['node']['pos'].append(bodies_pos.copy())
            self.log['node']['quat'].append(bodies_quat.copy())
            self.log['node']['xpos'].append(bodies_xpos.copy())
            self.log['node']['xquat'].append(bodies_xquat.copy())
            self.log['node']['xvelp'].append(bodies_xvelp.copy())
            self.log['node']['xvelr'].append(bodies_xvelr.copy())
            self.log['node']['ipos'].append(bodies_ipos.copy())
            self.log['node']['iquat'].append(bodies_iquat.copy())

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
            [bodies_mass.copy(),
             # bodies_inertia.copy(),
             # bodies_pos.copy(),
             # bodies_quat.copy(),
             # bodies_ipos.copy(),
             # bodies_iquat.copy(),
             bodies_xpos.copy(),
             bodies_xquat.copy(),
             bodies_xvelp.copy(),
             bodies_xvelr.copy()],
            axis=1)

    def extract_edge_features(self):
        """
        For extracting edge features which are joint features, we use accessor functions to access each joint feature
        by name. This data can be accessed through PyMjModel and PyMjData in env.sim.model and env.sim.data. These are
        static features like range of freedom, , etc.

        TODO: check dynamic features in runtime. Also compare xaxis and axis features during runtime
        ranges [2]: joint limits in range of motion form range[0] to range[1]
        # axis [3]: local joint axis (removed for now)
        xaxis [3]: Cartesian joint axis
        xanchor [3]: Cartesian position of joint anchor
        qpos [1]: angle of each joint
        qvel [1]: open or close velocity of each joint
        """
        mask = [self.sim.model.joint_name2id(joint_name.attrib['name']) for joint_name in self.edge_list if joint_name
                is not None]

        # axis is a static constant feature which does not change during the runtime. Thus we remove it and use
        # dynamic features instead.

        feature_list = []

        for edge in self.edge_list:
            if edge is not None:
                id = self.sim.model.joint_name2id(edge.attrib['name'])
                jnt_ranges = self.sim.model.jnt_range[id]
                # jnt_axis = self.sim.model.jnt_axis[id]
                jnt_xaxis = self.sim.data.xaxis[id]
                jnt_xanchor = self.sim.data.xanchor[id]
                jnt_qpos = self.sim.data.qpos[id]
                jnt_qvel = self.sim.data.qvel[id]

                # DEBUG
                # print('jnt_ranges', jnt_ranges)
                # # print('jnt_axis', jnt_axis.shape)
                # print('jnt_xaxis', jnt_xaxis)
                # print('jnt_xanchor', jnt_xanchor)
                # print('jnt_qpos', jnt_qpos)
                # print('jnt_qvel', jnt_qvel)
                # END DEBUG

                edge_feature = np.concatenate([jnt_ranges.copy(),
                                               # jnt_axis.copy(),
                                               jnt_xaxis.copy(),
                                               jnt_xanchor.copy(),
                                               [jnt_qpos.copy()],
                                               [jnt_qvel.copy()]])
            else:
                edge_feature = np.zeros(10)

            feature_list.append(edge_feature)

        # LOGGING
        jnt_ranges = self.sim.model.jnt_range[mask, :]
        jnt_axis = self.sim.model.jnt_axis[mask, :]
        jnt_xaxis = self.sim.data.xaxis[mask, :]
        jnt_xanchor = self.sim.data.xanchor[mask, :]
        jnt_qpos = self.sim.data.qpos[mask, np.newaxis]
        jnt_qvel = self.sim.data.qvel[mask, np.newaxis]

        if self.plot_log:
            self.log['edge']['ranges'].append(jnt_ranges.copy())
            self.log['edge']['axis'].append(jnt_axis.copy())
            self.log['edge']['xaxis'].append(jnt_xaxis.copy())
            self.log['edge']['xanchor'].append(jnt_xanchor.copy())
            self.log['edge']['qpos'].append(jnt_qpos.copy())
            self.log['edge']['qvel'].append(jnt_qvel.copy())
        # END OF LOGGING

        self.edge_features = np.array(feature_list)

    def log_plot(self):
        for n in self.log['node'].keys():
            y = np.array(self.log['node'][n])
            plt.figure()
            for i in range(y.shape[1]):
                for j in range(y.shape[2]):
                    plt.plot(y[:, i, j])
            plt.title(n)

        for n in self.log['edge'].keys():
            y = np.array(self.log['edge'][n])
            plt.figure()
            for i in range(y.shape[1]):
                for j in range(y.shape[2]):
                    plt.plot(y[:, i, j])
            plt.title(n)
        plt.show()


if __name__ == '__main__':
    import gym
    from pathlib import Path

    env = gym.make('FetchReachEnv-v0')
    home = str(Path.home())
    g = RobotGraph(env.sim,
                   home + '/Documents/SAC_GCN/CustomGymEnvs/envs/fetchreach/CustomFetchReach/assets/fetch/',
                   ['robot0:torso_lift_joint', 'robot0:head_pan_joint', 'robot0:head_tilt_joint',
                    'robot0:shoulder_pan_joint'])

    print(g.node_features.shape)
    print(g.edge_features.shape)
    node_id_list = []
    for n in g.node_list:
        node_id_list.append(env.sim.model.body_name2id(n.attrib['name']))

    for id in sorted(node_id_list):
        print(id, env.sim.model.body_id2name(id))
    print('#' * 100)

    edge_id_list = []
    for e in g.edge_list:
        if e is not None:
            edge_id_list.append(int(env.sim.model.joint_name2id(e.attrib['name'])))

    for id in sorted(edge_id_list):
        print(id, env.sim.model.joint_id2name(id))

    # print(env.sim.data.get_body_xpos("robot0:forearm_roll_link"))
    # print(env.sim.model.body_name2id("robot0:forearm_roll_link"))
    # id = env.sim.model.body_name2id("robot0:forearm_roll_link")
    # print(env.sim.model.body_mass[id])
