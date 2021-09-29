from RobotGraphModel.ModelParser import ModelParser
import numpy as np
import matplotlib.pyplot as plt


class RobotGraph:
    def __init__(self, sim, model_path, save_log=False):
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

        TODO: add gripper as a node which is a leaf node attached to a body. And also check the
         gripper coordinate with the robot0:r_gripper_finger_joint and robot0:l_gripper_finger_joint.
         This part is mandatory because our reward function depends on that and the agent should have
         the position of the gripper as part of its observation.

        :@param sim simulator of the environment
        :@param model_path path to the robot's xml file
        :@param save_log if true, a log of the change in node and edge features would be saved.
        """
        self.sim = sim
        self.parser = ModelParser(model_path)
        self.node_list = set()
        self.edge_list = set()
        self.edges_from = []
        self.edges_to = []
        self.node_features = None
        self.edge_features = None
        self.save_log = save_log
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
        TODO: check if static features don't change, remove the call to reclaiming those values like range in joint.
        Returns:
        """
        self.extract_node_features()
        self.extract_edge_features()

        self.plot_itr_counter += 1
        print(self.plot_itr_counter)
        if self.plot_itr_counter == self.plot_itr:
            self.log_plot()
            self.plot_itr_counter = 0

        return {'node_features': self.node_features.copy(),
                'edge_features': self.edge_features.copy()}

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

        TODO: check dynamic features in runtime.
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
        bodies_inertia = self.sim.model.body_inertia[mask, :]
        bodies_pos = self.sim.model.body_pos[mask, :]
        bodies_quat = self.sim.model.body_quat[mask, :]
        bodies_xpos = self.sim.data.body_xpos[mask, :]
        bodies_xquat = self.sim.data.body_xquat[mask, :]
        bodies_xvelp = self.sim.data.body_xvelp[mask, :]
        bodies_xvelr = self.sim.data.body_xvelr[mask, :]
        bodies_ipos = self.sim.model.body_ipos[mask, :]
        bodies_iquat = self.sim.model.body_iquat[mask, :]

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
            [bodies_mass.copy(), bodies_inertia.copy(), bodies_pos.copy(), bodies_quat.copy(), bodies_xpos.copy(),
             bodies_xquat.copy(), bodies_xvelp.copy(), bodies_xvelr.copy(), bodies_ipos.copy(), bodies_iquat.copy()],
            axis=1)

    def extract_edge_features(self):
        """
        For extracting edge features which are joint features, we use accessor functions to access each joint feature
        by name. This data can be accessed through PyMjModel and PyMjData in env.sim.model and env.sim.data. These are
        static features like range of freedom, , etc.

        TODO: write the detailed documentation of each feature
        TODO: check dynamic features in runtime. Also compare xaxis and axis features during runtime
        ranges [2]: joint limits in range of motion form range[0] to range[1]
        axis [3]: local joint axis
        xaxis [3]: Cartesian joint axis
        xanchor [3]: Cartesian position of joint anchor
        qpos [1]: angle of each joint
        qvel [1]: open or close velocity of each joint
        """
        mask = [self.sim.model.joint_name2id(joint_name.attrib['name']) for joint_name in self.edge_list]

        jnt_ranges = self.sim.model.jnt_range[mask, :]
        jnt_axis = self.sim.model.jnt_axis[mask, :]
        jnt_xaxis = self.sim.data.xaxis[mask, :]
        jnt_xanchor = self.sim.data.xanchor[mask, :]
        jnt_qpos = self.sim.data.qpos[mask, np.newaxis]
        jnt_qvel = self.sim.data.qvel[mask, np.newaxis]

        self.log['edge']['ranges'].append(jnt_ranges.copy())
        self.log['edge']['axis'].append(jnt_axis.copy())
        self.log['edge']['xaxis'].append(jnt_xaxis.copy())
        self.log['edge']['xanchor'].append(jnt_xanchor.copy())
        self.log['edge']['qpos'].append(jnt_qpos.copy())
        self.log['edge']['qvel'].append(jnt_qvel.copy())

        # DEBUG
        # print('jnt_ranges', jnt_ranges.shape)
        # print('jnt_axis', jnt_axis.shape)
        # print('jnt_xaxis', jnt_xaxis.shape)
        # print('jnt_xanchor', jnt_xanchor.shape)
        # print('jnt_qpos', jnt_qpos.shape)
        # print('jnt_qvel', jnt_qvel.shape)
        # END DEBUG

        self.edge_features = np.concatenate([jnt_ranges.copy(), jnt_axis.copy(), jnt_xaxis.copy(),
                                             jnt_xanchor.copy(), jnt_qpos.copy(), jnt_qvel.copy()], axis=1)

    def log_plot(self):
        for n in self.log['node'].keys():
            y = np.array(self.log['node'][n])
            plt.figure()
            for i in range(y.shape[1]):
                for j in range(y.shape[2]):
                    plt.plot(y[:, i, j])
            plt.title(n)
            # print(y.shape)

        for n in self.log['edge'].keys():
            y = np.array(self.log['edge'][n])
            plt.figure()
            for i in range(y.shape[1]):
                for j in range(y.shape[2]):
                    plt.plot(y[:, i, j])
            plt.title(n)
            # print(y.shape)
        plt.show()


if __name__ == '__main__':
    import gym
    from pathlib import Path

    env = gym.make('FetchReachEnv-v0')
    home = str(Path.home())
    g = RobotGraph(env.sim,
                   home + '/Documents/SAC_GCN/CustomGymEnvs/envs/fetchreach/CustomFetchReach/assets/fetch/')
    # print(env.sim.data.get_body_xpos("robot0:forearm_roll_link"))
    # print(env.sim.model.body_name2id("robot0:forearm_roll_link"))
    # id = env.sim.model.body_name2id("robot0:forearm_roll_link")
    # print(env.sim.model.body_mass[id])
