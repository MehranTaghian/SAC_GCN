from RobotGraphModel.ModelParser import ModelParser
import numpy as np
import matplotlib.pyplot as plt


class RobotGraph:
    def __init__(self, sim, env_name, weld_joints=None, bidirectional=False):
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
        Furthermore, we remove those body parts that contain 'camera' or 'laser' in their name.

        Note: in order for the model to be parsed, all the bodies should be named properly except the world body. The
        world body can have not name.

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
            <default> tag (e.g. for fetchreach.sh, the default values are in the shared.xml file).
            The edge_features list contains the features of edges in the order the came in the edge_list. (also
            in the order they came in edge_from and edge_to)

        :@param sim simulator of the environment
        :@param model_path path to the robot's xml file
        :@param weld_joints weld some specific joints in order to prevent them from moving
        :@param save_log if true, a log of the change in node and edge features would be saved.
        """
        self.weld_joints = weld_joints
        self.bidirectional = bidirectional
        self.sim = sim
        self.parser = ModelParser(self.sim.model.get_xml(), env_name)
        self.node_list = set()
        self.edge_list = {}
        # edges_from and edges_to are based on the index of the node in the node_list, not the joint_id or body_id
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

    def get_graph_obs(self):
        """
        :return:
        """
        self.node_features = self.extract_node_features()
        self.edge_features = self.extract_edge_features()

        return {'node_features': self.node_features.copy(),
                'edge_features': self.edge_features.copy(),
                'edges_from': self.edges_from,
                'edges_to': self.edges_to}

    def fill_node_edge_lists(self):
        # removing welded joints
        # for node1, node2, joint in self.parser.connections:
        #     # Note: all the body parts except the world body MUST be named.
        #     if 'name' not in node1.attrib:
        #         node1.attrib['name'] = 'world'
        #     elif 'name' not in node2.attrib:
        #         node2.attrib['name'] = 'world'
        #     if joint is not None and self.weld_joints is not None:
        #         if joint.attrib['name'] not in self.weld_joints:
        #             self.node_list.add(node1)
        #             self.node_list.add(node2)
        #             self.edge_list.append(joint)
        #         else:  # weld that joint
        #             self.node_list.add(node1)
        #             self.node_list.add(node2)
        #             self.edge_list.append(None)
        #     else:
        #         self.node_list.add(node1)
        #         self.node_list.add(node2)
        #         self.edge_list.append(joint)

        # with welded parts
        for node1, node2, _ in self.parser.connections:
            self.node_list.add(node1)
            self.node_list.add(node2)

        self.node_list = list(self.node_list)

    def generate_adjacency_matrix(self):

        # removing welded parts
        # for n1, n2, j in self.parser.connections:
        #     if j is not None and self.weld_joints is not None:
        #         if j.attrib['name'] not in self.weld_joints:
        #             edge_from = self.node_list.index(n1)
        #             edge_to = self.node_list.index(n2)
        #             self.edges_from.append(edge_from)
        #             self.edges_to.append(edge_to)
        #             # self.edges_from.append(edge_to)
        #             # self.edges_to.append(edge_from)
        #     else:
        #         edge_from = self.node_list.index(n1)
        #         edge_to = self.node_list.index(n2)
        #         self.edges_from.append(edge_from)
        #         self.edges_to.append(edge_to)
        #         # self.edges_from.append(edge_to)
        #         # self.edges_to.append(edge_from)

        # with welded parts
        for n1, n2, j in self.parser.connections:
            edge_from = self.node_list.index(n1)
            edge_to = self.node_list.index(n2)
            j = j if self.weld_joints is None or (j is not None and self.weld_joints is not None
                                                  and j.attrib['name'] not in self.weld_joints) else None

            if (edge_from, edge_to) not in self.edge_list.keys():
                self.edge_list[(edge_from, edge_to)] = []
                self.edges_from.append(edge_from)
                self.edges_to.append(edge_to)
                if self.bidirectional:
                    self.edge_list[(edge_to, edge_from)] = []
                    self.edges_from.append(edge_to)
                    self.edges_to.append(edge_from)

            if j is not None:
                self.edge_list[(edge_from, edge_to)].append(j)
                if self.bidirectional:
                    self.edge_list[(edge_to, edge_from)].append(j)

        self.edges_from = np.array(self.edges_from)
        self.edges_to = np.array(self.edges_to)

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
        raise NotImplementedError()

    def extract_edge_features(self):
        """
        For extracting edge features which are joint features, we use accessor functions to access each joint feature
        by name. This data can be accessed through PyMjModel and PyMjData in env.sim.model and env.sim.data. These are
        static features like range of freedom, , etc.

        ranges [2]: joint limits in range of motion form range[0] to range[1]
        # axis [3]: local joint axis (removed for now)
        xaxis [3]: Cartesian joint axis
        xanchor [3]: Cartesian position of joint anchor
        qpos [1]: angle of each joint
        qvel [1]: open or close velocity of each joint
        """
        raise NotImplementedError()
