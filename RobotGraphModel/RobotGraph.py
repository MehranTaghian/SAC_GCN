from ModelParser import ModelParser
import numpy as np


class RobotGraph:
    def __init__(self, model_path):
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
        self.parser = ModelParser(model_path)
        self.node_list = set()
        self.edge_list = set()
        self.edges_from = []
        self.edges_to = []
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
        # TODO generate feature vectors for nodes and edges
        # print(self.edges_from)
        # print(self.edges_to)

    def generate_adjacency_matrix(self):
        # DEBUG
        # for i in range(len(self.node_list)):
        #     print(i, self.node_list[i].attrib['name'])
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


if __name__ == '__main__':
    g = RobotGraph(
        '/CustomGymEnvs/envs/fetchreach/FetchReachEnv_v0_Normal/assets/fetch/')
