import xml.etree.ElementTree as et
import CustomGymEnvs
import gym


# TODO: receive a list of keywords to extract those bodies containing that keyword in their name
class ModelParser:
    def __init__(self, robot_directory):
        self.root = et.parse(robot_directory + 'robot.xml').getroot()
        self.element_list = [e for e in self.root.iter()]
        self.parent_map = {c: p for p in self.root.iter() for c in p}

        # Fill joints list with xml objects of <joint> tag. This list shows the edges of the graph
        self.joints = [j for j in self.root.iter() if j.tag == 'joint']
        self.bodies = [b for b in self.root.iter() if b.tag == 'body']

        # This dictionary represents a joint along with two separate bodies whom it attached together.
        # Those nodes whose super parent is 'mujoco' element, they are not connected to another body. So just ignore
        # them
        # TODO this way, robot0:slide0, robot0:slide1, and robot0:slide2 are being ignored
        # Here we have two types of connections, one is a connection between two bodies with a joint between them
        # and another is the second body (child body) is welded to the parent body. Each connection is shown using
        # a triple (n1, n2, e) where n1 is the child body, n2 is the parent body and e is the edge which can be either
        # a joint or welded.
        self.connections_joint = [(self.parent_map[j], self.parent_map[self.parent_map[j]], j) for j in self.joints
                                  if self.parent_map[self.parent_map[j]].tag != 'mujoco'
                                  and 'camera' not in self.parent_map[self.parent_map[j]].attrib['name']
                                  and 'camera' not in self.parent_map[j].attrib['name']
                                  and 'laser' not in self.parent_map[self.parent_map[j]].attrib['name']
                                  and 'laser' not in self.parent_map[j].attrib['name']]
        self.connections_welded = [(b, self.parent_map[b], None) for b in self.bodies
                                   if self.parent_map[b].tag != 'mujoco'
                                   and 'camera' not in b.attrib['name']
                                   and 'camera' not in self.parent_map[b].attrib['name']
                                   and 'laser' not in b.attrib['name']
                                   and 'laser' not in self.parent_map[b].attrib['name']]

        for p1, p2, _ in self.connections_joint:
            con = (p1, p2, None)
            if con in self.connections_welded:
                self.connections_welded.remove(con)

        self.connections = self.connections_joint + self.connections_welded


if __name__ == "__main__":
    p = ModelParser(
        '/home/mehran/Documents/SAC_GCN/CustomGymEnvs/envs/fetchreach/CustomFetchReach/assets/fetch/')

    # env = gym.make('FetchReachEnv-v0')
    # print(env.sim.data.qpos)
    # for j in p.joints:
    #     print(j.attrib['name'], env.sim.data.get_joint_qpos(j.attrib['name']))
    #     print(j)
    #     parent1 = p.parent_map[j]
    #     parent2 = p.parent_map[parent1]
    #     try:
    #         print("Parent 1:", parent1.attrib['name'])
    #         print("Parent 2:", parent2.attrib['name'])
    #     except KeyError:
    #         pass
