from xml.etree import ElementTree

class ModelParser:
    def __init__(self, model):
        self.root = ElementTree.fromstring(model)
        self.element_list = [e for e in self.root.iter()]
        self.parent_map = {c: p for p in self.root.iter() for c in p}

        # Fill joints list with xml objects of <joint> tag. This list shows the edges of the graph
        self.joints = [j for j in self.root.iter() if j.tag == 'joint' and 'name' in j.attrib]
        self.bodies = [b for b in self.root.iter() if b.tag == 'body']

        if 'name' not in self.parent_map[self.bodies[0]].attrib:
            self.parent_map[self.bodies[0]].attrib['name'] = 'world'

        for i in range(len(self.bodies)):
            if 'name' not in self.bodies[i].attrib:
                self.bodies[i].attrib['name'] = f'body_{i}'

        # This dictionary represents a joint along with two separate bodies whom it attached together.
        # Those nodes whose super parent is 'mujoco' element, they are not connected to another body. So just ignore
        # them
        # Here we have two types of connections, one is a connection between two bodies with a joint between them
        # and another is the second body (child body) is welded to the parent body. Each connection is shown using
        # a triple (n1, n2, e) where n1 is the child body, n2 is the parent body and e is the edge which can be either
        self.connections_joint = [(self.parent_map[j], self.parent_map[self.parent_map[j]], j) for j in self.joints
                                  if ('name' not in self.parent_map[self.parent_map[j]].attrib
                                      or ('camera' not in self.parent_map[self.parent_map[j]].attrib['name']
                                          and 'laser' not in self.parent_map[self.parent_map[j]].attrib['name'])
                                      ) and
                                  ('name' not in self.parent_map[j].attrib
                                   or ('camera' not in self.parent_map[j].attrib['name']
                                       and 'laser' not in self.parent_map[j].attrib['name'])
                                   )]
        # a joint or welded.
        self.connections_welded = [(b, self.parent_map[b], None) for b in self.bodies
                                   if ('name' not in b.attrib or
                                       ('camera' not in b.attrib['name'] and 'laser' not in b.attrib['name'])
                                       )
                                   and
                                   ('name' not in self.parent_map[b].attrib or
                                    ('camera' not in self.parent_map[b].attrib['name']
                                     and 'laser' not in self.parent_map[b].attrib['name'])
                                    )]

        # Removing those connections that have a joint from welded connections list
        for p1, p2, _ in self.connections_joint:
            con = (p1, p2, None)
            if con in self.connections_welded:
                self.connections_welded.remove(con)
        self.connections = self.connections_joint + self.connections_welded

        # self.connections = self.connections_joint


if __name__ == "__main__":
    env_name = 'FetchReachDense-v1'
    # env_name = 'AntEnv_v0_Normal'
    # env = gym.make(env_name)
    # p = ModelParser(env.sim.model.get_xml(), env_name)

    # env = gym.make('FetchReachDense-v1')
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
