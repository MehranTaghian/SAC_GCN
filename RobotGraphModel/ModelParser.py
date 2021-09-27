import xml.etree.ElementTree as et
import custom_gym_envs
import gym


class ModelParser:
    def __init__(self, robot_directory):
        self.root = et.parse(robot_directory + 'robot.xml').getroot()
        self.element_list = [e for e in self.root.iter()]
        self.parent_map = {c: p for p in self.root.iter() for c in p}

        # Fill joints list with xml objects of <joint> tag. This list shows the edges of the graph
        self.joints = [j for j in self.root.iter() if j.tag == 'joint']

        # This dictionary represents a joint along with two separate bodies whom it attached together.
        # Those nodes whose super parent is 'mujoco' element, they are not connected to another body. So just ignore
        # them
        # TODO this way, robot0:slide0, robot0:slide1, and robot0:slide2 are being ignored
        self.joints_connections = {j: (self.parent_map[j], self.parent_map[self.parent_map[j]]) for j in self.joints
                                   if self.parent_map[self.parent_map[j]].tag != 'mujoco'}


if __name__ == "__main__":
    p = ModelParser(
        '/home/mehran/Documents/SAC_GCN/custom_gym_envs/envs/fetchreach/FetchReachEnv_v0_Normal/assets/fetch/')

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
