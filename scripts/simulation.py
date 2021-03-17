import mujoco_py as mj
import os
from mujoco_py.modder import TextureModder
import numpy as np


class Simulation(object):
    def __init__(self, xml_file, s_dim, a_dim, goal, render):
        self.render = render

        self.model = mj.load_model_from_path(xml_file)
        self.sim = mj.MjSim(self.model)

        self.a_dim = a_dim
        self.s_dim = s_dim


        self.goal = goal
        if self.render:
            self.viewer = mj.MjViewer(self.sim)
            g = self.sim.data.get_site_xpos("target")
            self.goal=np.zeros((self.s_dim, 1))
            for i in range(int(self.s_dim/2)):
                self.goal[2*i] = g[i]
            self.viewer.add_marker(pos=np.array([0, 1 , 1]), label="shpere")

    def getTime(self):
        return self.sim.data.time

    def getGoal(self):
        return self.goal


    def getState(self):
        x = np.zeros((self.s_dim, 1))
        for i in range(int(self.s_dim/2)):
            x[2*i] = self.sim.data.qpos[i]
            x[2*i+1] = self.sim.data.qvel[i]
        return x


    def step(self, u, goal=None):
        for i in range(self.a_dim):
            self.sim.data.ctrl[i]=u[0, i]

        if self.render:
            if goal is not None:
                self.viewer.add_marker(pos=np.array([goal[0, 0], goal[2, 0] , 0.3]), size=np.array([0.05, 0.05, 0.05]), label="shpere")
            else:
                print("Warning no goal provided for rendering")
            self.viewer.render()
        self.sim.step()

        return self.getState()


def main():
    sim = Simulation("point_mass1d.xml", True)
    t = 1
    for i in range(t):

        x = sim.step(np.array([0.02]))


if __name__ == '__main__':
    main()
