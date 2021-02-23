import mujoco_py as mj
import os
from mujoco_py.modder import TextureModder
import numpy as np


class Simulation(object):
    def __init__(self, xml_file, goal, render):
        self.render = render
        self.modder = None

        self.model = mj.load_model_from_path(xml_file)
        self.sim = mj.MjSim(self.model)

        self.a_dim = self.sim.data.ctrl.shape[0]
        self.s_dim = 2*self.sim.data.qpos.shape[0]


        self.goal = goal
        if self.render:
            self.viewer = mj.MjViewer(self.sim)
            g = self.sim.data.get_site_xpos("target")
            self.goal=np.zeros((self.s_dim, 1))
            for i in range(int(self.s_dim/2)):
                self.goal[2*i] = g[i]


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


    def step(self, u):
        for i in range(self.a_dim):
            self.sim.data.ctrl[i]=u[0, i]

        if self.render:
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
