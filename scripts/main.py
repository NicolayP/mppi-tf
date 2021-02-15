from controller_base import ControllerBase
from simulation import Simulation
from model_base import ModelBase
from cost_base import CostBase

import numpy as np

def parse_arg(args):
    pass

def parse_config(file):
    pass

def main():
    sim = Simulation("point_mass1d.xml", False)
    state_goal = sim.getGoal()
    goal = np.zeros((2, 1))
    goal[0] = state_goal[0]
    model = ModelBase(mass=5, dt=0.01, state_dim=2, act_dim=1)
    cost = CostBase(lam=1, sigma=np.array([[1]]), goal=goal, Q=np.array([[1., 0.], [0., 1.]]))

    cont = ControllerBase(model, cost,
                          k=100, tau=20, dt=0.01, s_dim=2, a_dim=1, lam=1.,
                          sigma=np.array([[1]]))

    i = 1
    log = True
    step = 0
    max_steps = 20
    while step < max_steps:
        x = sim.getState()
        u = cont.next(x)
        x_next = sim.step(u)
        cont.save(x, u, x_next, log)

        if i % 100 == 0:
            i = 0
            cont.train(log)
            step += 1

        i+=1
if __name__ == '__main__':
    main()
