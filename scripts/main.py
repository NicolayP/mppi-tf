from controller_base import ControllerBase
from simulation import Simulation
from model_base import ModelBase
from cost_base import CostBase

import numpy as np

import argparse
import yaml
import os

from tqdm import tqdm

def parse_arg():
    parser = argparse.ArgumentParser(prog="mppi", description="mppi-tensorflow")
    parser.add_argument('config', metavar='c', type=str, help='Config file path')
    parser.add_argument('-r', '--render', action='store_true', help="render the simulation")
    parser.add_argument('-l', '--log', action='store_true', help="log in tensorboard")
    parser.add_argument('-s', '--steps', type=int, help='number of training steps', default=200)
    parser.add_argument('-t', '--train', type=int, help='training step iterations', default=10)
    args = parser.parse_args()
    return args.config, args.render, args.log, args.steps, args.train

def parse_config(file):
    with open(file) as file:
        conf = yaml.load(file, Loader=yaml.FullLoader)
        env = conf['env']
        s_dim = conf['state-dim']
        a_dim = conf['action-dim']
        goal = np.expand_dims(np.array(conf['goal']), -1)
        dt = conf['dt']
        tau = conf['horizon']
        init = np.array(conf['init-act'])
        lam = conf['lambda']
        maxu = np.array(conf['max-a'])
        noise = np.array(conf['noise'])
        samples = conf['samples']
        cost = conf['cost']
        q = np.array(cost['w'])


    return env, goal, dt, tau, init, lam, maxu, noise, samples, s_dim, a_dim, q

def main():
    conf_file, render, log, max_steps, train_iter = parse_arg()
    env, goal, dt, tau, init, lam, maxu, noise, samples, s_dim, a_dim, q = parse_config(conf_file)

    sim = Simulation(env, goal, render)
    state_goal = sim.getGoal()

    model = ModelBase(mass=5,
                      dt=dt,
                      state_dim=s_dim,
                      act_dim=a_dim,
                      name=os.path.splitext(os.path.basename(env))[0])

    cost = CostBase(lam=lam,
                    sigma=noise,
                    goal=goal,
                    Q=q)

    cont = ControllerBase(model, cost,
                          k=samples, tau=tau, dt=dt, s_dim=s_dim, a_dim=a_dim, lam=lam,
                          sigma=noise, log=log)


    prev_time = sim.getTime()
    time = sim.getTime()

    for step in tqdm(range(max_steps)):
        x = sim.getState()
        u, cost = cont.next(x)
        while time-prev_time < dt:
            x_next = sim.step(u)
            time=sim.getTime()
        prev_time = time
        cont.save(x, u, x_next, cost)

        if step % train_iter == 0:
            cont.train()

if __name__ == '__main__':
    main()
