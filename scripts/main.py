from controller_base import ControllerBase
from simulation import Simulation
from model_base import ModelBase
from cost_base import CostBase

import numpy as np

from utile import  parse_config, gif_path, plt_paths
import argparse
import os

from tqdm import tqdm

import matplotlib.pyplot as plt

def next_goal(x, y, r, w, t):
    x = (x + r*np.cos(w*t))
    vx = (-r*w*np.sin(w*t))
    y = (y + r*np.sin(w*t))
    vy = (r*w*np.cos(w*t))
    return np.hstack([x, vx, y, vy]).reshape(-1, 1)

def parse_arg():
    parser = argparse.ArgumentParser(prog="mppi", description="mppi-tensorflow")
    parser.add_argument('config', metavar='c', type=str, help='Config file path')
    parser.add_argument('-r', '--render', action='store_true', help="render the simulation")
    parser.add_argument('-l', '--log', action='store_true', help="log in tensorboard")
    parser.add_argument('-s', '--steps', type=int, help='number of training steps', default=200)
    parser.add_argument('-t', '--train', type=int, help='training step iterations', default=10)
    parser.add_argument('-g', '--gif', type=str, default=None, help="Save all the trajectories in a gif file (takes a lot of time)")
    args = parser.parse_args()
    return args.config, args.render, args.log, args.steps, args.train, args.gif

def main():
    conf_file, render, log, max_steps, train_iter, gif = parse_arg()
    env, goal, dt, tau, init, lam, maxu, noise, samples, s_dim, a_dim, q = parse_config(conf_file)

    sim = Simulation(env, s_dim, a_dim, goal, render)
    #state_goal = sim.getGoal()
    cx = 0.
    cy = 0.5
    cr = 1.
    omega = np.pi/100.

    goal =  next_goal(cx, cy, cr, 0, omega)

    model = ModelBase(mass=5,
                      dt=dt,
                      state_dim=s_dim,
                      act_dim=a_dim,
                      name=os.path.splitext(os.path.basename(env))[0])

    cost = CostBase(lam=lam,
                    sigma=noise,
                    goal=goal,
                    tau=tau,
                    Q=q)

    cont = ControllerBase(model, cost,
                          k=samples, tau=tau, dt=dt, s_dim=s_dim, a_dim=a_dim, lam=lam,
                          sigma=noise, log=log, config_file=conf_file)


    prev_time = sim.getTime()
    time = sim.getTime()
    paths_list = []
    weights_list = []
    for step in tqdm(range(max_steps)):
        x = sim.getState()
        u, cost, cost_state, cost_act, noises, paths, weights, action_seq = cont.next(x)
        ng = next_goal(cx, cy, cr, step, omega)
        cont.advanceGoal("next_goal", ng)
        if gif is not None:
            plt_paths(paths, weights, noises, action_seq, step, cont.getGoal())
        while time-prev_time < dt:
            x_next = sim.step(u, ng)
            time=sim.getTime()
        prev_time = time
        cont.save(x, u, x_next, cost, cost_state, cost_act)

        if step % train_iter == 0:
            cont.train()
    gif_path(max_steps, gif)

if __name__ == '__main__':
    main()
