from simulation import Simulation
import numpy as np

from cpprb import ReplayBuffer

import argparse
from mppi_tf.scripts.utile import parse_config
import os

from tqdm import tqdm

import matplotlib.pyplot as plt

def parse_arg():
    parser = argparse.ArgumentParser(prog="log_straight", description="logs q and q_dot for a constant speed")
    parser.add_argument('config', metavar='c', type=str, help='Config file path')
    parser.add_argument('-r', '--render', action='store_true', help="render the simulation")
    parser.add_argument('-l', '--log', action='store_true', help="log in tensorboard")
    args = parser.parse_args()
    return args.config, args.render, args.log




def log_straight():
    config_file, render, log = parse_arg()
    env, _, dt, _, _, _, _, _, _, _, _, s_dim, a_dim, _ = parse_config(config_file)
    sim = Simulation(env, s_dim, a_dim, None, True)
    u = np.ones((1, a_dim, 1))*3
    length = 500
    rb = ReplayBuffer(length,
                           env_dict={"obs": {"shape": (s_dim, 1)},
                           "act": {"shape": (a_dim, 1)},
                           "rew": {},
                           "next_obs": {"shape": (s_dim, 1)},
                           "done": {}})

    x = sim.getState()
    for i in range(length):
        x_next = sim.step(u)
        rb.add(obs=x, act=u, rew=0, next_obs=x_next, done=False)
        x = x_next

    if render:
        data = rb.get_all_transitions()
        plot_logs(data['obs'], data['next_obs'], length)


def plot_logs(x, x_next, length):
    plt.figure(0)
    plt.plot(range(length), x[:, 0, 0], range(length), x_next[:, 0, 0])
    plt.figure(1)
    plt.plot(range(length), x[:, 1, 0], range(length), x_next[:, 1, 0])
    plt.show()

def main():
    log_straight()

if __name__ == '__main__':
    main()
