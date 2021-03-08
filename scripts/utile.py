import numpy as np
import yaml
import os

from tqdm import tqdm

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
