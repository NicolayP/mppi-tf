import numpy as np
import yaml
import os

from tqdm import tqdm

import matplotlib.pyplot as plt
import imageio

def parse_config(file):
    with open(file) as file:
        conf = yaml.load(file, Loader=yaml.FullLoader)
        env = conf['env']
        s_dim = conf['state-dim']
        a_dim = conf['action-dim']
        dt = conf['dt']
        tau = conf['horizon']
        init = np.array(conf['init-act'])
        lam = conf['lambda']
        maxu = np.array(conf['max-a'])
        noise = np.array(conf['noise'])
        samples = conf['samples']


    return env, dt, tau, init, lam, maxu, noise, samples, s_dim, a_dim


def plt_paths(paths, weights, noises, action_seq, j, goal_seq):

    n_bins=100
    best_idx = np.argmax(weights)
    #todo extract a_sape from tensor
    noises = noises.numpy().reshape(-1, 2)

    fig = plt.figure()
    ax1 = plt.subplot(333)
    ax2 = plt.subplot(336)
    ax3 = plt.subplot(221)
    ax4 = plt.subplot(337)
    ax5 = plt.subplot(338)
    ax6 = plt.subplot(339)
    # We can set the number of bins with the `bins` kwarg
    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-0.1, 2)
    ax1.hist(noises[:, 0], bins=n_bins, density=True)

    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-0.1, 2)
    ax2.hist(noises[:, 1], bins=n_bins, density=True)

    ax3.set_ylim(-2, 2)
    ax3.set_xlim(-2, 2)
    for i, sample in enumerate(paths):
        ax3.plot(sample[:, 0], sample[:, 2], "-b")
        ax3.plot(paths[best_idx, :, 0], paths[best_idx, :, 2], "-r")
    gx = goal_seq[0, 0]
    gy = goal_seq[2, 0]
    ax3.scatter(gx, gy, c="k")

    ax4.set_xlim(-1, 60)
    ax4.set_ylim(-0.3, 0.3)
    ax4.plot(action_seq[:, 0])

    ax5.set_xlim(-1, 60)
    ax5.set_ylim(-0.3, 0.3)
    ax5.plot(action_seq[:, 1])

    #ax6.set_xlim(-0.1, 1.1)
    ax6.plot(weights.numpy().reshape(-1))

    plt.savefig('/tmp/mppi_{}.png'.format(j))
    plt.close("all")


def gif_path(len, gif):
    if gif is None:
        return
    with imageio.get_writer(gif, mode='I') as writer:
        files = ["/tmp/mppi_{}.png".format(i) for i in range(len)]
        for filename in files:
            image = imageio.imread(filename)
            writer.append_data(image)
            os.remove(filename)
    return
