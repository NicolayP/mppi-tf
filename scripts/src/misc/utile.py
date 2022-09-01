import tensorflow as tf
import numpy as np
from numpy import *
import yaml
import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

import scipy.signal
from scipy.spatial.transform import Rotation as R

dtype = tf.float32
npdtype = np.float32

control_items: dict = {}
control_step = 0


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)


def assert_shape(array, shape):
    ashape = array.shape
    if len(ashape) != len(shape):
        return False
    for i, j in zip(ashape, shape):
        if j != -1 and i != j:
            return False
    return True


def parse_config(file):
    with open(file) as file:
        conf = yaml.load(file, Loader=yaml.FullLoader)
    return conf


def list_files(log_dir):
    for file in os.listdir(log_dir):
        if os.path.isfile(os.path.join(log_dir, file)):
            yield file


def parse_dir(log_dir):
    for file in list_files(log_dir):
        if file == "config.yaml":
            config_file = os.path.join(log_dir, file)
        elif file == "task.yaml":
            task_file = os.path.join(log_dir, file)
    return parse_config(config_file), task_file


def plt_sgf(action_seq):
    print(action_seq.numpy()[:, :, 0].shape)
    _ = plt.figure()
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)

    for i in np.arange(9, 49, 4):
        y = scipy.signal.savgol_filter(action_seq.numpy()[:, :, 0], i, 7,
                                       deriv=0, delta=1.0, axis=0)
        ax1.plot(y[:, 0], label="{}".format(i))
        ax2.plot(y[:, 1], label="{}".format(i))
    plt.legend()
    plt.show()


def plt_paths(paths, weights, noises, action_seq, cost):
    global control_step
    n_bins = 100
    best_idx = np.argmax(weights)
    # todo extract a_sape from tensor
    noises = noises.numpy().reshape(-1, 2)

    _ = plt.figure()
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

    ax3.set_ylim(-5, 5)
    ax3.set_xlim(-5, 5)
    for _, sample in enumerate(paths):
        ax3.plot(sample[:, 0], sample[:, 2], "-b")
        ax3.plot(paths[best_idx, :, 0], paths[best_idx, :, 2], "-r")

    gx, gy = cost.draw_goal()
    ax3.scatter(gx, gy, c="k")

    ax4.set_xlim(-1, 60)
    ax4.set_ylim(-0.3, 0.3)
    ax4.plot(action_seq[:, 0])

    ax5.set_xlim(-1, 60)
    ax5.set_ylim(-0.3, 0.3)
    ax5.plot(action_seq[:, 1])

    # ax6.set_xlim(-0.1, 1.1)
    ax6.plot(weights.numpy().reshape(-1))

    plt.savefig('/tmp/mppi_{}.png'.format(control_step-1))
    plt.close("all")


def push_to_tensor(tensor, element):
    tmp = tf.expand_dims(element, axis=1) # shape [k, 1, dim, 1]
    return tf.concat([tensor[:, 1:], tmp], axis=1)


def plot_traj(trajs, seq=None, histories=None, plotStateCols=None, plotActionCols=None, horizon=50, dir="."):
    '''
        Plot trajectories and action sequence.
        inputs:
        -------
            - trajs: dict with model name as key and trajectories entry. If key is "gt" then it is assumed to be
                the ground truth trajectory.
            - seq: Action Sequence associated to the generated trajectoires. If not None, plots the 
                action seqence.
            - h: list of history used for the different models, ignored when model entry is "gt".
            - plotStateCols: Dict containing the state axis name as key and index as entry
            - plotAcitonCols: Dict containing the action axis name as key and index as entry.
            - horizon: The horizon of the trajectory to plot.
            - dir: The saving directory for the generated images.
    '''
    maxS = len(plotStateCols)
    maxA = len(plotActionCols)
    fig_state = plt.figure(figsize=(50, 50))
    for k, h in zip(trajs, histories):
        t = trajs[k]
        for i, name in enumerate(plotStateCols):
            m, n = np.unravel_index(i, (2, 6))
            idx = 1*m + 2*n + 1
            plt.subplot(6, 2, idx)
            plt.ylabel(f'{name}')
            if k == "gt":
                plt.plot(t[:horizon], marker='.', zorder=-10)
            else:
                plt.scatter(
                    np.arange(h, horizon+h), t[:, plotStateCols[name]],
                    marker='X', edgecolors='k', s=64, label=k
                )
    plt.legend()
    #plt.tight_layout()
    if dir is not None:
        name = os.path.join(dir, f"{k}.png")
        plt.savefig(name)
        plt.close()

    if seq is not None:
        fig_act = plt.figure(figsize=(30, 30))
        for i, name in enumerate(plotActionCols):
            plt.subplot(maxA, 1, i+1)
            plt.ylabel(f'{name}')
            plt.plot(seq[0, :horizon+h, plotActionCols[name]])

        #plt.tight_layout()
        if dir is not None:
            name = os.path.join(dir, f"{k}-actions.png")
            plt.savefig(name)
            plt.close()
        
        plt.show()


def traj_to_euler(traj, rep="rot"):
    if rep == "rot":
        rot = traj[:, 3:3+9].reshape((-1, 3, 3))
        r = R.from_matrix(rot)
    elif rep == "quat":
        quat = traj[:, 3:3+4]
        r = R.from_quat(quat)
    else:
        raise NotImplementedError
    pos = traj[:, :3]
    euler = r.as_euler('XYZ', degrees=True)
    vel = traj[:, -6:]

    traj = np.concatenate([pos, euler, vel], axis=-1)
    return traj
