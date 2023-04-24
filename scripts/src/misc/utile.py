import tensorflow as tf
import numpy as np
from numpy import *
import yaml
import os

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})

from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

import scipy.signal
from scipy.spatial.transform import Rotation as R

dtype = tf.float32
#dtype = tf.float64
npdtype = np.float32
#npdtype = np.float64

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


def plot_traj(trajs, seq=None, plotStateCols=None, plotActionCols=None, title="Traj", dir=".", filename=None):
    '''
        Plot trajectories and action sequence.
        inputs:
        -------
            - trajs: dict with model name as key and trajectories entry. If key is "gt" then it is assumed to be
                the ground truth trajectory.
            - seq: Action Sequence associated to the generated trajectoires. If not None, plots the 
                action seqence.
            - histories: list of history used for the different models, ignored when model entry is "gt".
            - frequencies: list of history used for the different models, ignored when model entry is "gt".
            - plotStateCols: Dict containing the state axis name as key and index as entry
            - plotAcitonCols: Dict containing the action axis name as key and index as entry.
            - title: String the name of fthe figure.
            - horizon: The horizon of the trajectory to plot.
            - dir: The saving directory for the generated images.
    '''
    maxS = len(plotStateCols)
    maxA = len(plotActionCols)
    # fig_state = plt.figure(figsize=(50, 50))
    fig, axes = plt.subplots(6, 2, figsize=(50, 50))
    fig.suptitle(title)
    for k in trajs:
        t, h, freq, tau = trajs[k]
        for i, name in enumerate(plotStateCols):
            m, n = np.unravel_index(i, (2, 6))
            #idx = 1*m + 2*n + 1
            axes[n, m].set_ylabel(f'{name}')
            if k == "gt":
                time_steps = np.linspace(0., freq*tau, tau)
                axes[n, m].plot(time_steps, t[:tau, plotStateCols[name]],
                    marker='.', zorder=-10, label=k)
            else:
                time_steps = np.linspace(0, freq*(tau+h), (tau+h))
                axes[n, m].plot(time_steps, t[:, plotStateCols[name]],
                    marker='X', label=k
                )
    plt.legend()
    #plt.tight_layout()
    if dir is not None:
        name = os.path.join(dir, f"{filename}.png")
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
            name = os.path.join(dir, f"{filename}-actions.png")
            plt.savefig(name)
            plt.close()
        
    plt.show()


def plot_6d(trajs, ColNames=None, title="Foo", dir=".", filename=None):
    '''
        Plot trajectories and action sequence.
        inputs:
        -------
            - trajs: dict with model name as key and entries being [traj, delta t, steps]. If key is "gt" then it is assumed to be
                the ground truth trajectory.
            - seq: Action Sequence associated to the generated trajectoires. If not None, plots the 
                action seqence.
            - plotStateCols: Dict containing the state axis name as key and index as entry
            - plotAcitonCols: Dict containing the action axis name as key and index as entry.
            - dir: The saving directory for the generated images.
    '''
    maxS = len(ColNames)
    #fig_state = plt.figure(figsize=(50, 50))
    fig, axes = plt.subplots(3, 2, figsize=(50, 50))
    fig.suptitle(title)
    for k in trajs:
        t, freq, tau = trajs[k]
        for i, name in enumerate(ColNames):
            m, n = np.unravel_index(i, (2, 3))
            axes[n, m].set_ylabel(f'{name}')
            x = np.linspace(0, freq*tau, tau)
            axes[n, m].plot(
                x, t[:, ColNames[name]],
                marker='X', label=k
            )
    plt.legend()
    #plt.tight_layout()
    if dir is not None:
        name = os.path.join(dir, f"{filename}.png")
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


def traj_to_forces(model, traj, rep="rot", dt=0.1):
    '''
        Takes as input a trajectory composed of
        pose and velocity. Using a AUV model, it
        computes the different forces acting on the
        vehicle and returns them

        inputs:
        -------
            traj: trajectory compose of [pose, vel], shape [tau, sDim, 1]
            rep: the representation used for the rotation
        
        outputs:
        --------
            - Cv: the coriolis component. Shape [tau, 6]
            - Dv: the damping component. Shape [tau, 6]
            - g: the restoring forces. Shape [tau, 6]
            - tau: the control input. Shape [tau, 6]
    '''

    # First step: we need to compute the acceleration of the
    # auv a each steps.
    if rep == "euler":
        angle_len = 3
    elif rep == "quat":
        angle_len = 4
    elif rep == "rot":
        angle_len = 9

    traj = traj[..., None]

    pose = traj[:, :3 + angle_len]
    vel = traj[:, 3+angle_len:]

    acc = (vel[2: ] - vel[:-2])/dt
    pose = pose[1:-1]
    vel = vel[1:-1]

    cvs = []
    dvs = []
    gs = []
    fs = []
    # Use the acceleration together with the state to
    # compute the different values.

    for p, v, a in zip(pose, vel, acc):
        c, cv, d, dv, g, f = model.get_forces(p[None], v[None], a[None])
        cvs.append(cv)
        dvs.append(dv)
        gs.append(g)
        fs.append(f)
    
    cvs = np.concatenate(cvs, axis=0)
    dvs = np.concatenate(dvs, axis=0)
    gs = np.concatenate(gs, axis=0)
    fs = np.concatenate(fs, axis=0)

    return cvs, dvs, gs, fs