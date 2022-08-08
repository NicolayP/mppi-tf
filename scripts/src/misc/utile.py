import tensorflow as tf
import numpy as np
from numpy import *
import yaml
import os

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

import imageio
import scipy.signal

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


def push_to_tensor(tensor, element):
    tmp = tf.expand_dims(element, axis=1) # shape [k, 1, dim, 1]
    return tf.concat([tensor[:, 1:], tmp], axis=1)