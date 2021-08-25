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


def log_control(writer, control_items, pose_id, speed_id):
    # TODO: Log position error, goal distance, predicted cost.
    # TODO: generate gifs if asked.
    global control_step
    cost = control_items["cost"]
    k = cost.shape[0]
    avg_cost = np.mean(cost)
    best_id = np.argmin(cost)
    best_cost = cost[best_id, 0, 0]

    #print(control_items.keys())

    with writer.as_default():
        for key in control_items:
            if key == "cost":
                tf.summary.scalar("cost/all/best_all_cost", best_cost, step=control_step)
                tf.summary.scalar("cost/all/average_all_cost", avg_cost, step=control_step)
                tf.summary.histogram("cost/all/all_cost", cost, step=control_step)

            elif key == "state_cost":
                state_cost = control_items[key]
                avg_state = np.mean(state_cost)
                best_state = state_cost[best_id, 0, 0]
                tf.summary.scalar("cost/state/best_state_cost", best_state, step=control_step)
                tf.summary.scalar("cost/state/avg_state_cost", avg_state, step=control_step)
                tf.summary.histogram("cost/state/state_cost", state_cost, step=control_step)

            elif key == "action_cost":
                action_cost = control_items[key]
                avg_act = np.mean(action_cost)
                best_act = action_cost[best_id, 0, 0]
                tf.summary.scalar("cost/action/avg_action_cost", avg_act, step=control_step)
                tf.summary.scalar("cost/action/best_acttion_cost", best_act, step=control_step)
                tf.summary.histogram("cost/action/action_cost", action_cost, step=control_step)

            elif key == "position_cost":
                position_cost = control_items[key]
                avg_pos = np.mean(position_cost)
                best_pos = position_cost[best_id, 0, 0]
                tf.summary.scalar("cost/elipse/avg_position_cost", avg_pos, step=control_step)
                tf.summary.scalar("cost/elipse/best_position_cost", best_pos, step=control_step)
                tf.summary.histogram("cost/elipse/position_cost", position_cost, step=control_step)

            elif key == "speed_cost":
                speed_cost = control_items[key]
                avg_speed = np.mean(speed_cost)
                best_speed = speed_cost[best_id, 0, 0]
                tf.summary.scalar("cost/elipse/avg_speed_cost", avg_speed, step=control_step)
                tf.summary.scalar("cost/elipse/best_speed_cost", best_speed, step=control_step)
                tf.summary.histogram("cost/elipse/speed_cost", speed_cost, step=control_step)

            elif key == "a_cost":
                a_cost = control_items[key]
                tf.summary.scalar("cost/action/clean_input_cost", a_cost[0, 0], step=control_step)

            elif key == "n_cost":
                n_cost = control_items[key]
                avg_n = np.mean(n_cost)
                best_n = n_cost[best_id, 0, 0]
                tf.summary.scalar("cost/action/avg_noisy_input_cost", avg_n, step=control_step)
                tf.summary.scalar("cost/action/best_noisy_input_cost", best_n, step=control_step)
                tf.summary.histogram("cost/action/noise_input_cost", n_cost, step=control_step)

            elif key == "mix_cost":
                mix_cost = control_items[key]
                avg_mix = np.mean(mix_cost)
                best_mix = mix_cost[best_id, 0, 0]
                tf.summary.scalar("cost/action/avg_mix_cost", avg_mix, step=control_step)
                tf.summary.scalar("cost/action/best_mix_cost", best_mix, step=control_step)
                tf.summary.histogram("cost/action/mix_cost", mix_cost, step=control_step)

            elif key == "control_cost":
                control_cost = control_items[key]
                avg_control = np.mean(control_cost)
                best_control = control_cost[best_id, 0, 0]
                tf.summary.scalar("cost/action/avg_control_cost", avg_control, step=control_step)
                tf.summary.scalar("cost/action/best_control_cost", best_control, step=control_step)
                tf.summary.histogram("cost/action/control_cost", control_cost, step=control_step)

            elif key == "nabla":
                nabla = control_items[key]
                tf.summary.scalar("controller/nabla_percent", nabla[0, 0]/k,
                                  step=control_step)

            elif key == "norm_cost":
                norm_cost = control_items[key]
                tf.summary.histogram("norm_cost", norm_cost, step=control_step)

            elif key == "weighted_noises":
                weighted_noises = control_items[key]
                tf.summary.histogram("controller/Weighted_noises", weighted_noises,
                                     step=control_step)

            elif key == "weights":
                weights = control_items[key]
                tf.summary.histogram("controller/Controller_weights", weights,
                                     step=control_step)

            elif key == "error_pos":
                error_pos = control_items[key]
                tf.summary.scalar("model_error/position_prediciton_error", error_pos, step=control_step)
            
            elif key == "error_vel":
                error_vel = control_items[key]
                tf.summary.scalar("model_error/speed_prediction_error", error_vel, step=control_step)

            elif key == "predicted_speed_cost":
                ps_cost = control_items[key]
                tf.summary.scalar("predicted/speed_cost", tf.squeeze(ps_cost), step=control_step)
            
            elif key == "predicted_state_cost":
                pstate_cost = control_items[key]
                tf.summary.scalar("predicted/state_cost", tf.squeeze(pstate_cost), step=control_step)
            
            elif key == "predicted_position_cost":
                pp_cost = control_items[key]
                tf.summary.scalar("predicted/position_cost", tf.squeeze(pp_cost), step=control_step)

            elif key == "next":
                action = control_items[key][0]
                for i in range(action.shape[0]):
                    tf.summary.scalar("input_{}".format(i), action[i, 0], step=control_step)
            
            elif key == "state":
                state = control_items[key]
                for i in pose_id:
                    tf.summary.scalar("state/position_{}".format(i), state[i, 0], step=control_step)
                for i in speed_id:
                    tf.summary.scalar("state/speed_{}".format(i), state[i, 0], step=control_step)
                
            elif key == "x_dist":
                x_dist = control_items[key]
                tf.summary.scalar("goal/position_to_goal_distance", x_dist, step=control_step)

            elif key == "v_dist":
                v_dist = control_items[key]
                tf.summary.scalar("goal/speed_to_goal_distance", v_dist, step=control_step)

            # The following remaining entries are used foor the gif generation
            # just put them here for completness.
            elif key == "noises":
                noises = control_items[key]
                tf.summary.histogram("noise", noises, step=control_step)
                pass

            elif key == "dist":
                dist = control_items[key]
                for i, el in enumerate(dist.numpy()):
                    tf.summary.scalar("goal/dist_{}".format(i), el[0], step=control_step)

            elif key == "action_seq":
                pass

            elif key == "paths":
                pass
            
            elif key == "update":
                pass

            elif key == "next_state":
                pass

            elif key == "arg":
                pass
    # reset for next summary
    control_items = {}
    control_step += 1
