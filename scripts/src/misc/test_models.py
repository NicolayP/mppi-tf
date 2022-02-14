import matplotlib.pyplot as plt
import numpy as np
from quaternion import as_euler_angles
import os

def to_euler(state):
    steps = state.shape[0]
    pos = state[:, 0:3, :]
    quats = np.squeeze(state[:, 3:7, :], axis=-1)
    euler = np.zeros(shape=(steps, 3, 1))
    for i, q in enumerate(quats):
        quat = np.quaternion(q[3], q[0], q[1], q[2])
        euler[i, :, :] = np.expand_dims(as_euler_angles(quat), axis=-1)*180/np.pi

    vel = state[:, 7:13, :]
    state_euler =  np.concatenate([pos, euler, vel], axis=1)
    return state_euler


def run_model(model, initalState, sequence):
    traj = [initalState]
    state = np.expand_dims(initalState, axis=0)
    steps = sequence.shape[1]
    for i in range(steps-1):
        toApply = sequence[:, i]
        nextState = model.predict(state, toApply)
        traj.append(np.squeeze(nextState, axis=0))
        state=nextState
    traj = np.array(traj)
    return traj


def plot_traj(fig, ax, traj, label, time, vel=False):
    x = traj[:, 0]
    y = traj[:, 1]
    z = traj[:, 2]

    roll = traj[:, 3]
    pitch = traj[:, 4]
    yaw = traj[:, 5]

    steps = traj.shape[0]
    absc = np.linspace(0, time, steps)

    if not vel:
        textLin = " {} (m) "
        textAng = " {} (deg) "
    else:
        textLin = " v{} (m/s) "
        textAng = " v{} (deg/s) "

    ax[0, 0].plot(absc, x, label=label)
    ax[0, 0].title.set_text(textLin.format("X"))
    ax[0, 0].set_xlabel(' Steps (0.1 sec)')

    ax[1, 0].plot(absc, y)
    ax[1, 0].title.set_text(textLin.format("Y"))
    ax[1, 0].set_xlabel(' Steps (0.1 sec)')

    ax[2, 0].plot(absc, z)
    ax[2, 0].title.set_text(textLin.format("Z"))
    ax[2, 0].set_xlabel(' Steps (0.1 sec)')

    ax[0, 1].plot(absc, roll)
    ax[0, 1].title.set_text(textAng.format("Roll"))
    ax[0, 1].set_xlabel(' Steps (0.1 sec)')

    ax[1, 1].plot(absc, pitch)
    ax[1, 1].title.set_text(textAng.format("Pitch"))
    ax[1, 1].set_xlabel(' Steps (0.1 sec)')

    ax[2, 1].plot(absc, yaw)
    ax[2, 1].title.set_text(textAng.format("Yaw"))
    ax[2, 1].set_xlabel(' Steps (0.1 sec)')

    return ax


def get_sequence(file):
    if not os.path.isfile(file):
        raise "Aciton sequence file doesn't exists"
    with open(file, "rb") as f:
        applied = np.expand_dims(np.expand_dims(np.load(f), axis=-1), axis=0)
    return applied


def get_init(file):
    if not os.path.isfile(file):
        raise "Inital state file doesn't exists"
    with open(file, "rb") as f:
        init = np.load(f)
    return init


def get_traj(file):
    if not os.path.isfile(file):
        raise "Trajectory file doesn't exists"
    with open(file, "rb") as f:
        traj = np.load(f)
    return traj


def get_ax(title):
    fig, axs = plt.subplots(3, 2)
    fig.suptitle(title)
    fig.tight_layout()
    return fig, axs


def test_models(sequenceFile, trajFile, models, labels, time):
    sequence = get_sequence(sequenceFile)
    gtTraj = get_traj(trajFile)
    figPose, axPose = get_ax("Pose")
    figVel, axVel = get_ax("Vel")
    axPose = plot_traj(figPose, axPose, to_euler(gtTraj[0:-2]), "gt", time)
    axVel = plot_traj(figVel, axVel, gtTraj[:, 7:13], "gt", time, vel=True)
    for model, label in zip(models, labels):
        traj = run_model(model, gtTraj[0], sequence)
        axPose = plot_traj(figPose, axPose, to_euler(traj), label, time)
        axVel = plot_traj(figVel, axVel, traj[:, 7:13], label, time, vel=True)

    figPose.suptitle("Position")
    figPose.legend(loc="center right", title="Legend", borderaxespad=0.1)

    figVel.suptitle("Velocities")
    figVel.legend(loc="center right", title="Legend", borderaxespad=0.1)
    plt.show()
