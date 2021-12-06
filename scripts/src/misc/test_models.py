import matplotlib.pyplot as plt
import numpy as np
from quaternion import as_euler_angles
import os
'''

def dummpy_plot(traj=None, applied=None, accs=None, labels=[""], time=0):
    print("\n\n" + "*"*5 + " Dummy plot " + "*"*5)
    fig, axs = plt.subplots(3, 2)
    fig.suptitle("Pose")
    fig.tight_layout()
    lines = []
    for states in traj:
        print(states.shape)
        x = states[:, :, 0]
        y = states[:, :, 1]
        z = states[:, :, 2]

        r = states[:, :, 3]*180/np.pi
        p = states[:, :, 4]*180/np.pi
        ya = states[:, :, 5]*180/np.pi

        steps = states.shape[1]
        absc = np.linspace(0, time, steps)
        lines.append(axs[0, 0].plot(absc, x[0, :])[0])
        #axs[0, 0].set_ylim(-5, 5)
        axs[0, 0].title.set_text(' X (m)')
        axs[0, 0].set_xlabel(' Steps (0.1 sec)')

        axs[1, 0].plot(absc, y[0, :])
        #axs[1, 0].set_ylim(-5, 5)
        axs[1, 0].title.set_text(' Y (m)')
        axs[1, 0].set_xlabel(' Steps (0.1 sec)')

        axs[2, 0].plot(absc, z[0, :])
        #axs[2, 0].set_ylim(-20, 5)
        axs[2, 0].title.set_text(' Z (m)')
        axs[2, 0].set_xlabel(' Steps (0.1 sec)')

        axs[0, 1].plot(absc, r[0, :])
        axs[0, 1].title.set_text(' Roll (Degrees)')
        axs[0, 1].set_xlabel(' Steps (0.1 sec)')

        axs[1, 1].plot(absc, p[0, :])
        axs[1, 1].title.set_text(' Pitch (Degrees)')
        axs[1, 1].set_xlabel(' Steps (0.1 sec)')

        axs[2, 1].plot(absc, ya[0, :])
        axs[2, 1].title.set_text(' Yaw (Degrees)')
        axs[2, 1].set_xlabel(' Steps (0.1 sec)')

    fig.legend(lines, labels=labels, loc="center right", title="Legend Title", borderaxespad=0.1)


    fig1, axs1 = plt.subplots(3, 2)
    fig1.suptitle("Velocity")
    fig1.tight_layout()
    lines1 = []
    for states in traj:
        vx = states[:, :, 6]
        vy = states[:, :, 7]
        vz = states[:, :, 8]

        vr = states[:, :, 9]*180/np.pi
        vp = states[:, :, 10]*180/np.pi
        vya = states[:, :, 11]*180/np.pi

        steps = states.shape[1]
        absc = np.linspace(0, time, steps)

        lines1.append(axs1[0, 0].plot(absc, vx[0, :])[0])
        #axs1[0, 0].set_ylim(-5, 5)
        axs1[0, 0].title.set_text(' Vel_x (m/s)')
        axs1[0, 0].set_xlabel(' Steps (0.1 sec)')

        axs1[1, 0].plot(absc, vy[0, :])
        #axs1[1, 0].set_ylim(-5, 5)
        axs1[1, 0].title.set_text(' Vel_y (m/s)')
        axs1[1, 0].set_xlabel(' Steps (0.1 sec)')

        axs1[2, 0].plot(absc, vz[0, :])
        #axs1[2, 0].set_ylim(-5, 5)
        axs1[2, 0].title.set_text(' Vel_z (m/s)')
        axs1[2, 0].set_xlabel(' Steps (0.1 sec)')

        axs1[0, 1].plot(absc, vr[0, :])
        #axs1[0, 1].set_ylim(-5, 5)
        axs1[0, 1].title.set_text(' Ang vel_p (deg/s)')
        axs1[0, 1].set_xlabel(' Steps (0.1 sec)')

        axs1[1, 1].plot(absc, vp[0, :])
        #axs1[1, 1].set_ylim(-5, 5)
        axs1[1, 1].title.set_text(' Ang_vel_q (deg/s)')
        axs1[1, 1].set_xlabel(' Steps (0.1 sec)')

        axs1[2, 1].plot(absc, vya[0, :])
        #axs1[2, 1].set_ylim(-5, 5)
        axs1[2, 1].title.set_text(' Ang_vel_r (deg/s)')
        axs1[2, 1].set_xlabel(' Steps (0.1 sec)')

    fig1.legend(lines, labels=labels, loc="center right", title="Legend Title", borderaxespad=0.1)


def test_data(auv_model_rk1, auv_model_rk2, auv_model_rk2_gz, auv_model_rk4, auv_model_proxy):
    with open("/home/pierre/workspace/uuv_ws/src/mppi-ros/log/init_state.npy", "rb") as f:
        inital_state = np.expand_dims(np.load(f), axis=0)
    with open("/home/pierre/workspace/uuv_ws/src/mppi-ros/log/applied.npy", "rb") as f:
        applied = np.load(f)
    with open("/home/pierre/workspace/uuv_ws/src/mppi-ros/log/states.npy", "rb") as f:
        states_ = np.load(f)

    tau = applied.shape[0]

    horizon = 20

    applied = np.expand_dims(applied, axis=(-1, 0))
    state_rk1 = inital_state
    state_rk2_gz = inital_state
    state_rk2 = inital_state
    state_rk4 = inital_state
    state_proxy = inital_state

    auv_model_rk1.set_prev_vel(inital_state[:, 7:13])
    rot_rk1 = auv_model_rk1.rotBtoI_np(np.squeeze(state_rk1[0, 3:7]))
    rot_rk1 = np.expand_dims(rot_rk1, axis=0)

    auv_model_rk2.set_prev_vel(inital_state[:, 7:13])
    rot_rk2 = auv_model_rk2.rotBtoI_np(np.squeeze(state_rk2[0, 3:7]))
    rot_rk2 = np.expand_dims(rot_rk2, axis=0)

    auv_model_rk2_gz.set_prev_vel(inital_state[:, 7:13])
    rot_rk2_gz = auv_model_rk2_gz.rotBtoI_np(np.squeeze(state_rk2_gz[0, 3:7]))
    rot_rk2_gz = np.expand_dims(rot_rk2_gz, axis=0)

    auv_model_rk4.set_prev_vel(inital_state[:, 7:13])
    rot_rk4 = auv_model_rk4.rotBtoI_np(np.squeeze(state_rk4[0, 3:7]))
    rot_rk4 = np.expand_dims(rot_rk4, axis=0)

    auv_model_proxy.set_prev_vel(inital_state[:, 7:13])
    rot_proxy = auv_model_proxy.rotBtoI_np(np.squeeze(state_rk4[0, 3:7]))
    rot_proxy = np.expand_dims(rot_proxy, axis=0)

    states_rk1=[]
    accs_rk1=[]

    states_rk2_gz=[]
    accs_rk2_gz=[]

    states_rk2=[]
    accs_rk2=[]

    states_rk4=[]
    accs_rk4=[]

    accs_est=[]
    states_uuv=[]

    states_proxy=[]

    # get Gazebo-uuv_sim data
    for t in range(tau-1):
        rot1 = auv_model_rk1.rotBtoI_np(np.squeeze(states_[t, 3:7]))
        lin_vel = states_[t, 7:10]

        s_uuv = np.expand_dims(np.concatenate([states_[t, 0:7], lin_vel, states_[t, 10:13]]), axis=0)

        acc_est = np.expand_dims((states_[t+1, 7:13] - states_[t, 7:13])/0.1, axis=0)

        rot1 = np.expand_dims(rot1, axis=0)

        states_uuv.append(euler_rot(np.expand_dims(s_uuv, axis=0), rot1))
        accs_est.append(np.expand_dims(acc_est, axis=0))

    # get different dt plots.
    for t in range(tau):
        next_state_rk2_gz, acc_rk2_gz = auv_model_rk2_gz.build_step_graph("foo", state_rk2_gz, applied[:, t, :, :], ret_acc=True)
        states_rk2_gz.append(euler_rot(np.expand_dims(next_state_rk2_gz, axis=1), auv_model_rk2_gz._rotBtoI))
        accs_rk2_gz.append(np.expand_dims(acc_rk2_gz, axis=0))
        state_rk2_gz = next_state_rk2_gz.numpy().copy()

    tau = int(horizon/auv_model_rk1.dt)

    x = np.linspace(0, horizon, applied.shape[1])
    y = applied
    applied_interp = np.zeros(shape=(y.shape[0], tau, y.shape[2], y.shape[3]))
    xvals = np.linspace(0, horizon, tau)
    for i in range(y.shape[2]):
        applied_interp[:, :, i, :] = np.expand_dims(np.interp(xvals, x, y[0, :, i, 0]), axis=-1)


    for t in range(tau):
        next_state_rk1, acc_rk1 = auv_model_rk1.build_step_graph("foo", state_rk1, applied_interp[:, t, :, :], ret_acc=True)
        states_rk1.append(euler_rot(np.expand_dims(next_state_rk1, axis=1), auv_model_rk1._rotBtoI))
        accs_rk1.append(np.expand_dims(acc_rk1, axis=0))
        state_rk1 = next_state_rk1.numpy().copy()

        next_state_rk2, acc_rk2 = auv_model_rk2.build_step_graph("foo", state_rk2, applied_interp[:, t, :, :], ret_acc=True)
        states_rk2.append(euler_rot(np.expand_dims(next_state_rk2, axis=1), auv_model_rk2._rotBtoI))
        accs_rk2.append(np.expand_dims(acc_rk2, axis=0))
        state_rk2 = next_state_rk2.numpy().copy()

        #next_state_rk4, acc_rk4 = auv_model_rk4.build_step_graph("foo", state_rk4, applied[:, 0, :, :], acc=True)
        #states_rk4.append(euler_rot(np.expand_dims(next_state_rk4, axis=1), auv_model_rk4._rotBtoI))
        #accs_rk4.append(np.expand_dims(acc_rk4, axis=0))
        #state_rk4 = next_state_rk4.numpy().copy()

    states_rk1 = np.concatenate(states_rk1, axis=1)
    states_rk2 = np.concatenate(states_rk2, axis=1)
    states_rk2_gz = np.concatenate(states_rk2_gz, axis=1)
    #states_rk4 = np.concatenate(states_rk4, axis=1)

    accs_rk1 = np.concatenate(accs_rk1, axis=1)
    accs_rk2 = np.concatenate(accs_rk2, axis=1)
    accs_rk2_gz = np.concatenate(accs_rk2_gz, axis=1)
    #accs_rk4 = np.concatenate(accs_rk4, axis=1)

    states_uuv = np.concatenate(states_uuv, axis=1)
    accs_est = np.concatenate(accs_est, axis=1)

    print("*"*5 + " States " + "*"*5)
    print(states_rk1.shape)
    print(states_rk2.shape)
    print(states_rk2_gz.shape)
    #print(states_rk4.shape)
    print(states_uuv.shape)

    print("*"*5 + " Acc " + "*"*5)
    print(accs_rk1.shape)
    print(accs_rk2.shape)
    print(accs_rk2_gz.shape)
    #print(accs_rk4.shape)
    print(accs_est.shape)

    dummpy_plot([states_rk2, states_rk2_gz, states_uuv], applied,
                [accs_rk2, accs_rk2_gz, accs_est],
                labels=["rk2", "state_rk2_gz", "uuv_sim"],
                time=horizon)

    plt.show()

'''

def to_euler(state_quat):
    steps = state_quat.shape[0]
    pos = state_quat[:, 0:3, :]
    quats = np.squeeze(state_quat[:, 3:7, :], axis=-1)
    euler = np.zeros(shape=(steps, 3, 1))
    for i, q in enumerate(quats):
        quat = np.quaternion(q[0], q[1], q[2], q[3])
        euler[i, :, :] = np.expand_dims(as_euler_angles(quat), axis=-1)*180/np.pi

    vel = state_quat[:, 7:13, :]
    state_euler =  np.concatenate([pos, euler, vel], axis=1)
    return state_euler


def run_model(model, initalState, sequence):
    traj = [initalState]
    state = np.expand_dims(initalState, axis=0)
    steps = sequence.shape[1]
    for i in range(steps):
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


def get_ax():
    fig, axs = plt.subplots(3, 2)
    fig.suptitle("Pose")
    fig.tight_layout()
    return fig, axs


def test_models(sequenceFile, trajFile, models, labels, time):
    sequence = get_sequence(sequenceFile)
    gtTraj = get_traj(trajFile)
    figPose, axPose = get_ax()
    figVel, axVel = get_ax()
    axPose = plot_traj(figPose, axPose, to_euler(gtTraj), "gt", time)
    axVel = plot_traj(figVel, axVel, gtTraj[:, 7:13], "gt", time, vel=True)
    for model, label in zip(models, labels):
        traj = run_model(model, gtTraj[0], sequence)
        axPose = plot_traj(figPose, axPose, to_euler(traj), label, time)
        axVel = plot_traj(figVel, axVel, traj[:, 7:13], label, time, vel=True)

    figPose.legend(loc="center right", title="Legend Title", borderaxespad=0.1)
    figVel.legend(loc="center right", title="Legend Title", borderaxespad=0.1)
    plt.show()
