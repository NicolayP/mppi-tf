import matplotlib.pyplot as plt

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


    fig3, axs3 = plt.subplots(3, 2)
    fig3.suptitle("Acceleration")
    fig3.tight_layout()
    lines3 = []
    if accs is not None:

        for acc in accs:
            accx = acc[:, :, 0]
            accy = acc[:, :, 1]
            accz = acc[:, :, 2]
            accp = acc[:, :, 3]
            accq = acc[:, :, 4]
            accr = acc[:, :, 5]

            steps = accx.shape[1]
            absc = np.linspace(0, time, steps)

            lines3.append(axs3[0, 0].plot(absc, accx[0, :])[0])

            #axs1[0, 0].set_ylim(-5, 5)
            axs3[0, 0].title.set_text(' Acc_x (m/s^2)')
            axs3[0, 0].set_xlabel(' Steps (0.1 sec)')

            axs3[1, 0].plot(absc, accy[0, :])
            #axs1[1, 0].set_ylim(-5, 5)
            axs3[1, 0].title.set_text(' Acc_y (m/s^2)')
            axs3[1, 0].set_xlabel(' Steps (0.1 sec)')

            axs3[2, 0].plot(absc, accz[0, :])
            #axs1[2, 0].set_ylim(-5, 5)
            axs3[2, 0].title.set_text(' Acc_z (m/s^2)')
            axs3[2, 0].set_xlabel(' Steps (0.1 sec)')

            axs3[0, 1].plot(absc, accp[0, :])
            #axs1[0, 1].set_ylim(-5, 5)
            axs3[0, 1].title.set_text(' Ang acc_p (deg/s^2)')
            axs3[0, 1].set_xlabel(' Steps (0.1 sec)')

            axs3[1, 1].plot(absc, accq[0, :])
            #axs1[1, 1].set_ylim(-5, 5)
            axs3[1, 1].title.set_text(' Ang acc_q (deg/s^2)')
            axs3[1, 1].set_xlabel(' Steps (0.1 sec)')
            axs3[2, 1].plot(absc, accr[0, :])
            #axs1[2, 1].set_ylim(-5, 5)
            axs3[2, 1].title.set_text(' Ang acc_r (deg/s^2)')
            axs3[2, 1].set_xlabel(' Steps (0.1 sec)')

    fig3.legend(lines, labels=labels, loc="center right", title="Legend Title", borderaxespad=0.1)
    return

def to_quat(state_euler):
    k = state_euler.shape[0]
    pos = state_euler[:, 0:3]
    euler = np.squeeze(state_euler[:, 3:6])
    vel = state_euler[:, 6:12]
    
    quat = from_euler_angles(euler)
    quats = np.zeros(shape=(k, 4, 1))
    for i, q in enumerate(quat):
        quats[i, 0, 0] = q.w
        quats[i, 1, 0] = q.x
        quats[i, 2, 0] = q.y
        quats[i, 3, 0] = q.z

    return np.concatenate([pos, quats, vel], axis=1)

def euler_rot(state, rotBtoI):
    """`list`: Orientation in Euler angles in radians 
    as described in Fossen, 2011.
    """
    # Rotation matrix from BODY to INERTIAL
    rot = rotBtoI
    # Roll
    roll = np.expand_dims(np.arctan2(rot[:, 2, 1], rot[:, 2, 2]), axis=-1)
    # Pitch, treating singularity cases
    den = np.sqrt(1 - rot[:, 2, 0]**2)
    pitch = np.expand_dims(-np.arctan(rot[:, 2, 0] / den), axis=-1)
    # Yaw
    yaw = np.expand_dims(np.arctan2(rot[:, 1, 0], rot[:, 0, 0]), axis=-1)
    pos = state[:, :, 0:3, :]
    vel = state[:, :, 7:13, :]
    euler = np.expand_dims(np.expand_dims(np.concatenate([roll, pitch, yaw], axis=-1), axis=-1), axis=1)

    state_euler =  np.concatenate([pos, euler, vel], axis=2)

    return state_euler

def to_euler(state_quat):
    k = state_quat.shape[0]
    steps = state_quat.shape[1]
    #print(state_quat.shape)
    pos = state_quat[:, :, 0:3, :]
    quats_samp = np.squeeze(state_quat[:, :, 3:7, :], axis=-1)
    euler = np.zeros(shape=(k, steps, 3, 1))
    for j, quats in enumerate(quats_samp):
        for i, q in enumerate(quats):
            quat = np.quaternion(q[0], q[1], q[2], q[3])
            euler[j, i, :, :] = np.expand_dims(as_euler_angles(quat), axis=-1)

    vel = state_quat[:, :, 7:13, :]
    #print(pos.shape)
    #print(euler.shape)
    #print(vel.shape)
    state_euler =  np.concatenate([pos, euler, vel], axis=2)
    return state_euler

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

def main():
    k = 1
    tau = 1000
    params = dict()
    params["mass"] = 1862.87
    params["volume"] = 1.83826
    params["volume"] = 1.8121303501945525 # makes the vehicle neutraly buoyant.
    params["density"] = 1028.0
    params["height"] = 1.6
    params["length"] = 2.6
    params["width"] = 1.5
    params["cog"] = [0.0, 0.0, 0.0]
    params["cob"] = [0.0, 0.0, 0.3]
    #params["cob"] = [0.0, 0.0, 0.0]
    params["Ma"] = [[779.79, -6.8773, -103.32,  8.5426, -165.54, -7.8033],
                    [-6.8773, 1222, 51.29, 409.44, -5.8488, 62.726],
                    [-103.32, 51.29, 3659.9, 6.1112, -386.42, 10.774],
                    [8.5426, 409.44, 6.1112, 534.9, -10.027, 21.019],
                    [-165.54, -5.8488, -386.42, -10.027,  842.69, -1.1162],
                    [-7.8033, 62.726, 10.775, 21.019, -1.1162, 224.32]]
    params["linear_damping"] = [-74.82, -69.48, -728.4, -268.8, -309.77, -105]
    params["quad_damping"] = [-748.22, -992.53, -1821.01, -672, -774.44, -523.27]
    params["linear_damping_forward_speed"] = [0., 0., 0., 0., 0., 0.]
    inertial = dict()
    inertial["ixx"] = 525.39
    inertial["iyy"] = 794.2
    inertial["izz"] = 691.23
    inertial["ixy"] = 1.44
    inertial["ixz"] = 33.41
    inertial["iyz"] = 2.6
    params["inertial"] = inertial
    dt = 0.1
    dt_gz = 0.1
    auv_quat = AUVModel(quat=True, action_dim=6, dt=dt, k=k, parameters=params)
    auv_quat_rk2 = AUVModel(quat=True, action_dim=6, dt=dt, k=k, rk=2, parameters=params)
    auv_quat_rk2_gz = AUVModel(quat=True, action_dim=6, dt=dt_gz, k=k, rk=2, parameters=params)
    auv_quat_rk4 = AUVModel(quat=True, action_dim=6, dt=dt, k=k, rk=4, parameters=params)
    auv_quat_proxy = AUVModel(quat=True, action_dim=6, dt=dt, k=k, rk=4, parameters=params)

    test_data(auv_quat, auv_quat_rk2, auv_quat_rk2_gz, auv_quat_rk4, auv_quat_proxy)
    exit()

if __name__ == "__main__":
    main()


def plotFrame(t, x, y, z):

    plt.figure()
    ax = plt.axes(projection="3d")
    x_ar = Arrow3D([0, 1],
                   [0, 0],
                   [0, 0],
                   mutation_scale=20,
                   lw=1, arrowstyle="->", color="r")

    y_ar = Arrow3D([0, 0],
                   [0, 1],
                   [0, 0],
                   mutation_scale=20,
                   lw=1, arrowstyle="->", color="g")

    z_ar = Arrow3D([0, 0],
                   [0, 0],
                   [0, 1],
                   mutation_scale=20,
                   lw=1, arrowstyle="->", color="b")

    ax.add_artist(x_ar)
    ax.add_artist(y_ar)
    ax.add_artist(z_ar)


    for i in range(t.shape[0]):
        x_ar = Arrow3D([t[i, 0, 0], x[i, 0, 0]],
                       [t[i, 1, 0], x[i, 1, 0]],
                       [t[i, 2, 0], x[i, 2, 0]],
                       mutation_scale=20,
                       lw=1, arrowstyle="->", color="r")
        y_ar = Arrow3D([t[i, 0, 0], y[i, 0, 0]],
                       [t[i, 1, 0], y[i, 1, 0]],
                       [t[i, 2, 0], y[i, 2, 0]],
                       mutation_scale=20,
                       lw=1, arrowstyle="->", color="g")
        z_ar = Arrow3D([t[i, 0, 0], z[i, 0, 0]],
                       [t[i, 1, 0], z[i, 1, 0]],
                       [t[i, 2, 0], z[i, 2, 0]],
                       mutation_scale=20,
                       lw=1, arrowstyle="->", color="b")
        ax.add_artist(x_ar)
        ax.add_artist(y_ar)
        ax.add_artist(z_ar)

def visuTrainData(nn_auv):
    pose_t0_Bt0 = np.array([
                            [
                             [1.0], [1.0], [0.5], #Position
                             [1.0], [0.0], [0.0], [0.0], #Quaterinon
                            ],
                            [
                             [1.0], [-1.75],[0.5],
                             [0.6532815], [0.6532815], [0.2705981], [0.2705981],
                            ]
                           ])

    pose_t1_Bt1 = np.array([
                            [
                             [1.5], [2.3], [0.9], #Position
                             [-0.1913417], [0.8001031], [0.4619398], [0.3314136] #Quaterinon
                            ],
                            [
                             [0.0], [-1.],[2.5],
                             [0.4615897], [0.8446119], [0.0560099], [0.2653839]
                            ]
                           ])

    inv = nn_auv.invTransform(pose_t0_Bt0)
    pose_t1_Bt0 = nn_auv.transform(inv, pose_t1_Bt1)

    t_t0_Bt0 = pose_t0_Bt0[:, 0:3]
    q_t0_Bt0 = pose_t0_Bt0[:, 3:7]

    t_t1_Bt1 = pose_t1_Bt1[:, 0:3]
    q_t1_Bt1 = pose_t1_Bt1[:, 3:7]

    t_t1_Bt0 = pose_t1_Bt0[:, 0:3]
    q_t1_Bt0 = pose_t1_Bt0[:, 3:7]

    x = np.array([[[1.0], [0.0], [0.0]]])
    y = np.array([[[0.0], [1.0], [0.0]]])
    z = np.array([[[0.0], [0.0], [1.0]]])

    x_t1_Bt0 = (t_t1_Bt0 + nn_auv.qMul(nn_auv.qMul(q_t1_Bt0, nn_auv.pureQ(x)), nn_auv.invQ(q_t1_Bt0))[:, 1:4]).numpy()
    y_t1_Bt0 = (t_t1_Bt0 + nn_auv.qMul(nn_auv.qMul(q_t1_Bt0, nn_auv.pureQ(y)), nn_auv.invQ(q_t1_Bt0))[:, 1:4]).numpy()
    z_t1_Bt0 = (t_t1_Bt0 + nn_auv.qMul(nn_auv.qMul(q_t1_Bt0, nn_auv.pureQ(z)), nn_auv.invQ(q_t1_Bt0))[:, 1:4]).numpy()

    x_t1_Bt1 = (t_t1_Bt1 + nn_auv.qMul(nn_auv.qMul(q_t1_Bt1, nn_auv.pureQ(x)), nn_auv.invQ(q_t1_Bt1))[:, 1:4]).numpy()
    y_t1_Bt1 = (t_t1_Bt1 + nn_auv.qMul(nn_auv.qMul(q_t1_Bt1, nn_auv.pureQ(y)), nn_auv.invQ(q_t1_Bt1))[:, 1:4]).numpy()
    z_t1_Bt1 = (t_t1_Bt1 + nn_auv.qMul(nn_auv.qMul(q_t1_Bt1, nn_auv.pureQ(z)), nn_auv.invQ(q_t1_Bt1))[:, 1:4]).numpy()

    x_t0_Bt0 = (t_t0_Bt0 + nn_auv.qMul(nn_auv.qMul(q_t0_Bt0, nn_auv.pureQ(x)), nn_auv.invQ(q_t0_Bt0))[:, 1:4]).numpy()
    y_t0_Bt0 = (t_t0_Bt0 + nn_auv.qMul(nn_auv.qMul(q_t0_Bt0, nn_auv.pureQ(y)), nn_auv.invQ(q_t0_Bt0))[:, 1:4]).numpy()
    z_t0_Bt0 = (t_t0_Bt0 + nn_auv.qMul(nn_auv.qMul(q_t0_Bt0, nn_auv.pureQ(z)), nn_auv.invQ(q_t0_Bt0))[:, 1:4]).numpy()

    plotFrame(np.concatenate([t_t0_Bt0, t_t1_Bt1], axis=0),
              np.concatenate([x_t0_Bt0, x_t1_Bt1], axis=0),
              np.concatenate([y_t0_Bt0, y_t1_Bt1], axis=0),
              np.concatenate([z_t0_Bt0, z_t1_Bt1], axis=0))

    plotFrame(t_t1_Bt0,
              x_t1_Bt0,
              y_t1_Bt0,
              z_t1_Bt0)

    #plt.show()

def visuPred2Inertial(nn_auv):
    pose_t0_Bt0 = np.array([
                        [
                            [1.0], [1.0], [0.5], #Position
                            [1.0], [0.0], [0.0], [0.0], #Quaterinon
                        ],
                        [
                            [1.0], [-1.75],[0.5],
                            [0.6532815], [0.6532815], [0.2705981], [0.2705981],
                        ]
                        ])

    pose_t1_Bt0 = np.array([
                            [
                             [1.5], [2.3], [0.9], #Position
                             [-0.1913417], [0.8001031], [0.4619398], [0.3314136] #Quaterinon
                            ],
                            [
                             [0.0], [-1.],[2.5],
                             [0.4615897], [0.8446119], [0.0560099], [0.2653839]
                            ]
                           ])

    pose_t1_Bt1 = nn_auv.transform(pose_t0_Bt0, pose_t1_Bt0)

    t_t0_Bt0 = pose_t0_Bt0[:, 0:3]
    q_t0_Bt0 = pose_t0_Bt0[:, 3:7]

    t_t1_Bt1 = pose_t1_Bt1[:, 0:3]
    q_t1_Bt1 = pose_t1_Bt1[:, 3:7]

    t_t1_Bt0 = pose_t1_Bt0[:, 0:3]
    q_t1_Bt0 = pose_t1_Bt0[:, 3:7]

    x = np.array([[[1.0], [0.0], [0.0]]])
    y = np.array([[[0.0], [1.0], [0.0]]])
    z = np.array([[[0.0], [0.0], [1.0]]])

    x_t1_Bt0 = (t_t1_Bt0 + nn_auv.qMul(nn_auv.qMul(q_t1_Bt0, nn_auv.pureQ(x)), nn_auv.invQ(q_t1_Bt0))[:, 1:4]).numpy()
    y_t1_Bt0 = (t_t1_Bt0 + nn_auv.qMul(nn_auv.qMul(q_t1_Bt0, nn_auv.pureQ(y)), nn_auv.invQ(q_t1_Bt0))[:, 1:4]).numpy()
    z_t1_Bt0 = (t_t1_Bt0 + nn_auv.qMul(nn_auv.qMul(q_t1_Bt0, nn_auv.pureQ(z)), nn_auv.invQ(q_t1_Bt0))[:, 1:4]).numpy()

    x_t1_Bt1 = (t_t1_Bt1 + nn_auv.qMul(nn_auv.qMul(q_t1_Bt1, nn_auv.pureQ(x)), nn_auv.invQ(q_t1_Bt1))[:, 1:4]).numpy()
    y_t1_Bt1 = (t_t1_Bt1 + nn_auv.qMul(nn_auv.qMul(q_t1_Bt1, nn_auv.pureQ(y)), nn_auv.invQ(q_t1_Bt1))[:, 1:4]).numpy()
    z_t1_Bt1 = (t_t1_Bt1 + nn_auv.qMul(nn_auv.qMul(q_t1_Bt1, nn_auv.pureQ(z)), nn_auv.invQ(q_t1_Bt1))[:, 1:4]).numpy()

    x_t0_Bt0 = (t_t0_Bt0 + nn_auv.qMul(nn_auv.qMul(q_t0_Bt0, nn_auv.pureQ(x)), nn_auv.invQ(q_t0_Bt0))[:, 1:4]).numpy()
    y_t0_Bt0 = (t_t0_Bt0 + nn_auv.qMul(nn_auv.qMul(q_t0_Bt0, nn_auv.pureQ(y)), nn_auv.invQ(q_t0_Bt0))[:, 1:4]).numpy()
    z_t0_Bt0 = (t_t0_Bt0 + nn_auv.qMul(nn_auv.qMul(q_t0_Bt0, nn_auv.pureQ(z)), nn_auv.invQ(q_t0_Bt0))[:, 1:4]).numpy()

    plotFrame(np.concatenate([t_t0_Bt0, t_t1_Bt1], axis=0),
              np.concatenate([x_t0_Bt0, x_t1_Bt1], axis=0),
              np.concatenate([y_t0_Bt0, y_t1_Bt1], axis=0),
              np.concatenate([z_t0_Bt0, z_t1_Bt1], axis=0))

    plotFrame(t_t1_Bt0,
              x_t1_Bt0,
              y_t1_Bt0,
              z_t1_Bt0)
    plt.show()

def main():
    '''
        Debugging main function, not the main use of this program:

        1. Test prepareTrainingData for: 1 element, 2 elements and N batchesize.
        2. Test prepareData for: 1 element, 2 elements and N batchesize.
        3. Test pred2Inertial for: 1 element, 2 elements and N batchsize.

    '''
    # Test 1: Visualize the training data in the world frame as given by gazebo/robot.
    # And in the body_t frame. Mostly visual verification. After that still need to verify the
    # Unit tests.
    nn_auv = NNAUVModel()
    #visuTrainData(nn_auv)
    visuPred2Inertial(nn_auv)



    pass

if __name__ == "__main__":
    main()