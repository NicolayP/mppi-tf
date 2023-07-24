import tensorflow as tf
#gpu = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(device=gpu[0], enable=True)

import numpy as np
from .model_base import ModelBase
from ..misc.utile import assert_shape, dtype, npdtype


def skew_op(vec):
    S = np.zeros(shape=(3, 3))

    S[0, 1] = -vec[2]
    S[0, 2] = vec[1]

    S[1, 0] = vec[2]
    S[1, 2] = -vec[0]

    S[2, 0] = -vec[1]
    S[2, 1] = vec[0]
    return tf.constant(S, dtype=dtype)


def tf_skew_op(scope, vec):
    with tf.name_scope(scope) as scope:
        vec = tf.expand_dims(vec, axis=-1)
        OPad = tf.zeros(shape=(1), dtype=dtype)
        r0 = tf.expand_dims(tf.concat([OPad, -vec[2], vec[1]],
                                      axis=-1),
                            axis=1)

        r1 = tf.expand_dims(tf.concat([vec[2], OPad, -vec[0]],
                                      axis=-1),
                            axis=1)

        r2 = tf.expand_dims(tf.concat([-vec[1], vec[0], OPad],
                                      axis=-1),
                            axis=1)

        S = tf.concat([r0, r1, r2], axis=1)
        return S


def skew_op_k(batch):
    k = batch.shape[0]
    S = np.zeros(shape=(k, 3, 3))

    S[:, 0, 1] = -batch[:, 2]
    S[:, 0, 2] = batch[:, 1]

    S[:, 1, 0] = batch[:, 2]
    S[:, 1, 2] = -batch[:, 0]

    S[:, 2, 0] = -batch[:, 1]
    S[:, 2, 1] = batch[:, 0]
    return S


def tf_skew_op_k(scope, batch):
    with tf.name_scope(scope) as scope:
        k = tf.shape(batch)[0]
        vec = tf.expand_dims(batch, axis=-1)

        OPad = tf.zeros(shape=(k, 1), dtype=dtype)
        r0 = tf.expand_dims(tf.concat([OPad, -vec[:, 2], vec[:, 1]],
                                      axis=-1),
                            axis=1)

        r1 = tf.expand_dims(tf.concat([vec[:, 2], OPad,  -vec[:, 0]],
                                      axis=-1),
                            axis=1)

        r2 = tf.expand_dims(tf.concat([-vec[:, 1], vec[:, 0], OPad],
                                      axis=-1),
                            axis=1)

        S = tf.concat([r0, r1, r2], axis=1)
        return S


'''
    AUV dynamical model based on the UUV_sim vehicle model
    and implemented in tensorflow to be used with the MPPI
    controller.
'''
class AUVModel(ModelBase):
    def __init__(self, modelDict,
                 inertialFrameId='world',
                 actionDim=6,
                 limMax=np.ones(shape=(6, 1), dtype=npdtype),
                 limMin=-np.ones(shape=(6, 1), dtype=npdtype),
                 name="AUV",
                 k=tf.Variable(1),
                 dt=0.1,
                 rk=2, # Deprecated
                 parameters=dict()):
        '''
            Class constructor
        '''
        self._rk = rk
        stateDim = 13

        ModelBase.__init__(self, modelDict,
                           stateDim=stateDim,
                           actionDim=actionDim,
                           limMax=tf.constant(limMax, dtype=dtype),
                           limMin=tf.constant(limMin, dtype=dtype),
                           name=name,
                           k=k,
                           dt=dt,
                           inertialFrameId=inertialFrameId)

        assert inertialFrameId in ['world', 'world_ned']
        if "rk" in parameters:
            self._rk = parameters["rk"]
        else:
            self._rk = 1

        # TODO: Not used.
        if inertialFrameId == 'world':
            self._bodyFrameId = 'base_link'
        else:
            self._bodyFrameId = 'base_link_ned'

        self._mass = 0
        if "mass" in parameters:
            self._mass = tf.Variable(parameters['mass'],
                                     trainable=True,
                                     dtype=dtype)
        assert (self._mass > 0), "Mass has to be positive."

        self._volume = 0
        if "volume" in parameters:
            self._volume = parameters["volume"]
        assert (self._volume > 0), "Volume has to be positive."

        self._density = 0
        if "density" in parameters:
            self._density = parameters["density"]
        assert (self._density > 0), "Liquid density has to be positive."

        # TODO: Not used.
        self._height = 0
        if "height" in parameters:
            self._height = parameters["height"]
        assert (self._height > 0), "Height has to be positive."

        # TODO: Not used.
        self._length = 0
        if "length" in parameters:
            self._length = parameters["length"]
        assert (self._length > 0), "Length has to be positive."

        # TODO: Not used.
        self._width = 0
        if "width" in parameters:
            self._width = parameters["width"]
        assert (self._width > 0), "Width has to be positive."

        if "cog" in parameters:
            self._cog = tf.constant(parameters["cog"], dtype=dtype)
            assert (len(self._cog) == 3), 'Invalid center of \
                                           gravity vector. Size != 3'
        else:
            raise AssertionError("need to define the center of \
                                 gravity in the body frame")

        if "cob" in parameters:
            self._cob = tf.constant(parameters["cob"], dtype=dtype)
            assert (len(self._cob) == 3), "Invalid center of buoyancy \
                                           vector. Size != 3"
        else:
            raise AssertionError("need to define the center of \
                                 buoyancy in the body frame")

        addedMass = np.zeros((6, 6))
        if "Ma" in parameters:
            addedMass = np.array(parameters["Ma"])
            assert (addedMass.shape == (6, 6)), "Invalid add mass matrix"
        self._addedMass = tf.constant(addedMass, dtype=dtype)

        damping = np.zeros(shape=(6, 6))
        if "linear_damping" in parameters:
            damping = np.array(parameters["linear_damping"])
            if damping.shape == (6,):
                damping = np.diag(damping)
            assert (damping.shape == (6, 6)), "Linear damping must be \
                                               given as a 6x6 matrix or \
                                               the diagonal coefficients"

        self._linearDamping = tf.constant(damping, dtype=dtype)

        qaudDamping = np.zeros(shape=(6,), dtype=npdtype)
        if "quad_damping" in parameters:
            qaudDamping = np.array(parameters["quad_damping"], dtype=npdtype)
            assert (qaudDamping.shape == (6,)), "Quadratic damping must \
                                                 be given defined with 6 \
                                                 coefficients"

        self._quadDamping = tf.linalg.diag(qaudDamping)

        dampingForward = np.zeros(shape=(6, 6), dtype=npdtype)
        if "linear_damping_forward_speed" in parameters:
            dampingForward = np.array(
                                parameters["linear_damping_forward_speed"],
                                dtype=npdtype)
            if dampingForward.shape == (6,):
                dampingForward = np.diag(dampingForward)
            assert (dampingForward.shape == (6, 6)), "Linear damping \
                                                      proportional to the \
                                                      forward speed must \
                                                      be given as a 6x6 \
                                                      matrix or the diagonal \
                                                      coefficients"

        self._linearDampingForwardSpeed = tf.expand_dims(
                                            tf.constant(dampingForward,
                                                        dtype=dtype),
                                            axis=0)

        inertial = dict(ixx=0, iyy=0, izz=0, ixy=0, ixz=0, iyz=0)
        if "inertial" in parameters:
            inertialArg = parameters["inertial"]
            for key in inertial:
                if key not in inertialArg:
                    raise AssertionError('Invalid moments of inertia')

        self._inertial = self.get_inertial(inertialArg)

        self._unitZ = tf.constant([[[0.], [0.], [1.]]], dtype=dtype)

        self._gravity = 9.81

        self._massEye = self._mass * tf.eye(3, dtype=dtype)
        self._massLower = self._mass * tf_skew_op("Skew_cog", self._cog)

        self._rbMass = self.rigid_body_mass()
        self._mTot = self.total_mass()
        self._invMTot = tf.linalg.inv(self._mTot)

        # TODO: Not used.
        self._invMRb = tf.linalg.inv(self._rbMass)

    def print_info(self):
        """Print the vehicle's parameters."""
        print("="*5, " Model Info ", "="*5)
        print('Body frame: {}'.format(self._bodyFrameId))
        print('Mass: {0:.3f} kg'.format(self._mass.numpy()))
        print('System inertia matrix:\n{}'.format(self._rbMass))
        print('Added-mass:\n{}'.format(self._addedMass))
        print('Inertial:\n{}'.format(self._inertial))
        print('Volume: {}'.format(self._volume))
        print('M:\n{}'.format(self._mTot))
        print('M_inv:\n{}'.format(self._invMTot))
        print('Linear damping:\n{}'.format(self._linearDamping))
        print('Quad. damping:\n{}'.format(self._quadDamping))
        print('Center of gravity:\n{}'.format(self._cog))
        print('Center of buoyancy:\n{}'.format(self._cob))

    def rigid_body_mass(self):
        upper = tf.concat([self._massEye, -self._massLower], axis=1)
        lower = tf.concat([self._massLower, self._inertial], axis=1)
        return tf.concat([upper, lower], axis=0)

    def total_mass(self):
        return tf.add(self._rbMass, self._addedMass)

    def get_inertial(self, dict):
        # buid the inertial matrix
        ixx = tf.Variable(dict['ixx'], trainable=True, dtype=dtype)
        ixy = tf.Variable(dict['ixy'], trainable=True, dtype=dtype)
        ixz = tf.Variable(dict['ixz'], trainable=True, dtype=dtype)
        iyy = tf.Variable(dict['iyy'], trainable=True, dtype=dtype)
        iyz = tf.Variable(dict['iyz'], trainable=True, dtype=dtype)
        izz = tf.Variable(dict['izz'], trainable=True, dtype=dtype)

        row0 = tf.expand_dims(tf.stack([ixx, ixy, ixz], axis=0), axis=0)
        row1 = tf.expand_dims(tf.stack([ixy, iyy, iyz], axis=0), axis=0)
        row2 = tf.expand_dims(tf.stack([ixz, iyz, izz], axis=0), axis=0)

        inertial = tf.concat([row0, row1, row2], axis=0)

        return inertial

    def build_step_graph(self, scope, state, action, dev=False):
        # Assume input is between -1; 1
        action = self.action_to_input(action)
        if len(tf.shape(state)) == 4:
            # From lagged controller. Remove history (should be equal to 1)
            state = tf.squeeze(state, axis=1)
        if len(tf.shape(action)) == 4:
            action = tf.squeeze(action, axis=1)
        if dev:
            step, Cv, Dv, g = self.step(scope, state, action, rk=self._rk, dev=dev)
            return step, Cv, Dv, g
        return self.step(scope, state, action, rk=self._rk)

    def step_gt_vel(self, scope, state, vel):
        '''
            Performs integration given the ground truth velocity.

            inputs:
            -------
                - scope: string, the tensorflow scope.
                - state: tensor. The state of the vehicle. (pose, velocity)
                    shape: [1, 13, 1]
                - velocity: the next velocity expressed in body frame.
                    shape: [1, 6, 1]
        '''
        with tf.name_scope(scope) as scope:
            pose, speed = self.prepare_data(state)
            rotBtoI, TBtoI = self.body2inertial_transform(pose)
            poseDot = tf.matmul(self.get_jacobian(rotBtoI, TBtoI), speed)

            next_pose = pose + poseDot*self._dt
            next_speed = vel
            next_state = tf.concat([next_pose, next_speed], axis=1)
            next_state = self.normalize_quat(next_state)
            return next_state

    def step(self, scope, state, action, rk=1, dev=False):
        # Forward and backward euler integration.
        if dev:
            rk =1
        with tf.name_scope(scope) as scope:
            if dev:
                k1, Cv, Dv, g = self.state_dot(state, action, dev)
            else:
                k1 = self.state_dot(state, action)
            if rk == 1:
                tmp = k1*self._dt

            elif rk == 2:
                k2 = self.state_dot(tf.add(state, self._dt*k1), action)
                tmp = self._dt/2. * tf.add(k1, k2)

            elif rk == 4:
                k2 = self.state_dot(tf.add(state, self._dt*k1/2.), action)
                k3 = self.state_dot(tf.add(state, self._dt*k2/2.), action)
                k4 = self.state_dot(tf.add(state, self._dt*k3), action)
                tmp = 1./6. * tf.add(tf.add(k1, 2.*k2),
                                     tf.add(2.*k3, k4*self._dt))*self._dt

            nextState = tf.add(state, tmp)
            nextState = self.normalize_quat(nextState)

            if dev:
                return nextState, Cv, Dv, g

            return nextState

    def state_dot(self, state, action, dev=False):
        '''
            Computes x_dot = f(x, u)

                    nextState = tf.add(state, tmp)
            - input:
            --------
                - state: The state of the system
                    tf.Tensor shape [k, 13, 1].
                - action: The action applied to the system.
                    tf.Tensor shape [k, 6, 1].
            
            - output:
            ---------
                - state_dot: The first order derivate of the state
                after applying action to state.
                    tf.Tensor shape [k, 13, 1].
        '''
        # mostily used for rk integration methods.
        pose, speed = self.prepare_data(state)
        rotBtoI, TBtoIquat = self.body2inertial_transform(pose)

        poseDot = tf.matmul(self.get_jacobian(rotBtoI, TBtoIquat), speed)
        with tf.name_scope("acceleration") as acc:
            if dev:
                speedDot, Cv, Dv, g = self.acc(acc, speed, action, rotBtoI, TBtoIquat, dev)
                return self.get_state_dot(poseDot, speedDot), Cv, Dv, g
            speedDot = self.acc(acc, speed, action, rotBtoI, TBtoIquat)
            return self.get_state_dot(poseDot, speedDot)

    def get_jacobian(self, rotBtoI, tBtoIquat):
        '''
        Returns J(nu) in $mathbb{R}^{7 cross 7}$
                     ---------------------------------------
            J(nu) = | q_{n}^{b}(Theta) 0^{3 cross 3}    |
                     | 0^{3 cross 3} T_{theta}(theta)   |
                     ---------------------------------------
        '''
        OPad3x3 = tf.zeros(shape=(self._k, 3, 3), dtype=dtype)
        OPad4x3 = tf.zeros(shape=(self._k, 4, 3), dtype=dtype)

        jacR1 = tf.concat([rotBtoI, OPad3x3], axis=-1)

        jacR2 = tf.concat([OPad4x3, tBtoIquat], axis=-1)
        jac = tf.concat([jacR1, jacR2], axis=1)

        return jac

    def body2inertial_transform(self, pose):
        '''
            Computes the rotational transform from
            body to inertial Rot_{n}^{b}(q)
            and the attitude transformation T_{q}(q).

            input:
            ------
                - pose the robot pose expressed in inertial frame.
                    Shape [k, 7, 1]

        '''
        quat = pose[:, 3:7, :]
        w = quat[:, 3]
        x = quat[:, 0]
        y = quat[:, 1]
        z = quat[:, 2]

        r1 = tf.expand_dims(tf.concat([1 - 2 * (tf.pow(y, 2) + tf.pow(z, 2)),
                                       2 * (x * y - z * w),
                                       2 * (x * z + y * w)], axis=-1),
                            axis=1)

        r2 = tf.expand_dims(tf.concat([2 * (x * y + z * w),
                                       1 - 2 * (tf.pow(x, 2) + tf.pow(z, 2)),
                                       2 * (y * z - x * w)], axis=-1),
                            axis=1)

        r3 = tf.expand_dims(tf.concat([2 * (x * z - y * w),
                                       2 * (y * z + x * w),
                                       1 - 2 * (tf.pow(x, 2) + tf.pow(y, 2))],
                                      axis=-1),
                            axis=1)

        rotBtoI = tf.concat([r1, r2, r3], axis=1)

        rwt = tf.expand_dims(tf.concat([-x, -y, -z], axis=-1), axis=1)
        rxt = tf.expand_dims(tf.concat([w, -z, y], axis=-1), axis=1)
        ryt = tf.expand_dims(tf.concat([z, w, -x], axis=-1), axis=1)
        rzt = tf.expand_dims(tf.concat([-y, x, w], axis=-1), axis=1)
        TBtoIquat = 0.5 * tf.concat([rxt, ryt, rzt, rwt], axis=1)

        return rotBtoI, TBtoIquat

    def prepare_data(self, state):
        pose = state[:, 0:7]
        speed = state[:, 7:13]
        return pose, speed

    def get_state_dot(self, poseDot, speedDot):
        '''
            Return the state of the system after
            propagating it for one timestep.

            input:
            ------
                - poseDot: float64 tensor.
                    Shape [k/1, 7, 1].
                - speedDot: float64 tensor. Shape [k, 6, 1]

            output:
            -------
                - nextState: float64 tensor. Shape [k, 13, 1]
        '''
        kS = tf.shape(speedDot)[0]
        # On the first step, the pose is only of size k=1.
        poseDot = tf.broadcast_to(poseDot, [kS, 7, 1])
        stateDot = tf.concat([poseDot, speedDot], axis=1)
        return stateDot

    def normalize_quat(self, pose):
        '''
            Normalizes the quaternions.

            input:
            ------
                - pose. Float64 Tensor. Shape [k, 13, 1]

            ouput:
            ------
                - the pose with normalized quaternion. Float64 Tensor.
                    Shape [k, 13, 1]
        '''

        pos = pose[:, 0:3]
        quat = tf.squeeze(pose[:, 3:7], axis=-1)
        vel = pose[:, 7:13]

        quat = tf.math.l2_normalize(quat, axis=-1)
        quat = tf.expand_dims(quat, axis=-1)
        #quat = tf.divide(quat, tf.linalg.norm(quat, axis=1, keepdims=True))
        pose = tf.concat([pos, quat, vel], axis=1)
        return pose

    def restoring_forces(self, scope, rotBtoI):
        with tf.name_scope(scope) as scope:
            cog = tf.expand_dims(self._cog, axis=0)
            cob = tf.expand_dims(self._cob, axis=0)

            fng = - self._mass*self._gravity*self._unitZ
            fnb = self._volume*self._density*self._gravity*self._unitZ

            rotItoB = tf.transpose(rotBtoI, perm=[0, 2, 1])

            fbg = tf.matmul(rotItoB, fng)
            fbb = tf.matmul(rotItoB, fnb)

            cog = tf.broadcast_to(cog, [self._k, 3])
            cob = tf.broadcast_to(cob, [self._k, 3])

            mbg = tf.expand_dims(tf.linalg.cross(cog,
                                                 tf.squeeze(fbg, axis=-1)),
                                 axis=-1)
            mbb = tf.expand_dims(tf.linalg.cross(cob,
                                                 tf.squeeze(fbb, axis=-1)),
                                 axis=-1)

            restoring = -tf.concat([fbg+fbb,
                                    mbg+mbb], axis=1)

            return restoring

    def damping_matrix(self, scope, vel=None):
        '''
            Computes the damping matrix.

            input:
            ------
                - speed, the speed tensor shape [k, 6, 1]

            output:
            -------
                - $D(\nu)$ the damping matrix. Shape [k, 6, 6]
        '''
        with tf.name_scope(scope) as scope:
            D = - self._linearDamping - tf.multiply(
                                           tf.expand_dims(vel[:, 0],
                                                          axis=-1),
                                           self._linearDampingForwardSpeed)
            a = tf.expand_dims(self._quadDamping, axis=0)
            b = tf.abs(tf.linalg.diag(tf.squeeze(vel, axis=-1)))
            tmp = - tf.linalg.matmul(a, b)
            D = tf.add(D, tmp)

            return D

    def coriolis_matrix(self, scope, vel=None):
        '''
            Computes the coriolis matrix

            input:
            ------
                - speed, the speed tensor. Shape [k, 6]

            ouput:
            ------
                - $ C(\nu)\nu $ the coriolis matrix. Shape [k, 6, 6]
        '''

        with tf.name_scope(scope) as scope:

            OPad = tf.zeros(shape=(self._k, 3, 3), dtype=dtype)

            skewCori = tf.squeeze(tf.matmul(self._mTot[0:3, 0:3],
                                            vel[:, 0:3]) +
                                  tf.matmul(self._mTot[0:3, 3:6],
                                            vel[:, 3:6]),
                                  axis=-1)
            S12 = -tf_skew_op_k("Skew_coriolis", skewCori)

            skewCoriDiag = tf.squeeze(tf.matmul(self._mTot[3:6, 0:3],
                                                vel[:, 0:3]) +
                                      tf.matmul(self._mTot[3:6, 3:6],
                                                vel[:, 3:6]),
                                      axis=-1)
            S22 = -tf_skew_op_k("Skew_coriolis_diag", skewCoriDiag)

            r1 = tf.concat([OPad, S12], axis=-1)
            r2 = tf.concat([S12, S22], axis=-1)
            C = tf.concat([r1, r2], axis=1)

            return C

    def acc(self, scope, vel, genForce=None, rotBtoI=None, tBtoIqat=None, dev=False):
        with tf.name_scope(scope) as scope:
            tensGenForce = np.zeros(shape=(6, 1))
            if genForce is not None:
                tensGenForce = genForce

            D = self.damping_matrix("Damping", vel)
            Dv = tf.matmul(D, vel)
            C = self.coriolis_matrix("Coriolis", vel)
            Cv = tf.matmul(C, vel)
            g = self.restoring_forces("Restoring", rotBtoI)
            rhs = tensGenForce - Cv - Dv - g
            lhs = tf.broadcast_to(self._invMTot, [self._k, 6, 6])
            acc = tf.matmul(lhs, rhs)

            #print("*"*5, " C ", "*"*5)
            #print(C)
            #print("*"*5, " g ", "*"*5)
            #print(g)
            if dev:
                return acc, Cv, Dv, g
            return acc

    def get_forces(self, pose, vel, acc):
        rotBtoI, tBtoIquat = self.body2inertial_transform(pose)
        C = self.coriolis_matrix("Coriolis", vel)
        Cv = tf.matmul(C, vel)

        D = self.damping_matrix("Damping", vel)
        Dv = tf.matmul(D, vel)

        g = self.restoring_forces("Restoring", rotBtoI)
        Ma = tf.matmul(self._mTot, acc)
        force = Ma + Cv + Dv + g

        return C, Cv, D, Dv, g, force

    def save_params(self, path, step):
        pass

    def load_params(self, path):
        pass

    def fake_input(self):
        dummy_state = np.array([
            [0.], [0.], [0.],
            [0.], [0.], [0.], [1.],
            [0.], [0.], [0.],
            [0.], [0.], [0.]
        ])

    def compare_gt_pred(self, state, gt_state, action=None, gt_acc=None):
        pose_hat, vel_hat = self.prepare_data(state)
        rotBtoI_hat, tBtoIquat_hat = self.body2inertial_transform(pose_hat)
        C_hat = self.coriolis_matrix("Coriolis", vel_hat)
        D_hat = self.damping_matrix("Damping", vel_hat)
        g_hat = self.restoring_forces("Restoring", rotBtoI_hat)
        Cv_hat = tf.matmul(C_hat, vel_hat)
        Dv_hat = tf.matmul(D_hat, vel_hat)

        pose, vel = self.prepare_data(gt_state)
        rotBtoI, tBtoIquat = self.body2inertial_transform(pose)
        C = self.coriolis_matrix("Coriolis", vel)
        D = self.damping_matrix("Damping", vel)
        g = self.restoring_forces("Restoring", rotBtoI)
        Cv = tf.matmul(C, vel)
        Dv = tf.matmul(D, vel)
        
        print("C hat:        ", C_hat)
        print("C:            ", C)
        print("C diff:       ", C_hat - C)

        print("Cv hat:       ", Cv_hat)
        print("Cv:           ", Cv)
        print("Cv diff:      ", Cv_hat - Cv)

        print("D hat:        ", D_hat)
        print("D:            ", D)
        print("D diff:       ", D_hat - D)

        print("Dv hat:       ", Dv_hat)
        print("Dv:           ", Dv)
        print("Dv diff:      ", Dv_hat - Dv)

        print("g hat:        ", g_hat)
        print("g:            ", g)
        print("g diff:       ", g_hat - g)
        print("Sum hat:      ", Cv_hat + Dv_hat + g_hat)
        print("Sum:          ", Cv + Dv + g)
        if action is not None:
            acc_hat = tf.matmul(self._invMTot, action - Cv_hat - Dv_hat - g_hat)
            acc = tf.matmul(self._invMTot, action - Cv - Dv - g)
            print("Ma_hat:       ", action - (Cv_hat + Dv_hat + g_hat))
            print("Ma:           ", action - (Cv + Dv + g))
            print("Acc_hat:      ", acc_hat)
            print("Acc:          ", acc)

            if gt_acc is not None:
                print("Diff Acc_hat: ", gt_acc - acc_hat)
                print("Diff Acc:     ", gt_acc - acc)

    def action_to_input(self, action):
        # assume input between [-1; 1]
        action = tf.clip_by_value(action, -1, 1)
        return self._actMax * action

class AUVModelDebug(AUVModel):
    def __init__(self, modelDict,
                 inertialFrameId='world',
                 actionDim=6,
                 limMax=tf.ones(shape=(1,), dtype=tf.float64),
                 limMin=-tf.ones(shape=(1,), dtype=tf.float64),
                 name="AUV",
                 k=tf.Variable(1),
                 dt=0.1,
                 rk=2, # Deprecated
                 parameters=dict()):

        AUVModel.__init__(self, modelDict,
                        actionDim=actionDim,
                        limMax=limMax,
                        limMin=limMin,
                        name=name,
                        k=k,
                        dt=dt,
                        inertialFrameId=inertialFrameId,
                        parameters=parameters)
        
    def coriolis_matrix(self, scope, vel=None, dev=False):
        with tf.name_scope(scope) as scope:
            OPad = tf.zeros(shape=(self._k, 3, 3), dtype=dtype)

            skewCoriAdded = tf.squeeze(tf.matmul(self._addedMass[0:3, 0:3],
                                            vel[:, 0:3]) +
                                tf.matmul(self._addedMass[0:3, 3:6],
                                            vel[:, 3:6]),
                                axis=-1)
            S12_added = -tf_skew_op_k("Skew_coriolis_added", skewCoriAdded)

            skewCoriRigid = tf.squeeze(tf.matmul(self._rbMass[0:3, 0:3],
                                            vel[:, 0:3]) +
                                tf.matmul(self._rbMass[0:3, 3:6],
                                            vel[:, 3:6]),
                                axis=-1)
            S12_rigid = -tf_skew_op_k("Skew_coriolis_added", skewCoriRigid)

            skewCoriDiagAdded = tf.squeeze(tf.matmul(self._addedMass[3:6, 0:3],
                                                vel[:, 0:3]) +
                                    tf.matmul(self._addedMass[3:6, 3:6],
                                                vel[:, 3:6]),
                                    axis=-1)
            S22_added = -tf_skew_op_k("Skew_coriolis_diag", skewCoriDiagAdded)

            skewCoriDiagRigid = tf.squeeze(tf.matmul(self._rbMass[3:6, 0:3],
                                                vel[:, 0:3]) +
                                    tf.matmul(self._rbMass[3:6, 3:6],
                                                vel[:, 3:6]),
                                    axis=-1)
            S22_rigid = -tf_skew_op_k("Skew_coriolis_diag", skewCoriDiagRigid)

            r1_added = tf.concat([OPad, S12_added], axis=-1)
            r2_added = tf.concat([S12_added, S22_added], axis=-1)
            C_added = tf.concat([r1_added, r2_added], axis=1)

            r1_rigid = tf.concat([OPad, S12_rigid], axis=-1)
            r2_rigid = tf.concat([S12_rigid, S22_rigid], axis=-1)
            C_rigid = tf.concat([r1_rigid, r2_rigid], axis=1)

            C = C_added + C_rigid

            if dev:
                return C, C_added, C_rigid

            return C

    def acc(self, scope, vel, genForce=None, rotBtoI=None, tBtoIqat=None, dev=False, debug=False):
        with tf.name_scope(scope) as scope:
            tensGenForce = np.zeros(shape=(6, 1))
            if genForce is not None:
                tensGenForce = genForce

            D = self.damping_matrix("Damping", vel)
            Dv = tf.matmul(D, vel)
            if dev:
                C, C_added, C_rigid = self.coriolis_matrix("Coriolis", vel, dev=dev)
                Cv_added = tf.matmul(C_added, vel)
                Cv_rigid = tf.matmul(C_rigid, vel)
            else:
                C = self.coriolis_matrix("Coriolis", vel, dev=dev)
            Cv = tf.matmul(C, vel)
            g = self.restoring_forces("Restoring", rotBtoI)
            rhs = tensGenForce - Cv - Dv - g
            lhs = tf.broadcast_to(self._invMTot, [self._k, 6, 6])
            acc = tf.matmul(lhs, rhs)

            added = tf.matmul(self._addedMass, acc)
            if debug:
                print("-"*8, "Debug model step", "-"*8)
                print("| ", "vel:       ", vel)
                print("| ", "acc:       ", acc)
                print("| ", "added:     ", added)
                print("| ", "Cv:        ", Cv)
                print("| ", "Cv_added:  ", Cv_added)
                print("| ", "Cv_rigid:  ", Cv_rigid)
                print("| ", "Dv:        ", Dv)
                print("| ", "restoring: ", g)
                print("-"*32)
                input()

            if dev:
                return acc, added, Cv, Cv_added, Cv_rigid, Dv, g
            return acc

    def step(self, scope, state, action, rk=1, dev=False):
        # Forward and backward euler integration.
        if dev:
            rk =1
        with tf.name_scope(scope) as scope:
            if dev:
                k1, Cv, Dv, g = self.state_dot(state, action, dev)
            else:
                k1 = self.state_dot(state, action)
            if rk == 1:
                tmp = k1*self._dt

            elif rk == 2:
                k2 = self.state_dot(tf.add(state, self._dt*k1), action)
                tmp = self._dt/2. * tf.add(k1, k2)

            elif rk == 4:
                k2 = self.state_dot(tf.add(state, self._dt*k1/2.), action)
                k3 = self.state_dot(tf.add(state, self._dt*k2/2.), action)
                k4 = self.state_dot(tf.add(state, self._dt*k3), action)
                tmp = 1./6. * tf.add(tf.add(k1, 2.*k2),
                                     tf.add(2.*k3, k4*self._dt))*self._dt

            nextState = tf.add(state, tmp)
            nextState = self.normalize_quat(nextState)

            if dev:
                return nextState, Cv, Dv, g

            return nextState

    def state_dot(self, state, action, dev=False):
        '''
            Computes x_dot = f(x, u)

                    nextState = tf.add(state, tmp)
            - input:
            --------
                - state: The state of the system
                    tf.Tensor shape [k, 13, 1].
                - action: The action applied to the system.
                    tf.Tensor shape [k, 6, 1].
            
            - output:
            ---------
                - state_dot: The first order derivate of the state
                after applying action to state.
                    tf.Tensor shape [k, 13, 1].
        '''
        # mostily used for rk integration methods.
        pose, speed = self.prepare_data(state)
        rotBtoI, TBtoIquat = self.body2inertial_transform(pose)

        poseDot = tf.matmul(self.get_jacobian(rotBtoI, TBtoIquat), speed)
        with tf.name_scope("acceleration") as acc:
            if dev:
                speedDot, added, Cv, Cv_added, Cv_rigid, Dv, g = self.acc(acc, speed, action, rotBtoI, TBtoIquat, dev)
                return self.get_state_dot(poseDot, speedDot), Cv, Dv, g
            speedDot = self.acc(acc, speed, action, rotBtoI, TBtoIquat)
            return self.get_state_dot(poseDot, speedDot)

    def compare_gt_pred(self, state, gt_state, action=None, gt_acc=None):
        pose_hat, vel_hat = self.prepare_data(state)
        rotBtoI_hat, tBtoIquat_hat = self.body2inertial_transform(pose_hat)
        C_hat, C_hat_added, C_hat_rigid = self.coriolis_matrix("Coriolis", vel_hat, dev=True)
        D_hat = self.damping_matrix("Damping", vel_hat)
        g_hat = self.restoring_forces("Restoring", rotBtoI_hat)
        Cv_hat = tf.matmul(C_hat, vel_hat)
        Dv_hat = tf.matmul(D_hat, vel_hat)

        pose, vel = self.prepare_data(gt_state)
        rotBtoI, tBtoIquat = self.body2inertial_transform(pose)
        C, C_added, C_rb = self.coriolis_matrix("Coriolis", vel, dev=True)
        D = self.damping_matrix("Damping", vel)
        g = self.restoring_forces("Restoring", rotBtoI)
        Cv = tf.matmul(C, vel)
        Dv = tf.matmul(D, vel)
        
        print("C hat:        ", C_hat)
        print("C:            ", C)
        print("C diff:       ", C_hat - C)

        print("Cv hat:       ", Cv_hat)
        print("Cv:           ", Cv)
        print("Cv diff:      ", Cv_hat - Cv)

        print("D hat:        ", D_hat)
        print("D:            ", D)
        print("D diff:       ", D_hat - D)

        print("Dv hat:       ", Dv_hat)
        print("Dv:           ", Dv)
        print("Dv diff:      ", Dv_hat - Dv)

        print("g hat:        ", g_hat)
        print("g:            ", g)
        print("g diff:       ", g_hat - g)
        print("Sum hat:      ", Cv_hat + Dv_hat + g_hat)
        print("Sum:          ", Cv + Dv + g)
        if action is not None:
            acc_hat = tf.matmul(self._invMTot, action - Cv_hat - Dv_hat - g_hat)
            acc = tf.matmul(self._invMTot, action - Cv - Dv - g)
            print("Ma_hat:       ", action - (Cv_hat + Dv_hat + g_hat))
            print("Ma:           ", action - (Cv + Dv + g))
            print("Acc_hat:      ", acc_hat)
            print("Acc:          ", acc)

            if gt_acc is not None:
                print("Diff Acc_hat: ", gt_acc - acc_hat)
                print("Diff Acc:     ", gt_acc - acc)