import tensorflow as tf
import numpy as np
from model_base import ModelBase

import time as t


def skew_op(vec):
    S = np.zeros(shape=(3, 3))

    S[0, 1] = -vec[2]
    S[0, 2] = vec[1]

    S[1, 0] = vec[2]
    S[1, 2] = -vec[0]

    S[2, 0] = -vec[1]
    S[2, 1] = vec[0]
    return tf.constant(S, dtype=tf.float64)


def tf_skew_op(scope, vec):
    with tf.name_scope(scope) as scope:
        vec = tf.expand_dims(vec, axis=-1)
        OPad = tf.zeros(shape=(1), dtype=tf.float64)
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
        k = batch.shape[0]
        vec = tf.expand_dims(batch, axis=-1)

        OPad = tf.zeros(shape=(k, 1), dtype=tf.float64)
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


class AUVModel(ModelBase):
    '''
        AUV dynamical model based on the UUV_sim vehicle model
        and implemented in tensorflow to be used with the MPPI
        controller.
    '''

    def __init__(self,
                 inertialFrameId='world',
                 actionDim=6,
                 name="AUV", k=1,
                 dt=0.1,
                 rk=1,
                 parameters=dict()):
        '''
            Class constructor
        '''
        self._rk = rk
        stateDim = 13

        ModelBase.__init__(self,
                           stateDim,
                           actionDim,
                           name,
                           k,
                           dt,
                           inertialFrameId)

        assert inertialFrameId in ['world', 'world_ned']

        # TODO: Not used.
        if inertialFrameId == 'world':
            self._bodyFrameId = 'base_link'
        else:
            self._bodyFrameId = 'base_link_ned'

        self._mass = 0
        if "mass" in parameters:
            self._mass = tf.Variable(parameters['mass'],
                                     trainable=True,
                                     dtype=tf.float64)
        assert (self._mass > 0), "Mass has to be positive."

        self._volume = 0
        if "volume" in parameters:
            self._volume = parameters["volume"]
        assert (self._volume > 0), "Volume has to be positive."

        self._densityu = 0
        if "density" in parameters:
            self._densityu = parameters["density"]
        assert (self._densityu > 0), "Liquid density has to be positive."

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
            self._cog = tf.constant(parameters["cog"], dtype=tf.float64)
            assert (len(self._cog) == 3), 'Invalid center of \
                                           gravity vector. Size != 3'
        else:
            raise AssertionError("need to define the center of \
                                 gravity in the body frame")

        if "cob" in parameters:
            self._cob = tf.constant(parameters["cob"], dtype=tf.float64)
            assert (len(self._cob) == 3), "Invalid center of buoyancy \
                                           vector. Size != 3"
        else:
            raise AssertionError("need to define the center of \
                                 buoyancy in the body frame")

        addedMass = np.zeros((6, 6))
        if "Ma" in parameters:
            addedMass = np.array(parameters["Ma"])
            assert (addedMass.shape == (6, 6)), "Invalid add mass matrix"
        self._addedMass = tf.constant(addedMass, dtype=tf.float64)

        damping = np.zeros(shape=(6, 6))
        if "linear_damping" in parameters:
            damping = np.array(parameters["linear_damping"])
            if damping.shape == (6,):
                damping = np.diag(damping)
            assert (damping.shape == (6, 6)), "Linear damping must be \
                                               given as a 6x6 matrix or \
                                               the diagonal coefficients"

        self._linearDamping = tf.constant(damping, dtype=tf.float64)

        qaudDamping = np.zeros(shape=(6,))
        if "quad_damping" in parameters:
            qaudDamping = np.array(parameters["quad_damping"])
            assert (qaudDamping.shape == (6,)), "Quadratic damping must \
                                                 be given defined with 6 \
                                                 coefficients"

        self._quadDamping = tf.linalg.diag(qaudDamping)

        dampingForward = np.zeros(shape=(6, 6))
        if "linear_damping_forward_speed" in parameters:
            dampingForward = np.array(
                                parameters["linear_damping_forward_speed"])
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
                                                        dtype=tf.float64),
                                            axis=0)

        inertial = dict(ixx=0, iyy=0, izz=0, ixy=0, ixz=0, iyz=0)
        if "inertial" in parameters:
            inertialArg = parameters["inertial"]
            for key in inertial:
                if key not in inertialArg:
                    raise AssertionError('Invalid moments of inertia')

        self._inertial = self.get_inertial(inertialArg)

        self._unitZ = tf.constant([[[0.], [0.], [1.]]], dtype=tf.float64)

        self._gravity = 9.81

        self._massEye = self._mass * tf.eye(3, dtype=tf.float64)
        self._massLower = self._mass * tf_skew_op("Skew_cog", self._cog)
        self._rbMass = self.rigid_body_mass()
        self._mTot = self.total_mass()
        self._invMTot = tf.linalg.inv(self._mTot)

        # TODO: Not used.
        self._invMRb = tf.linalg.inv(self._rbMass)

        self._timingDict = {}
        self._timingDict["total"] = 0.
        self._timingDict["b2i_trans"] = 0.
        self._timingDict["pose_dot"] = 0.
        self._timingDict["calls"] = 0.

        self._accTimingDict = {}
        self._accTimingDict['total'] = 0.
        self._accTimingDict["cori"] = 0.
        self._accTimingDict["rest"] = 0.
        self._accTimingDict["solv"] = 0.
        self._accTimingDict["damp"] = 0.
        self._accTimingDict["calls"] = 0.

    def print_info(self):
        """Print the vehicle's parameters."""
        print('Body frame: {}'.format(self._bodyFrameId))
        print('Mass: {0:.3f} kg'.format(self._mass.numpy()))
        print('System inertia matrix:\n{}'.format(self._rbMass))
        print('Added-mass:\n{}'.format(self._addedMass))
        print('M:\n{}'.format(self._mTot))
        print('Linear damping: {}'.format(self._linearDamping))
        print('Quad. damping: {}'.format(self._quadDamping))
        print('Center of gravity: {}'.format(self._cog))
        print('Center of buoyancy: {}'.format(self._cob))
        print('Inertial:\n{}'.format(self._inertial))

    def rigid_body_mass(self):
        upper = tf.concat([self._massEye, -self._massLower], axis=1)
        lower = tf.concat([self._massLower, self._inertial], axis=1)
        return tf.concat([upper, lower], axis=0)

    def total_mass(self):
        return tf.add(self._rbMass, self._addedMass)

    def get_inertial(self, dict):
        # buid the inertial matrix
        ixx = tf.Variable(dict['ixx'], trainable=True, dtype=tf.float64)
        ixy = tf.Variable(dict['ixy'], trainable=True, dtype=tf.float64)
        ixz = tf.Variable(dict['ixz'], trainable=True, dtype=tf.float64)
        iyy = tf.Variable(dict['iyy'], trainable=True, dtype=tf.float64)
        iyz = tf.Variable(dict['iyz'], trainable=True, dtype=tf.float64)
        izz = tf.Variable(dict['izz'], trainable=True, dtype=tf.float64)

        row0 = tf.expand_dims(tf.concat([ixx, ixy, ixz], axis=0), axis=0)
        row1 = tf.expand_dims(tf.concat([ixy, iyy, iyz], axis=0), axis=0)
        row2 = tf.expand_dims(tf.concat([ixz, iyz, izz], axis=0), axis=0)

        inertial = tf.concat([row0, row1, row2], axis=0)

        return inertial

    def build_step_graph(self, scope, state, action, retAcc=False):
        return self.step(scope, state, action, rk=self._rk, acc=retAcc)

    def step(self, scope, state, action, rk=1, acc=False):
        # Forward and backward euler integration.
        with tf.name_scope(scope) as scope:
            k1 = self.state_dot(state, action)
            if rk == 1:
                nextState = tf.add(state, k1*self.dt)

            elif rk == 2:
                k2 = self.state_dot(tf.add(state, self.dt*k1), action)
                tmp = self.dt/2. * tf.add(k1, k2)

            elif rk == 4:
                k2 = self.state_dot(tf.add(state, self.dt*k1/2.), action)
                k3 = self.state_dot(tf.add(state, self.dt*k2/2.), action)
                k4 = self.state_dot(tf.add(state, self.dt*k3), action)
                tmp = 1./6. * tf.add(tf.add(k1, 2.*k2),
                                     tf.add(2.*k3, k4*self.dt))*self.dt

            nextState = tf.add(state, tmp)
            nextState = self.normalize_quat(nextState)

            if not acc:
                return nextState
            return nextState, k1[:, 7:13]

    def state_dot(self, state, action):
        '''
            Computes x_dot = f(x, u)
        '''
        # mostily used for rk integration methods.
        pose, speed = self.prepare_data(state)

        start = t.perf_counter()
        self.body2inertial_transform(pose)
        end = t.perf_counter()
        self._timingDict["b2i_trans"] += end-start

        start = t.perf_counter()
        poseDot = tf.matmul(self.get_jacobian(), speed)
        end = t.perf_counter()
        self._timingDict["pose_dot"] += end-start

        start = t.perf_counter()
        speedDot = self.acc("acceleration", speed, action)
        end = t.perf_counter()
        self._accTimingDict["total"] += end-start
        self._accTimingDict["calls"] += 1

        self._timingDict["calls"] += 1

        return self.get_state_dot(poseDot, speedDot)

    def get_jacobian(self):
        '''
        Returns J(nu) in $mathbb{R}^{7 cross 7}$
                     ---------------------------------------
            J(nu) = | q_{n}^{b}(Theta) 0^{3 cross 3}    |
                     | 0^{3 cross 3} T_{theta}(theta)   |
                     ---------------------------------------
        '''
        k = self._rotBtoI.shape[0]

        OPad3x3 = tf.zeros(shape=(k, 3, 3), dtype=tf.float64)
        OPad4x3 = tf.zeros(shape=(k, 4, 3), dtype=tf.float64)

        jacR1 = tf.concat([self._rotBtoI, OPad3x3], axis=-1)

        jacR2 = tf.concat([OPad4x3, self._TBtoIquat], axis=-1)
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

        w = quat[:, 0]
        x = quat[:, 1]
        y = quat[:, 2]
        z = quat[:, 3]

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

        self._rotBtoI = tf.concat([r1, r2, r3], axis=1)

        r1t = tf.expand_dims(tf.concat([-x, -y, -z], axis=-1), axis=1)

        r2t = tf.expand_dims(tf.concat([w, -z, y], axis=-1), axis=1)

        r3t = tf.expand_dims(tf.concat([z, w, -x], axis=-1), axis=1)

        r4t = tf.expand_dims(tf.concat([-y, x, w], axis=-1), axis=1)

        self._TBtoIquat = 0.5 * tf.concat([r1t, r2t, r3t, r4t], axis=1)

    def rotBtoI_np(self, quat):
        w = quat[0]
        x = quat[1]
        y = quat[2]
        z = quat[3]

        return np.array([
                         [1 - 2 * (y**2 + z**2),
                          2 * (x * y - z * w),
                          2 * (x * z + y * w)],
                         [2 * (x * y + z * w),
                          1 - 2 * (x**2 + z**2),
                          2 * (y * z - x * w)],
                         [2 * (x * z - y * w),
                          2 * (y * z + x * w),
                          1 - 2 * (x**2 + y**2)]
                        ])

    def tItoB_np(self, euler):
        r = euler[0]
        p = euler[1]
        T = np.array([[1., 0., -np.sin(p)],
                      [0., np.cos(r), np.cos(p) * np.sin(r)],
                      [0., -np.sin(r), np.cos(p) * np.cos(r)]])
        return T

    def prepare_data(self, state):
        pose = state[:, 0:7]
        speed = state[:, 7:13]
        return pose, speed

    def get_state(self, poseNext, speedNext):
        '''
            Return the state of the system after
            propagating it for one timestep.

            input:
            ------
                - poseNext: float64 tensor.
                    Shape [k/1, 7, 1].
                - speedNext: float64 tensor.
                    Shape [k, 6, 1]

            output:
            -------
                - nextState: float64 tensor. Shape [k, 12/13, 1]
        '''

        poseNext = self.normalize_quat(poseNext)
        k_p = poseNext.shape[0]
        k_s = speedNext.shape[0]
        # On the first step, the pose is only of size k=1.
        if k_p < k_s:
            poseNext = tf.broadcast_to(poseNext, [k_s, 7, 1])
        state = tf.concat([poseNext, speedNext], axis=1)
        return state

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
                - nextState: float64 tensor. Shape [k, 12/13, 1]
        '''
        k_p = poseDot.shape[0]
        k_s = speedDot.shape[0]
        # On the first step, the pose is only of size k=1.
        if k_p < k_s:
            poseDot = tf.broadcast_to(poseDot, [k_s, 7, 1])
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
        quat = pose[:, 3:7]
        vel = pose[:, 7:13]
        quat = tf.divide(quat, tf.linalg.norm(quat, axis=1, keepdims=True))
        pose = tf.concat([pos, quat, vel], axis=1)
        return pose

    def restoring_forces(self, scope):
        with tf.name_scope(scope) as scope:
            cog = tf.expand_dims(self._cog, axis=0)
            cob = tf.expand_dims(self._cob, axis=0)

            fng = - self._mass*self._gravity*self._unitZ
            fnb = self._volume*self._densityu*self._gravity*self._unitZ

            rotItoB = tf.transpose(self._rotBtoI, perm=[0, 2, 1])

            fbg = tf.matmul(rotItoB, fng)
            fbb = tf.matmul(rotItoB, fnb)

            k = rotItoB.shape[0]

            cog = tf.broadcast_to(cog, [k, 3])
            cob = tf.broadcast_to(cob, [k, 3])

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
                - $D(\nu)\nu$ the damping matrix. Shape [k, 6, 6]
        '''
        with tf.name_scope(scope) as scope:
            D = -1*self._linearDamping - tf.multiply(
                                           tf.expand_dims(vel[:, 0],
                                                          axis=-1),
                                           self._linearDampingForwardSpeed)

            tmp = -1*tf.linalg.matmul(tf.expand_dims(self._quadDamping,
                                                     axis=0),
                                      tf.abs(
                                        tf.linalg.diag(
                                          tf.squeeze(vel, axis=-1))))
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

            OPad = tf.zeros(shape=(self._k, 3, 3), dtype=tf.float64)

            S12 = -tf_skew_op_k("Skew_coriolis",
                                tf.squeeze(
                                    tf.matmul(self._mTot[0:3, 0:3],
                                              vel[:, 0:3]) +
                                    tf.matmul(self._mTot[0:3, 3:6],
                                              vel[:, 3:6]),
                                    axis=-1))

            S22 = -tf_skew_op_k("Skew_coriolis_diag",
                                tf.squeeze(
                                    tf.matmul(self._mTot[3:6, 0:3],
                                              vel[:, 0:3]) +
                                    tf.matmul(self._mTot[3:6, 3:6],
                                              vel[:, 3:6]),
                                    axis=-1))

            r1 = tf.concat([OPad, S12], axis=-1)
            r2 = tf.concat([S12, S22], axis=-1)
            C = tf.concat([r1, r2], axis=1)

            return C

    def acc(self, scope, vel, genForce=None):
        with tf.name_scope(scope) as scope:
            tensGenForce = np.zeros(shape=(6, 1))
            if genForce is not None:
                tensGenForce = genForce

            start = t.perf_counter()
            D = self.damping_matrix("Damping", vel)
            end = t.perf_counter()
            self._accTimingDict["damp"] += end-start

            start = t.perf_counter()
            C = self.coriolis_matrix("Coriolis", vel)
            end = t.perf_counter()
            self._accTimingDict["cori"] += end-start

            start = t.perf_counter()
            g = self.restoring_forces("Restoring")
            end = t.perf_counter()
            self._accTimingDict["rest"] += end-start

            start = t.perf_counter()

            rhs = tensGenForce - tf.matmul(C, vel) - tf.matmul(D, vel) - g
            lhs = tf.broadcast_to(self._invMTot, [self.k, 6, 6])

            acc = tf.matmul(lhs, rhs)

            end = t.perf_counter()
            self._accTimingDict["solv"] += end-start

            return acc

    def get_profile(self):
        profile = self._timingDict.copy()
        profile['acc'] = self._accTimingDict.copy()
        return profile
