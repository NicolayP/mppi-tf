#from typing import Optional
import tensorflow as tf
import numpy as np
#from quaternion import as_euler_angles
from quaternion import from_euler_angles, as_euler_angles
from model_base import ModelBase

import matplotlib.pyplot as plt

gpu = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(device=gpu[0], enable=True)

import time as t

def skew_op(vec):     
    S = np.zeros(shape=(3, 3))
    
    S[0, 1] = -vec[2]
    S[0, 2] =  vec[1]
    
    S[1, 0] =  vec[2]
    S[1, 2] = -vec[0]

    S[2, 0] = -vec[1]
    S[2, 1] =  vec[0]
    return tf.constant(S, dtype=tf.float64)

def tf_skew_op(scope, vec):
    with tf.name_scope(scope) as scope:
        vec = tf.expand_dims(vec, axis=-1)
        O_pad = tf.zeros(shape=(1), dtype=tf.float64)
        r0 = tf.expand_dims(tf.concat([O_pad, -vec[2], vec[1]], axis=-1), axis=1)
        r1 = tf.expand_dims(tf.concat([vec[2], O_pad,  -vec[0]], axis=-1), axis=1)
        r2 = tf.expand_dims(tf.concat([-vec[1], vec[0], O_pad], axis=-1), axis=1)

        S = tf.concat([r0, r1, r2], axis=1)
        return S

def skew_op_k(batch):
    k = batch.shape[0]
    S = np.zeros(shape=(k, 3, 3))
    
    S[:, 0, 1] = -batch[:, 2]
    S[:, 0, 2] =  batch[:, 1]
    
    S[:, 1, 0] =  batch[:, 2]
    S[:, 1, 2] = -batch[:, 0]

    S[:, 2, 0] = -batch[:, 1]
    S[:, 2, 1] =  batch[:, 0]
    return S

def tf_skew_op_k(scope, batch):
    with tf.name_scope(scope) as scope:
        k = batch.shape[0]
        vec = tf.expand_dims(batch, axis=-1)
    
        O_pad = tf.zeros(shape=(k, 1), dtype=tf.float64)
        r0 = tf.expand_dims(tf.concat([O_pad, -vec[:, 2], vec[:, 1]], axis=-1), axis=1)
        r1 = tf.expand_dims(tf.concat([vec[:, 2], O_pad,  -vec[:, 0]], axis=-1), axis=1)
        r2 = tf.expand_dims(tf.concat([-vec[:, 1], vec[:, 0], O_pad], axis=-1), axis=1)
    
        S = tf.concat([r0, r1, r2], axis=1)
        return S

'''
    AUV dynamical model based on the UUV_sim vehicle model
    and implemented in tensorflow to be used with the MPPI
    controller.
'''
class AUVModel(ModelBase):
    def __init__(self, inertial_frame_id='world', quat=False, action_dim=6, name="AUV", k=1, dt=0.1, rk=1, parameters=dict()):
        '''
            Class constructor
        '''
        self.rk = rk
        self.dt = dt
        if quat:
            self._quat=True
            state_dim=13
        else:
            self._quat=False
            state_dim=12

        ModelBase.__init__(self, state_dim, action_dim, name, k, inertial_frame_id)

        assert  inertial_frame_id in ['world', 'world_ned']

        self.inertial_frame_id = inertial_frame_id
        if self.inertial_frame_id == 'world':
            self.body_frame_id = 'base_link'
        else:
            self.body_frame_id = 'base_link_ned'

        self.mass = 0
        if "mass" in parameters:
            self.mass = tf.Variable(parameters['mass'], trainable=True, dtype=tf.float64)
        assert (self.mass > 0), "Mass has to be positive."
        
        self.volume = 0
        if "volume" in parameters:
            self.volume = parameters["volume"]
        assert (self.volume > 0), "Volume has to be positive."

        self.density = 0
        if "density" in parameters:
            self.density = parameters["density"]
        assert (self.density > 0), "Liquid density has to be positive."

        self.height = 0
        if "height" in parameters:
            self.height = parameters["height"]
        assert (self.height > 0), "Height has to be positive."

        self.length = 0
        if "length" in parameters:
            self.length = parameters["length"]
        assert (self.length > 0), "Length has to be positive."

        self.width = 0
        if "width" in parameters:
            self.width = parameters["width"]
        assert (self.width > 0), "Width has to be positive."

        if "cog" in parameters:
            self.cog = tf.constant(parameters["cog"], dtype=tf.float64)
            assert (len(self.cog) == 3), 'Invalid center of gravity vector. Size != 3'
        else:
            raise AssertionError("need to define the center of gravity in the body frame")

        if "cob" in parameters:
            self.cob = tf.constant(parameters["cob"], dtype=tf.float64)
            assert (len(self.cob) == 3), "Invalid center of buoyancy vector. Size != 3"
        else:
            raise AssertionError("need to define the center of buoyancy in the body frame")

        added_mass = np.zeros((6, 6))
        if "Ma" in parameters:
            added_mass = np.array(parameters["Ma"])
            assert (added_mass.shape == (6, 6)), "Invalid add mass matrix"
        self.added_mass = tf.constant(added_mass, dtype=tf.float64)
        
        damping = np.zeros(shape=(6, 6))
        if "linear_damping" in parameters:
            damping = np.array(parameters["linear_damping"])
            if damping.shape == (6,):
                damping = np.diag(damping)
            assert (damping.shape == (6, 6)), "Linear damping must be given as a 6x6 matrix or the diagonal coefficients"
        self.linear_damping = tf.constant(damping, dtype=tf.float64)

        quad_damping = np.zeros(shape=(6,))
        if "quad_damping" in parameters:
            quad_damping = np.array(parameters["quad_damping"])
            assert (quad_damping.shape == (6,)), "Quadratic damping must be given defined with 6 coefficients"
        self.quad_damping = tf.linalg.diag(quad_damping)

        damping_forward = np.zeros(shape=(6, 6))
        if "linear_damping_forward_speed" in parameters:
            damping_forward = np.array(parameters["linear_damping_forward_speed"])
            if damping_forward.shape == (6,):
                damping_forward = np.diag(damping_forward)
            assert (damping_forward.shape == (6, 6)), "Linear damping proportional to the \
                                                      forward speed must be given as a 6x6 \
                                                      matrix or the diagonal coefficients"
        self.linear_damping_forward_speed = tf.expand_dims(tf.constant(damping_forward, dtype=tf.float64), axis=0)

        inertial = dict(ixx=0, iyy=0, izz=0, ixy=0, ixz=0, iyz=0)
        if "inertial" in parameters:
            inertial_arg = parameters["inertial"]
            for key in inertial:
                if key not in inertial_arg:
                    raise AssertionError('Invalid moments of inertia')

        self.inertial = self.get_inertial(inertial_arg)

        self.unit_z = tf.constant([[[0.], [0.], [1.]]], dtype=tf.float64)

        self.gravity = 9.81

        self.mass_eye = self.mass * tf.eye(3, dtype=tf.float64)
        self.mass_lower =  self.mass * tf_skew_op("Skew_cog", self.cog)
        self.rb_mass = self.rigid_body_mass()
        self._Mtot = self.total_mass()
        self.invMtot = tf.linalg.inv(self._Mtot)
        self.inv_mrb = tf.linalg.inv(self.rb_mass)


        self.timing_dict = {}
        self.timing_dict["total"] = 0.
        self.timing_dict["b2i_trans"] = 0.
        self.timing_dict["pose_dot"] = 0.
        self.timing_dict["calls"] = 0.

        self.acc_timing_dict = {}
        self.acc_timing_dict['total'] = 0.
        self.acc_timing_dict["cori"] = 0.
        self.acc_timing_dict["rest"] = 0.
        self.acc_timing_dict["solv"] = 0.
        self.acc_timing_dict["damp"] = 0.
        self.acc_timing_dict["calls"] = 0.
        

    def print_info(self):
        """Print the vehicle's parameters."""
        print('Body frame: {}'.format(self.body_frame_id))
        print('Mass: {0:.3f} kg'.format(self.mass.numpy()))
        print('System inertia matrix:\n{}'.format(self.rb_mass))
        print('Added-mass:\n{}'.format(self.added_mass))
        print('M:\n{}'.format(self._Mtot))
        print('Linear damping: {}'.format(self.linear_damping))
        print('Quad. damping: {}'.format(self.quad_damping))
        print('Center of gravity: {}'.format(self.cog))
        print('Center of buoyancy: {}'.format(self.cob))
        print('Inertial:\n{}'.format(self.inertial))

    def rigid_body_mass(self):
        upper = tf.concat([self.mass_eye, -self.mass_lower], axis=1)
        lower = tf.concat([self.mass_lower, self.inertial], axis=1)
        return tf.concat([upper, lower], axis=0)

    def total_mass(self):
        return tf.add(self.rb_mass, self.added_mass)

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

    def build_step_graph(self, scope, state, action, ret_acc=False):
        if self._quat:
            return self.step_q(scope, state, action, rk=self.rk, acc=ret_acc)
        return self.step(scope, state, action)

    def step(self, scope, state, action):
        with tf.name_scope(scope) as scope:
            pose, speed = self.prepare_data(state)
            self.body2inertial_transform(pose)

            pose_dot = tf.matmul(self.get_jacobian(), speed)
            speed_dot = self.acc("acceleration", speed, action)

            pose_next = tf.add(self.dt*pose_dot, pose)
            speed_next = tf.add(self.dt*speed_dot, speed)
            return self.get_state(pose_next, speed_next)

    def step_q(self, scope, state, action, rk=1, acc=False):
        # Forward and backward euler integration.
        with tf.name_scope(scope) as scope:
            if rk == 1:
                k1 = self.state_dot(state, action)
                next_state = tf.add(state, k1*self.dt)


            elif rk == 2:
                k1 = self.state_dot(state, action)
                k2 = self.state_dot(tf.add(state, self.dt*k1), action)
                next_state = tf.add(state, self.dt/2. * tf.add(k1, k2))

            elif rk == 4:
                k1 = self.state_dot(state, action)
                k2 = self.state_dot(tf.add(state, self.dt*k1/2.), action)
                k3 = self.state_dot(tf.add(state, self.dt*k2/2.), action)
                k4 = self.state_dot(tf.add(state, self.dt*k3), action)
                tmp = 1./6. * tf.add(tf.add(k1, 2.*k2), tf.add(2.*k3, k4*self.dt))*self.dt
                next_state = tf.add(state, tmp)

            if self._quat:
                next_state = self.normalize_quat(next_state)

            if not acc:
                return next_state
            return next_state, k1[:, 7:13]

    def state_dot(self, state, action):
        '''
            Computes x_dot = f(x, u)
        '''
        # mostily used for rk integration methods.
        pose, speed = self.prepare_data(state)

        start = t.perf_counter()
        self.body2inertial_transform_q(pose)
        end = t.perf_counter()
        self.timing_dict["b2i_trans"] += end-start

        start = t.perf_counter()
        pose_dot = tf.matmul(self.get_jacobian_q(), speed)
        end = t.perf_counter()
        self.timing_dict["pose_dot"] += end-start

        start = t.perf_counter()
        speed_dot = self.acc("acceleration", speed, action)
        end = t.perf_counter()
        self.acc_timing_dict["total"] += end-start
        self.acc_timing_dict["calls"] += 1

        self.timing_dict["calls"] += 1

        return self.get_state_dot(pose_dot, speed_dot)

    def set_prev_vel(self, vel):
            self.prev_vel = vel

    def get_jacobian(self):
        '''
        Returns J(\nu) \in $\mathbb{R}^{6 \cross 6}$ 
                     ---------------------------------------
            J(\nu) = | Rot_{n}^{b}(\Theta) 0^{3 \cross 3}  |
                     | 0^{3 \cross 3} T_{\theta}(\theta)   |
                     ---------------------------------------
        '''
        k = self._rotBtoI.shape[0]
        O_pad = tf.zeros(shape=(k, 3, 3), dtype=tf.float64)
        jac_r1 = tf.concat([self._rotBtoI, O_pad], axis=-1)
        jac_r2 = tf.concat([O_pad, self._TBtoIeuler], axis=-1)
        jac = tf.concat([jac_r1, jac_r2], axis=1)
        return jac

    def get_jacobian_q(self):
        '''
        Returns J(\nu) \in $\mathbb{R}^{7 \cross 7}$
                     ---------------------------------------
            J(\nu) = | q_{n}^{b}(\Theta) 0^{3 \cross 3}    |
                     | 0^{3 \cross 3} T_{\theta}(\theta)   |
                     ---------------------------------------
        '''
        k = self._rotBtoI.shape[0]

        O_pad3x3 = tf.zeros(shape=(k, 3, 3), dtype=tf.float64)
        O_pad4x3 = tf.zeros(shape=(k, 4, 3), dtype=tf.float64)

        jac_r1 = tf.concat([self._rotBtoI, O_pad3x3], axis=-1)

        jac_r2 = tf.concat([O_pad4x3, self._TBtoIquat], axis=-1)
        jac = tf.concat([jac_r1, jac_r2], axis=1)

        return jac

    def body2inertial_transform(self, pose):
        '''
            Computes the rotational transform from body to inertial Rot_{n}^{b}(\Theta)
            and the attitude transformation T_{\theta}(\theta).

            input:
            ------
                - pose the robot pose expressed in inertial frame. Shape [k, 6, 1]

        '''
        k = pose.shape[0]
        angles = tf.squeeze(pose[:, 3:6, :], axis=-1)

        c = tf.expand_dims(tf.math.cos(angles), axis=-1)
        s = tf.expand_dims(tf.math.sin(angles), axis=-1)

        # cos(roll)/sin(roll)
        cr = c[:, 0]
        sr = s[:, 0]
        
        # cos(pitch)/sin(pitch)
        cp = c[:, 1]
        sp = s[:, 1]
        tp = tf.divide(sp, cp)

        # cos(yaw)/sin(yaw)
        cy = c[:, 2]
        sy = s[:, 2]

        # build rotation matrix.
        O_pad = tf.zeros(shape=(k, 1), dtype=tf.float64)
        I_pad = tf.ones(shape=(k, 1), dtype=tf.float64)

        rz_r1 = tf.expand_dims(tf.concat([cy, -sy, O_pad], axis=-1), axis=1)
        rz_r2 = tf.expand_dims(tf.concat([sy, cy, O_pad], axis=-1), axis=1)
        rz_r3 = tf.expand_dims(tf.concat([O_pad, O_pad, I_pad], axis=-1), axis=1)

        Rz = tf.concat([rz_r1, rz_r2, rz_r3], axis=1)

        ry_r1 = tf.expand_dims(tf.concat([cp, O_pad, sp], axis=-1), axis=1)
        ry_r2 = tf.expand_dims(tf.concat([O_pad, I_pad, O_pad], axis=-1), axis=1)
        ry_r3 = tf.expand_dims(tf.concat([-sp, O_pad, cp], axis=-1), axis=1)

        Ry = tf.concat([ry_r1, ry_r2, ry_r3], axis=1)

        rx_r1 = tf.expand_dims(tf.concat([I_pad, O_pad, O_pad], axis=-1), axis=1)
        rx_r2 = tf.expand_dims(tf.concat([O_pad, cr, -sr], axis=-1), axis=1)
        rx_r3 = tf.expand_dims(tf.concat([O_pad, sr, cr], axis=-1), axis=1)

        Rx = tf.concat([rx_r1, rx_r2, rx_r3], axis=1)

        self._rotBtoI = tf.linalg.matmul(tf.matmul(Rz, Ry), Rx)

        TBtoI_r1 = tf.expand_dims(tf.concat([I_pad, tf.divide(tf.multiply(sr, sp), cp), tf.divide(tf.multiply(cr, sp), cp)], axis=-1), axis=1)
        TBtoI_r2 = tf.expand_dims(tf.concat([O_pad, cr, -sr], axis=-1), axis=1)
        TBtoI_r3 = tf.expand_dims(tf.concat([O_pad, tf.divide(sr, cp), tf.divide(cr, cp)], axis=-1), axis=1)

        self._TBtoIeuler = tf.concat([TBtoI_r1, TBtoI_r2, TBtoI_r3], axis=1)

    def body2inertial_transform_q(self, pose):
        '''
            Computes the rotational transform from body to inertial Rot_{n}^{b}(q)
            and the attitude transformation T_{q}(q).


            input:
            ------
                - pose the robot pose expressed in inertial frame. Shape [k, 7, 1]

        '''
        k = pose.shape[0]
        quat = pose[:, 3:7, :]

        w = quat[:, 0]
        x = quat[:, 1]
        y = quat[:, 2]
        z = quat[:, 3]

        r1 = tf.expand_dims(tf.concat([1 - 2 * (tf.pow(y, 2) + tf.pow(z, 2)),
                                        2 * (x * y - z * w),
                                        2 * (x * z + y * w)], axis=-1), axis=1)

        r2 = tf.expand_dims(tf.concat([2 * (x * y + z * w),
                                        1 - 2 * (tf.pow(x, 2) + tf.pow(z, 2)),
                                        2 * (y * z - x * w)], axis=-1), axis=1)

        r3 = tf.expand_dims(tf.concat([2 * (x * z - y * w),
                                        2 * (y * z + x * w),
                                        1 - 2 * (tf.pow(x, 2) + tf.pow(y, 2))], axis=-1), axis=1)
        self._rotBtoI = tf.concat([r1, r2, r3], axis=1)


        r1_t = tf.expand_dims(tf.concat([-x, -y, -z], axis=-1), axis=1)

        r2_t = tf.expand_dims(tf.concat([w, -z, y], axis=-1), axis=1)

        r3_t = tf.expand_dims(tf.concat([z, w, -x], axis=-1), axis=1)

        r4_t = tf.expand_dims(tf.concat([-y, x, w], axis=-1), axis=1)

        self._TBtoIquat = 0.5 * tf.concat([r1_t, r2_t, r3_t, r4_t], axis=1)

    def rotBtoI_np(self, quat):
        w = quat[0]
        x = quat[1]
        y = quat[2]
        z = quat[3]

        return np.array([
                         [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                         [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
                         [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)]
                        ])

    def tItoB_np(self, euler):
        r = euler[0]
        p = euler[1]
        y = euler[2]
        T = np.array([[1., 0., -np.sin(p)],
                      [0., np.cos(r), np.cos(p) * np.sin(r)],
                      [0., -np.sin(r), np.cos(p) * np.cos(r)]])
        return T

    def prepare_data(self, state):
        if self._quat:
            pose = state[:, 0:7]
            speed = state[:, 7:13]
        else:
            pose = state[:, 0:6]
            speed = state[:, 6:12]
        return pose, speed

    def get_state(self, pose_next, speed_next):
        '''
            Return the state of the system after propagating it for one timestep.

            input:
            ------
                - pose_next: float64 tensor. Shape [k/1, 6/7, 1]. 6 If euler representation, 7 if quaternion.
                - speed_next: float64 tensor. Shape [k, 6, 1]

            output:
            -------
                - next_state: float64 tensor. Shape [k, 12/13, 1]
        '''
        if self._quat:
            pose_next = self.normalize_quat(pose_next)
        k_p = pose_next.shape[0]
        k_s = speed_next.shape[0]
        # On the first step, the pose is only of size k=1.
        if k_p < k_s:
            if self._quat:
                pose_next = tf.broadcast_to(pose_next, [k_s, 7, 1])
            else:
                pose_next = tf.broadcast_to(pose_next, [k_s, 6, 1])
        state = tf.concat([pose_next, speed_next], axis=1)
        return state

    def get_state_dot(self, pose_dot, speed_dot):
        '''
            Return the state of the system after propagating it for one timestep.

            input:
            ------
                - pose_next: float64 tensor. Shape [k/1, 6/7, 1]. 6 If euler representation, 7 if quaternion.
                - speed_next: float64 tensor. Shape [k, 6, 1]

            output:
            -------
                - next_state: float64 tensor. Shape [k, 12/13, 1]
        '''
        k_p = pose_dot.shape[0]
        k_s = speed_dot.shape[0]
        # On the first step, the pose is only of size k=1.
        if k_p < k_s:
            if self._quat:
                pose_dot = tf.broadcast_to(pose_dot, [k_s, 7, 1])
            else:
                pose_dot = tf.broadcast_to(pose_dot, [k_s, 6, 1])
        state_dot = tf.concat([pose_dot, speed_dot], axis=1)
        return state_dot

    def normalize_quat(self, pose):
        '''
            Normalizes the quaternions.

            input:
            ------
                - pose. Float64 Tensor. Shape [k, 13, 1]
            
            ouput:
            ------
                - the pose with normalized quaternion. Float64 Tensor. Shape [k, 13, 1]
        '''

        pos = pose[:, 0:3]
        quat = pose[:, 3:7]
        vel = pose[:, 7:13]
        quat = tf.divide(quat, tf.linalg.norm(quat, axis=1, keepdims=True))
        pose = tf.concat([pos, quat, vel], axis=1)
        return pose

    def restoring_forces(self, scope):
        with tf.name_scope(scope) as scope:
            cog = tf.expand_dims(self.cog, axis=0)
            cob = tf.expand_dims(self.cob, axis=0)

            fng = - self.mass*self.gravity*self.unit_z
            fnb = self.volume*self.density*self.gravity*self.unit_z

            rotItoB = tf.transpose(self._rotBtoI, perm=[0, 2, 1])

            fbg = tf.matmul(rotItoB, fng)
            fbb = tf.matmul(rotItoB, fnb)

            k = rotItoB.shape[0]

            cog = tf.broadcast_to(cog, [k, 3])
            cob = tf.broadcast_to(cob, [k, 3])

            mbg = tf.expand_dims(tf.linalg.cross(cog, tf.squeeze(fbg, axis=-1)), axis=-1)
            mbb = tf.expand_dims(tf.linalg.cross(cob, tf.squeeze(fbb, axis=-1)), axis=-1)

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
            D = -1*self.linear_damping-tf.multiply(tf.expand_dims(vel[:, 0], axis=-1), self.linear_damping_forward_speed)
            tmp = -1 * tf.linalg.matmul(tf.expand_dims(self.quad_damping, axis=0), tf.abs(tf.linalg.diag(tf.squeeze(vel, axis=-1))))
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
        k = vel.shape[0]
        with tf.name_scope(scope) as scope:

            O_pad = tf.zeros(shape=(k, 3, 3), dtype=tf.float64)

            S_12 = -tf_skew_op_k("Skew_coriolis",
                tf.squeeze(
                    tf.matmul(self._Mtot[0:3, 0:3], vel[:, 0:3]) +
                    tf.matmul(self._Mtot[0:3, 3:6], vel[:, 3:6])
                , axis=-1)
            )

            S_22 = -tf_skew_op_k("Skew_coriolis_diag",
                tf.squeeze(
                    tf.matmul(self._Mtot[3:6, 0:3], vel[:, 0:3]) +
                    tf.matmul(self._Mtot[3:6, 3:6], vel[:, 3:6])
                , axis=-1)
            )
            r1 = tf.concat([O_pad, S_12], axis=-1)
            r2 = tf.concat([S_12, S_22], axis=-1)
            C = tf.concat([r1, r2], axis=1)

            return C

    def acc(self, scope, vel, gen_forces=None):
        with tf.name_scope(scope) as scope:
            tens_gen_forces = np.zeros(shape=(6, 1))
            if gen_forces is not None:
                tens_gen_forces = gen_forces

            start = t.perf_counter()
            D = self.damping_matrix("Damping", vel)
            end = t.perf_counter()
            self.acc_timing_dict["damp"] += end-start

            start = t.perf_counter()
            C = self.coriolis_matrix("Coriolis", vel)
            end = t.perf_counter()
            self.acc_timing_dict["cori"] += end-start

            start = t.perf_counter()
            g = self.restoring_forces("Restoring")
            end = t.perf_counter()
            self.acc_timing_dict["rest"] += end-start

            start = t.perf_counter()

            rhs = tens_gen_forces - tf.matmul(C, vel) - tf.matmul(D, vel) - g
            lhs = tf.broadcast_to(self.invMtot, [self.k, 6, 6])

            acc = tf.matmul(lhs, rhs)

            end = t.perf_counter()
            self.acc_timing_dict["solv"] += end-start

            return acc

    def getProfile(self):
        profile = self.timing_dict.copy()
        profile['acc'] = self.acc_timing_dict.copy()
        return profile

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
