import tensorflow as tf
import numpy as np
from model_base import ModelBase

gpu = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(device=gpu[0], enable=True)

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
    def __init__(self, inertial_frame_id='world', state_dim=12, action_dim=5, name="AUV", k=1, dt=0.1, parameters=dict()):
        '''
            Class constructor
        '''
        self.k = k
        self.dt = dt

        ModelBase.__init__(self, state_dim, action_dim, name)
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

        self.inertial = self.getInertial(inertial_arg)

        self.unit_z = tf.constant([0., 0., 1.], dtype=tf.float64)        

        self.gravity = 9.81

        self.mass_eye = self.mass * tf.eye(3, dtype=tf.float64)
        self.mass_lower =  self.mass * tf_skew_op("Skew_cog", self.cog)
        self.rb_mass = self.rigid_body_mass()
        self._Mtot = self.total_mass()

    def rigid_body_mass(self):
        upper = tf.concat([self.mass_eye, -self.mass_lower], axis=1)
        lower = tf.concat([self.mass_lower, self.inertial], axis=1)
        return tf.concat([upper, lower], axis=0)

    def total_mass(self):
        return tf.add(self.rb_mass, self.added_mass)

    def getInertial(self, dict):
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

    @tf.function
    def buildStepGraph(self, scope, state, action):
        with tf.name_scope(scope) as scope:
            pose, speed = self.prepare_data(state)
            self.rotBtoI, self.TBtoI = self.body2inertial_transform(pose)
            
            pose_dot = tf.matmul(self.get_jacobian(), speed)
            
            speed_dot = self.acc("acceleration", speed, action)
            
            pose_next = tf.add(self.dt*pose_dot, pose)
            speed_next = tf.add(self.dt*speed_dot, speed)

            return self.get_state(pose_next, speed_next) 

    def get_jacobian(self):
        '''
        Returns J(\nu) \in $\mathbb{R}^{6 \cross 6}$ 
                     ---------------------------------------
            J(\nu) = | Rot_{n}^{b}(\Theta) 0^{3 \cross 3}  |
                     | 0^{3 \cross 3} T_{\theta}(\theta)   |
                     ---------------------------------------
        '''
        O_pad = tf.zeros(shape=(self.k, 3, 3), dtype=tf.float64)
        jac_r1 = tf.concat([self.rotBtoI, O_pad], axis=-1)
        jac_r2 = tf.concat([O_pad, self.TBtoI], axis=-1)
        jac = tf.concat([jac_r1, jac_r2], axis=1)
        return jac

    def body2inertial_transform(self, pose):
        '''
            Computes the rotational transform from body to inertial Rot_{n}^{b}(\Theta)
            and the attitude transformation T_{\theta}(\theta).

            input:
            ------
                - pose the robot pose expressed in inertial frame. Shape [k, 6, 1]
            
            output:
            ------- 
                - J(pose) shape [k, 6, 6]
        '''
        angles = tf.squeeze(pose[:, 3:6, :], axis=-1)

        c = tf.expand_dims(tf.math.cos(angles), axis=-1)
        s = tf.expand_dims(tf.math.sin(angles), axis=-1)

        # cos(roll)/sin(roll)
        cr = c[:, 0]
        sr = s[:, 0]
        
        # cos(pitch)/sin(pitch)
        cp = c[:, 1]
        sp = s[:, 1]

        # cos(yaw)/sin(yaw)
        cy = c[:, 2]
        sy = s[:, 2]

        # build rotation matrix.
        O_pad = tf.zeros(shape=(self.k, 1), dtype=tf.float64)
        I_pad = tf.ones(shape=(self.k, 1), dtype=tf.float64)

        rz_r1 = tf.expand_dims(tf.concat([cy, -sy, O_pad], axis=-1), axis=1)
        rz_r2 = tf.expand_dims(tf.concat([sy, cy, O_pad], axis=-1), axis=1)
        rz_r3 = tf.expand_dims(tf.concat([O_pad, O_pad, I_pad], axis=-1), axis=1)

        Rz = tf.concat([rz_r1, rz_r2, rz_r3], axis=1)


        ry_r1 = tf.expand_dims(tf.concat([cp, O_pad, sp], axis=-1), axis=1)
        ry_r2 = tf.expand_dims(tf.concat([O_pad, I_pad, O_pad], axis=-1), axis=1)
        ry_r3 = tf.expand_dims(tf.concat([-sp, O_pad, cp], axis=-1), axis=1)

        Ry = tf.concat([ry_r1, ry_r2, ry_r3], axis=1)


        rx_r1 = tf.expand_dims(tf.concat([cr, -sr, O_pad], axis=-1), axis=1)
        rx_r2 = tf.expand_dims(tf.concat([sr, cr, O_pad], axis=-1), axis=1)
        rx_r3 = tf.expand_dims(tf.concat([O_pad, O_pad, I_pad], axis=-1), axis=1)

        Rx = tf.concat([rx_r1, rx_r2, rx_r3], axis=1)

        rotBtoI = tf.linalg.matmul(tf.matmul(Rz, Ry), Rx)

        TBtoI_r1 = tf.expand_dims(tf.concat([I_pad, tf.divide(tf.multiply(sr, sp), cp), tf.divide(tf.multiply(cr, sp), cp)], axis=-1), axis=1)
        TBtoI_r2 = tf.expand_dims(tf.concat([O_pad, cr, -sr], axis=-1), axis=1)
        TBtoI_r3 = tf.expand_dims(tf.concat([O_pad, tf.divide(sr, cp), tf.divide(cr, cp)], axis=-1), axis=1)

        TBtoI = tf.concat([TBtoI_r1, TBtoI_r2, TBtoI_r3], axis=1)

        return rotBtoI, TBtoI

    def prepare_data(self, state):
        pose = state[:, 0:6]
        speed = state[:, 6:12]
        return pose, speed

    def get_state(self, pose_next, speed_next):
        state = tf.concat([pose_next, speed_next], axis=1)
        return state

    def to_SNAME(self, x):
        if self._body_frame_id == 'base_link_ned':
            return x
        try:
            if x.shape == (3,):
                return np.array([x[0], -1*x[1], -1*x[2]])
            elif x.shape == (6,):
                return np.array([x[0], -1*x[1], -1*x[2],
                                 x[3], -1*x[4], -1*x[5]])

        except Exception as e:
            print('Invalid input vector, v=' + str(x))
            print('Message=' + str(e))
            return None

    def from_SNAME(self, x):
        if self._body_frame_id == 'base_link_ned':
            return x
        return self.to_SNAME(x)

    def restoring_forces(self, scope, q=None, use_sname=False):
        '''
            computes the restoring forces. 

            input:
            ------
                - mass, float, the mass of the vehicle. in kg.
                - gravity, float, the gravity constant. in $\frac{N}{t^{2}}$
                - volume, float, the volume of the of the vehicle. In m^{3}.
                - density, float, the liquid density. In $\frac{kg}{m^{3}}$
                - cog center of gravity expressed in the body frame. In m. Shape [3].
                - cob center of boyency expressed in the body frame. In m. Shape [3].
                - rotItoB, rotational transform from inertial frame to the body frame. Shape [k, 3, 3]

        '''
        with tf.name_scope(scope) as scope:
            if use_sname:
                Fg = -self.mass * self.gravity * self.unit_z
                Fb = self.volume * self.density * self.gravity * self.unit_z
            else:
                Fg = self.mass * self.gravity * self.unit_z
                Fb = self.volume * self.density * self.gravity * self.unit_z
            
            return tf.concat([-1 * tf.matmul(tf.transpose(self.rotBtoI, perm=[0, 2, 1]), tf.expand_dims(Fg + Fb, axis=-1)), 
                            -1 * tf.matmul(tf.transpose(self.rotBtoI, perm=[0, 2, 1]), tf.expand_dims(tf.linalg.cross(self.cog, Fg) +
                                                                    tf.linalg.cross(self.cob, Fb), axis=-1))],
                            axis=1)
    
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
            D = tf.add(D, tf.add(self.quad_damping, tf.abs(tf.linalg.diag(tf.squeeze(vel, axis=-1)))))
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

            O_pad = tf.zeros(shape=(self.k, 3, 3), dtype=tf.float64)

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
    
    def acc(self, scope, vel, gen_forces=None, use_sname=True):
        with tf.name_scope(scope) as scope:
            tens_gen_forces = np.zeros(shape=(6, 1))
            if gen_forces is not None:
                tens_gen_forces = gen_forces
            
            D = self.damping_matrix("Damping", vel)
            C = self.coriolis_matrix("Coriolis", vel)
            g = self.restoring_forces("Restoring")

            rhs = tens_gen_forces - tf.matmul(C, vel) - tf.matmul(D, vel) - g

            acc = tf.linalg.solve(tf.broadcast_to(self._Mtot, [self.k, 6, 6]), rhs)
            return acc
