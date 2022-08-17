from .cost_base import CostBase
from ..misc.utile import assert_shape, dtype

import numpy as np
import tensorflow as tf
import tensorflow_graphics as tfg


class ElipseCost(CostBase):
    def __init__(self,
                 lam,
                 gamma,
                 upsilon,
                 sigma,
                 a,b,
                 center_x,
                 center_y,
                 speed,
                 m_state,
                 m_vel):
        '''
            2D eliptic cost function.
            - input:
            --------
                - lam (lambda) the inverse temperature. 
                - gamma: decoupling parameter between action and noise.
                - upsilon: covariance augmentation for noise generation.
                - sigma: the noise covariance matrix. shape [a_dim, a_dim].
                - a: the long axis of the elipse.
                - b: the short axis of the elipse.
                - center_x: the x value of the elipse center.
                - center_y: the y value of the elipse center.
                - speed: the target speed.
                - m_state: multiplier for the state error.
                - m_vel: multiplier for the vel error.
        '''
        CostBase.__init__(self, lam, gamma, upsilon, sigma)
        self.a = a
        self.b = b
        self.cx = center_x
        self.cy = center_y
        self.gv = speed
        self.mx = tf.cast(m_state, dtype)
        self.mv = tf.cast(m_vel, dtype)

    def state_cost(self, scope, state):
        '''
            Computes the state cost for the eliptic cost function.

            - input:
            --------
                - scope: the tensorflow scope.
                - state: current state. Shape: [k/1, 4, 1]

            - output:
            ---------
                dict with entry:
                "speed_cost" = m_vel * (speed - current_speed)^2
                "position_cost" =  m_state|\frac{x-cx}{a} + \frac{y-cy}{b} - 1|
                "state_cost" = speed_cost + position_cost

        '''
        if not assert_shape(state, (-1, 4, 1)):
            raise AssertionError("State tensor doesn't have the expected shape.\n Expected [k/1, 4, 1], got {}".format(state.shape))

        x = tf.slice(state, [0, 0, 0], [-1, 1, -1])
        y = tf.slice(state, [0, 2, 0], [-1, 1, -1])
        vx = tf.slice(state, [0, 1, 0], [-1, 1, -1])
        vy = tf.slice(state, [0, 3, 0], [-1, 1, -1])
        v = tf.sqrt(tf.pow(vx, 2) + tf.pow(vy, 2))
        diffx = tf.divide(tf.math.subtract(x, self.cx, name="diff"), self.a)
        diffy = tf.divide(tf.math.subtract(y, self.cy, name="diff"), self.b)
        d = tf.abs(tf.pow(diffx, 2) + tf.pow(diffy, 2) - 1)
        d = tf.math.multiply(self.mx, d)
        dv = tf.pow(v - self.gv, 2)
        dv = tf.math.multiply(self.mv, dv)
        state_cost = tf.add(d, dv)

        return state_cost

    def draw_goal(self):
        alpha = np.linspace(0, 2*np.pi, 1000)
        x = self.a*np.cos(alpha)
        y = self.b*np.sin(alpha)
        return x, y

    def dist(self, state):
        return_dict = {}
        x = state[0]
        vx = state[1]
        y = state[2]
        vy = state[3]
        v = np.sqrt(vx**2 + vy**2)
        x_dist = (((x-self.cx)/self.a)**2 + ((y-self.cy)/self.b)**2) - 1
        v_dist = np.abs(v-self.gv)
        return_dict["x_dist"] = x_dist[0]
        return_dict["v_dist"] = v_dist[0]
        return return_dict


class ElipseCost3D(CostBase):
    def __init__(self,
                 lam,
                 gamma,
                 upsilon,
                 sigma,
                 normal,
                 aVec,
                 axis,
                 center,
                 speed,
                 v_speed,
                 mState,
                 mVel):
        '''
            3D eliptic cost function.
            - input:
            --------
                - lam (lambda) the inverse temperature.
                - gamma: decoupling parameter between action and noise.
                - upsilon: covariance augmentation for noise generation.
                - sigma: the noise covariance matrix. shape [a_dim, a_dim].
                - a: the long axis of the elipse.
                - b: the short axis of the elipse.
                - center_x: the x value of the elipse center.
                - center_y: the y value of the elipse center.
                - speed: the target speed.
                - m_state: multiplier for the state error.
                - m_vel: multiplier for the vel error.
        '''
        CostBase.__init__(self, lam, gamma, upsilon, sigma)
        axis = np.concatenate([axis, np.array([[1.]])], axis=0)
        self.axis = tf.convert_to_tensor(axis, dtype=dtype)
        self.aVec = tf.convert_to_tensor(aVec, dtype=dtype)
        self.normal = tf.convert_to_tensor(normal, dtype=dtype)
        self.bVec = tf.expand_dims(
                        tf.linalg.cross(
                            tf.squeeze(self.normal, axis=-1),
                            tf.squeeze(self.aVec, axis=-1)
                        ), axis=-1
                    )
        self.center = tf.convert_to_tensor(center, dtype=dtype)

        self.mapping = tf.constant([
                                    [
                                     [-axis[0, 0]/axis[1, 0]],
                                     [axis[1, 0]/axis[0, 0]],
                                     [0.]
                                    ]
                                   ],
                                   dtype=dtype)
        self.prepare_consts()
        self.gv = tf.constant(speed, dtype=dtype)

        #self.gv = speed
        #self.vz = v_speed
        self.mS = tf.cast(mState, dtype)
        self.mV = tf.cast(mVel, dtype)

    def prepare_consts(self):
        N = tf.concat([self.aVec, self.bVec, self.normal], axis=-1)
        self.R = tf.transpose(tf.linalg.inv(N))
        self.q = tfg.geometry.transformation.quaternion.from_rotation_matrix(self.R)
        self.t = self.center

    def state_cost(self, scope, state):
        position = tf.squeeze(state[:, 0:3], axis=-1)
        quat = tf.squeeze(state[:, 3:7], axis=-1)
        # Express the point in the plane frame.
        posPf = tfg.geometry.transformation.quaternion.rotate(position, self.q)
        posPf = tf.expand_dims(posPf, axis=-1)
        quatPf = tfg.geometry.transformation.quaternion.multiply(self.q, quat)
        posePf = tf.concat([posPf, tf.expand_dims(quatPf, axis=-1)], axis=1)
        positionCost = self.position_error(posPf)
        orientationCost = self.orientation_error(posePf)
        velCost = self.velocity_error(state[:, 7:13])

        stateCost = self.mS*positionCost + self.mS*orientationCost + self.mV*velCost
        return stateCost

    def position_error(self, position):
        '''
            Computes the distance between a set of points and the elipse.
            It assumes that the elipse lives in a 2D plane (Z=0) and the
            points are expressed in the plane frame.

            - inputs:
            ---------
                - position: Tf tensor representing points position in the
                    plane frame. Shape [k, 3, 1]

            - outputs:
            ----------
                - distance in euclidian norm between the point and the elipse.
                    Shape [k, 1, 1]
        ''' 
        d = tf.pow(tf.divide(position, self.axis), 2)
        d = tf.reduce_sum(d, axis=1) # -> shape [k, 1, 1]
        d = tf.abs(d - 1.)
        return tf.expand_dims(d, axis=-1)

    def orientation_error(self, pose):
        '''
            Computes the distance between the orientation and the desired
            orientation. The desired orientation is defined by angle
            between the agent and the elipse tangent.

            - inputs:
            ---------
                - pose: Tf tensor representing the pose in the
                    plane frame. Shape [k, 7, 1]

            - outputs:
            ----------
                - orientation cost. Shape [k, 1, 1]
        '''
        position = pose[:, 0:3]
        # Rotation from elipse frame to point
        quaterion = tf.squeeze(pose[:, 3:7])
        tgVec = tf.gather(position, indices=[1, 0, 2], axis=1)
        tgVec = tf.squeeze(tf.multiply(tgVec, self.mapping), axis=-1)
        tgVec = tf.linalg.normalize(tgVec, axis=-1)[0]
        x = tf.constant([1., 0., 0.], dtype=dtype)
        q = tfg.geometry.transformation.quaternion.between_two_vectors_3d(x, tgVec)
        err = tfg.geometry.transformation.quaternion.relative_angle(q, quaterion)
        return err

    def velocity_error(self, velocity):
        '''
            Computed the "distance" between the desired velocity and the
            current velocity in absolute value.

            - inputs:
            ---------
                - velocity: Tf tensor representing the current velocity of
                    the agent. Shape [k, 6, 1]

            - outputs:
            ----------
                - distance between the desired velocity and the current one
                    Shape [k, 1, 1]
        '''
        # compute the normalized linear velocity
        v = tf.norm(velocity[:, 0:3], axis=1)
        dv = tf.abs(tf.pow(v, 2) - tf.pow(self.gv, 2))
        return tf.expand_dims(dv, axis=-1)
