from .cost_base import CostBase
from ..misc.utile import assert_shape, dtype

import numpy as np
from scipy.spatial.transform import Rotation as R
import tensorflow as tf
import tensorflow_graphics.geometry.transformation as tfgt


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

    '''
    Returns a list of 3D points representing the trajectory of the task. Doesn't handl
    obstacles.

    inputs:
    -------
        - state: the state of the agent.
        - pts: the number of points in the list
    '''
    def get_3D(self, state, pts=100):
        alpha = np.linspace(0, 2*np.pi, pts)
        x = self.a*np.cos(alpha) + self.cx
        y = self.b*np.sin(alpha) + self.cy
        return x, y


class ElipseCost3D(CostBase):
    '''
        3D eliptic cost function.
        - input:
        --------
            - lam (lambda) the inverse temperature.
            - gamma: decoupling parameter between action and noise.
            - upsilon: covariance augmentation for noise generation.
            - sigma: the noise covariance matrix. shape [a_dim, a_dim].
            - normal: np.array, the unit vector normal to the elipse plane. Shape (3, 1)
            - aVec: np.array, the unit vector indicating the direction of the main axis. Shape (3, 1)
            - axis: np.array, length of a axis and b axis. Shape (2, 1)
            - center: np.array, the center point of the elipse. Shape (2, 1)
            - speed: the target speed in absolute value.
            - m_state: multiplier for the state error.
            - m_vel: multiplier for the vel error.
    '''
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
                 mState,
                 mVel,
                 tg=True):
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
        # translation to bring points to the elipse origin
        self.t = tf.convert_to_tensor(center, dtype=dtype)

        # Normalized mapping of tangent vector
        self.mapping_tg = tf.constant([
                                    [
                                     [-axis[0, 0]/axis[1, 0]],
                                     [axis[1, 0]/axis[0, 0]],
                                     [0.]
                                    ]
                                   ],
                                   dtype=dtype)

        # Normalized mapping of perpendicular vector
        self.mapping_perp = tf.constant([
                                    [
                                     [-axis[1, 0]/axis[0, 0]],
                                     [-axis[0, 0]/axis[1, 0]],
                                     [0.]
                                    ]
                                   ],
                                   dtype=dtype)

        self.prepare_consts()
        self.gv = tf.constant(speed, dtype=dtype)

        self.mS = tf.cast(mState, dtype)
        self.mV = tf.cast(mVel, dtype)
        self.tg = tg

    def prepare_consts(self):
        N = tf.concat([self.aVec, self.bVec, self.normal], axis=-1)
        self.R = tf.transpose(tf.linalg.inv(N))
        # quaternion mapping points to the elipse plane.
        self.q = tfgt.quaternion.from_rotation_matrix(self.R)
        self.np_r = R.from_quat(self.q.numpy())
        self.x = tf.constant([[1., 0., 0.]], dtype=dtype)

    '''
        State cost for eliptic cost. The function transforms the state
        in the coordinate system made of the plane of the elipse and its normal.
        It then computes a cost that is made of:
            - distance to the elipse.
            - orientation (either tg or perpendiculare) to the elispe
            - speed.
        - inputs:
        ---------
            - scope: string. Tensorflow scope.
            - state: tensor. State of the vehicle.
                shape [k/1, 13, 1]

        - outputs:
        ----------
            - cost = alpha * position_cost + beta * orientation_cost + gamma * velocity_cost,
            shape [k, 1, 1]
    '''
    def state_cost(self, scope, state):
        posePf = self.to_elipse_frame(state)
        posPf = posePf[:, :3]
        positionCost = self.dist_to_elipse(posPf)
        orientationCost = self.orientation_error(posePf)
        #velCost = self.velocity_error(state[:, 7:13])
        velCost = self.velocity_cost(state[:, 7:13])
        stateCost = self.mS*positionCost + self.mS*orientationCost + self.mV*velCost
        return stateCost

    def velocity_cost(self, velocity):
        if self.tg:
            v = velocity[:, 0]
        else:
            v = velocity[:, 1]
        dv = tf.pow(v-self.gv, 2)
        dv = tf.expand_dims(dv, axis=-1)
        return dv

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
    def dist_to_elipse(self, position):
        d = tf.pow(tf.divide(position, self.axis), 2)
        d = tf.reduce_sum(d, axis=1) # -> shape [k, 1, 1]
        d = tf.abs(d - 1.)
        return tf.expand_dims(d, axis=-1)

    def orientation_error(self, pose):
        if self.tg:
            return self.orientation_error_tg(pose)
        return self.orientation_error_perp(pose)

    '''
        Computes the distance between the orientation and the desired
        orientation. The desired orientation is defined by angle
        between the agent and the elipse normal (directed towards the center).

        - inputs:
        ---------
            - pose: Tf tensor representing the pose in the
                plane frame. Shape [k, 7, 1]

        - outputs:
        ----------
            - orientation cost. Shape [k, 1, 1]
    '''
    def orientation_error_perp(self, pose):
        # to compute the tangant perpendicular vector to the elipse we differentiate
        # the elipse implicitly. See obsidian/MyPapers/Experiments#Elipse orientation cost.
        position = pose[:, 0:3]
        quaternion = pose[:, 3:7, 0]
        perp_vec = tf.squeeze(tf.multiply(position, self.mapping_perp), axis=-1)
        perp_vec = tf.linalg.normalize(perp_vec, axis=-1)[0]
        # operation to get the quaternion from the vector
        k = tf.shape(perp_vec)[0]
        x_broad = tf.broadcast_to(self.x, (k, 3))
        q = tfgt.quaternion.between_two_vectors_3d(x_broad, perp_vec)
        err = tfgt.quaternion.relative_angle(q, quaternion)[:, None, None]
        return err

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
    def orientation_error_tg(self, pose):
        position = pose[:, 0:3]
        # Rotation from elipse frame to point
        quaternion = pose[..., 3:7, 0]
        # to compute the tangeant vector to the elipse we differentiate
        # the elipse implicitly. See obsidian/MyPapers/Experiments#Elipse orientation cost.
        tgVec = tf.gather(position, indices=[1, 0, 2], axis=1)
        tgVec = tf.squeeze(tf.multiply(tgVec, self.mapping_tg), axis=-1)
        # normalize outputs tuple, normed vector and norm.
        tgVec = tf.linalg.normalize(tgVec, axis=-1)[0]
        # X-axis vector (or forward.)
        k = tf.shape(tgVec)[0]
        x_broad = tf.broadcast_to(self.x, (k, 3))
        # The desired quaternion is defiend by the rotation from the forward vector
        # to the tangant vector.
        q = tfgt.quaternion.between_two_vectors_3d(x_broad, tgVec)
        err = tfgt.quaternion.relative_angle(q, quaternion)[:, None, None]
        return err

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
    def velocity_error(self, velocity):
        # compute the norm of linear velocity
        return self.velocity_cost(velocity)
        v = tf.norm(velocity[:, :3], axis=1)
        dv = tf.abs(v - self.gv)
        return tf.expand_dims(dv, axis=-1)

    '''
    Returns a list of 3D points representing the trajectory of the task. Doesn't handl
    obstacles.

    inputs:
    -------
        - state: the state of the agent.
        - pts: the number of points in the list
    '''
    def get_3D(self, state, pts=100):
        alpha = np.linspace(0, 2*np.pi, pts)
        x = self.axis[0]*np.cos(alpha)
        y = self.axis[1]*np.sin(alpha)
        z = np.zeros(shape=alpha.shape)
        points = np.concatenate([x[..., None], y[..., None], z[..., None]], axis=-1)
        r_inv = self.np_r.inv()
        points = r_inv.apply(points) + self.t.numpy()[:, 0]
        return points

    def angle_error(self, state, split=False):
        posePf = self.to_elipse_frame(state)
        return self.orientation_error(posePf)

    def position_error(self, state, split=False):
        posPf = self.to_elipse_frame(state)[:, :3]
        return self.dist_to_elipse(posPf)

    def to_elipse_frame(self, state):
        # Express the points in the plane frame.
        k = tf.shape(state)[0]
        q = tf.broadcast_to(self.q, (k, 4))
        # Firsrt translate the point
        position = tf.squeeze(state[:, 0:3] - self.t, axis=-1)
        quat = tf.squeeze(state[:, 3:7], axis=-1)
        # Rotate the point in the elipse frame.
        posPf = tfgt.quaternion.rotate(position, q)
        posPf = tf.expand_dims(posPf, axis=-1)

        # Rotate the orientation in the elipse frame
        quatPf = tfgt.quaternion.multiply(q, quat)
        # The new pose.
        posePf = tf.concat([posPf, tf.expand_dims(quatPf, axis=-1)], axis=1)
        return posePf