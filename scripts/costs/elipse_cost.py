from cost_base import CostBase
from utile import assert_shape

import numpy as np
import tensorflow as tf

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


class ElipseCost(CostBase):
    def __init__(self, lam, gamma, upsilon, sigma, a, b, center_x, center_y, speed, m_state, m_vel):
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
        self.mx = tf.cast(m_state, tf.float64)
        self.mv = tf.cast(m_vel, tf.float64)

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

        return_dict = {}
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

        return_dict["speed_cost"]=dv
        return_dict["position_cost"]=d
        return_dict["state_cost"]=state_cost
        return return_dict

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
    def __init__(self, lam, gamma, upsilon, sigma, a, b, center_x, center_y, depth, speed, v_speed, m_state, m_vel):
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
        self.depth = depth
        self.gv = speed
        self.vz = v_speed
        self.mx = tf.cast(m_state, tf.float64)
        self.mv = tf.cast(m_vel, tf.float64)

    def state_cost(self, scope, state):
        '''
            Computes the state cost for the eliptic cost function.

            - input:
            --------
                - scope: the tensorflow scope.
                - State: current state. Shape: [k/1, 12, 1]

            - output:
            ---------
                dict with entry:
                "speed_cost" = m_vel * (speed - current_speed)^2
                "position_cost" =  m_state|\frac{x-cx}{a} + \frac{y-cy}{b} - 1|
                "state_cost" = speed_cost + position_cost

        '''
        if not assert_shape(state, (-1, 12, 1)):
            raise AssertionError("State tensor doesn't have the expected shape.\n Expected [k/1, 12, 1], got {}".format(state.shape))
        return_dict = {}
        x = tf.expand_dims(state[:, 0], axis=-1)
        y = tf.expand_dims(state[:, 1], axis=-1)
        z = tf.expand_dims(state[:, 2], axis=-1)

        vx = tf.expand_dims(state[:, 6], axis=-1)
        vy = tf.expand_dims(state[:, 7], axis=-1)
        vz = tf.expand_dims(state[:, 8], axis=-1)

        v = tf.sqrt(tf.pow(vx, 2) + tf.pow(vy, 2))
        diffx = tf.divide(tf.math.subtract(x, self.cx, name="diff"), self.a)
        diffy = tf.divide(tf.math.subtract(y, self.cy, name="diff"), self.b)
        d = tf.abs(tf.pow(diffx, 2) + tf.pow(diffy, 2) - 1)
        d = tf.math.multiply(self.mx, d)
        dv = tf.pow(v - self.gv, 2)
        dv = tf.math.multiply(self.mv, dv)
        state_cost = tf.add(d, dv)

        dvz = tf.pow(vz - self.vz, 2)
        dvz = tf.math.multiply(self.mv, dvz)

        diffz = tf.math.subtract(z, self.depth, name="depth_diff")
        diffz = tf.math.multiply(self.mx, diffz)

        state_cost = tf.add(state_cost, dvz)
        state_cost = tf.add(state_cost, diffz)

        return_dict["speed_cost"]=dv
        return_dict["position_cost"]=d
        return_dict["state_cost"]=state_cost
        return return_dict

    def draw_goal(self):
        alpha = np.linspace(0, 2*np.pi, 1000)
        x = self.a*np.cos(alpha)
        y = self.b*np.sin(alpha)
        return x, y

    def dist(self, state):
        return_dict = {}
        x = state[0]
        y = state[1]
        z = state[2]
        vx = state[6]
        vy = state[7]
        vz = state[8]
        v = np.sqrt(vx**2 + vy**2)
        x_dist = (((x-self.cx)/self.a)**2 + ((y-self.cy)/self.b)**2) - 1
        v_dist = np.abs(v-self.gv)
        return_dict["x_dist"] = x_dist[0]
        return_dict["v_dist"] = v_dist[0]
        return return_dict