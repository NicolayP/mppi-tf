import tensorflow as tf
import numpy as np
from .cost_base import CostBase
from ..misc.utile import assert_shape, dtype
from scipy.spatial.transform import Rotation as R

# TODO: compute all constants without tensorflow. Out of the graph computation.
class StaticCost(CostBase):
    def __init__(self, lam, gamma, upsilon, sigma, goal, Q, diag=False):
        '''
            Compute the cost for a static point.

            - input:
            --------
                - lam (lambda) the inverse temperature. 
                - gamma: decoupling parameter between action and noise.
                - upsilon: covariance augmentation for noise generation.
                - sigma: the noise covariance matrix. shape [aDim, aDim].
                - goal: target goal (psition; speed). shape [sDim, 1].
                - Q: weight matrix for the different part of the cost function. shape: [sDim, sDim]
        '''

        CostBase.__init__(self, lam, gamma, upsilon, sigma)
        
        self.Q = tf.convert_to_tensor(Q, dtype=dtype)

        if diag:
            self.Q = tf.linalg.diag(self.Q)
        
        self.q_shape = self.Q.shape
        self.setGoal(goal)
        
    def setGoal(self, goal):
        if not assert_shape(goal, (self.q_shape[0], 1)):
            raise AssertionError("Goal tensor shape error, expected: [{}, 1], got {}".format(self.q_shape[0], goal.shape))

        self.goal = tf.convert_to_tensor(goal, dtype=dtype)

    def getGoal(self):
        return self.goal

    def state_cost(self, scope, state):
        '''
            Computes state cost for the static point.

            - input:
            --------
                - scope: the tensorflow scope.
                - state: current state. Shape: [k/1, sDim, 1]

            - output:
            ---------
                - dict with entries:
                    "state_cost" = (state-goal)^T Q (state-goal)
        '''
        diff = tf.math.subtract(state, self.goal, name="diff")
        stateCost = tf.linalg.matmul(
                        diff,
                        tf.linalg.matmul(self.Q,
                                         diff,
                                         name="right"),
                        transpose_a=True,
                        name="left")

        return stateCost

    def draw_goal(self):
        np_goal = self.goal
        return np_goal[0], np_goal[1]

    def dist(self, state):
        return tf.subtract(state, self.goal)

    '''
    Returns a list of 3D points representing the trajectory of the task. Doesn't handl
    obstacles.

    inputs:
    -------
        - state: the state of the agent.
        - pts: the number of points in the list
    '''
    def get_3D(self, state, pts=100):
        x0 = self.goal[:3]
        x1 = state[:3]
        t = np.linspace(0., 1., pts)

        return t * x0 + (1 - t) * x1


class StaticQuatCost(CostBase):
    def __init__(self, lam, gamma, upsilon, sigma, goal, Q, diag=False):
        '''
            Compute the cost for a static point.

            - input:
            --------
                - lam (lambda) the inverse temperature. 
                - gamma: decoupling parameter between action and noise.
                - upsilon: covariance augmentation for noise generation.
                - sigma: the noise covariance matrix. shape [aDim, aDim].
                - goal: target goal (position; speed). shape [sDim, 1].
                - Q: weight matrix for the different part of the cost function. shape: [sDim, sDim]
        '''

        CostBase.__init__(self, lam, gamma, upsilon, sigma)
        
        self.Q = tf.convert_to_tensor(Q, dtype=dtype)

        if diag:
            self.Q = tf.linalg.diag(self.Q)
        
        self.q_shape = self.Q.shape
        if not assert_shape(self.Q, (10, 10)):
            raise AssertionError("Goal tensor shape error, expected: [10, 10], got {}".format(self.q_shape))

        self.goal = tf.Variable(
            goal,
            trainable=False,
            dtype=dtype,
            name="goal")

        self.set_goal(goal)
        
    def set_goal(self, goal):
        if not assert_shape(goal, (13, 1)):
            raise AssertionError("Goal tensor shape error, expected: [{}, 1], got {}".format(self.q_shape[0], goal.shape))
        self.goal.assign(goal)

    def get_goal(self):
        return self.goal

    def state_cost(self, scope, state):
        '''
            Computes state cost for the static point.

            - input:
            --------
                - scope: the tensorflow scope.
                - state: current state. Shape: [k/1, sDim, 1]

            - output:
            ---------
                - dict with entries:
                    "state_cost" = (state-goal)^T Q (state-goal)
        '''
        diff = self.dist(state)
        stateCost = tf.linalg.matmul(
                        diff,
                        tf.linalg.matmul(self.Q,
                                         diff,
                                         name="right"),
                        transpose_a=True,
                        name="left")

        return stateCost

    def draw_goal(self):
        np_goal = self.goal
        return np_goal[0], np_goal[1]

    def dist(self, state):
        state = tf.squeeze(state, axis=-1)
        goal = tf.squeeze(self.goal, axis=-1)
        quat = state[:, 3:7]
        goal_quat = goal[3:7]

        theta = tf.math.acos(2*tf.math.pow(tf.tensordot(quat, goal_quat, 1), 2) - 1)

        pos = state[:, :3]
        goal_pos = goal[:3]
        pos_dist = tf.subtract(pos, goal_pos)

        vel = state[:, -6:]
        goal_vel = goal[-6:]
        vel_dist = tf.subtract(vel, goal_vel)        
        return tf.concat([pos_dist, theta[..., None], vel_dist], axis=1)[..., None]

    def split_state_cost(self, state):
        diff = self.dist(state)
        rhs = tf.linalg.matmul(self.Q, diff, name="right")
        split_cost = tf.math.multiply(diff, rhs, name="split")
        return split_cost



class StaticRotCost(CostBase):
    def __init__(self, lam, gamma, upsilon, sigma, goal, Q, diag=False, rep="quat"):
        '''
            Compute the cost for a static point.

            - input:
            --------
                - lam (lambda) the inverse temperature. 
                - gamma: decoupling parameter between action and noise.
                - upsilon: covariance augmentation for noise generation.
                - sigma: the noise covariance matrix. shape [aDim, aDim].
                - goal: target goal (psition; speed). shape [sDim, 1].
                - Q: weight matrix for the different part of the cost function. shape: [sDim, sDim]
        '''

        CostBase.__init__(self, lam, gamma, upsilon, sigma)
        
        self.Q = tf.convert_to_tensor(Q, dtype=dtype)

        if diag:
            self.Q = tf.linalg.diag(self.Q)
        
        self.q_shape = self.Q.shape
        if not assert_shape(self.Q, (10, 10)):
            raise AssertionError("Goal tensor shape error, expected: [10, 10], got {}".format(self.q_shape))
        
        self.rep = rep

        goal = self.to_vec(goal)

        self.goal = tf.Variable(
            goal,
            trainable=False,
            dtype=dtype,
            name="goal")

        self.set_goal(goal)
        
    def set_goal(self, goal):
        if not assert_shape(goal, (18, 1)):
            raise AssertionError("Goal tensor shape error, expected: [{}, 1], got {}".format(self.q_shape[0], goal.shape))

        self.goal.assign(goal)

    def to_vec(self, goal):
        if self.rep == "quat":
            quat = goal[3:7, 0]
            r = R.from_quat(quat)
        elif self.rep == "euler":
            euler = goal[3:6, 0]
            r = R.from_euler(euler)
        mat = r.as_matrix().reshape((9, 1))
        pos = goal[:3]
        goal_vel = goal[-6:]
        return np.concatenate([pos, mat, goal_vel], axis=0)

    def get_goal(self):
        return self.goal

    def state_cost(self, scope, state):
        '''
            Computes state cost for the static point.

            - input:
            --------
                - scope: the tensorflow scope.
                - state: current state. Shape: [k/1, sDim, 1]

            - output:
            ---------
                - dict with entries:
                    "state_cost" = (state-goal)^T Q (state-goal)
        '''
        diff = self.dist(state)
        stateCost = tf.linalg.matmul(
                        diff,
                        tf.linalg.matmul(self.Q,
                                         diff,
                                         name="right"),
                        transpose_a=True,
                        name="left")
        return stateCost

    def draw_goal(self):
        np_goal = self.goal
        return np_goal[0], np_goal[1]

    def dist(self, state):
        state = tf.squeeze(state, axis=-1)
        goal = tf.squeeze(self.goal, axis=-1)
        rot = tf.reshape(state[:, 3:3+9], (-1, 3, 3))
        goal_rot = tf.reshape(goal[3:3+9], (3, 3))

        rot_angle = tf.linalg.matmul(goal_rot, rot, transpose_b=True)
        theta = tf.math.acos(tf.clip_by_value(((tf.linalg.trace(rot_angle) - 1.) / 2.), -1, 1))

        pos = state[:, :3]
        goal_pos = goal[:3]
        pos_dist = tf.subtract(pos, goal_pos)

        vel = state[:, -6:]
        goal_vel = goal[-6:]
        vel_dist = tf.subtract(vel, goal_vel)        
        diff = tf.concat([pos_dist, theta[..., None], vel_dist], axis=1)[..., None]
        return diff
