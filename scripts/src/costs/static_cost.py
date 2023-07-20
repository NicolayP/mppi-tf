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
        x0 = self.goal[:3, 0]
        x1 = state[:3, 0]
        t = np.linspace(0., 1., pts)[:, None]

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

    '''
        Returns a list of 3D points representing the trajectory of the task. Doesn't handl
        obstacles.

        inputs:
        -------
            - state: the state of the agent.
            - pts: the number of points in the list
    '''
    def get_3D(self, state, pts=100):
        x0 = self.goal[:3, 0]
        x1 = state[:3, 0]
        t = np.linspace(0., 1., pts)[:, None]

        return t * x0 + (1 - t) * x1

    def angle_error(self, state, split=False):
        state = tf.squeeze(state, axis=-1)
        goal = tf.squeeze(self.goal, axis=-1)
        quat = state[:, 3:7]
        goal_quat = goal[3:7]
        theta = tf.math.acos(2*tf.math.pow(tf.tensordot(quat, goal_quat, 1), 2) - 1)
        return theta

    def velocity_error(self, state, split=False):
        vel = tf.squeeze(state[:, -6:], axis=-1)
        goal_vel = tf.squeeze(self.goal[-6:], axis=-1)
        vel_dist = tf.linalg.norm(tf.subtract(vel, goal_vel), axis=-1)
        return vel_dist

    def position_error(self, state, split=False):
        pos = tf.squeeze(state[:, :3], axis=-1)
        state = tf.squeeze(state, axis=-1)
        goal_pos = tf.squeeze(self.goal[:3], axis=-1)
        pos_dist = tf.linalg.norm(tf.subtract(pos, goal_pos), axis=-1)
        return pos_dist

    '''
        Final step cost only dependant on the pose not on the velocity.
    '''
    def build_final_step_cost_graph(self, scope, state):
        state = tf.squeeze(state, axis=-1)
        goal = tf.squeeze(self.goal, axis=-1)
        quat = state[:, 3:7]
        goal_quat = goal[3:7]
        theta = tf.math.acos(2*tf.math.pow(tf.tensordot(quat, goal_quat, 1), 2) - 1)

        pos = state[:, :3]
        goal_pos = goal[:3]
        pos_dist = tf.subtract(pos, goal_pos)

        pose_dist = tf.concat([pos_dist, theta[..., None]], axis=1)[..., None]

        pose_cost = tf.linalg.matmul(
                pose_dist,
                tf.linalg.matmul(self.Q[0:4, 0:4],
                                    pose_dist,
                                    name="right"),
                transpose_a=True,
                name="left")

        return 100*pose_cost


class ListQuatCost(StaticQuatCost):
    '''
    '''
    def __init__(self, lam, gamma, upsilon, sigma, goals, Q, diag=False, min_dist=0.5):
        self.goals = goals
        self.i = 0
        goal = self.goals[self.i]
        self.i += 1
        self.min_dist = min_dist
        StaticQuatCost.__init__(self, lam, gamma, upsilon, sigma, goal, Q, diag)

    '''
    '''
    def update_goal(self, state):
        p_dist = self.position_dist(state)
        if p_dist < self.min_dist and self.i < len(self.goals):
            self.set_goal(self.goals[self.i])
            self.i += 1

    '''
    '''
    def get_3D(self, state, pts=100):
        pts_seg = int(pts / (len(self.goals[self.i:])+1)) # number of points per segment remaining
        s1 = state
        s2 = self.goals[self.i-1] # current goal in numpy
        segments = [self.gen_segment(s1, s2, pts_seg)]
        for i in range(self.i, len(self.goals)):
            s1 = self.goals[i-1]
            s2 = self.goals[i]
            segments.append(self.gen_segment(s1, s2, pts_seg))
        return np.concatenate(segments, axis=0)

    '''
    '''
    def gen_segment(self, s1, s2, pts):
        x0 = s1[:3, 0]
        x1 = s2[:3, 0]
        t = np.linspace(0., 1., pts)[:, None]

        return t * x0 + (1 - t) * x1

    '''
    '''
    def position_dist(self, state):
        p = state[:3]
        g_p = self.goal[:3]
        d = g_p - p
        return tf.linalg.norm(d)


class StaticRotCost(CostBase):
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
    def __init__(self, lam, gamma, upsilon, sigma, goal, Q, diag=False, rep="quat"):
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

    '''
    ''' 
    def set_goal(self, goal):
        if not assert_shape(goal, (18, 1)):
            raise AssertionError("Goal tensor shape error, expected: [{}, 1], got {}".format(self.q_shape[0], goal.shape))

        self.goal.assign(goal)

    '''
    '''
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

    '''
    '''
    def get_goal(self):
        return self.goal

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
    def state_cost(self, scope, state):
        diff = self.dist(state)
        stateCost = tf.linalg.matmul(
                        diff,
                        tf.linalg.matmul(self.Q,
                                         diff,
                                         name="right"),
                        transpose_a=True,
                        name="left")
        return stateCost

    '''
    '''
    def draw_goal(self):
        np_goal = self.goal
        return np_goal[0], np_goal[1]

    '''
    '''
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

    '''
    Returns a list of 3D points representing the trajectory of the task. Doesn't handl
    obstacles.

    inputs:
    -------
        - state: the state of the agent.
        - pts: the number of points in the list
    '''
    def get_3D(self, state, pts=100):
        x0 = self.goal[:3, 0]
        x1 = state[:3, 0]
        t = np.linspace(0., 1., pts)[:, None]

        return t * x0 + (1 - t) * x1

    def angle_error(self, state, split=False):
        pass

    def velocity_error(self, state, split=False):
        pass

    def position_error(self, state, split=False):
        pass