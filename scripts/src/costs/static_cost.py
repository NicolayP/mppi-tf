import tensorflow as tf
from .cost_base import CostBase
from ..misc.utile import assert_shape

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
        
        self.Q = tf.convert_to_tensor(Q, dtype=tf.float64)

        if diag:
            self.Q = tf.linalg.diag(self.Q)
        
        self.q_shape = self.Q.shape
        self.setGoal(goal)
        
    def setGoal(self, goal):
        if not assert_shape(goal, (self.q_shape[0], 1)):
            raise AssertionError("Goal tensor shape error, expected: [{}, 1], got {}".format(self.q_shape[0], goal.shape))

        self.goal = tf.convert_to_tensor(goal, dtype=tf.float64)

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
