import tensorflow as tf
from cost_base import CostBase

# TODO: compute all constants without tensorflow. Out of the graph computation.
class StaticCost(CostBase):
    def __init__(self, lam, gamma, upsilon, sigma, goal, tau, Q):
        CostBase.__init__(self, lam, gamma, upsilon, sigma, tau)
        self.setGoal(goal)
        self.Q = tf.convert_to_tensor(Q, dtype=tf.float64)

    def state_cost(self, scope, state):
        diff = tf.math.subtract(state, self.goal, name="diff")
        return tf.linalg.matmul(diff, tf.linalg.matmul(self.Q, diff, name="right"), transpose_a=True, name="left")

    def setGoal(self, goal):
        self.goal = tf.convert_to_tensor(goal, dtype=tf.float64)

    def getGoal(self):
        return self.goal.numpy()

    def advanceGoal(self, scope, next):
        with tf.name_scope("advance_goal") as ag:
            # shapes: in [s_dim, 1], out None
            remain = tf.slice(self.goal, [1, 0, 0], [self.tau-1, -1, -1])
            self.goal = tf.concat([remain, next], 0)

    def draw_goal(self):
        np_goal = self.getGoal()
        return np_goal[0], np_goal[1]

    def dist(self, state):
        return np.linalg.norm(state-self.getGoal(), axis=-1)
