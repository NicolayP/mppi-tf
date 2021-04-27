import tensorflow as tf
import numpy as np
from mppi_tf.scripts.model_base import ModelBase
import pandas as pd

def blockDiag(vec, pad, dim):
    vec_np = np.array([])
    for h in range(dim):

        tmp = np.array([])
        for w in range(dim):
            if w == 0 and h == 0:
                tmp = vec
            elif w == 0:
                tmp = pad
            elif w == h:
                tmp = np.hstack([tmp, vec])
            else:
                tmp = np.hstack([tmp, pad])

        if h == 0:
            vec_np = tmp
        else:
            vec_np = np.vstack([vec_np, tmp])
    return vec_np


def get_data(filename):
    df = pd.read_csv(filename)
    df_header = df.copy().columns[0: -1]

    # remove the two last rows as they seem erroneous.
    arr = df[df_header].to_numpy()[0: -2]
    x = np.expand_dims(arr[:, 0:2], -1)
    u = np.expand_dims(arr[:, 2:3], -1)
    gt = np.expand_dims(arr[:, 3:], -1)
    return gt, x, u


class PointMassModel(ModelBase):
    def __init__(self, mass=1, dt=0.1, state_dim=2, action_dim=1, name="point_mass"):
        ModelBase.__init__(self, dt, state_dim, action_dim, name)

        mass = tf.Variable([[mass]], name="mass",
                                trainable=True, dtype=tf.float64)

        self.addModelVars("mass", mass)

        with tf.name_scope("Const") as c:
            self.create_const(c)

    def buildStepGraph(self, scope, state, action):
        with tf.name_scope("Model_Step"):
            return tf.add(self.buildFreeStepGraph("free", state),
                          self.buildActionStepGraph("action", action))

    def buildFreeStepGraph(self, scope, state):
        with tf.name_scope(scope):
            return tf.linalg.matmul(self.A, state, name="A_x")
        
    def buildActionStepGraph(self, scope, action):
        with tf.name_scope(scope):
            return tf.linalg.matmul(tf.divide(self.B,
                                                self.model_vars["mass"],
                                                name="B"),
                                    action, name="B_u")

    def getMass(self):
        return self.model_vars["mass"].numpy()[0]

    def create_const(self, scope):
        a = np.array([[1., self.dt], [0., 1.]])
        a_pad = np.array([[0, 0], [0, 0]])
        a_np = blockDiag(a, a_pad, int(self.state_dim/2))
        self.A = tf.constant(a_np, dtype=tf.float64, name="A")

        b = np.array([[(self.dt*self.dt)/2.], [self.dt]])
        b_pad = np.array([[0], [0]])
        b_np = blockDiag(b, b_pad, self.action_dim)
        self.B = tf.constant(b_np, dtype=tf.float64, name="B")
