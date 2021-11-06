import tensorflow as tf
import numpy as np
from model_base import ModelBase
import pandas as pd

def block_diag(vec, pad, dim):
    vecNp = np.array([])
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
            vecNp = tmp
        else:
            vecNp = np.vstack([vecNp, tmp])
    return vecNp

class PointMassModel(ModelBase):
    '''
        Point mass model heritated from Model Base class.
        This model is a simple LTI model of a point mass where the mass
        is a trainable variable.
    '''
    def __init__(self, mass=1, dt=0.1, stateDim=2, actionDim=1, name="point_mass"):
        '''
            Constructor of the point mass model.

            - input:
            --------
                - mass. Float, the inital mass of the model.
                - state_dim. Int, the state space dimension.
                - action_dim. Int, the action space dimension.
                - name. String, model name.

        '''

        ModelBase.__init__(self, stateDim, actionDim, name)
        self._dt = dt
        mass = tf.Variable([[mass]], name="mass",
                                trainable=True, dtype=tf.float64)

        self.add_model_vars("mass", mass)

        with tf.name_scope("Const") as c:
            self.create_const(c)

    def build_step_graph(self, scope, state, action):
        '''
            Abstract method, need to be overwritten in child class.
            Step graph for the model. This computes the prediction for $\hat{f}(x, u)$

            - input:
            --------
                - scope: String, the tensorflow scope name.
                - state: State tensor. Shape [k, s_dim, 1]
                - action: Action tensor. Shape [k, a_dim, 1]

            - output:
            ---------
                - the next state.
        '''

        with tf.name_scope("Model_Step"):
            return tf.add(self.build_free_step_graph("free", state),
                          self.build_action_step_graph("action", action))

    def build_free_step_graph(self, scope, state):
        '''
            Control free update part of the model. From LTI notation this
            corresponds to A*x_{t}

            - input:
            --------
                - scope: String, the tensorflow scope name.
                - state: State tensor. Shape [k, s_dim, 1]

            - output:
            ---------
                - A*x_{t}: the input free update tensor. Shape [k, s_dim, 1]
        '''

        with tf.name_scope(scope):
            return tf.linalg.matmul(self._A, state, name="A_x")

    def build_action_step_graph(self, scope, action):
        '''
            Control update part of the model. From LTI notation this
            this corresponds to B*u_{t}

            - input:
            --------
                - scope: String, the tensorflow scope name.
                - action: the action tensor. Shape [k, a_dim, 1]

            - output:
            ---------
                - B*u_{t}: the input update tensor. Shape [k, s_dim, 1]
        '''

        with tf.name_scope(scope):
            return tf.linalg.matmul(tf.divide(self._B,
                                                self._modelVars["mass"],
                                                name="B"),
                                    action, name="B_u")

    def getMass(self):
        '''
            Return the mnodel estimated mass.
        '''

        return self._modelVars["mass"].numpy()[0]

    def create_const(self, scope):
        '''
            Creates the A and B matrix of the LTI system.

            - input:
            --------
                - scope. String, the tensorflow scope name
        '''

        a = np.array([[1., self.dt], [0., 1.]])
        aPad = np.array([[0, 0], [0, 0]])
        aNp = block_diag(a, aPad, int(self._stateDim/2))
        self._A = tf.constant(aNp, dtype=tf.float64, name="A")

        b = np.array([[(self.dt*self.dt)/2.], [self.dt]])
        bPad = np.array([[0], [0]])
        bNp = block_diag(b, bPad, self._actionDim)
        self._B = tf.constant(bNp, dtype=tf.float64, name="B")
