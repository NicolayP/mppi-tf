import tensorflow as tf
import numpy as np
from model_base import ModelBase

class NNModel(ModelBase):
    def __init__(self, dt=0.1, state_dim=2, action_dim=1, name="nn_model"):
        ModelBase.__init__(self, dt, state_dim, action_dim, name)
        self.leaky_relu_alpha = 0.2
        self.initializer = tf.initializers.glorot_uniform()
        self.addModelVars("first", self.getWeights((state_dim+action_dim, 5), "first"))
        self.addModelVars("second", self.getWeights((5, 5), "second"))
        self.addModelVars("final", self.getWeights((5, state_dim), "final"))

    def buildStepGraph(self, scope, state, action):
        # expand and broadcast state vector to match dim of action
        sshape = state.shape
        ashape = action.shape

        print(sshape)
        print(ashape)

        if len(sshape) < 3 and len(ashape) == 3:
            state = tf.broadcast_to(state, [ashape[0], sshape[0], sshape[1]])
        
        inputs = tf.squeeze(tf.concat([state, action], axis=1), -1)

        print(inputs.shape)

        init = self.dense(inputs, self.model_vars["first"])
        second = self.dense(init, self.model_vars["second"])
        return tf.expand_dims(self.final(second, self.model_vars["final"]), -1)

    def final(self, inputs, weights):
        return tf.matmul(inputs, weights)

    def dense(self, inputs, weights):
        return tf.nn.leaky_relu( tf.matmul(inputs, weights), alpha=self.leaky_relu_alpha)

    def getWeights(self, shape, name):
        return tf.Variable(self.initializer(shape, dtype=tf.float64), name=name, trainable=True, dtype=tf.float64)

