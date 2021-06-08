import tensorflow as tf
import numpy as np
from model_base import ModelBase

class NNModel(ModelBase):
    '''
        Neural network based model class.
    '''
    def __init__(self, state_dim=2, action_dim=1, name="nn_model"):
        '''
            Neural network model constructor.

            - input:
            --------
                - state_dim: int the state space dimension.
                - action_dim: int the action space dimension.
                - name: string the model name.
        '''

        ModelBase.__init__(self, state_dim, action_dim, name)
        self.leaky_relu_alpha = 0.2
        self.initializer = tf.initializers.glorot_uniform()
        self.addModelVars("first", self.getWeights((state_dim+action_dim, 5), "first"))
        self.addModelVars("second", self.getWeights((5, 5), "second"))
        self.addModelVars("final", self.getWeights((5, state_dim), "final"))

    def buildStepGraph(self, scope, state, action):
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
        # expand and broadcast state vector to match dim of action
        sshape = state.shape
        ashape = action.shape


        if len(sshape) < 3 and len(ashape) == 3:
            state = tf.broadcast_to(state, [ashape[0], sshape[0], sshape[1]])
        
        inputs = tf.squeeze(tf.concat([state, action], axis=1), -1)

        init = self.dense(inputs, self.model_vars["first"])
        second = self.dense(init, self.model_vars["second"])
        return tf.expand_dims(self.final(second, self.model_vars["final"]), -1)

    def final(self, inputs, weights):
        '''
            Computes the output layer of the neural network.

            - input:
            --------
                - inputs: the input tensor. Shape []
                - weights: the weights tensor. Shape []

            - output:
            ---------
                - the output tensor w^T*x. Shape []
        '''
        return tf.matmul(inputs, weights)

    def dense(self, inputs, weights):
        '''
            Computes the middle layers of the nn. Leaky relu activated.

            - input:
            --------
                - inputs: the input tensor. Shape []
                - weights: the weights tensor. Shape []

            - output:
            ---------
                - the output tensor leaky_relu(w^T*x). Shape []
        '''

        return tf.nn.leaky_relu( tf.matmul(inputs, weights), alpha=self.leaky_relu_alpha)

    def getWeights(self, shape, name):
        '''
            initalize the weights of a given shape

            - input:
            --------
                - shape: list, the shape of the weights.
                - name: string, the name of the shapes.
        '''
        return tf.Variable(self.initializer(shape, dtype=tf.float64), name=name, trainable=True, dtype=tf.float64)

