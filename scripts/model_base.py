import tensorflow as tf
import numpy as np
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt



class ModelBase(object):
    '''
        Model base class for the MPPI controller.
        Every model should inherit this class.
    '''
    def __init__(self, state_dim=2,
                 action_dim=1, k=1, name="model", inertial_frame_id="world"):
        '''
            Model constructor. 
            
            - input:
            --------
                - state_dim: int. the state space dimension.
                - action_dim: int. the action space dimension.
                - name: string. the model name. Used for logging.
        '''
        self.k = k
        self.inertial_frame_id = inertial_frame_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model_vars = {}

        self.optimizer = tf.optimizers.Adam(learning_rate=0.5)
        self.current_loss = None
        self.name = name

    def addModelVars(self, name, var):
        '''
            Add model variables to the dictionnary of variables.
            Used for logging and learning (if enabled)
            
            - input:
            --------
                - name: string. Unique variable name for identification.
                - var: the variable object. Tensorflow variable with trainable enabled.

            - output:
            ---------
                None.
        '''

        self.model_vars[name] = var

    def buildLossGraph(self, gt, x, a):
        '''
            Computes the loss function for a given batch of samples. Can be overwritten by child
            classes. Standard is l2 loss function.

            - input:
            --------
                - gt: the ground truth tensor. Shape [batch_size, s_dim, 1]
                - x: the previous state. Shape [batch_size, s_dim, 1]
                - a: the action to apply. Shape [batch_size, a_dim, 1]

            - output:
            ---------
                - the loss function between one step prediction "model(x, a)" and gt.
        '''

        pred = self.build_step_graph("train", x, a)
        return tf.reduce_mean(tf.math.squared_difference(pred, gt),
                              name="Loss")

    def is_trained(self):
        '''
            Tells whether the model is trained or not. 

            - output:
            ---------
                - bool, true if training finished.
        '''
        if (self.current_loss is not None) and self.current_loss < 5e-5:
            return True
        return False

    def train_step(self, gt, x, a, step=None, writer=None, log=False):
        '''
            Performs one step of training. 
            
            - input:
            --------
                - gt. the ground truth tensor. Shape [batch_size, s_dim, 1]
                - x. the input state tensor. Shape [batch_size, s_dim, 1]
                - a. the input action tensor. Shape [batch_size, a_dim, 1]
                - step. Int, The current learning step.
                - writer. Tensorflow summary writer. 
                - log. bool. If true, logs learning info in tensorboard.

            - output:
            ---------
                None
        '''

        gt = tf.convert_to_tensor(gt, dtype=tf.float64)
        x = tf.convert_to_tensor(x, dtype=tf.float64)
        a = tf.convert_to_tensor(a, dtype=tf.float64)
        with tf.GradientTape() as tape:
            for key in self.model_vars:
                tape.watch(self.model_vars[key])
            self.current_loss = self.buildLossGraph(gt, x, a)

        grads = tape.gradient(self.current_loss, list(self.model_vars.values()))
        self.optimizer.apply_gradients(list(zip(grads, list(self.model_vars.values()))))

        if log:
            with writer.as_default():
                for key in self.model_vars:
                    if tf.size(self.model_vars[key]).numpy() == 1:
                        tf.summary.scalar("training/{}".format(key),
                                          self.model_vars[key].numpy()[0, 0],
                                          step=step)
                    else:
                        tf.summary.histogram("training/{}".format(key),
                                                self.model_vars[key],
                                                step=step)

                tf.summary.scalar("training/loss",
                                  self.current_loss.numpy(),
                                  step=step)

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
        raise NotImplementedError

    def predict(self, state, action):
        '''
            Performs one step prediction for error visualisation.

            - input:
            --------
                - state: the state tensor. Shape [1, s_dim, 1]
                - action: the action tensor. Shape [1, a_dim, 1]

            - output:
            ---------
                - the predicted next state. Shape [1, s_dim, 1]
        '''
        self.k = 1
        return self.build_step_graph("step", state, action)

    def get_name(self):
        '''
            Get the name of the model.

            - output:
            ---------
                - String, the name of the model
        '''
        return self.name

    def get_stats(self):
        raise NotImplementedError

    def get_state_dim(self):
        return self.state_dim

    def get_action_dim(self):
        return self.action_dim

    def set_k(self, k):
        self.k = k