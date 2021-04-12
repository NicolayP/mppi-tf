import tensorflow as tf
import numpy as np
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt



class ModelBase(object):
    def __init__(self, dt=0.1, state_dim=2,
                 action_dim=1, name="point_mass"):

        self.dt = dt
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model_vars = {}

        self.optimizer = tf.optimizers.Adam(learning_rate=0.5)
        self.current_loss = None
        self.name = name

    def addModelVars(self, name, var):
        self.model_vars[name] = var

    def buildLossGraph(self, gt, x, a):
        pred = self.buildStepGraph("train", x, a)
        return tf.reduce_mean(tf.math.squared_difference(pred, gt),
                              name="Loss")

    def isTrained(self):
        if (self.current_loss is not None) and self.current_loss < 5e-5:
            return True
        return False

    def train_step(self, gt, x, a, step, writer=None, log=False):

        gt = tf.convert_to_tensor(gt, dtype=tf.float64)
        x = tf.convert_to_tensor(x, dtype=tf.float64)
        a = tf.convert_to_tensor(a, dtype=tf.float64)

        with tf.GradientTape() as tape:
            for key in self.model_vars:
                tape.watch(self.model_vars[key])
            self.current_loss = self.buildLossGraph(gt, x, a)

        grads = tape.gradient(self.current_loss, self.model_vars.items())
        self.optimizer.apply_gradients(list(zip(grads, self.model_vars.items())))

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

    def buildStepGraph(self, scope, state, action):
        raise NotImplementedError

    def predict(self, state, action):
        '''
        Performs one step prediction for prediction error
        '''
        return self.buildStepGraph("step", state, action)

    def getName(self):
        return self.name

