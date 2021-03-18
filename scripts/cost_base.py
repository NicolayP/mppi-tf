import tensorflow as tf
import numpy as np
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

# TODO: compute all constants without tensorflow. Out of the graph computation.
class CostBase(object):
    def __init__(self, lam, sigma, tau):

        with tf.name_scope("Cost_setup") as cs:
            self.lam = lam
            self.tau = tau
            s = tf.convert_to_tensor(sigma, dtype=tf.float64)
            self.invSig = tf.linalg.inv(s)


    def build_step_cost_graph(self, scope, state, action, noise):
        with tf.name_scope("step_cost") as s:
            state_cost = self.state_cost(s, state)
            action_cost = self.action_cost(s, action, noise)
            return tf.add(state_cost, action_cost, name="add"), state_cost, action_cost


    def build_final_step_cost_graph(self, scope, state):
        return self.state_cost(scope, state)


    def state_cost(self, scope, state):
        raise NotImplementedError


    def action_cost(self, scope, action, noise):
        noise_cost = tf.linalg.matmul(self.invSig, noise, name="noise")
        action_cost = tf.linalg.matmul(action, noise_cost, transpose_a=True, name="action")
        return tf.math.multiply(tf.cast(self.lam, dtype=tf.float64), action_cost)
