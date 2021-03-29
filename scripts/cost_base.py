import tensorflow as tf
import numpy as np
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from logger import addItem

# TODO: compute all constants without tensorflow. Out of the graph computation.
class CostBase(object):
    def __init__(self, lam, gamma, upsilon, sigma, tau):

        with tf.name_scope("Cost_setup") as cs:
            self.lam = lam
            self.gamma = gamma
            self.upsilon = upsilon
            self.tau = tau
            s = tf.convert_to_tensor(sigma, dtype=tf.float64)
            self.invSig = tf.linalg.inv(s)


    def build_step_cost_graph(self, scope, state, action, noise):
        return_dict = {}

        with tf.name_scope("step_cost") as s:
            state_cost_dict = self.state_cost(s, state)
            action_cost = self.action_cost(s, action, noise)
            step_cost = tf.add(state_cost_dict["state_cost"], action_cost, name="add")


        return_dict["action_cost"]=action_cost
        return_dict["cost"]=step_cost
        return_dict = {**return_dict, **state_cost_dict}
        return return_dict

    def add_cost(self, scope, input_dict, current_dict):
        for key in input_dict:
            if key in current_dict.keys():
                current_dict[key] = tf.add(current_dict[key], input_dict[key], "tmp_cost")
            else:
                current_dict[key] = input_dict[key]
        return current_dict

    def build_final_step_cost_graph(self, scope, state):
        return self.state_cost(scope, state)


    def state_cost(self, scope, state):
        raise NotImplementedError


    def action_cost(self, scope, action, noise):
        rhs_noise_cost = tf.linalg.matmul(self.invSig, noise, name="rhs_noise")
        rhs_action_cost = tf.linalg.matmul(self.invSig, action, name="rhs_action")
        # \u^{T}_t \Sigma^{-1} \epsilon_t
        mix_cost = tf.linalg.matmul(action, rhs_noise_cost, transpose_a=True, name="mix")
        # \epsilon^{T}_t \Sigma^{-1} \epsilon_t
        noise_cost = tf.linalg.matmul(noise, rhs_noise_cost, transpose_a=True, name="noise")
        # \u^{T}_t \Sigma^{-1} \u_t
        action_cost = tf.linalg.matmul(action, rhs_action_cost, transpose_a=True, name="action")

        # \gamma [action_cost + 2mix_cost]
        control_cost = tf.math.multiply(tf.cast(self.gamma, dtype=tf.float64),
            tf.add(action_cost, tf.math.multiply(tf.cast(2., dtype=tf.float64), mix_cost)))
        # \lambda(1-\upsilon^{-1})noise_cost
        pert_cost = tf.math.multiply(tf.cast(self.lam*(1.-1./self.upsilon), dtype=tf.float64),
            noise_cost)
        # \frac{1}{2}*(control_cost+pert_cost)
        return tf.math.multiply(tf.cast(0.5, dtype=tf.float64), tf.add(control_cost, pert_cost))


    def draw_goal(self):
        raise NotImplementedError


    def dist(self, state):
        raise NotImplementedError
