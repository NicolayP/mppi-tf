import tensorflow as tf
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu_devices[0], True)

from tensorflow.python.ops import summary_ops_v2
from cpprb import ReplayBuffer

import numpy as np
from cost_base import CostBase
from model_base import ModelBase
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import os
from shutil import copyfile
import scipy.signal
from logger import addItem, log_control

class ControllerBase(object):

    def __init__(self,
                 model,
                 cost,
                 k=1,
                 tau=1,
                 dt=0.01,
                 s_dim=1,
                 a_dim=1,
                 lam=1.,
                 upsilon=1.,
                 sigma=np.array([]),
                 init_seq=np.array([]),
                 normalize_cost=False,
                 log=False,
                 config_file=None,
                 task_file=None,
                 debug=False):
        # TODO: Check parameters
        self.k = k
        self.tau = tau
        self.dt = dt
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.lam = lam
        self.upsilon = upsilon
        self.normalize_cost = normalize_cost
        self.log=log
        self.debug=debug


        self.sigma = tf.convert_to_tensor(sigma, dtype=tf.float64)

        if init_seq.size == 0:
            self.action_seq = np.zeros((self.tau, self.a_dim, 1))
        else:
            self.action_seq = init_seq

        self.model = model
        self.cost = cost

        self.mass_init = self.model.getMass()

        self.buffer_size = 264
        self.batch_size = 32

        self.rb = ReplayBuffer(self.buffer_size,
                               env_dict={"obs": {"shape": (self.s_dim, 1)},
                                         "act": {"shape": (self.a_dim, 1)},
                                         "rew": {},
                                         "next_obs": {"shape": (self.s_dim, 1)},
                                         "done": {}})

        self.train_step = 0
        self.writer = None
        if self.log or self.debug:
            stamp = datetime.now().strftime("%Y.%m.%d-%H:%M:%S")
            path = '../graphs/python/'
            if self.debug:
                path = os.path.join(path, 'debug')
            logdir = os.path.join(path,
            model.getName(),
            "k" + str(k),
            "T" + str(tau),
            "L" + str(lam),
            stamp)

            self.writer = tf.summary.create_file_writer(logdir)

            self.error_step = 0
            self.cost_step = 0
            #self.save_graph()

            self.summary_name = ["x", "y", "z"]

            if config_file is not None:
                conf_dest = os.path.join(logdir, "config.yaml")
                task_dest = os.path.join(logdir, "task.yaml")
                copyfile(config_file, conf_dest)
                copyfile(task_file, task_dest)


    def save_graph(self):
        state=np.zeros((self.s_dim, 1))
        with self.writer.as_default():
            graph = self._next.get_concrete_function(state, self.action_seq).graph # get graph from function
            summary_ops_v2.graph(graph.as_graph_def()) # visualize


    def save(self, x, u, x_next):
        self.rb.add(obs=x, act=u, rew=0, next_obs=x_next, done=False)


    def plot_speed(self, v_gt, v, u, m, m_l, dt):
        y = (v_gt - v)
        x_pred = np.linspace(-4, 4, 100)
        y_pred = x_pred*dt/m
        y_pred_l = x_pred*dt/m_l
        plt.scatter(u, y)
        plt.plot(x_pred, y_pred, x_pred, y_pred_l)
        plt.show()


    def train(self):
        if self.rb.get_stored_size() < 32 or self.model.isTrained():
            return

        epochs = 500
        for e in range(epochs):
            sample = self.rb.sample(self.batch_size)
            #sample = self.rb.get_all_transitions()
            gt = sample['next_obs']
            x = sample['obs']
            u = sample['act']
            self.model.train_step(gt, x, u, self.train_step*epochs + e, self.writer, self.log)

        self.train_step += 1

        '''
        sample = self.rb.get_all_transitions()
        gt = sample['next_obs']
        x = sample['obs']
        u = sample['act']
        v_gt = np.squeeze(gt[:, 1, :], -1)
        v = np.squeeze(x[:, 1, :], -1)
        u_plot = np.squeeze(u, -1)
        '''


    @tf.function
    def _next(self, state, action_seq, normalize_cost=False):
        with tf.name_scope("Controller") as cont:
            state=tf.convert_to_tensor(state, dtype=tf.float64, name=cont)
            return self.buildGraph(cont, state, action_seq, normalize=normalize_cost)


    def next(self, state):
        return_dict = self._next(state, self.action_seq, self.normalize_cost)
        print(return_dict.keys())
        # FIRST GUESS window_length = 5, we don't want to filter out to much since we expect smooth inputs, need to be played around with.
        # FIRST GUESS polyorder = 3, think that a 3rd degree polynome is enough for this.

        return_dict["action_seq"] = np.expand_dims(scipy.signal.savgol_filter(return_dict["action_seq"].numpy()[:, :, 0], 29, 9, deriv=0, delta=1.0, axis=0), axis=-1)
        self.action_seq = return_dict["action_seq"]

        if self.log:
            log_control(self.writer, return_dict)

        return return_dict["next"]


    def beta(self, scope, cost):
        # shapes: in [k, 1, 1]; out [1, 1]
        return tf.reduce_min(cost, 0)


    def normArg(self, scope, cost, beta, normalize=False):
        shift = tf.math.subtract(cost, beta)

        if normalize:
            max = tf.reduce_max(shift, 0)
            return tf.divide(shift, max)
        return shift


    def expArg(self, scope, arg):
        # shapes: in [k, 1, 1], [1, 1]; out [k, 1, 1]
        return tf.math.multiply(np.array([-1./self.lam]),
                                arg)


    def exp(self, scope, arg):
        # shapes: in [k, 1, 1]; out [k, 1, 1]
        return tf.math.exp(arg)


    def nabla(self, scope, arg):
        # shapes: in [k, 1, 1]; out [k, 1, 1]
        return tf.math.reduce_sum(arg, 0)


    def weights(self, scope, arg, nabla):
        # shapes: in [k, 1, 1], [1, 1]; out [k, 1, 1]
        return tf.realdiv(arg, nabla)


    def weightedNoise(self, scope, weights, noises):
        # shapes: in [k, 1, 1], [k, tau, a_dim, 1]; out [tau, a_dim, 1]
        return tf.math.reduce_sum(tf.math.multiply(tf.expand_dims(weights, -1), noises), 0)


    def prepareAction(self, scope, actions, timestep):
        # shapes: in [tau, a_dim, 1]; out [a_dim, 1]
        return tf.squeeze(tf.slice(actions, [timestep, 0, 0], [1, -1, -1]), 0)


    def prepareNoise(self, scope, noises, timestep):
        # shapes: in [k,, tau, a_dim, 1]; out [k, a_dim, 1]
        return tf.squeeze(tf.slice(noises, [0, timestep, 0, 0], [-1, 1, -1, -1]), 1)


    def update(self, scope, cost, noises, normalize=False):
        return_dict = {}
        # shapes: in [k, 1, 1], [k, tau, a_dim, 1]; out [tau, a_dim, 1]
        with tf.name_scope("Beta"):
            beta = self.beta(scope, cost)
        with tf.name_scope("Expodential_arg"):
            arg = self.normArg(scope, cost, beta, normalize=normalize)
            exp_arg = self.expArg(scope, arg)
        with tf.name_scope("Expodential"):
            exp = self.exp(scope, exp_arg)
        with tf.name_scope("Nabla"):
            nabla = self.nabla(scope, exp)
        with tf.name_scope("Weights"):
            weights = self.weights(scope, exp, nabla)
        with tf.name_scope("Weighted_Noise"):
            weighted_noises = self.weightedNoise(scope, weights, noises)
        with tf.name_scope("Sequence_update"):
            update = tf.add(self.action_seq, weighted_noises)

        return_dict["weights"]=weights
        return_dict["nabla"]=nabla
        return_dict["arg"]=arg
        return_dict["weighted_noises"]=weighted_noises
        return_dict["update"]=update
        return return_dict


    def shift(self, scope, action_seq, init, length):
        # shapes: in [tau, a_dim, 1], [x, a_dim, 1], scalar; out [tau-len + x, a_dim, 1]
        remain = tf.slice(action_seq, [length, 0, 0], [self.tau-length, -1, -1])
        return tf.concat([remain, init], 0)


    def getNext(self, scope, current, length):
        # shapes: in [tau, a_dim, 1], scalar; out [scalar, a_dim, 1]
        return tf.slice(current, [0, 0, 0], [length, -1, -1])


    def advanceGoal(self, scope, next):
        self.cost.setGoal(next)


    def buildGraph(self, scope, state, action_seq, normalize=False):
        return_dict = {}
        with tf.name_scope(scope) as scope:
            with tf.name_scope("random") as rand:
                noises = self.buildNoise(rand)
            with tf.name_scope("Rollout") as roll:
                cost_dict = self.buildModel(roll, state, noises, action_seq)
            with tf.name_scope("Update") as up:
                update_dict = self.update(up, cost_dict["cost"], noises, normalize=normalize)
            with tf.name_scope("Next") as n:
                next = self.getNext(n, update_dict["update"], 1)
            with tf.name_scope("shift_and_init") as si:
                init = self.initZeros(si, 1)
                action_seq = self.shift(si, action_seq, init, 1)

        return_dict["noises"] = noises
        return_dict["action_seq"] = action_seq
        return_dict["next"] = next
        return_dict = {**return_dict, **cost_dict, **update_dict}

        return return_dict


    def buildModel(self, scope, state, noises, action_seq):

        return_dict = {}

        cost = tf.zeros([self.k, 1, 1], dtype=tf.float64)
        paths = []

        for i in range(self.tau):
            with tf.name_scope("Prepare_data_" + str(i)) as pd:
                action = self.prepareAction(pd, action_seq, i)
                noise = self.prepareNoise(pd, noises, i)
                to_apply = tf.add(action, noise, name="to_apply")
            with tf.name_scope("Step_" + str(i)) as s:
                next_state = self.model.buildStepGraph(s, state, to_apply)
            with tf.name_scope("Cost_" + str(i)) as c:
                tmp_dict = self.cost.build_step_cost_graph(c, next_state, action, noise)
                cost_dict = self.cost.add_cost(c, tmp_dict, return_dict)
            state = next_state

            paths.append(tf.expand_dims(state, 1))
        paths = tf.concat(paths, 1)

        with tf.name_scope("terminal_cost") as s:
            f_cost_dict = self.cost.build_final_step_cost_graph(s, next_state)
        with tf.name_scope("Rollout_cost"):
            sample_costs_dict = self.cost.add_cost(c, f_cost_dict, cost_dict)

        return_dict["paths"] = paths
        return_dict = {**return_dict, **sample_costs_dict}
        return return_dict


    def buildNoise(self, scope):
        # scales: in []; out [k, tau, a_dim, 1]
        rng = tf.random.normal(shape=(self.k, self.tau, self.a_dim, 1),
                               stddev=1.,
                               mean=0.,
                               dtype=tf.float64,
                               seed=1)

        return tf.linalg.matmul(self.sigma*self.upsilon, rng)


    def initZeros(self, scope, size):
        # shape: out [size, a_dim, 1]
        return tf.zeros([size, self.a_dim, 1], dtype=tf.float64)


def main():

    k = 5
    tau = 2
    mass = 1.
    dt = 0.01
    s_dim = 2
    a_dim = 1
    lam = 1.

    sigma = np.array([[1]])
    goal = np.array([[1.], [0.]])
    Q = np.array([[1., 0.], [0., 1.]])




    model = ModelBase(mass, dt, s_dim, a_dim)
    cost = CostBase(lam, sigma, goal, Q)


    cont = ControllerBase(model=model,
                          cost=cost,
                          k=k,
                          tau=tau,
                          dt=dt,
                          mass=mass,
                          s_dim=s_dim,
                          a_dim=a_dim,
                          sigma=sigma)
    state = np.array([[[0.], [0.]]])

    writer = tf.summary.create_file_writer('../graphs/python')

    with writer.as_default():
        graph = cont.next.get_concrete_function(state).graph # get graph from function
        summary_ops_v2.graph(graph.as_graph_def()) # visualize

if __name__ == '__main__':
    main()
