import tensorflow as tf
from tensorflow.python.ops import summary_ops_v2
from cpprb import ReplayBuffer

import numpy as np
from cost_base import CostBase
from model_base import ModelBase
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

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
                 sigma=np.array([]),
                 init_seq=np.array([]),
                 save_graph=False):
        # TODO: Check parameters
        self.k = k
        self.tau = tau
        self.dt = dt
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.lam = lam


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

        stamp = datetime.now().strftime("%Y.%m.%d-%H:%M:%S")
        print(stamp)
        logdir = '../graphs/python/%s' % stamp

        self.writer = tf.summary.create_file_writer(logdir)

        self.train_step = 0
        self.error_step = 0

        if save_graph:
            self.save_graph()


    def save_graph(self):
        state=np.zeros((self.s_dim, 1))
        with writer.as_default():
            graph = self.next.get_concrete_function(state).graph # get graph from function
            summary_ops_v2.graph(graph.as_graph_def()) # visualize

    def save(self, x, u, x_next, log):
        self.rb.add(obs=x, act=u, rew=0, next_obs=x_next, done=False)
        if log:
            x_next_pred = self.model.predict(x, u).numpy()[0]
            error = np.squeeze(x_next - x_next_pred, -1)
            with self.writer.as_default():
                tf.summary.scalar("position_error", error[0], step=self.error_step)
                tf.summary.scalar("speed_error", error[1], step=self.error_step)
            self.error_step += 1

    def plot_speed(self, v_gt, v, u, m, m_l, dt):
        y = (v_gt - v)
        x_pred = np.linspace(-4, 4, 100)
        y_pred = x_pred*dt/m
        y_pred_l = x_pred*dt/m_l
        plt.scatter(u, y)
        plt.plot(x_pred, y_pred, x_pred, y_pred_l)
        plt.show()

    def train(self, log):
        epochs = 500
        print(self.train_step)
        for e in range(epochs):
            sample = self.rb.sample(self.batch_size)
            #sample = self.rb.get_all_transitions()
            gt = sample['next_obs']
            x = sample['obs']
            u = sample['act']
            self.model.train_step(gt, x, u, self.train_step*epochs + e, self.writer, log)

        self.train_step += 1

        sample = self.rb.get_all_transitions()
        gt = sample['next_obs']
        x = sample['obs']
        u = sample['act']
        v_gt = np.squeeze(gt[:, 1, :], -1)
        v = np.squeeze(x[:, 1, :], -1)
        u_plot = np.squeeze(u, -1)
        #self.plot_speed(v_gt, v, u_plot, self.mass_init, self.model.getMass(), self.dt)

    def setGoal(self, goal):
        # shape [s_dim, s_dim]
        self.goal = goal

    @tf.function
    def next(self, state):
        with tf.name_scope("Controller") as cont:
            state=tf.convert_to_tensor(state, dtype=tf.float64, name=cont)
            return self.buildGraph(cont, state)


    def beta(self, scope, cost):
        # shapes: in [k, 1, 1]; out [1, 1]
        return tf.reduce_min(cost, 0)


    def expArg(self, scope, cost, beta):
        # shapes: in [k, 1, 1], [1, 1]; out [k, 1, 1]
        return tf.math.multiply(np.array([-1./self.lam]),
                                tf.math.subtract(cost, beta))


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


    def update(self, scope, cost, noises):
        # shapes: in [k, 1, 1], [k, tau, a_dim, 1]; out [tau, a_dim, 1]
        with tf.name_scope("Beta"):
            beta = self.beta(scope, cost)
        with tf.name_scope("Expodential_arg"):
            exp_arg = self.expArg(scope, cost, beta)
        with tf.name_scope("Expodential"):
            exp = self.exp(scope, exp_arg)
        with tf.name_scope("Nabla"):
            nabla = self.nabla(scope, exp)
        with tf.name_scope("Weights"):
            weight = self.weights(scope, exp, nabla)
        with tf.name_scope("Weighted_Noise"):
            weighted_noise = self.weightedNoise(scope, weight, noises)
        with tf.name_scope("Sequence_update"):
            return tf.add(self.action_seq, weighted_noise)


    def shift(self, scope, action_seq, init, length):
        # shapes: in [tau, a_dim, 1], [x, a_dim, 1], scalar; out [tau-len + x, a_dim, 1]
        remain = tf.slice(action_seq, [length, 0, 0], [self.tau-length, -1, -1])
        return tf.concat([remain, init], 0)


    def getNext(self, scope, current, length):
        # shapes: in [tau, a_dim, 1], scalar; out [scalar, a_dim, 1]
        return tf.slice(current, [0, 0, 0], [length, -1, -1])


    def buildGraph(self, scope, state):
        with tf.name_scope(scope) as scope:
            with tf.name_scope("random") as rand:
                noises = self.buildNoise(rand)
            with tf.name_scope("Rollout") as roll:
                cost = self.buildModel(roll, state, noises)
            with tf.name_scope("Update") as up:
                self.action_seq = self.update(up, cost, noises)

            with tf.name_scope("Next") as n:
                next = self.getNext(n, self.action_seq, 1)
            with tf.name_scope("shift_and_init") as si:
                init = self.initZeros(si, 1)
                self.action_seq = self.shift(si, self.action_seq, init, 1)
            return next


    def buildModel(self, scope, state, noises):
        cost = tf.zeros([self.k, 1, 1], dtype=tf.float64)
        for i in range(self.tau):
            with tf.name_scope("Prepare_data_" + str(i)) as pd:
                action = self.prepareAction(pd, self.action_seq, i)
                noise = self.prepareNoise(pd, noises, i)
                to_apply = tf.add(action, noise, name="to_apply")
            with tf.name_scope("Step_" + str(i)) as s:
                next_state = self.model.buildStepGraph(s, state, to_apply)
            with tf.name_scope("Cost_" + str(i)) as c:
                tmp = self.cost.build_step_cost_graph(c, next_state, action, noise)
                cost = tf.add(cost, tmp, name="tmp_cost")
            state = next_state

        with tf.name_scope("terminal_cost") as s:
            f_cost = self.cost.build_final_step_cost_graph(s, next_state)
        with tf.name_scope("Rollout_cost"):
            return tf.add(f_cost, cost, name="final_cost")


    def buildNoise(self, scope):
        # scales: in []; out [k, tau, a_dim, 1]
        rng = tf.random.normal(shape=(self.k, self.tau, self.a_dim, 1),
                               stddev=1.,
                               mean=0.,
                               dtype=tf.float64,
                               seed=1)

        return tf.linalg.matmul(self.sigma, rng)


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


    print(cont.next(state))

    writer = tf.summary.create_file_writer('../graphs/python')

    with writer.as_default():
        graph = cont.next.get_concrete_function(state).graph # get graph from function
        summary_ops_v2.graph(graph.as_graph_def()) # visualize

if __name__ == '__main__':
    main()