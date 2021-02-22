import tensorflow as tf
import numpy as np
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt





def blockDiag(vec, pad, dim):
    vec_np = np.array([])
    for h in range(dim):

        tmp = np.array([])
        for w in range(dim):
            if w==0 and h==0:
                tmp = vec
            elif w==0:
                tmp = pad
            elif w==h:
                tmp = np.hstack([tmp, vec])
            else:
                tmp = np.hstack([tmp, pad])

        if h==0:
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

class ModelBase(object):
    def __init__(self, mass=1, dt=0.1, state_dim=2, act_dim=1, name="point_mass"):
        self.dt = dt
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.mass = tf.Variable([[mass]], name="mass", trainable=True, dtype=tf.float64)
        with tf.name_scope("Const") as c:
            self.create_const(c)
        self.optimizer = tf.optimizers.Adam()

        self.name= name


    def getMass(self):
        return self.mass.numpy()[0]


    def create_const(self, scope):
        a = np.array([[1., self.dt],[0., 1.]])
        a_pad = np.array([[0, 0], [0, 0]])
        a_np = blockDiag(a, a_pad, int(self.state_dim/2))
        self.A = tf.constant(a_np, dtype=tf.float64, name="A")

        b = np.array([[(self.dt*self.dt)/2.], [self.dt]])
        b_pad = np.array([[0], [0]])
        b_np = blockDiag(b, b_pad, self.act_dim)
        self.B = tf.constant(b_np, dtype=tf.float64, name="B")


    def buildLossGraph(self, gt, x, a):
        pred = self.buildStepGraph("train", x, a)
        tmp = tf.math.squared_difference(pred, gt)
        return tf.reduce_mean(tf.math.squared_difference(pred, gt), name="Loss")


    def train_step(self, gt, x, a, step, writer, log):

        gt = tf.convert_to_tensor(gt, dtype=tf.float64)
        x = tf.convert_to_tensor(x, dtype=tf.float64)
        a = tf.convert_to_tensor(a, dtype=tf.float64)

        with tf.GradientTape() as tape:
            tape.watch(self.mass)
            current_loss = self.buildLossGraph(gt, x, a)
        grads = tape.gradient(current_loss, [self.mass])
        self.optimizer.apply_gradients(list(zip(grads, [self.mass])))
        if log:
            with writer.as_default():
                tf.summary.scalar("mass", self.mass.numpy()[0,0], step=step)
                tf.summary.scalar("loss", current_loss.numpy(), step=step)


    def buildStepGraph(self, scope, state, action):
        with tf.name_scope("Model_Step"):
            return tf.add(self.buildFreeStepGraph("free", state),
                          self.buildActionStepGraph("action", action))


    def buildFreeStepGraph(self, scope, state):
        with tf.name_scope(scope):
            return tf.linalg.matmul(self.A, state, name="A_x")


    def buildActionStepGraph(self, scope, action):
        with tf.name_scope(scope):
            return tf.linalg.matmul(tf.divide(self.B, self.mass, name="B"), action, name="B_u")


    def predict(self, state, action):
        '''
        Performs one step prediction for prediction error
        '''
        return self.buildStepGraph("step", state, action)

    def getName(self):
        return self.name

def plot(v_gt, v, u, m, m_l, dt):
    y = (v_gt - v)
    x_pred = np.linspace(-4, 4, 100)
    y_pred = x_pred*dt/m
    y_pred_l = x_pred*dt/m_l
    plt.scatter(u, y)
    plt.plot(x_pred, y_pred, x_pred, y_pred_l)
    plt.show()

def main():
    # Set up logging.
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = '../graphs/python/%s' % stamp
    writer = tf.summary.create_file_writer(logdir)

    state = tf.convert_to_tensor(np.array([[[0.5], [0.1]]]), dtype=tf.float64)
    action = tf.convert_to_tensor(np.array([[[0.1]]]), dtype=tf.float64)

    tf.summary.trace_on(graph=True, profiler=False)
    model = ModelBase("model", 0.5, 0.1, 2, 1)
    next = model.buildStepGraph("model", state, action)

    with writer.as_default():
        tf.summary.trace_export(
            name="model_trace",
            step=0,
            profiler_outdir=logdir
        )

    gt, x, a = get_data("../data.csv")
    gt_plot = np.squeeze(gt, -1)
    x_plot = np.squeeze(x, -1)
    a_plot = np.squeeze(a, -1)
    print(model.mass.numpy()[0, 0])

    epochs = 500
    for e in range(epochs):
        model.train_step(gt, x, a, e, writer)

    plot(gt_plot[:, -1], x_plot[:, -1], a_plot, 0.25, model.mass.numpy()[0], 0.1)

if __name__ == '__main__':
    main()
