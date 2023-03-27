import tensorflow as tf
import numpy as np
from ..misc.utile import assert_shape, dtype, plot_traj, traj_to_euler

class ToSE3Mat(tf.Module):
    def __init__(self):
        self.pad = tf.constant([[[0., 0., 0., 1.]]], dtype=dtype)

    def forward(self, x):
        k = tf.shape(x)[0]
        p = tf.expand_dims(x[:, :3], axis=-1)
        r = tf.reshape(x[:, 3:3+9], (-1, 3, 3))
        noHomo = tf.concat([r, p], axis=-1)
        pad = tf.broadcast_to(
                self.pad,
                [k, 1, 4],
        )
        homo = tf.concat([noHomo, pad], axis=-2)
        return homo


class SE3int(tf.Module):
    def __init__(self):
        self.skew = Skew()
        self.so3int = SO3int(self.skew)
        self.pad = tf.constant([[[0., 0., 0., 1.]]], dtype=dtype)
        self.eps = 1e-10

    def forward(self, M, tau):
        return M @ self.exp(tau)

    def exp(self, tau):
        k = tf.shape(tau)[0]
        rho_vec = tau[:, :3]
        theta_vec = tau[:, 3:]

        r = self.so3int.exp(theta_vec)
        p = self.v(theta_vec) @ tf.expand_dims(rho_vec, axis=-1)

        noHomo = tf.concat([r, p], axis=-1)
        pad = tf.broadcast_to(
                self.pad,
                [k, 1, 4],
        )
        homo = tf.concat([noHomo, pad], axis=-2)
        return homo

    def v(self, theta_vec):
        theta = tf.linalg.norm(theta_vec, axis=-1)
        mask = tf.cast(theta > self.eps, dtype=dtype)
        skewT = self.skew.forward(theta_vec)
        a = tf.eye(3, dtype=dtype)
        b = ((1-tf.cos(theta))/tf.pow(theta, 2))[:, None, None] * skewT
        c = ((theta - tf.sin(theta))/tf.pow(theta, 3))[:, None, None] * (skewT @ skewT)
        res = a + tf.math.multiply_no_nan((b + c), mask[:, None, None])
        return res


class SO3int(tf.Module):
    def __init__(self, skew=None):
        if skew is None:
            self.skew = Skew()
        else:
            self.skew = skew

    def forward(self, R, tau):
        return R @ self.exp(tau)

    def exp(self, tau):
        theta = tf.linalg.norm(tau, axis=-1) # [k,]
        u = tf.math.l2_normalize(tau, axis=-1) #[k, 3]
        skewU = self.skew.forward(u)
        a = tf.eye(3, dtype=dtype)
        b = tf.sin(theta)[:, None, None]*skewU
        c = (1-tf.cos(theta))[:, None, None]*(skewU @ skewU)
        return a + b + c


class Skew(tf.Module):
    def __init__(self):
        self.e1 = tf.constant([
            [0., 0., 0.],
            [0., 0., -1.],
            [0., 1., 0.]
        ], dtype=dtype)

        self.e2 = tf.constant([
            [0., 0., 1.],
            [0., 0., 0.],
            [-1., 0., 0.]
        ], dtype=dtype)

        self.e3 = tf.constant([
            [0., -1., 0.],
            [1., 0., 0.],
            [0., 0., 0.]
        ], dtype=dtype)

    def forward(self, vec):
        a = self.e1 * vec[:, 0, None, None]
        b = self.e2 * vec[:, 1, None, None]
        c = self.e3 * vec[:, 2, None, None]
        return a + b + c


class FlattenSE3(tf.Module):
    def __init__(self):
        pass

    def forward(self, M, vel):
        p = M[:, 0:3, 3]
        r = tf.reshape(M[:, 0:3, 0:3], (-1, 9))
        x = tf.concat([p, r, vel], axis=1)
        return x


def load_onnx_model(dir):
    tf_model = tf.saved_model.load(dir)
    f = tf_model.signatures["serving_default"]
    return f


def push_to_tensor(tensor, x):
    tmp = tf.expand_dims(x, axis=1) # shape [k, 1, dim, 1]
    return tf.concat([tensor[:, 1:], tmp], axis=1)


def rollout(model, init, seq, h, horizon, dev=False, debug=True):
    state = init
    pred = []
    if dev:
        Cvs = []
        Dvs = []
        gs = []
    for i in range(h, horizon+h):
        if dev:
            nextState, Cv, Dv, g = model.build_step_graph("foo", state, seq[:, i-h:i], dev)
            Cvs.append(Cv)
            Dvs.append(Dv)
            gs.append(g)
        else:
            nextState = model.build_step_graph("foo", state, seq[:, i-h:i])
        pred.append(nextState)
        state = push_to_tensor(state, nextState)
    traj = tf.concat(pred, axis=0)

    if dev:
        Cvs = tf.concat(Cvs, axis=0)
        Dvs = tf.concat(Dvs, axis=0)
        gs = tf.concat(gs, axis=0)
        return traj, Cvs, Dvs, gs
    return traj


def rand_rollout(models, histories, plotStateCols, plotActionCols, horizon, dir):
    trajs = {}
    seq = 5. * tf.random.normal(
        shape=(1, horizon+10, 6, 1),
        mean=0.,
        stddev=1.,
        dtype=dtype,
        seed=1
    )
    for model, h in zip(models, histories):
        init = np.zeros((1, h, 18, 1))
        rot = np.eye(3)
        init[:, :, 3:3+9, :] = np.reshape(rot, (9, 1))
        init = init.astype(np.float32)
        init = tf.convert_to_tensor(init, dtype=dtype)
        pred = rollout(model, init, seq, h, horizon)
        pred = np.squeeze(pred.numpy(), axis=-1)
        trajs[model.name + "_rand"] = traj_to_euler(pred, rep="rot")

    plot_traj(trajs, seq, histories, plotStateCols, plotActionCols, horizon, dir)

