import tensorflow as tf


class ToSE3Mat(tf.Module):
    def __init__(self):
        self.pad = tf.constant([[[0., 0., 0., 1.]]], dtype=tf.float64)

    def forward(self, x):
        k = x.shape[0]
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
        self.pad = tf.constant([[[0., 0., 0., 1.]]], dtype=tf.float64)

    def forward(self, M, tau):
        return M @ self.exp(tau)

    def exp(self, tau):
        k = tau.shape[0]
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
        theta = tf.linalg.norm(theta_vec)
        skewT = self.skew.forward(theta_vec)
        a = tf.eye(3, dtype=tf.float64)
        b = (1-tf.cos(theta))/tf.pow(theta, 2) * skewT
        c = (theta - tf.sin(theta))/tf.pow(theta, 3) * tf.pow(skewT, 2)
        return a + b + c


class SO3int(tf.Module):
    def __init__(self, skew=None):
        if skew is None:
            self.skew = Skew()
        else:
            self.skew = skew

    def forward(self, R, tau):
        return R @ self.exp(tau)

    def exp(self, tau):
        theta = tf.linalg.norm(tau, axis=1)
        u = tau/theta[:, None]
        skewU = self.skew.forward(u)
        a = tf.eye(3, dtype=tf.float64)
        b = tf.sin(theta)[:, None, None]*skewU
        c = (1-tf.cos(theta))[:, None, None]*tf.pow(skewU, 2)
        return a + b + c


class Skew(tf.Module):
    def __init__(self):
        self.e1 = tf.constant([
            [0., 0., 0.],
            [0., 0., -1.],
            [0., 1., 0.]
        ], dtype=tf.float64)

        self.e2 = tf.constant([
            [0., 0., 1.],
            [0., 0., 0.],
            [-1., 0., 0.]
        ], dtype=tf.float64)

        self.e3 = tf.constant([
            [0., -1., 0.],
            [1., 0., 0.],
            [0., 0., 0.]
        ], dtype=tf.float64)

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