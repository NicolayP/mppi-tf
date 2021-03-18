from cost_base import CostBase
import numpy as np
import tensorflow as tf


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

class ElipseCost(CostBase):
    def __init__(self, lam, sigma, tau, a, b, center_x, center_y, radius, speed, m_state, m_vel):
        CostBase.__init__(self, lam, sigma, tau)
        self.a = a
        self.b = b
        self.cx = center_x
        self.cy = center_y
        self.r = radius
        self.gv = speed
        self.mx = tf.cast(m_state, tf.float64)
        self.mv = tf.cast(m_vel, tf.float64)

    def state_cost(self, scope, state):
        x = tf.slice(state, [0, 0, 0], [-1, 1, -1])
        y = tf.slice(state, [0, 1, 0], [-1, 1, -1])
        vx = tf.slice(state, [0, 2, 0], [-1, 1, -1])
        vy = tf.slice(state, [0, 3, 0], [-1, 1, -1])
        v = tf.sqrt(tf.pow(vx, 2) + tf.pow(vy, 2))

        diffx = tf.divide(tf.math.subtract(x, self.cx, name="diff"), self.a)
        diffy = tf.divide(tf.math.subtract(y, self.cy, name="diff"), self.b)
        d = tf.abs(tf.pow(diffx, 2) + tf.pow(diffy, 2) - self.r)
        dv = tf.pow(v - self.gv, 2)
        return tf.add(tf.math.multiply(self.mx, d), tf.math.multiply(self.mv, dv))


def main():
    x_sam = 1001
    y_sam = 1001
    coord_x = np.linspace(-5, 5, x_sam)
    coord_y = np.linspace(-5, 5, y_sam)
    xv, yv = np.meshgrid(coord_x, coord_y, sparse=False, indexing='ij')
    x = np.reshape(xv, (-1, 1))
    y = np.reshape(yv, (-1, 1))
    vx = np.zeros(x.shape)
    vy = np.zeros(y.shape)
    nptensor = np.stack([x, y, vx, vy], axis=1)
    tensor = tf.convert_to_tensor(nptensor, dtype=tf.float64)

    Sigma=np.array([[1., 0., 0.],
                    [0., 1., 0.],
                    [0., 0., 1.]])

    tau = 1
    a = 4
    b = 3
    cx = 1
    cy = 2
    r = 2
    ms = 1
    ma = 1
    gv = 10
    cost = ElipseCost(1, Sigma, tau, a, b, cx, cy, r, gv, ms, ma)
    c = cost.state_cost("main", tensor).numpy()
    cv = c.reshape((x_sam, y_sam))

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(xv, yv, cv, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)


    # Add a color bar which maps values to colors.
    ax.set_zlim(80, 110)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


if __name__ == '__main__':
    main()
