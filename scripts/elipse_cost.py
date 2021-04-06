from cost_base import CostBase
import numpy as np
import tensorflow as tf

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


class ElipseCost(CostBase):
    def __init__(self, lam, gamma, upsilon, sigma, tau, a, b, center_x, center_y, speed, m_state, m_vel):
        CostBase.__init__(self, lam, gamma, upsilon, sigma, tau)
        self.a = a
        self.b = b
        self.cx = center_x
        self.cy = center_y
        self.gv = speed
        self.mx = tf.cast(m_state, tf.float64)
        self.mv = tf.cast(m_vel, tf.float64)

    def state_cost(self, scope, state):
        return_dict = {}
        x = tf.slice(state, [0, 0, 0], [-1, 1, -1])
        y = tf.slice(state, [0, 2, 0], [-1, 1, -1])
        vx = tf.slice(state, [0, 1, 0], [-1, 1, -1])
        vy = tf.slice(state, [0, 3, 0], [-1, 1, -1])
        v = tf.sqrt(tf.pow(vx, 2) + tf.pow(vy, 2))
        diffx = tf.divide(tf.math.subtract(x, self.cx, name="diff"), self.a)
        diffy = tf.divide(tf.math.subtract(y, self.cy, name="diff"), self.b)
        d = tf.abs(tf.pow(diffx, 2) + tf.pow(diffy, 2) - 1)
        d = tf.math.multiply(self.mx, d)
        dv = tf.pow(v - self.gv, 2)
        dv = tf.math.multiply(self.mv, dv)
        state_cost = tf.add(d, dv)

        return_dict["speed_cost"]=dv
        return_dict["position_cost"]=d
        return_dict["state_cost"]=state_cost
        return return_dict

    def draw_goal(self):
        alpha = np.linspace(0, 2*np.pi, 1000)
        x = self.a*np.cos(alpha)
        y = self.b*np.sin(alpha)
        return x, y

    def dist(self, state):
        return_dict = {}
        x = state[0]
        vx = state[1]
        y = state[2]
        vy = state[3]
        v = np.sqrt(vx**2 + vy**2)
        x_dist = (((x-self.cx)/self.a)**2 + ((y-self.cy)/self.b)**2) - 1
        v_dist = np.abs(v-self.gv)
        return_dict["x_dist"] = x_dist[0]
        return_dict["v_dist"] = v_dist[0]
        return return_dict


def main():
    x_sam = 1001
    y_sam = 1001
    coord_x = np.linspace(-6, 6, x_sam)
    coord_y = np.linspace(-7, 7, y_sam)
    xv, yv = np.meshgrid(coord_x, coord_y, sparse=False, indexing='ij')
    x = np.reshape(xv, (-1, 1))
    y = np.reshape(yv, (-1, 1))
    vx = np.zeros(x.shape)
    vy = np.zeros(y.shape)
    nptensor = np.stack([x, vx, y, vy], axis=1)
    tensor = tf.convert_to_tensor(nptensor, dtype=tf.float64)

    Sigma=np.array([[1., 0., 0.],
                    [0., 1., 0.],
                    [0., 0., 1.]])

    tau = 1
    a = 2
    b = 1.5
    cx = 0
    cy = 0
    ms = 100
    ma = 5
    gv = 5
    cost = ElipseCost(1, 1, 1, Sigma, tau, a, b, cx, cy, gv, ms, ma)
    c = cost.state_cost("main", tensor)["state_cost"].numpy()
    cv = c.reshape((x_sam, y_sam))

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(xv, yv, cv,
                       linewidth=0, antialiased=False)


    # Add a color bar which maps values to colors.
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


if __name__ == '__main__':
    main()
