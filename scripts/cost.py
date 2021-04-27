import numpy as np
import yaml

from mppi_tf.scripts.static_cost import StaticCost
from mppi_tf.scripts.elipse_cost import ElipseCost



def static(task_dic, lam, gamma, upsilon, sigma, tau):
    goal = np.expand_dims(np.array(task_dic['goal']), -1)
    Q = np.array(task_dic['Q'])
    return StaticCost(lam, gamma, upsilon, sigma, goal, tau, Q)


def elipse(task_dic, lam, gamma, upsilon, sigma, tau):
    a = task_dic['a']
    b = task_dic['b']
    center_y = task_dic['center_x']
    center_x = task_dic['center_y']
    speed = task_dic['speed']
    m_state = task_dic['m_state']
    m_vel = task_dic['m_vel']
    return ElipseCost(lam, gamma, upsilon, sigma, tau, a, b, center_x,
                      center_y, speed, m_state, m_vel)


def getCost(task, lam, gamma, upsilon, sigma, tau):

    switcher = {
        "static": static,
        "elipse": elipse
    }

    cost_type = task['type']
    getter = switcher.get(cost_type, lambda: "invalid cost type, check\
                spelling, supporter are: static, elipse")
    return getter(task, lam, gamma, upsilon, sigma, tau)
