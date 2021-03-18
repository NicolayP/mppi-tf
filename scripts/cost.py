import numpy as np
import yaml
import os

from elipse_cost import ElipseCost
from static_cost import StaticCost

def static(task_dic, lam, sigma, tau):
    goal = np.expand_dims(np.array(task_dic['goal']), -1)
    Q = np.array(task_dic['Q'])
    return StaticCost(lam, sigma, goal, tau, Q)

def elipse(task_dic, lam, sigma, tau):
    a = task_dic['a']
    b = task_dic['b']
    center_y = task_dic['center_x']
    center_x = task_dic['center_y']
    radius = task_dic['radius']
    speed = task_dic['speed']
    m_state = task_dic['m_state']
    m_vel = task_dic['m_vel']
    return ElipseCost(lam, sigma, tau, a, b, center_x, center_y, radius, speed, m_state, m_vel)

def getCost(task_file, lam, sigma, tau):

    switcher = {
        "static": static,
        "elipse": elipse
    }

    with open(task_file) as file:
        task = yaml.load(file, Loader=yaml.FullLoader)
        cost_type = task['type']
        getter = switcher.get(cost_type, lambda: "invalid cost type, check spelling, supporter are: static, elipse")
        return getter(task, lam, sigma, tau)
