import numpy as np

from .costs.static_cost import StaticCost, StaticRotCost
from .costs.elipse_cost import ElipseCost, ElipseCost3D



def static(task_dic, lam, gamma, upsilon, sigma):
    goal = np.expand_dims(np.array(task_dic['goal']), -1)
    Q = np.array(task_dic['Q'])
    diag = task_dic["diag"]
    return StaticCost(lam, gamma, upsilon, sigma, goal, Q, diag)

def static_rot(task_dic, lam, gamma, upsilon, sigma):
    goal = np.expand_dims(np.array(task_dic['goal']), -1)
    Q = np.array(task_dic['Q'])
    diag = task_dic["diag"]
    rep = task_dic['rep']
    return StaticRotCost(lam, gamma, upsilon, sigma, goal, Q, diag, rep)


def elipse(task_dic, lam, gamma, upsilon, sigma):
    a = task_dic['a']
    b = task_dic['b']
    center_y = task_dic['center_x']
    center_x = task_dic['center_y']
    speed = task_dic['speed']
    m_state = task_dic['m_state']
    m_vel = task_dic['m_vel']
    return ElipseCost(lam, gamma, upsilon, sigma, a, b, center_x,
                      center_y, speed, m_state, m_vel)


def elipse3d(task_dic, lam, gamma, upsilon, sigma):
    a = task_dic['a']
    b = task_dic['b']
    center_y = task_dic['center_x']
    center_x = task_dic['center_y']
    speed = task_dic['speed']
    m_state = task_dic['m_state']
    m_vel = task_dic['m_vel']
    return ElipseCost3D(lam, gamma, upsilon, sigma, a, b, center_x,
                      center_y, -10., speed, 0., m_state, m_vel)


def waypoints(task_dict, lam, gamma, upsilon, sigma):
    waypoins = task_dict['waypoints']
    dist = task_dict['dist']
    return WaypointCost(lam, gamma, upsilon, sigma, waypoins, dist)


def get_cost(task, lam, gamma, upsilon, sigma):

    switcher = {
        "static": static,
        "static_rot": static_rot,
        "elipse": elipse,
        "elipse3d": elipse3d,
        "waypoints": waypoints
    }

    cost_type = task['type']

    getter = switcher.get(cost_type, lambda: "invalid cost type, check\
                spelling, supporter are: static, elipse, elipse3d")
    return getter(task, lam, gamma, upsilon, sigma)
