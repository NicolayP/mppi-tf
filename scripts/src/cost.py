import numpy as np

from .costs.static_cost import StaticCost, StaticQuatCost, StaticRotCost
from .costs.elipse_cost import ElipseCost, ElipseCost3D
from .costs.cost_base import CylinderObstacle


'''
    Cost function section
'''
def static(task_dic, lam, gamma, upsilon, sigma):
    goal = np.expand_dims(np.array(task_dic['goal']), -1)
    Q = np.array(task_dic['Q'])
    diag = task_dic["diag"]
    return StaticCost(lam, gamma, upsilon, sigma, goal, Q, diag)


def static_rot(task_dic, lam, gamma, upsilon, sigma):
    goal = np.expand_dims(np.array(task_dic['goal']), axis=-1)
    Q = np.array(task_dic['Q'])
    diag = task_dic["diag"]
    rep = task_dic['rep']
    return StaticRotCost(lam, gamma, upsilon, sigma, goal, Q, diag, rep)


def static_quat(task_dic, lam, gamma, upsilon, sigma):
    goal = np.expand_dims(np.array(task_dic['goal']), -1)
    Q = np.array(task_dic['Q'])
    diag = task_dic["diag"]
    return StaticQuatCost(lam, gamma, upsilon, sigma, goal, Q, diag)


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

'''
    Obstacle section
'''
def cylinder(obs_dict):
    p1 = np.array(obs_dict["p1"])
    p2 = np.array(obs_dict["p2"])
    r = np.array(obs_dict["r"])
    return CylinderObstacle(p1, p2, r)


'''
    Object instanciation
'''
def get_cost(task, lam, gamma, upsilon, sigma):

    switcher = {
        "static": static,
        "static_quat": static_quat,
        "static_rot": static_rot,
        "elipse": elipse,
        "elipse3d": elipse3d,
        "waypoints": waypoints
    }

    cost_type = task['type']

    getter = switcher.get(cost_type, lambda: "invalid cost type, check\
                spelling, supporter are: static, elipse, elipse3d")
    cost = getter(task, lam, gamma, upsilon, sigma)

    obs_switcher = {
        "cylinder": cylinder
    }


    for obs in task['obs']:
        obs_getter = obs_switcher.get(task["obs"][obs]["type"], lambda: "invalid obstacle type,\
            supported are: cylinder")
        new_obs = obs_getter(task["obs"][obs])
        cost.add_obstacle(new_obs)
    
    return cost