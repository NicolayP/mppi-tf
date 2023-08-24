from controllers.mppi_base import ControllerBase
from models.auv_torch import AUVFossen
from models.rnn_auv import 
from costs.static import Static

import numpy as np

####################################
#       Controller seciton         #
####################################

def state(model, cost, observer, k, tau, lam, upsilon, sigma):
    return ControllerBase(model=model, cost=cost, observer=observer,
                          k=k, tau=tau, lam=lam, upsilon=upsilon, sigma=sigma)

def get_controller(cont_dict, model, cost, observer,
                   k, tau, lam, upsilon, sigma):
    switcher = {
        "state_controller": state,
    }
    controller_type = cont_dict["type"]
    getter = switcher.get(controller_type, lambda: "invalid controller type, \
                          check spelling. Supported are: lagged_controller")

    return getter(
        model=model, cost=cost, observer=observer, 
        k=k, tau=tau, lam=lam, upsilon=upsilon, sigma=sigma
    )

####################################
#          Model seciton           #
####################################

def auv(model_dict, dt, limMax, limMin):
    return AUVFossen(model_dict, dt)

def rnn(model_dict, dt, limMax, limMin):
    pass

def get_model(model_dict, dt, limMax, limMin):
    switcher = {
        "auv_fossen": auv,
        "auv_rnn": rnn,
    }
    model_type = model_dict["type"]
    getter = switcher.get(model_type, lambda: "invalid model type, \
                          check spelling. Supported are: auv_fossen|auv_rnn")

    return getter(
        model_dict=model_dict, dt=dt,
        limMax=limMax, limMin=limMin
    )


####################################
#           Cost seciton           #
####################################

def static(cost_dict, lam, gamma, upsilon, sigma):
    Q = np.array(cost_dict['Q'])
    goal = np.array(cost_dict['goal'])[..., None]
    diag = cost_dict['diag']
    return Static(lam, gamma, upsilon, sigma, goal, Q, diag)

def get_cost(cost_dict, lam, gamma, upsilon, sigma):
    switcher = {
        "static": static,
    }
    cost_type = cost_dict["type"]
    getter = switcher.get(cost_type, lambda: "invalid cost type, \
                          check spelling. Supported are: static|elipse")

    return getter(
        cost_dict=cost_dict, lam=lam, gamma=gamma, upsilon=upsilon, sigma=sigma
    )