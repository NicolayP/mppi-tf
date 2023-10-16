from scripts.controllers.mppi_base import ControllerBase
from scripts.models.fossen import AUVFossen
from scripts.models.nn_auv import AUVStep, AUVTraj
from scripts.costs.static import Static

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

def fossen(model_dict, dt, limMax, limMin, training=False):
    return AUVFossen(model_dict, dt)

def ml_model(model_dict, dt, limMax, limMin, training=False):
    if training:
        return AUVTraj(model_dict)
    return AUVStep(model_dict, dt)

def get_model(model_dict, dt, limMax, limMin):
    switcher = {
        "auv_fossen": fossen,
        "auv_rnn": ml_model,
        "auv_lstm": ml_model,
        "auv_nn": ml_model,
    }
    model_type = model_dict["type"]
    # if training is true we need to return a AUV trajectory.
    # otherwise we return AUV Step object with the right 
    training = model_dict["training"]

    getter = switcher.get(model_type, lambda: "invalid model type, \
                          check spelling. Supported are: auv_fossen|auv_rnn|auv_lstm|auv_nn")

    return getter(
        model_dict=model_dict, dt=dt,
        limMax=limMax, limMin=limMin, training=training
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
