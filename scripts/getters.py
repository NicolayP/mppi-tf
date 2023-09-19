from controllers.mppi_base import ControllerBase
from models.fossen import AUVFossen
from models.rnn_auv import AUVStep, AUVTraj, AUVRNNDeltaV, AUVLSTMDeltaV, AUVNNDeltaV
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

def fossen(model_dict, dt, limMax, limMin, training=False):
    return AUVFossen(model_dict, dt)

def rnn(model_dict, dt, limMax, limMin, training=False):
    pass

def lstm(model_dict, dt, limMax, limMin, training=False):
    pass

def nn(model_dict, dt, limMax, limMin, training=False):
    pass

def get_model(model_dict, dt, limMax, limMin):
    switcher = {
        "auv_fossen": fossen,
        "auv_rnn": rnn,
        "auv_lstm": lstm,
        "auv_nn": nn,
    }
    model_type = model_dict["type"]
    # if training is true we need to return a AUV trajectory.
    # otherwise we return AUV Step object with the right 
    training = model_dict["training"]

    getter = switcher.get(model_type, lambda: "invalid model type, \
                          check spelling. Supported are: auv_fossen|auv_rnn")

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