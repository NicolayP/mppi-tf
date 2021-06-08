import numpy as np
import yaml

from nn_model import NNModel
from point_mass_model import PointMassModel
from auv_model import AUVModel

def nn(model_dic, samples, dt, state_dim, action_dim, name):
    return NNModel(state_dim=state_dim, 
                   action_dim=action_dim,
                   name=name)


def pm(model_dic, samples, dt, state_dim, action_dim, name):
    return PointMassModel(mass=model_dic["mass"],
                          dt=dt, 
                          state_dim=state_dim, 
                          action_dim=action_dim,
                          name=name)

def auv(model_dic, samples, dt, state_dim, action_dim, name):
    return AUVModel(inertial_frame_id=model_dic['frame_id'],
                    state_dim=state_dim,
                    action_dim=action_dim,
                    name=name,
                    k=samples,
                    dt=dt,
                    parameters=model_dic)


def getModel(model_dict, samples, dt, state_dim, action_dim, name):

    switcher = {
        "point_mass": pm,
        "neural_net": nn,
        "auv": auv
    }

    model_type = model_dict['type']
    getter = switcher.get(model_type, lambda: "invalid model type, check\
                spelling, supporter are: point_mass, neural_net, auv")
    return getter(model_dict, samples, dt, state_dim, action_dim, name)
