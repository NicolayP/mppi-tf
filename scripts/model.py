import numpy as np
import yaml

from nn_model import NNModel
from point_mass_model import PointMassModel

def nn(model_dic):
    return NNModel(dt=model_dic['dt'], 
                   state_dim=model_dic['state_dim'], 
                   action_dim=model_dic['action_dim'],
                   name=model_dic['name'])


def pm(model_dic):
    return PointMassModel(mass=model_dic["mass"],
                          dt=model_dic['dt'], 
                          state_dim=model_dic['state_dim'], 
                          action_dim=model_dic['action_dim'],
                          name=model_dic['name'])

def auv(model_dic):
    raise NotImplementedError()


def getCost(model_file):

    switcher = {
        "point_mass": pm,
        "neural_net": nn,
        "auv": auv
    }

    with open(model_file) as file:
        model = yaml.load(file, Loader=yaml.FullLoader)
        model_type = task['model']
        getter = switcher.get(model_type, lambda: "invalid model type, check\
                    spelling, supporter are: point_mass, neural_net, auv")
        return getter(model)
