from .models.nn_model import NNAUVModel, NNModel
from .models.point_mass_model import PointMassModel
from .models.auv_model import AUVModel

import numpy as np

def nn(model_dic, samples, dt, state_dim, action_dim, name, paramFile=None):
    return NNModel(state_dim=state_dim,
                   action_dim=action_dim,
                   name=name,
                   weightFile=paramFile)

def auv_nn(modelDict, samples, dt, state_dim, action_dim, name, paramFile=None):
    return NNAUVModel(k=samples,
                      stateDim=state_dim,
                      actionDim=action_dim,
                      mask=np.array(modelDict["mask"]),
                      weightFile=paramFile)

def pm(model_dic, samples, dt, state_dim, action_dim, name, paramFile=None):
    return PointMassModel(mass=model_dic["mass"],
                          dt=dt, 
                          state_dim=state_dim, 
                          action_dim=action_dim,
                          name=name)

def auv(modelDict, samples, dt, stateDim, actionDim, name, paramFile=None):
    return AUVModel(inertialFrameId=modelDict['frame_id'],
                    actionDim=actionDim,
                    name=name,
                    k=samples,
                    dt=dt,
                    parameters=modelDict)


def get_model(model_dict, samples, dt, state_dim, action_dim, name, paramFile=None):

    switcher = {
        "point_mass": pm,
        "neural_net": nn,
        "auv": auv,
        "auv_nn": auv_nn
    }

    model_type = model_dict['type']
    getter = switcher.get(model_type, lambda: "invalid model type, check\
                spelling, supporter are: point_mass, neural_net, auv")
    return getter(model_dict, samples, dt, state_dim, action_dim, name, paramFile)
