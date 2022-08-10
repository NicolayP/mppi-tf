from .models.nn_model import NNAUVModel, NNModel, NNAUVModelSpeed
from .models.point_mass_model import PointMassModel
from .models.auv_model import AUVModel

import numpy as np
import tensorflow as tf

def nn(modelDict, samples, dt, state_dim, action_dim, name, paramFile=None):
    return NNModel(modelDict=modelDict,
                   state_dim=state_dim,
                   action_dim=action_dim,
                   limMax=limMax,
                   limMin=limMin,
                   name=name,
                   weightFile=paramFile)

def auv_nn(modelDict, samples, dt, state_dim, action_dim, name, paramFile=None):
    return NNAUVModel(modelDict=modelDict,
                      k=samples,
                      stateDim=state_dim,
                      actionDim=action_dim,
                      mask=np.array(modelDict["mask"]),
                      weightFile=paramFile)

def auv_nn_speed(modelDict, samples, dt, state_dim, action_dim, name, paramFile=None):
    return NNAUVModelSpeed(modelDict=modelDict,
                           k=samples,
                           stateDim=state_dim,
                           actionDim=action_dim,
                           mask=np.array(modelDict["mask"]),
                           weightFile=paramFile)

def pm(modelDict, samples, dt, state_dim, action_dim, name, paramFile=None):
    return PointMassModel(modelDict=modelDict,
                          mass=modelDict["mass"],
                          dt=dt, 
                          state_dim=state_dim, 
                          action_dim=action_dim,
                          limMax=limMax,
                          limMin=limMin,
                          name=name)

def auv(modelDict, samples, dt, stateDim, actionDim, name, paramFile=None):
    return AUVModel(modelDict=modelDict,
                    inertialFrameId=modelDict['frame_id'],
                    actionDim=actionDim,
                    name=name,
                    k=samples,
                    dt=dt,
                    parameters=modelDict)


def get_model(model_dict, samples, dt, state_dim, action_dim,
              name, paramFile=None):

    switcher = {
        "point_mass": pm,
        "neural_net": nn,
        "auv": auv,
        "auv_nn": auv_nn,
        "auv_nn_speed": auv_nn_speed
    }
    model_type = model_dict['type']
    getter = switcher.get(model_type, lambda: "invalid model type, check\
                spelling, supporter are: point_mass, neural_net, auv")
    return getter(model_dict, samples, dt, state_dim, action_dim,
                  name, paramFile)


def copy_model(model):
    modelDict = model._modelDict

    return get_model(modelDict,
                     tf.Variable(model._k.value()),
                     model._dt,
                     model._stateDim,
                     model._actionDim,
                     model._name)
