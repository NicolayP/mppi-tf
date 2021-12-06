from .src.models.nn_model import NNAUVModel, NNModel
from .src.models.point_mass_model import PointMassModel
from .src.models.auv_model import AUVModel



def nn(model_dic, samples, dt, state_dim, action_dim, name):
    return NNModel(state_dim=state_dim,
                   action_dim=action_dim,
                   name=name)

def auv_nn(modelDict, samples, dt, state_dim, action_dim, name):
    return NNAUVModel()

def pm(model_dic, samples, dt, state_dim, action_dim, name):
    return PointMassModel(mass=model_dic["mass"],
                          dt=dt, 
                          state_dim=state_dim, 
                          action_dim=action_dim,
                          name=name)

def auv(modelDict, samples, dt, stateDim, actionDim, name):
    return AUVModel(inertialFrameId=modelDict['frame_id'],
                    actionDim=actionDim,
                    name=name,
                    k=samples,
                    dt=dt,
                    parameters=modelDict)


def get_model(model_dict, samples, dt, state_dim, action_dim, name):

    switcher = {
        "point_mass": pm,
        "neural_net": nn,
        "auv": auv,
        "auv_nn": auv_nn
    }

    model_type = model_dict['type']
    getter = switcher.get(model_type, lambda: "invalid model type, check\
                spelling, supporter are: point_mass, neural_net, auv")
    return getter(model_dict, samples, dt, state_dim, action_dim, name)
