import torch
#from utils import dtype

from observers.observer_base import ObserverBase
from utils import load_param, get_device
from getters import get_controller, get_model, get_cost

def main():
    #######################
    ###  Configuration  ###
    #######################
    samples = 2000
    tau = 50
    lam = 0.5
    upsilon = 1.
    gamma = 0.1

    device = get_device()

    model_config = "../config/models/rexrov2.default.yaml"
    model_dict = load_param(model_config)

    cost_config = "../config/tasks/static_cost_auv.yaml"
    cost_dict = load_param(cost_config)

    cont_config = "../config/controller/state.default.yaml"
    cont_dict = load_param(cont_config)

    sigma = cont_dict["noise"]
    dt = cont_dict["dt"]

    ############################
    ### Instanciate objects: ###
    ############################
    cost = get_cost(cost_dict, lam, gamma, upsilon, sigma).to(device)
    print("Cost loaded")

    model = get_model(model_dict, dt, 0., 0.).to(device)
    print("Model loaded")

    observer = ObserverBase(log=False, k=samples)
    print("observer loaded")

    controller = get_controller(cont_dict, model, cost, observer,
                                samples, tau, lam, upsilon, sigma).to(device)
    print("Controller loaded")

    s = torch.tensor([0., 0., 0.,
                      0., 0., 0., 1.,
                      0., 0., 0.,
                      0., 0., 0.])[..., None].to(device)

    # Call the controller:
    a = controller(s)
    print("Next aciton: ", a.shape)

if __name__ == "__main__":
    main()
