import torch
#from utils import dtype
import numpy as np
import scipy.signal
import yaml

from models.auv_torch import AUVFossen
from costs.static import Static
from controllers.mppi_base import ControllerBase
from observers.observer_base import ObserverBase

import time as t


def main():
    upsilon = 1.
    lam = 0.25
    gamma = 0.1
    samples = 3000
    #samples = 2
    horizon = 50
    #horizon = 3
    task_file = "/home/pierre/workspace/uuv_ws/src/mppi_ros/scripts/mppi_tf/config/tasks/static_cost_auv.yaml"
    with open(task_file, "r") as stream:
              task_dict = yaml.safe_load(stream)
    env_file = "/home/pierre/workspace/uuv_ws/src/mppi_ros/scripts/mppi_tf/config/envs/uuv_sim.default.yaml"
    with open(env_file, "r") as stream:
              env_dict = yaml.safe_load(stream)

    fossen_params_gt = dict()
    fossen_params_gt["mass"] = (1862.87, False)
    fossen_params_gt["volume"] = (1.8121303501945525, False)
    fossen_params_gt["cog"] = ([0., 0., 0.], False)
    fossen_params_gt["cob"] = ([0., 0., 0.3], False)
    fossen_params_gt["mtot"] = ([[2.64266e+03, -6.87730e+00, -1.03320e+02,  8.54260e+00, -1.65540e+02, -7.80330e+00],
                            [-6.87730e+00,  3.08487e+03,  5.12900e+01,  4.09440e+02, -5.84880e+00, 6.27260e+01],
                            [-1.03320e+02,  5.12900e+01,  5.52277e+03,  6.11120e+00, -3.86420e+02, 1.07740e+01],
                            [8.54260e+00,  4.09440e+02,  6.11120e+00,  1.06029e+03, -8.58700e+00, 5.44290e+01],
                            [-1.65540e+02, -5.84880e+00, -3.86420e+02, -8.58700e+00,  1.63689e+03, 1.48380e+00],
                            [-7.80330e+00,  6.27260e+01,  1.07750e+01,  5.44290e+01,  1.48380e+00, 9.15550e+02]], False)
    fossen_params_gt["linear_damping"] = ([-74.82, -69.48, -728.4, -268.8, -309.77, -105.], False)
    fossen_params_gt["quad_damping"] = ([-748.22, -992.53, -1821.01, -672, -774.44, -523.27], False)
    fossen_params_gt["linear_damping_forward"] = ([0., 0., 0., 0., 0., 0.], False)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    model = AUVFossen(fossen_params_gt)
    print("Model loaded")

    cost = Static(lam, gamma, upsilon, env_dict["noise"], task_dict["goal"], task_dict["Q"])
    print("Cost Loaded")

    observer = ObserverBase(logpath="./mppi_torch", k=samples)
    print("observer loaded")

    mppi = ControllerBase(model, cost, observer, samples, horizon, lam, upsilon, env_dict["noise"]).to(device)
    print("controller Loaded")
    
    s = torch.tensor([0., 0., 0.,
                      1., 0., 0., 0.,
                      0., 0., 0.,
                      0., 0., 0.])
    
    device = "cpu"
    mppi.to(device)
    for i in range (1):
        mppi(s.to(device))
    print("*"*5, f" {device} ", "*"*5)
    mppi.stats()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    mppi.to(device)
    for i in range (1):
        mppi(s.to(device))
    print("*"*5, f" {device} ", "*"*5)
    mppi.stats()

    mppi_cuda = torch.cuda.make_graphed_callables(mppi, (s,))
    for i in range (1):
        mppi_cuda(s.to(device))
    print("*"*5, f" {device} ", "*"*5)
    mppi_cuda.stats()

if __name__ == "__main__":
    main()
