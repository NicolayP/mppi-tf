import torch
from utils import dtype

from observers.observer_base import ObserverBase
from utils import load_param, get_device
from getters import get_controller, get_model, get_cost
import numpy as np


def timed(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    restul = fn()
    end.record()
    torch.cuda.synchronize()
    return restul, start.elapsed_time(end) / 1000


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


    s = torch.tensor([0., 0., 0.,
                      0., 0., 0., 1.,
                      0., 0., 0.,
                      0., 0., 0.], dtype=dtype)[..., None].double().to(device)
    
    ############################
    ### Instanciate normal controller: ###
    ############################
    cost = get_cost(cost_dict, lam, gamma, upsilon, sigma).to(device)
    print("Cost loaded")

    cost = cost

    model = get_model(model_dict, dt, 0., 0.).to(device)
    print("Model loaded")

    model = model

    observer = ObserverBase(log=False, k=samples)
    print("Observer loaded")

    controller = get_controller(cont_dict, model, cost, observer,
                                samples, tau, lam, upsilon, sigma).to(device)
    print("Controller loaded")


    ############################
    ### Instanciate scripted controller: ###
    ############################
    cost = get_cost(cost_dict, lam, gamma, upsilon, sigma).to(device)
    cost = torch.jit.script(cost)

    model = get_model(model_dict, dt, 0., 0.).to(device)
    model = torch.jit.script(model)

    observer = ObserverBase(log=False, k=samples)
    scripted_controller = get_controller(cont_dict, model, cost, observer,
                                samples, tau, lam, upsilon, sigma).to(device)

    scripted_controller = torch.jit.script(scripted_controller,s )

    print("\n"+"~" * 10)

    print("eager:", timed(lambda: controller(s))[1])
    print("compile:", timed(lambda: scripted_controller(s))[1])


    N_ITERS = 10
    eager_times = []
    compile_times = []
    for i in range(N_ITERS):
        _, eager_time = timed(lambda: controller(s))
        eager_times.append(eager_time)
        print(f"eager eval time {i}: {eager_time}")

    print("~" * 10)

    compile_times = []
    for i in range(N_ITERS):
        _, compile_time = timed(lambda: scripted_controller(s))
        compile_times.append(compile_time)
        print(f"compile eval time {i}: {compile_time}")
    print("~" * 10)


    eager_med = np.median(eager_times)
    compile_med = np.median(compile_times)
    speedup = eager_med / compile_med
    print(f"(eval) eager median: {eager_med}, compile median: {compile_med}, speedup: {speedup}x")
    print("~" * 10)

if __name__ == "__main__":
    main()
