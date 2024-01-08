import torch
import pypose as pp
from scripts.utils.utils import tdtype

from scripts.utils.utils import load_param, get_device
from scripts.getters import get_controller, get_model, get_cost

from scripts.utils.onnx_utils import load_onnx_model, create_onnx_bound_model
import numpy as np


def timed(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    restul = fn()
    end.record()
    torch.cuda.synchronize()
    return restul, start.elapsed_time(end) / 1000

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def main():
    #######################
    ###  Configuration  ###
    #######################
    samples = 20
    tau = 5
    lam = 0.5
    upsilon = 1.
    gamma = 0.1

    device = get_device()

    model_config = "config/models/rnn.default.yaml" # RNN/LSTM SETUP
    #model_config = "config/models/rexrov2.default.yaml" # FOSSEN SETUP
    model_dict = load_param(model_config)

    cost_config = "config/tasks/static_pypose.default.yaml" # RNN/LSTM SETUP
    #cost_config = "config/tasks/static.default.yaml" # FOSSEN SETUP
    cost_dict = load_param(cost_config)
    cost_dict["Q"] = np.array(cost_dict["Q"])

    cont_config = "config/controller/pypose.default.yaml" # RNN/LSTM SETUP
    #cont_config = "config/controller/state.default.yaml" # FOSSEN SETUP
    cont_dict = load_param(cont_config)

    sigma = cont_dict["noise"]
    dt = cont_dict["dt"]


    pose = pp.identity_SE3(1).to(device) # RNN/LSTM SETUP
    #pose = torch.tensor([[0., 0., 0., 0., 0., 0., 1.]], dtype=tdtype).to(device) # FOSSEN SETUP
    vel = torch.tensor([[0., 0., 0., 0., 0., 0.]], dtype=tdtype).to(device)

    ############################
    ### Instanciate normal controller: ###
    ############################
    cost = get_cost(cost_dict, lam, gamma, upsilon, sigma).to(device)
    print("Cost loaded")
    cost = cost.to(device)

    model = get_model(model_dict, dt, 0., 0.).to(device)
    print("Model loaded")

    # NOTE FROM PIERRE: Why is this instruction here?
    model = model.to(device)

    #observer = ObserverBase(log=False, k=samples)
    observer = None
    print("Observer loaded")

    controller = get_controller(cont_dict, model, cost, observer,
                                samples, tau, lam, upsilon, sigma).to(device)
    print("Controller loaded")


    print("\n"+"~" * 10)
    print("eager:", timed(lambda: controller(pose, vel))[1])
    N_ITERS = 10
    eager_times = []
    compile_times = []
    for i in range(N_ITERS):
        _, eager_time = timed(lambda: controller(pose, vel))
        eager_times.append(eager_time)
        print(f"eager eval time {i}: {eager_time}")

    print("~" * 10)

    # ############################
    # ### Instanciate scripted controller: ###
    # ############################
    # cost = get_cost(cost_dict, lam, gamma, upsilon, sigma).to(device)

    # model = get_model(model_dict, dt, 0., 0.).to(device)

    # observer = None
    # scripted_controller = get_controller(cont_dict, model, cost, observer,
    #                             samples, tau, lam, upsilon, sigma).to(device)

    # scripted_controller = torch.jit.script(scripted_controller)

    # ################# UNCOMMENT TO EXPORT ONNX MODEL #################
    # # torch.onnx.export(controller,
    # #         args=s,
    # #         f="model.onnx",
    # #         export_params=True,
    # #         opset_version=15, # Check different opsets. Must be >=11
    # #         input_names=["state"],
    # #         output_names=["action"]
    # #         )
    
    # ort_session = load_onnx_model("model.onnx", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    # ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(s)}

    # print("\n"+"~" * 10)

    # print("eager:", timed(lambda: controller(pose, vel))[1])
    # print("compile:", timed(lambda: scripted_controller(pose, vel))[1])
    # print("onnx:", timed(lambda:ort_session.run(None, ort_inputs))[1])

    # print("\n"+"~" * 10)

    # N_ITERS = 10
    # eager_times = []
    # compile_times = []
    # for i in range(N_ITERS):
    #     _, eager_time = timed(lambda: controller(pose, vel))
    #     eager_times.append(eager_time)
    #     print(f"eager eval time {i}: {eager_time}")

    # print("~" * 10)

    # compile_times = []
    # for i in range(N_ITERS):
    #     _, compile_time = timed(lambda: scripted_controller(s))
    #     compile_times.append(compile_time)
    #     print(f"compile eval time {i}: {compile_time}")
    # print("~" * 10)

    # onnx_times = []
    # for i in range(N_ITERS):
    #     _, compile_time = timed(lambda:ort_session.run(None, ort_inputs))
    #     onnx_times.append(compile_time)
    #     print(f"onnx eval time {i}: {compile_time}")
    # print("~" * 10)


    # eager_med = np.median(eager_times)
    # compile_med = np.median(compile_times)
    # onnx_med = np.median(onnx_times)
    # speedup_comp = eager_med / compile_med
    # speedup_onnx = eager_med / onnx_med
    # print(f"(eval) eager median: {eager_med}, compile median: {compile_med}, speedup: {speedup_comp}x")
    # print(f"(eval) eager median: {eager_med}, onnx median: {onnx_med}, speedup: {speedup_onnx}x")
    # print("~" * 10)

if __name__ == "__main__":
    main()
