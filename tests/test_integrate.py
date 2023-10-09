import torch
import pypose as pp
import numpy as np
import pandas as pd

import os

from scripts.utils.utils import get_device, to_euler, plot_traj
from scripts.models.nn_auv import AUVDeltaVProxy, AUVTraj
from scripts.inputs.ModelInput import ModelInputPypose
from scripts.training.loss_fct import TrajLoss

import time
import matplotlib.pyplot as plt


def integrate(data_dir, tau=450, visu=True):
    tau = tau
    device = get_device(gpu=0)
    files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]

    trajs, trajs_plot, Bvels, Bdvs, acts = [], [], [], [], [],
    for file in files:
        df = pd.read_csv(os.path.join(data_dir, file))
        trajs.append(torch.Tensor(df[['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']].to_numpy())[None])
        trajs_plot.append(df[['x', 'y', 'z', 'roll', 'pitch', 'yaw']].to_numpy()[None])
        Bvels.append(torch.Tensor(df[['Bu', 'Bv', 'Bw', 'Bp', 'Bq', 'Br']].to_numpy())[None])
        Bdvs.append(torch.Tensor(df[['Bdu', 'Bdv', 'Bdw', 'Bdp', 'Bdq', 'Bdr']].to_numpy())[None])
        acts.append(torch.Tensor(df[['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']].to_numpy())[None])

    trajs = pp.SE3(torch.concat(trajs, dim=0)).to(device)
    Bvels = torch.concat(Bvels, dim=0).to(device)
    Bdvs = torch.concat(Bdvs, dim=0).to(device)
    acts = torch.concat(acts, dim=0).to(device)
    trajs_plot = np.concatenate(trajs_plot, axis=0)

    pred = AUVTraj().to(device)
    dv_pred_proxy = AUVDeltaVProxy(Bdvs).to(device)
    pred.step.dv_pred = dv_pred_proxy

    input = ModelInputPypose(len(files), 1).to(device)
    input.init_from_states(trajs[:, :1].to(device), Bvels[:, :1].to(device), acts[:, :0].to(device))
    
    s = time.time()
    # predict up to the last state as we don't know tau+1
    pred_trajs, pred_vs, pred_dvs = pred(input, acts[:, :tau-1])
    e = time.time()
    print(f"Prediciton time: {e-s}")

    loss_fc = TrajLoss().to(device)
    l = loss_fc(trajs[:, 1:tau], pred_trajs,
                Bvels[:, 1:tau], pred_vs,
                Bdvs[:, 0:tau-1], pred_dvs)
    print(f"Loss: {l}, {l.shape}")

    pred_trajs = pred_trajs.detach().cpu()
    pred_vs = pred_vs.detach().cpu()
    Bvels = Bvels.cpu()
    pred_dvs = pred_dvs.detach().cpu()
    Bdvs = Bdvs.cpu()

    pred_traj_euler = to_euler(pred_trajs[0])
    traj_euler = trajs_plot[0]

    if visu:
        s_col = {"x": 0, "y": 1, "z": 2, "roll": 3, "pitch": 4, "yaw": 5}
        plot_traj({"pred": pred_traj_euler, "gt": traj_euler[1:tau]}, s_col, tau, True, title="State")
        v_col = {"u": 0, "v": 1, "w": 2, "p": 3, "q": 4, "r": 5}
        plot_traj({"pred": pred_vs[0], "gt": Bvels[0, 1:tau]}, v_col, tau, True, title="Velocities")
        dv_col = {"Bdu": 0, "Bdv": 1, "Bdw": 2, "Bdp": 3, "Bdq": 4, "Bdr": 5}
        plot_traj({"pred": pred_dvs[0], "gt": Bdvs[0, :tau-1]}, dv_col, tau, True, title="Velocities Deltas")
        plt.show()