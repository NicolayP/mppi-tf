import torch
from torch.utils.tensorboard import SummaryWriter
import pypose as pp
import numpy as np
import pandas as pd

from scripts.utils.utils import parse_param, save_param, get_device
from scripts.models.nn_auv_v2 import AUVTrajV2, AUVPROXYDeltaV
from scripts.training.loss_fct import TrajLoss
from scripts.training_v2.training_utils import get_datasets, traj_eval, train_v2


import os
import argparse
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(prog="DEEP-AUV",
                                     description="Trains AUV models with pypose.")
    parser.add_argument("-p", "--parameters", type=str,
                        help="Path to a yaml file containing the training parameters.\
                        the file will be copied to the log directory.",
                        default=None)
    parser.add_argument("-g", "--gpu", type=int,
                        help="Chose GPU number. Automatically uses a GPU if available",
                        default=None)
    args = parser.parse_args()
    return args

def training(parameters, gpu):
    log_path = parameters["log"]["path"]
    if parameters["log"]["stamped"]:
        stamp = datetime.now().strftime("%d.%m.%Y_%H:%M:%S")
        log_path = os.path.join(log_path, stamp)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    save_param(os.path.join(log_path, "parameters.yaml"), parameters)

    ckpt_dir = os.path.join(log_path, "train_ckpt")
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    ckpt_steps = parameters["log"]["ckpt_steps"]

    datasets = get_datasets(parameters["dataset_params"])
    writer = SummaryWriter(log_path)

    device = get_device(cpu=True)
    if gpu is not None:
        device = get_device(gpu=gpu)

    model = AUVTrajV2(parameters["model"], dt=0.1, limMax=None, limMin=None).to(device)
    loss = TrajLoss(parameters["loss"]["traj"],
                    parameters["loss"]["vel"],
                    parameters["loss"]["dv"]).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=parameters["optim"]["lr"])
    epochs = parameters["optim"]["epochs"]

    # dv = model.auv_step.dv_pred

    # writer.add_graph(dv, [pose, vel, seq[:, 0:1]], verbose=True)

    #model.auv_step.dv_pred = proxy

    # print("Pose:    ", type(pose), " ", pose.shape)
    # print("Vel:     ", type(vel), " ", vel.shape)
    # print("Actions: ", type(seq), " ", seq.shape)

    # print("poses:   ", type(pose_target), " ", pose_target.shape)
    # print("vels:    ", type(vel_target), " ", vel_target.shape)
    # print("Dvs:     ", type(dv_target), " ", dv_target.shape)

    # pose_pred, vel_pred, dv_pred = model(pose, vel, seq)

    # print("poses:   ", type(pose_pred), " ", pose_pred.shape)
    # print("vels:    ", type(vel_pred), " ", vel_pred.shape)
    # print("Dvs:     ", type(dv_pred), " ", dv_pred.shape)

    train_v2(datasets, model, loss, optim, writer, epochs, device, ckpt_dir, ckpt_steps)
    pass

def main():
    args = parse_args()

    params = parse_param(args.parameters)
    training(params, args.gpu)

    pass

if __name__ == "__main__":
    main()