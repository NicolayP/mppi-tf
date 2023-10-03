import torch
from torch.utils.tensorboard import SummaryWriter
import pypose as pp
import numpy as np
import pandas as pd

from scripts.training.loss_fct import TrajLoss
from scripts.training.datasets import DatasetList3D
from scripts.utils.utils import parse_param, read_files, save_param, get_device
from scripts.training.learning_utils import train

from scripts.getters import get_model

import os
import random
import argparse
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(prog="ML-AUV",
                                     description="Trains AUV model with pypose")
    parser.add_argument("-p", "--parameters", type=str,
                        help="Path to a yaml file containing the training parameters.\
                        the file will be copied to the log directory.",
                        default=None)
    parser.add_argument("-g", "--gpu", type=int,
                        help="Choose gpu number. Automatically uses a GPU if available",
                        default=0)

    args = parser.parse_args()
    return args

def training(parameters, gpu):
    #-----------------------------------------------------#
    #----------------------- SETUP -----------------------#
    #-----------------------------------------------------#
    dp = parameters["dataset_params"]
    files = [f for f in os.listdir(dp["dir"]) if os.path.isfile(os.path.join(dp["dir"], f))]
    nb_files = min(len(files), dp["samples"])

    random.shuffle(files)
    files = random.sample(files, nb_files)
    train_size = int(dp["split"]*len(files))

    train_files = files[:train_size]
    val_files = files[train_size:]

    train_params = parameters["data_loader_params"]
    stats_file = os.path.join(dp['dir'], "stats", "stats.yaml")
    stats = parse_param(stats_file)
    ds = []

    for mode, files in zip(["train", "val"], [train_files, val_files]):
        dfs = read_files(dp["dir"], files, mode)
        ds.append(torch.utils.data.DataLoader(
            DatasetList3D(
                dfs,
                steps=dp["steps"],
                #history=dp["history"],
                v_frame=dp["v_frame"],
                dv_frame=dp["dv_frame"],
                act_normed=dp["act_normed"],
                se3=dp["se3"],
                out_normed=dp["out_normed"],
                stats=stats
            ), **train_params))

    log_path = parameters["log"]["path"]
    if parameters["log"]["stamped"]:
        stamp = datetime.now().strftime("%d.%m.%Y_%H:%M:%S")
        log_path = os.path.join(log_path, stamp)

    if not os.path.exists(log_path):
        os.makedirs(log_path)

    save_param(os.path.join(log_path, "parameters.yaml"), parameters)

    ckpt_dir = os.path.join(log_path, "train_ckpt")
    if not os.path.exist(ckpt_dir):
        os.makedirs(ckpt_dir)
    ckpt_steps = parameters["log"]["ckpt_steps"]

    #-----------------------------------------------------#
    ##----------------------- TRAIN ----------------------#
    #-----------------------------------------------------#

    writer = SummaryWriter(log_path)
    device = get_device(True, gpu)
    model = get_model(parameters["model"]).to(device)
    loss = TrajLoss(parameters["loss"]["traj"],
                    parameters["loss"]["vel"],
                    parameters["loss"]["dv"]).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=parameters["optim"]["lr"])
    epochs = parameters["optim"]["epochs"]

    train(ds, model, loss, optim, writer, epochs, device, ckpt_dir, ckpt_steps)

def main():
    args = parse_args()

    params = parse_param(args.parameters)
    training(params, args.gpu)


if __name__ == "__main__":
    main()