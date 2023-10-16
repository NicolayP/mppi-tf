import torch
import wandb

from scripts.utils.utils import parse_param, save_param, get_device
from scripts.models.nn_auv_v2 import AUVTraj
from scripts.training.loss_fct import TrajLoss
from scripts.training_v2.training_utils import get_datasets, train_v2


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

    log_conf = parameters["logging"]
    datasets = get_datasets(parameters["dataset_params"])

    wandb.init(config=log_conf)

    device = get_device(cpu=True)
    if gpu is not None:
        device = get_device(gpu=gpu)

    model = AUVTraj(parameters["model"], dt=0.1, limMax=None, limMin=None).to(device)
    loss = TrajLoss(parameters["loss"]["traj"],
                    parameters["loss"]["vel"],
                    parameters["loss"]["dv"]).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=parameters["optim"]["lr"])
    epochs = parameters["optim"]["epochs"]

    wandb.watch(model, criterion=loss, log="all", log_freq=1000, log_graph=True)

    train_v2(datasets, model, loss, optim, epochs, device, ckpt_dir, ckpt_steps)
    wandb.save(ckpt_dir+"/chkpt*")


def main():
    args = parse_args()

    params = parse_param(args.parameters)
    training(params, args.gpu)

    pass


if __name__ == "__main__":
    main()