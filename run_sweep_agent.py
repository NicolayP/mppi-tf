import torch
import wandb
import os, argparse
from datetime import datetime


from scripts.utils.utils import parse_param, save_param, get_device
from scripts.models.nn_auv import AUVTraj
from scripts.training.loss_fct import TrajLoss
from scripts.training.training_utils import get_datasets, train


def training():
    run = wandb.init()
    config = wandb.config
    parameters = {"log": config.log,
                  "loss": config.loss,
                  "optim": config.optim,
                  "model": config.model,
                  "dataset_params": config.dataset_params}

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

    device = get_device(cpu=False)

    model = AUVTraj(parameters["model"], dt=0.1, limMax=None, limMin=None).to(device)
    loss = TrajLoss(parameters["loss"]["pose"],
                    parameters["loss"]["vel"],
                    parameters["loss"]["dv"]).to(device)
    val_loss = TrajLoss(1., 0., 0.).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=parameters["optim"]["lr"])
    epochs = parameters["optim"]["epochs"]

    wandb.watch(model, criterion=loss, log="all", log_freq=1000, log_graph=True)

    train(datasets, model, loss, optim, epochs, device, ckpt_dir, ckpt_steps, val_loss)
    wandb.save(ckpt_dir+"/chkpt*")


training()