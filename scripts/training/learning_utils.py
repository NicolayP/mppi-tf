import torch
from torch.utils.data import DataLoader
import pypose as pp
import numpy as np
import pandas as pd
import os

from scipy.spatial.transform import Rotation as R
from scripts.utils.utils import tdtype, npdtype, to_euler, gen_imgs_3D, disable_tqdm, parse_param, read_files
from scripts.training.datasets import DatasetListModelInput
from scripts.inputs.ModelInput import ModelInputPypose
import random

import matplotlib.pyplot as plt

from tqdm import tqdm
import wandb


############################################
#                                          #
#            DATALOADER UTILS              #
#                                          #
############################################
def get_datasets(parameters):
    files = [f for f in os.listdir(parameters["dir"]) if os.path.isfile(os.path.join(parameters["dir"], f))]
    nb_files = min(len(files), parameters["samples"])

    random.shuffle(files)
    files = random.sample(files, nb_files)
    train_size = int(parameters["split"]*len(files))

    train_files = files[:train_size]
    val_files = files[train_size:]

    stats_file = os.path.join(parameters['dir'], "stats", "stats.yaml")
    stats = parse_param(stats_file)
    ds = []

    for mode, files in zip(["train", "val"], [train_files, val_files]):
        dfs = read_files(parameters["dir"], files, mode)
        ds.append(get_dataloader_model_input(dfs, parameters["steps"], parameters["history"],
                                             frame=parameters["frame"], 
                                             stats=stats,
                                             batch_size=parameters["batch_size"],
                                             shuffle=parameters["shuffle"],
                                             num_workers=parameters["num_workers"]))
    return ds


def get_dataloader_model_input(datafiles, steps, history, frame,
                               act_normed=True, se3=True, out_normed=True, stats=None,
                               batch_size=512, shuffle=True, num_workers=8):
    ds = DatasetListModelInput(
            data_list=datafiles,
            steps=steps,
            history=history,
            v_frame=frame,
            dv_frame=frame,
            act_normed=act_normed,
            se3=se3,
            out_normed=out_normed,
            stats=stats)
    dl = DataLoader(ds, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers)
    return dl


############################################
#                                          #
#                 LOGGING                  #
#                                          #
############################################

'''
    Computes the loss on an entire trajectory. If plot is true, it also plots the
    predicted trajecotry for different horizons.

    input:
    ------
        - dataset: torch.utils.data.Dataset with a methods called get_trajs() that
        returns full trajectory contained in the dataset.
        - model: the dynamical model used for predicitons.
        - loss: torch.function, the loss function used to measure the performance of the model.
        - tau: list of ints, the horizons we want to measure the performance on in increasing order.
        - writer: torch.summarywriter. Writer used to log the data
        - step: the current step in the training process used for logging.
        - device: string, the device to run the model on.
        - mode: string (default: "train") or val. Defines the mode in which this funciton is called.
        - plot: bool (default: False) if true, plots the first trajectory of the dataset as well as
            the on predicted by the model.
'''
def traj_loss(dataset, model, loss, tau, step, device, mode="train", plot=False):
    pose_init, vel_init, act_init, gt_trajs, gt_vels, gt_dvs, aciton_seqs = dataset.get_trajs()

    k = pose_init.shape[0]
    h = pose_init.shape[1]
    # Crop traj to desired size
    gt_trajs = gt_trajs[:, h:tau+h]
    gt_vels = gt_vels[:, h:tau+h]
    gt_dvs = gt_dvs[:, h:tau+h]

    model_input = ModelInputPypose(k, h).to(device)
    model_input.init_from_states(pose_init.to(device), vel_init.to(device), act_init.to(device))
    A = aciton_seqs[:, h:tau+h].to(device)

    pred_trajs, pred_vels, pred_dvs = model(model_input, A[:, :tau])

    losses = loss(
            pred_trajs[:, :tau], gt_trajs[:, :tau].to(device),
            pred_vels[:, :tau], gt_vels[:, :tau].to(device),
            pred_dvs[:, :tau], gt_dvs[:, :tau].to(device)
        )
    losses_split = [loss(
            pred_trajs[:, :tau], gt_trajs[:, :tau].to(device),
            pred_vels[:, :tau], gt_vels[:, :tau].to(device),
            pred_dvs[:, :tau], gt_dvs[:, :tau].to(device), split=True
        )]

    name = [["x", "y", "z", "vec_x", "vec_y", "vec_z"],
            ["u", "v", "w", "p", "q", "r"],
            ["du", "dv", "dw", "dp", "dq", "dr"]]
    
    log_data = {}
    for i, (l, l_split, t) in enumerate(zip(losses, losses_split, tau)):
        log_data[f"{mode}/{t}-multi-step-loss-all"] = l 
        for d in range(6):
            for j in range(3):
                log_data[f"{mode}/{t}-multi-step-loss-{name[j][d]}"] = l_split[i][j][d]

    wandb.log(log_data, step = step)
    if not plot:
        return

    t_dict = {
        "model": to_euler(pred_trajs[0].detach().cpu().data),
        "gt": to_euler(gt_trajs[0].data)
    }

    v_dict = {
        "model": pred_vels[0].detach().cpu(),
        "gt": gt_vels[0]
    }

    dv_dict = {
        "model": pred_dvs[0].detach().cpu(),
        "gt": gt_dvs[0]
    }

    t_imgs, v_imgs, dv_imgs = gen_imgs_3D(t_dict, v_dict, dv_dict, tau=tau)

    for t_img, v_img, dv_img, t in zip(t_imgs, v_imgs, dv_imgs, tau):
        images = [wandb.Image(t_img, caption=f"traj-{t}"), wandb.Image(v_img, caption=f"vel-{t}"),wandb.Image(dv_img, caption=f"dv-{t}")]
        wandb.log({f"{mode}/{t}": images}, step = step)

############################################
#                                          #
#        TRAINING AND VALIDATION           #
#                                          #
############################################
'''
    Validation Step. Computes and logs different metrics to validate
    the performances of the network.

    input:
    ------
        - dataset: torch.utils.data.Dataset with a methods called get_trajs() that
        returns full trajectory contained in the dataset.
        - model: the dynamical model used for predicitons.
        - loss: torch.function, the loss function used to measure the performance of the model.
        - epoch: The current training epoch.
        - device: string, the device to run the model on.
'''
def val_step(dataloader, model, loss, epoch, device):
    torch.autograd.set_detect_anomaly(True)
    size = len(dataloader.dataset)
    t = tqdm(enumerate(dataloader), desc=f"Val: {epoch}", ncols=200, colour="red", leave=False, disable=disable_tqdm)
    model.eval()
    for batch, data in t:
        X, U, traj, vel, dv = data
        X, U = X.to(device), U.to(device)
        traj, vel, dv = traj.to(device), vel.to(device), dv.to(device)

        pred, pred_vel, pred_dv = model(X, U)
        l = loss(traj, pred, vel, pred_vel, dv, pred_dv)

        wandb.add_scalar("val/loss", l, epoch*size+batch*len(X))

    # Trajectories generation for validation
    tau = [50]
    traj_loss(dataloader.dataset, model, loss, tau, epoch, device, "val", True)

'''
    Training Step. Update the networks and logs different training metrics.

    input:
    ------
        - dataset: torch.utils.data.Dataset with a methods called get_trajs() that
        returns full trajectory contained in the dataset.
        - model: the dynamical model used for predicitons.
        - loss: torch.function, the loss function used to measure the performance of the model.
        - optim: torch.optimizer, the optimizer used to update the nn weights.
        - epoch: The current training epoch.
        - device: string, the device to run the model on.
'''
def train_step(dataloader, model, loss, optim, epoch, device):
    #print("\n", "="*5, "Training", "="*5)
    torch.autograd.set_detect_anomaly(True)
    size = len(dataloader.dataset)
    t = tqdm(enumerate(dataloader), desc=f"Epoch: {epoch}", ncols=200, colour="red", leave=False, disable=disable_tqdm)
    model.train()
    for batch, data in t:
        X, U, traj, vel, dv = data
        X, U, traj, vel, dv = X.to(device), U.to(device), traj.to(device), vel.to(device), dv.to(device)

        pred, pred_v, pred_dv = model(X, U)

        optim.zero_grad()
        l = loss(traj, pred, vel, pred_v, dv, pred_dv)
        l.backward()
        optim.step()

        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         writer.add_histogram("train/" + name, param, epoch*size+batch*len(X))
        l_split = loss(traj, pred, vel, pred_v, dv, pred_dv, split=True)
        name = [["x", "y", "z", "vec_x", "vec_y", "vec_z"],
            ["u", "v", "w", "p", "q", "r"],
            ["du", "dv", "dw", "dp", "dq", "dr"]]
        for d in range(6):
            for i in range(3):
                wandb.add_scalar("train/split-loss-" + name[i][d], l_split[i][d], epoch*size+batch*len(X))
        wandb.add_scalar("train/loss", l, epoch*size+batch*len(X))

    return l.item(), batch*len(X)

'''
    Train procedure for SE3 Neural Network.

    inputs:
    -------
        - ds: pairs of datatsets. The first one will be used for training, the second
        one is for validation.
        - model: the dynamical model used for predicitons.
        - loss: torch.function, the loss function used to measure the performance of the model.
        - optim: torch.optimizer, the optimizer used to update the nn weights.
        - epoch: The current training epoch.
        - device: string, the device to run the model on.
        - ckpt_dir: String, Directory to use to save the Model in.
        - ckpt_check: Int, interval at which to save the model.
'''
def train(ds, model, loss_fc, optim, epochs, device, ckpt_dir=None, ckpt_steps=2):
    size = len(ds[0].dataset)
    l = np.nan
    cur = 0
    t = tqdm(range(epochs), desc="Training", ncols=150, colour="blue",
     postfix={"loss": f"Loss: {l:>7f} [{cur:>5d}/{size:>5d}]"}, disable=disable_tqdm)
    for e in t:
        if (e % ckpt_steps == 0) and ckpt_dir is not None:
            tau=50
            traj_loss(ds[0].dataset, model, loss_fc, tau, e, device, "train", True)
            val_step(ds[1], model, loss_fc, e, device)

            if ckpt_steps > 0:
                tmp_path = os.path.join(ckpt_dir, f"step_{e}.pth")
                torch.save(model.state_dict(), tmp_path)

        l, cur = train_step(ds[0], model, loss_fc, optim, e, device)
        t.set_postfix({"loss": f"Loss: {l:>7f} [{cur:>5d}/{size:>5d}]"})