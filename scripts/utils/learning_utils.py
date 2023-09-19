import torch
import pypose as pp
import numpy as np
import pandas as pd
import os

from scipy.spatial.transform import Rotation as R
from utile import tdtype, npdtype, to_euler, gen_imgs_3D, disable_tqdm
import random

import matplotlib.pyplot as plt

from tabulate import tabulate
from tqdm import tqdm



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
def traj_loss(dataset, model, loss, tau, writer, step, device, mode="train", plot=False):
    gt_trajs, gt_vels, gt_dv, aciton_seqs = dataset.get_trajs()
    x_init = gt_trajs[:, 0:1].to(device)
    v_init = gt_vels[:, 0:1].to(device)
    A = aciton_seqs[:, :tau[-1]].to(device)
    init = torch.concat([x_init.data, v_init], dim=-1)

    pred_trajs, pred_vels, pred_dvs = model(init, aciton_seqs.to(device))


    losses = [loss(
            pred_trajs[:, :h], gt_trajs[:, :h].to(device),
            pred_vels[:, :h], gt_vels[:, :h].to(device),
            pred_dvs[:, :h], gt_dv[:, :h].to(device)
        ) for h in tau]
    losses_split = [[loss(
            pred_trajs[:, :h], gt_trajs[:, :h].to(device),
            pred_vels[:, :h], gt_vels[:, :h].to(device),
            pred_dvs[:, :h], gt_dv[:, :h].to(device), split=True
        )] for h in tau]

    name = [["x", "y", "z", "vec_x", "vec_y", "vec_z"],
            ["u", "v", "w", "p", "q", "r"],
            ["du", "dv", "dw", "dp", "dq", "dr"]]
    if writer is not None:
        for i, (l, l_split, t) in enumerate(zip(losses, losses_split, tau)):
            writer.add_scalar(f"{mode}/{t}-multi-step-loss-all", l, step)
            for d in range(6):
                for j in range(3):
                    writer.add_scalar(f"{mode}/{t}-multi-step-loss-{name[j][d]}", l_split[i][j][d], step)

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
        "gt": gt_dv[0]
    }

    t_imgs, v_imgs, dv_imgs = gen_imgs_3D(t_dict, v_dict, dv_dict, tau=tau)

    for t_img, v_img, dv_img, t in zip(t_imgs, v_imgs, dv_imgs, tau):
        writer.add_image(f"{mode}/traj-{t}", t_img, step, dataformats="HWC")
        writer.add_image(f"{mode}/vel-{t}", v_img, step, dataformats="HWC")
        writer.add_image(f"{mode}/dv-{t}", dv_img, step, dataformats="HWC")



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
        - writer: torch.summarywriter. Writer used to log the data
        - epoch: The current training epoch.
        - device: string, the device to run the model on.
'''
def val_step(dataloader, model, loss, writer, epoch, device):
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

        if writer is not None:
            writer.add_scalar("val/loss", l, epoch*size+batch*len(X))

    # Trajectories generation for validation
    tau = [50]
    traj_loss(dataloader.dataset, model, loss, tau, writer, epoch, device, "val", True)

'''
    Training Step. Update the networks and logs different training metrics.

    input:
    ------
        - dataset: torch.utils.data.Dataset with a methods called get_trajs() that
        returns full trajectory contained in the dataset.
        - model: the dynamical model used for predicitons.
        - loss: torch.function, the loss function used to measure the performance of the model.
        - optim: torch.optimizer, the optimizer used to update the nn weights.
        - writer: torch.summarywriter. Writer used to log the data
        - epoch: The current training epoch.
        - device: string, the device to run the model on.
'''
def train_step(dataloader, model, loss, optim, writer, epoch, device):
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

        if writer is not None:
            # for name, param in model.named_parameters():
            #     if param.requires_grad:
            #         writer.add_histogram("train/" + name, param, epoch*size+batch*len(X))
            l_split = loss(traj, pred, vel, pred_v, dv, pred_dv, split=True)
            name = [["x", "y", "z", "vec_x", "vec_y", "vec_z"],
                ["u", "v", "w", "p", "q", "r"],
                ["du", "dv", "dw", "dp", "dq", "dr"]]
            for d in range(6):
                for i in range(3):
                    writer.add_scalar("train/split-loss-" + name[i][d], l_split[i][d], epoch*size+batch*len(X))
            writer.add_scalar("train/loss", l, epoch*size+batch*len(X))

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
        - writer: torch.summarywriter. Writer used to log the data
        - epoch: The current training epoch.
        - device: string, the device to run the model on.
        - ckpt_dir: String, Directory to use to save the Model in.
        - ckpt_check: Int, interval at which to save the model.
'''
def train(ds, model, loss_fc, optim, writer, epochs, device, ckpt_dir=None, ckpt_steps=2):
    if writer is not None:
        s = torch.Tensor(np.zeros(shape=(1, 1, 13))).to(device)
        s[..., 6] = 1.
        A = torch.Tensor(np.zeros(shape=(1, 10, 6))).to(device)
        writer.add_graph(model, (s, A))
    size = len(ds[0].dataset)
    l = np.nan
    cur = 0
    t = tqdm(range(epochs), desc="Training", ncols=150, colour="blue",
     postfix={"loss": f"Loss: {l:>7f} [{cur:>5d}/{size:>5d}]"}, disable=disable_tqdm)
    for e in t:
        if (e % ckpt_steps == 0) and ckpt_dir is not None:
            tau=[50]
            traj_loss(ds[0].dataset, model, loss_fc, tau, writer, e, device, "train", True)
            val_step(ds[1], model, loss_fc, writer, e, device)

            if ckpt_steps > 0:
                tmp_path = os.path.join(ckpt_dir, f"step_{e}.pth")
                torch.save(model.state_dict(), tmp_path)

        l, cur = train_step(ds[0], model, loss_fc, optim, writer, e, device)
        t.set_postfix({"loss": f"Loss: {l:>7f} [{cur:>5d}/{size:>5d}]"})

        if writer is not None:
            writer.flush()
