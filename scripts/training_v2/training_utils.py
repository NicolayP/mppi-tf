import torch
from torch.utils.data import DataLoader
import pypose as pp
import numpy as np
import pandas as pd
import os
import random
from tqdm import tqdm

from scipy.spatial.transform import Rotation as R
from scripts.utils.utils import read_files, tdtype, npdtype, to_euler, gen_img_3D_v2, disable_tqdm, parse_param
from scripts.training_v2.datasets import DatasetTensor

import matplotlib.pyplot as plt
##########################
### DATALOADER SECTION ###
##########################
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
        ds.append(get_dataloader_model_input(datafiles=dfs,
                                             tau=parameters["steps"],
                                             frame=parameters["frame"],
                                             stats=stats,
                                             batch_size=parameters["batch_size"],
                                             shuffle=parameters["shuffle"],
                                             num_workers=parameters["num_workers"]))
    return ds

def get_dataloader_model_input(datafiles, tau, frame,
                               act_normed=True, out_normed=True, stats=None,
                               batch_size=512, shuffle=True, num_workers=8):
    ds = DatasetTensor(
            data_list=datafiles,
            tau=tau,
            frame=frame,
            act_normed=act_normed,
            out_normed=out_normed,
            stats=stats)
    dl = DataLoader(ds, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers)
    return dl

##########################
###       LOGGING      ###
##########################
def log(mode, writer, loss, targets, predictions, step, horizon):
    pred_p, pred_v, pred_dv = predictions
    target_p, target_v, target_dv = targets
    l = loss(target_p, pred_p, target_v, pred_v, target_dv, pred_dv)
    split = loss(target_p, pred_p, target_v, pred_v, target_dv, pred_dv, split=True)
    axis = [["x", "y", "z", "vec_x", "vec_y", "vec_z"],
            ["u", "v", "w", "p", "q", "r"],
            ["du", "dv", "dv", "dp", "dq", "dr"]]
    split_name = [f"{mode}/{entry}/{horizon}-steps-" for entry in ["pose", "vel", "deltaV"]]
    entry_name = f"{mode}/{horizon}-steps_loss"
    writer.add_scalar(entry_name, l, step)
    for d in range(6):
        for i in range(3):
            writer.add_scalar(split_name[i] + axis[i][d], split[i][d], step)

def traj_eval(mode, dataset, model, loss, tau, writer, step, device, plot=False):
    X, Y = dataset.get_trajs(tau)

    pose, vel, seq = X
    poses_target, vels_target, dvs_target = Y
    poses_pred, vels_pred, dvs_pred = model(pose.to(device), vel.to(device), seq.to(device))

    poses_pred = poses_pred.detach().cpu()
    vels_pred = vels_pred.detach().cpu()
    dvs_pred = dvs_pred.detach().cpu()

    log(mode, writer, loss,
        poses_target, poses_pred,
        vels_target, vels_pred,
        dvs_target, dvs_pred,
        step, tau)

    if not plot:
        return

    p_dict = {
        "model": to_euler(poses_pred[0].data),
        "gt": to_euler(poses_target[0].data)
    }

    v_dict = {
        "model": vels_pred[0],
        "gt": vels_target[0]
    }

    dv_dict = {
        "model": dvs_pred[0],
        "gt": dvs_target[0]
    }

    p_img, v_img, dv_img = gen_img_3D_v2(p_dict, v_dict, dv_dict, tau=tau)
    writer.add_image(f"{mode}/traj-{tau}", p_img, step, dataformats="HWC")
    writer.add_image(f"{mode}/vel-{tau}", v_img, step, dataformats="HWC")
    writer.add_image(f"{mode}/dv-{tau}", dv_img, step, dataformats="HWC")

##########################
### TRAIN & VALIDATION ###
##########################
def train_step(dataloader, model, loss, optim, writer, epoch, device):
    model.train()
    torch.autograd.set_detect_anomaly(True)
    dataset_size = len(dataloader.dataset)
    t = tqdm(enumerate(dataloader), desc=f"Train: {epoch}", ncols=120, colour="green", leave=False, disable=disable_tqdm)
    for batch, data in t:
        X, Y = data
        k = X[0].shape[0]
        pose, vel, seq = X[0].to(device), X[1].to(device), X[2].to(device)
        preds = model(pose, vel, seq)

        optim.zero_grad()
        l = loss(Y[0].to(device), preds[0], Y[1].to(device), preds[1], Y[2].to(device), preds[2])
        l.backward()
        optim.step()

        step = epoch*dataset_size + dataloader.batch_size*batch + k
        if writer is not None:
            log("train", writer, loss, Y, preds, step, dataloader.dataset.s)
    return l.item(), step


def val_step(dataloader, model, loss, writer, epoch, device):
    model.eval()
    torch.autograd.set_detect_anomaly(True)
    dataset_size = len(dataloader.dataset)
    t = tqdm(enumerate(dataloader), desc=f"Val: {epoch}", ncols=100, colour="red", leave=False, disable=disable_tqdm)
    for batch, data in t:
        X, Y = data
        k = X[0].shape[0]
        pose, vel, seq = X[0].to(device), X[1].to(device), X[2].to(device)
        predicitons = model(pose, vel, seq)
        if writer is not None:
            step = epoch*dataset_size + dataloader.batch_size*batch + k
            log("val", writer, loss, Y, predicitons, step, dataloader.dataset.s)
    tau = 150
    traj_eval("val", dataloader.dataset, model, loss, tau, writer, epoch, device, True)


def train_v2(dataloaders, model, loss, optim, writer, epochs, device, ckpt_dir=None, ckpt_steps=2):
    if writer is not None:
        #pose = pp.identity_SE3(1, 1).to(device)
        state = torch.zeros(1, 1, 7+6).to(device)
        state[..., 7] = 1.
        seq = torch.zeros(1, 50, 6).to(device)
        writer.add_graph(model, [state, seq])
    return

    size = len(dataloaders[0].dataset)*epochs
    l = np.nan
    current_step = 0
    val_epoch = 0
    t = tqdm(range(epochs), desc="Training", ncols=150, colour="blue",
             postfix={"loss": f"Loss: {l:>7f} [{current_step:>5d}/{size:>5d}]"}, disable=disable_tqdm)
    for e in t:
        l, current_step = train_step(dataloaders[0], model, loss, optim, writer, e, device)
        t.set_postfix({"loss": f"Loss: {l:>7f} [{current_step:>5d}/{size:>5d}]"})

        if (e % ckpt_steps == 0) and ckpt_dir is not None:
            tau = 150
            traj_eval("train-test", dataloaders[0].dataset, model, loss, tau, writer, val_epoch, device, True)
            val_step(dataloaders[1], model, loss, writer, val_epoch, device)

            if ckpt_steps > 0:
                tmp_path = os.path.join(ckpt_dir, f"step_{e}.pth")
                torch.save(model.state_dict(), tmp_path)
        if writer is not None:
            writer.flush()
