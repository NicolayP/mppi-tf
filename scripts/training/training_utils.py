import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import random
from tqdm import tqdm
import wandb

from scripts.utils.utils import read_files, to_euler, gen_img_3D, disable_tqdm, parse_param
from scripts.training.datasets import DatasetTensor

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
        ds.append(get_dataloader_tensor(datafiles=dfs,
                                             tau=parameters["steps"],
                                             frame=parameters["frame"],
                                             stats=stats,
                                             batch_size=parameters["batch_size"],
                                             shuffle=parameters["shuffle"],
                                             num_workers=parameters["num_workers"]))
    return ds

def get_dataloader_tensor(datafiles, tau, frame,
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
def log(mode, loss, targets, predictions, step, horizon):
    pred_p, pred_v, pred_dv = predictions
    target_p, target_v, target_dv = targets
    l = loss(target_p, pred_p, target_v, pred_v, target_dv, pred_dv)
    split = loss(target_p, pred_p, target_v, pred_v, target_dv, pred_dv, split=True)
    axis = [["x", "y", "z", "vec_x", "vec_y", "vec_z"],
            ["u", "v", "w", "p", "q", "r"],
            ["du", "dv", "dw", "dp", "dq", "dr"]]
    split_name = [f"{mode}/{entry}/{horizon}-steps-" for entry in ["pose", "vel", "deltaV"]]
    entry_name = f"{mode}/{horizon}-steps-loss"
    log_data = {}
    log_data[entry_name] = l
    for d in range(6):
        for i in range(3):
            log_data[split_name[i] + axis[i][d]] = split[i][d]
    wandb.log(log_data)

def traj_eval(mode, dataset, model, loss, tau, step, device, plot=False):
    X, Y = dataset.get_trajs(tau)

    pose, vel, seq = X
    poses_target, vels_target, dvs_target = Y
    poses_pred, vels_pred, dvs_pred = model(pose.to(device), vel.to(device), seq.to(device))

    poses_pred = poses_pred.detach().cpu()
    vels_pred = vels_pred.detach().cpu()
    dvs_pred = dvs_pred.detach().cpu()

    log(mode, loss,
        Y, (poses_pred, vels_pred, dvs_pred),
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

    p_img, v_img, dv_img = gen_img_3D(p_dict, v_dict, dv_dict, tau=tau)
    images = [wandb.Image(p_img, caption=f"traj-{tau}"), wandb.Image(v_img, caption=f"vel-{tau}"),wandb.Image(dv_img, caption=f"dv-{tau}")]
    wandb.log({f"{mode}/{tau}": images})


##########################
### TRAIN & VALIDATION ###
##########################
def train_step(dataloader, model, loss, optim, epoch, device):
    model.train()
    torch.autograd.set_detect_anomaly(True)
    dataset_size = len(dataloader.dataset)
    t = tqdm(enumerate(dataloader), desc=f"Train: {epoch}", ncols=120, colour="green", leave=False, disable=disable_tqdm)
    for batch, data in t:
        (pose, vel, seq), (poses_t, vels_t, dvs_t) = data
        k = pose.shape[0]
        pose, vel, seq = pose.to(device), vel.to(device), seq.to(device)
        poses_t, vels_t, dvs_t = poses_t.to(device), vels_t.to(device), dvs_t.to(device)
        targets = (poses_t, vels_t, dvs_t)

        optim.zero_grad()
        preds = model(pose, vel, seq)
        l = loss(poses_t, preds[0], vels_t, preds[1], dvs_t, preds[2])
        l.backward()
        optim.step()

        step = epoch*dataset_size + dataloader.batch_size*batch + k
        log("train", loss, targets, preds, step, dataloader.dataset.prediction_steps)
    return l.item(), step

def val_step(dataloader, model, loss, epoch, device):
    torch.autograd.set_detect_anomaly(True)
    dataset_size = len(dataloader.dataset)
    t = tqdm(enumerate(dataloader), desc=f"Val: {epoch}", ncols=200, colour="red", leave=False, disable=disable_tqdm)
    model.eval()
    for batch, data in t:
        (pose, vel, seq), (poses_t, vels_t, dvs_t) = data
        k = pose.shape[0]
        pose, vel, seq = pose.to(device), vel.to(device), seq.to(device)
        poses_t, vels_t, dvs_t = poses_t.to(device), vels_t.to(device), dvs_t.to(device)
        targets = (poses_t, vels_t, dvs_t)

        predicitons = model(pose, vel, seq)

        step = epoch*dataset_size + dataloader.batch_size*batch + k
        log("val", loss, targets, predicitons, step, dataloader.dataset.prediction_steps)
    tau = 50
    traj_eval("traj-val", dataloader.dataset, model, loss, tau, epoch, device, True)

def train(dataloaders, model, loss, optim, epochs, device, ckpt_dir=None, ckpt_steps=2, val_loss=None):
    if val_loss is None:
        val_loss = loss
    size = len(dataloaders[0].dataset)*epochs
    l = np.nan
    current_step = 0
    val_epoch = 0
    t = tqdm(range(epochs), desc="Training", ncols=150, colour="blue",
             postfix={"loss": f"Loss: {l:>7f} [{current_step:>5d}/{size:>5d}]"}, disable=disable_tqdm)
    for e in t:
        if (e % ckpt_steps == 0) and ckpt_dir is not None:
            tau = 150
            traj_eval("train-test", dataloaders[0].dataset, model, loss, tau, val_epoch, device, True)
            val_step(dataloaders[1], model, val_loss, val_epoch, device)
            if ckpt_steps > 0:
                tmp_path = os.path.join(ckpt_dir, f"step_{e}.pth")
                torch.save(model.state_dict(), tmp_path)

        l, current_step = train_step(dataloaders[0], model, loss, optim, e, device)
        print(f"[{e+1}/{epochs}] Loss: {l}")
        t.set_postfix({"loss": f"Loss: {l:>7f} [{current_step:>5d}/{size:>5d}]"})
