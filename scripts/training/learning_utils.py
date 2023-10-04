import torch
from torch.utils.data import DataLoader
import pypose as pp
import numpy as np
import pandas as pd
import os

from scipy.spatial.transform import Rotation as R
from scripts.inputs.ModelInput import ModelInputPypose
from scripts.utils.utils import read_files, tdtype, npdtype, to_euler, gen_imgs_3D, disable_tqdm, parse_param
from scripts.training.datasets import DatasetListModelInput
import random

import matplotlib.pyplot as plt

from tqdm import tqdm


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

def log(mode, writer, loss, target_traj, pred_traj, target_vel, pred_vel, target_dv, pred_dv, step, horizon=None):
    l = loss(target_traj, pred_traj, target_vel, pred_vel, target_dv, pred_dv)
    l_split = loss(target_traj, pred_traj, target_vel, pred_vel, target_dv, pred_dv, split=True)
    axis = [["x", "y", "z", "vec_x", "vec_y", "vec_z"],
            ["u", "v", "w", "p", "q", "r"],
            ["du", "dv", "dw", "dp", "dq", "dr"]]
    
    split_name = [f"{mode}/{horizon}-steps_{name}-loss-" for name in ["pose", "vel", "delta-vel"]]
    entry_name = f"{mode}/{horizon}-steps_loss"
    for d in range(6):
        for i in range(3):
            writer.add_scalar(split_name[i] + axis[i][d], l_split[i][d], step)
    writer.add_scalar(entry_name, l, step)

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
def traj_loss(mode, dataset, model, loss, tau, writer, step, device, plot=False):
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

    pred_trajs, pred_vels, pred_dvs = model(model_input, A)

    log(mode, writer, loss, 
        gt_trajs, pred_trajs.detach().cpu(),
        gt_vels, pred_vels.detach().cpu(),
        gt_dvs, pred_dvs.detach().cpu(),
        step, tau)

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

    for t_img, v_img, dv_img in zip(t_imgs, v_imgs, dv_imgs):
        writer.add_image(f"{mode}/traj-{tau}", t_img, step, dataformats="HWC")
        writer.add_image(f"{mode}/vel-{tau}", v_img, step, dataformats="HWC")
        writer.add_image(f"{mode}/dv-{tau}", dv_img, step, dataformats="HWC")


############################################
#                                          #
#        TRAINING AND VALIDATION           #
#                                          #
############################################
'''
    Training Step. Runs the training through one epoch of the dataset.
    Update the networks and logs different training metrics.

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
    torch.autograd.set_detect_anomaly(True)
    dataset_size = len(dataloader.dataset)
    model.train()
    
    t = tqdm(enumerate(dataloader), desc=f"Epoch: {epoch}", ncols=150, colour="red", leave=False, disable=disable_tqdm)
    
    for batch, data in t:
        pose_past, vel_past, action_past, action_seq, target_traj, target_vel, target_dv = data
        pose_past, vel_past, action_past, action_seq = pose_past.to(device), vel_past.to(device), action_past.to(device), action_seq.to(device)
        target_traj, target_vel, target_dv = target_traj.to(device), target_vel.to(device), target_dv.to(device)
        k = pose_past.shape[0]
        h = pose_past.shape[1]
        model_input = ModelInputPypose(k, h).to(device)
        model_input.init_from_states(pose_past, vel_past, action_past)
        model_input = model_input.to(device)
        pred_traj, pred_vel, pred_dv = model(model_input, action_seq)

        optim.zero_grad()
        l = loss(target_traj, pred_traj, target_vel, pred_vel, target_dv, pred_dv)
        l.backward(retain_graph=True)
        optim.step()

        step = epoch*dataset_size+dataloader.batch_size*batch+k
        if writer is not None:
            log("train", writer, loss, target_traj, pred_traj, target_vel, pred_vel, target_dv, pred_dv, step, dataloader.dataset.s)

    return l.item(), step

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
    dataset_size = len(dataloader.dataset)
    t = tqdm(enumerate(dataloader), desc=f"Val: {epoch}", ncols=150, colour="red", leave=False, disable=disable_tqdm)
    model.eval()

    for batch, data in t:
        pose_past, vel_past, action_past, action_seq, target_traj, target_vel, target_dv = data
        pose_past, vel_past, action_past, action_seq = pose_past.to(device), vel_past.to(device), action_past.to(device), action_seq.to(device)
        target_traj, target_vel, target_dv = target_traj.to(device), target_vel.to(device), target_dv.to(device)
        k = pose_past.shape[0]
        h = pose_past.shape[1]
        model_input = ModelInputPypose(k, h).to(device)
        model_input.init_from_states(pose_past, vel_past, action_past)
        model_input = model_input.to(device)
        pred_traj, pred_vel, pred_dv = model(model_input, action_seq)

        if writer is not None:
            step = epoch*dataset_size+dataloader.batch_size*batch+k
            log("val", writer, loss, target_traj, pred_traj, target_vel, pred_vel, target_dv, pred_dv, step, dataloader.dataset.s)
            pass
    # Trajectory evaluaton:
    tau = 50
    traj_loss("val", dataloader.dataset, model, loss, tau, writer, epoch, device, True)


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
    # TRACING OF THE GRAPH SEEMS TO FAIL FOR MODELINPUTPYPOSE. WHY???
    # Does tracing only allows for tensor as input?
    # if writer is not None:
    #     h = ds[0].dataset.h
    #     model_input = ModelInputPypose(2, h).to(device)
    #     A = torch.Tensor(np.zeros(shape=(2, 10, 6))).to(device)
    #     model(model_input, A)
    #     writer.add_graph(model, (model_input, A))


    size = len(ds[0].dataset)*epochs
    loss = np.nan
    current_step = 0
    t = tqdm(range(epochs), desc="Training", ncols=150, colour="blue",
     postfix={"loss": f"Loss: {loss:>7f} [{current_step:>5d}/{size:>5d}]"}, disable=disable_tqdm)

    for e in t:
        loss, current_step = train_step(ds[0], model, loss_fc, optim, writer, e, device)
        t.set_postfix({"loss": f"Loss: {loss:>7f} [{current_step:>5d}/{size:>5d}]"})
        # Saving checkpoint and perform validation step.
        if (e % ckpt_steps == 0) and ckpt_dir is not None:
            tau=50
            traj_loss("train", ds[0].dataset, model, loss_fc, tau, writer, e, device, True)
            val_step(ds[1], model, loss_fc, writer, e, device)

            if ckpt_steps > 0:
                tmp_path = os.path.join(ckpt_dir, f"step_{e}.pth")
                torch.save(model.state_dict(), tmp_path)

        if writer is not None:
            writer.flush()
