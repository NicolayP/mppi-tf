import torch
from torch.nn.functional import normalize
import numpy as np
from scipy.spatial.transform import Rotation as R
import pandas as pd
import random

import matplotlib.pyplot as plt

from tabulate import tabulate
from tqdm import tqdm
import os
# import onnx
# from onnx_tf.backend import prepare

# import lietorch as lie

class ListDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, steps=1, history=1, rot='rot'):
        self.data_list = data_list
        self.s = steps
        self.h = history
        if rot == "rot":
            self.rot = ["r00", "r01", "r02", "r10", "r11", "r12", "r20", "r21", "r22"]
        elif rot == "quat":
            self.rot = ["qw", "qx", "qy", "qz"]
        elif rot == "euler":
            self.rot = ["roll", "pitch", "yaw"]
        else:
            raise TypeError
        self.pos = ["x", "y", "z"]
        self.lin_vel = ["u", "v", "w"]
        self.ang_vel = ["p", "q", "r"]
        self.x_labels = self.pos + self.rot + self.lin_vel + self.ang_vel
        self.u_labels = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]
        self.y_labels = self.x_labels.copy()
        
        self.nb_traj = len(data_list)
        
        self.samples = [traj.shape[0] - self.h - self.s + 1 for traj in data_list]
        
        self.len = sum(self.samples)
        
        self.bins = self.create_bins()
        
    def create_bins(self):
        bins = [0]
        cummul = 0
        for s in self.samples:
            cummul += s
            bins.append(cummul)
        return bins
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        i = (np.digitize([idx], self.bins) -1)[0]
        traj = self.data_list[i]
        j = idx - self.bins[i]
        sub_frame = traj.iloc[j:j+self.s+self.h]
        s = sub_frame[self.x_labels].to_numpy()
        u = sub_frame[self.u_labels].to_numpy()
        x = s[:self.h]
        u = u[:self.h+self.s-1]
        y = s[self.h:self.h+self.s]
        return x, u, y

    def getTrajs(self):
        traj_list = []
        action_seq_list = []
        for data in self.data_list:
            traj = data[self.x_labels]
            traj_list.append(traj)
            action_seq = data[self.u_labels]
            action_seq_list.append(action_seq)
        return traj_list, action_seq_list


class Dataset(torch.utils.data.Dataset):
    '''
    Constructor.
        input:
        ------
            - data: trajectories [h, Tau, x]
                where h is the number of trajectories, 
                tau is the number of timesteps and x
                the dimension of the state.
            - history: int (default: 1) the number 
                of past timesteps to use
            - steps: int (default: 1) the number of
                futur steps to predict.
    '''
    def __init__(self, data, steps=1, history=1):
        self.w = steps
        self.h = history
        self.x = data[:, :, :-6]
        self.u = data[:, :, -6:]
        self.trajs = self.x.shape[0]
        self.samples = self.x.shape[1] - self.h - self.w + 1

    def __len__(self):
        return self.trajs*self.samples

    def __getitem__(self, idx):
        ij = np.unravel_index([idx], (self.trajs, self.samples))
        h = ij[0][0]
        j = ij[1][0]
        x = self.x[h, j:j+self.h]
        u = self.u[h, j:j+self.h+self.w-1]
        y = self.x[h, j+self.h:j+self.h+self.w]
        return x, u, y

    def getTraj(self, idx):
        return self.x[idx], self.u[idx] 

    def getTrajs(self):
        return self.x, self.u


def push_to_tensor(tensor, x):
    return torch.cat((tensor[:, 1:], x.unsqueeze(dim=1)), dim=1)


def get_dataloader(filename, params, angleFormat="rot", normalize=False, maxInput=100., steps=1, history=1, split=.7):
    '''
        Creates a dataloader from a datafile that contains a set of trajectories.

        input:
        ------
            - filename: string pointing to the file containing the trajectories.
                Format expected to be a csv file.
            - angleFormat: string: "rot" (Default), "quat" or "euler".
                Expresses the format to use for the orientation.
            - normalized: Bool, default: False. If true, the input force vectore
                and the velocity vectors are normalized.
            - maxInput: 
            - step: Int > 0, default: 1. The number of steps to take from the
                trajectory. If 1 the dataloader will generate single transition,
                otherwise the dataloader will generate longer sequence.
            - history: Int > 0, default: 1. The number of previous steps to use
                for the prediciton.
            - split: Float \in [0, 1]. The percentage of the dataset to use for
                training and validation set. The value represents the size of the
                training set.

        output:
        -------
            - Pair of training Dataloader and validation dataloader.
    '''

    # load from file
    trajs = pd.read_csv(filename, index_col=[0,1], skipinitialspace=True)
    trajs = trajs.astype(np.float32)
    multiIndex = trajs.index
    # Create single entries for the multiIndex.
    trajs_index = multiIndex.get_level_values(0).drop_duplicates()

    features = trajs.columns
    # number of steps in each trajectories. Is assumed to be constant!
    steps_index = trajs.loc[trajs_index[0]].index

    # Feature names
    position_name = ['x (m)', 'y (m)', 'z (m)']
    euler_name = ['roll (rad)', 'pitch (rad)', 'yaw (rad)']
    quat_name = ['qx', 'qy', 'qz', 'qw']
    rot_name = ['r00', 'r01', 'r02', 'r10', 'r11', 'r12', 'r20', 'r21', 'r22']
    vel_name = ['u (m/s)', 'v (m/s)', 'w (m/s)', 'p (rad/s)', 'q (rad/s)', 'r (rad/s)']
    in_name = ['Fx (N)', 'Fy (N)', 'Fz (N)', 'Tx (Nm)', 'Ty (Nm)', 'Tz (Nm)']

    if angleFormat == "rot":
        ang_rep = rot_name
    elif angleFormat == "quat":
        ang_rep = quat_name
    elif angleFormat == "euler":
        ang_rep = euler_name

    learn_features = position_name + ang_rep + vel_name + in_name
    learning = trajs[learn_features]

    if normalize:
        vel_norm = learning[vel_name]

        vel_mean = vel_norm.mean()
        vel_std = vel_norm.std()
        vel_norm = (vel_norm-vel_mean)/vel_std

        in_norm = learning[in_name]
        in_norm = in_norm/maxInput

        learning[vel_name] = (learning[vel_name]-vel_mean)/vel_std
        learning[in_name] = learning[in_name]/maxInput

    # Spliting data:
    k = int(split*len(trajs_index))
    train_index = random.choices(trajs_index, k=k)
    val_index = trajs_index.drop(train_index)
    train_index = trajs_index.drop(val_index)

    train = learning.loc[train_index]
    val = learning.loc[val_index]

    train_data = train.to_numpy().reshape(train_index.size, steps_index.size, train.columns.size)
    val_data = val.to_numpy().reshape(val_index.size, steps_index.size, train.columns.size)

    Ds = torch.utils.data.DataLoader(
        Dataset(train_data, steps=steps, history=history),
        **params)

    DsVal = torch.utils.data.DataLoader(
        Dataset(val_data, steps=steps, history=history),
        **params)

    return (Ds, DsVal)


def train(dataloader, model, loss, opti, writer=None, epoch=None, device="cpu", verbose=True):
    torch.autograd.set_detect_anomaly(True)
    size = len(dataloader.dataset)
    model.train()
    t = tqdm(enumerate(dataloader), desc=f"Epoch: {epoch}", ncols=150, colour="red", leave=False)
    for batch, data in t:

        X, U, Y = data
        X, U, Y = X.to(device), U.to(device), Y.to(device)
        X = X[:, :, 3:] # remove all x, y and z.
        h = X.shape[1]
        w = Y.shape[1]
        # TODO: Expand for w > 1
        pred = model(X.flatten(1), U.flatten(1))[:, -6:]
        y = Y.flatten(1)[:, -6:]

        opti.zero_grad()
        l = loss(pred, y)
        l.backward()
        opti.step()

        if writer is not None:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    writer.add_histogram("train/" + name, param, epoch*size + batch)
            for dim in range(6):
                lossDim = loss(pred[:, dim], y[:, dim])
                writer.add_scalar("loss/"+ str(dim), lossDim, epoch*size + batch)
    return l.item(), batch*len(X)


def learn(dataLoaders, model, loss, opti, writer=None, maxEpochs=1, device="cpu", encoding="lie"):
    # if encoding == "lie":
    #     train_fct = train_lie
    # else:
    train_fct = train
    dls = dataLoaders
    size = len(dls[0].dataset)
    l = np.nan
    current = 0
    t = tqdm(range(maxEpochs), desc="Training", ncols=150, colour="blue", postfix={"loss": f"Loss: {l:>7f} [{current:>5d}/{size:>5d}]"})
    for e in t:
        l, current = train_fct(
            dataloader=dls[0],
            model=model,
            loss=loss,
            opti=opti,
            writer=writer,
            epoch=e,
            device=device)
        t.set_postfix({"loss": f"Loss: {l:>7f} [{current:>5d}/{size:>5d}]"})
    print("Done!\n")


def test(dataLoaders, model, loss, writer=None, step=None, device="cpu"):
    pass


def val(dataLoader, models, metric, histories=None, device="cpu",
        plotStateCols=None, plotActionCols=None, horizon=50, dir=".",
        plot=True):
    gtTrajs, actionSeqs = dataLoader.dataset.getTrajs()
    histories.append(1)
    errs = {}
    for j, (traj, seq) in enumerate(zip(gtTrajs, actionSeqs)):
        trajs = {}
        traj = torch.from_numpy(traj.to_numpy()).to(device)
        seq = torch.from_numpy(seq.to_numpy()).to(device)
        for model, h in zip(models, histories):
            init = traj[0:h][None, ...]
            pred = rollout(model, init, seq[None, ...], h, device, horizon)
            trajs[model.name + f"_traj_{j}"] = traj_to_euler(pred.cpu().numpy(), rep="quat")
            errs[model.name + f"_traj_{j}"] = [metric(pred, traj[h:horizon+h]).cpu().numpy()]
        trajs["gt"] = traj_to_euler(traj.cpu().numpy(), rep="quat")
        if plot:
            plot_traj(trajs, seq[None, ...].cpu(), histories, plotStateCols, plotActionCols, horizon, dir, f"traj_{j}")
    headers = ["name", "l2"]
    print(tabulate([[k,] + v for k, v in errs.items()], headers=headers))


def rollout(model, init, seq, h=1, device="cpu", horizon=50):
    state = init
    with torch.no_grad():
        pred = []
        for i in range(h, horizon+h):
            nextState = model(state, seq[:, i-h:i])
            pred.append(nextState)
            state = push_to_tensor(state, nextState)
        traj = torch.concat(pred, dim=0)
        return traj


def plot_traj(trajs, seq=None, histories=None, plotStateCols=None, plotActionCols=None, horizon=50, dir=".", file_name="foo"):
    '''
        Plot trajectories and action sequence.
        inputs:
        -------
            - trajs: dict with model name as key and trajectories entry. If key is "gt" then it is assumed to be
                the ground truth trajectory.
            - seq: Action Sequence associated to the generated trajectoires. If not None, plots the 
                action seqence.
            - h: list of history used for the different models, ignored when model entry is "gt".
            - plotStateCols: Dict containing the state axis name as key and index as entry
            - plotAcitonCols: Dict containing the action axis name as key and index as entry.
            - horizon: The horizon of the trajectory to plot.
            - dir: The saving directory for the generated images.
    '''
    maxS = len(plotStateCols)
    maxA = len(plotActionCols)
    fig_state = plt.figure(figsize=(50, 50))
    for k, h in zip(trajs, histories):
        t = trajs[k]
        for i, name in enumerate(plotStateCols):
            m, n = np.unravel_index(i, (2, 6))
            idx = 1*m + 2*n + 1
            plt.subplot(6, 2, idx)
            plt.ylabel(f'{name}')
            if k == "gt":
                plt.plot(t[:horizon, i], marker='.', zorder=-10)
            else:
                plt.scatter(
                    np.arange(h, horizon+h), t[:, plotStateCols[name]],
                    marker='X', edgecolors='k', s=64
                )
    #plt.tight_layout()
    if dir is not None:
        name = os.path.join(dir, f"{file_name}.png")
        plt.savefig(name)
        plt.close()

    if seq is not None:
        fig_act = plt.figure(figsize=(30, 30))
        for i, name in enumerate(plotActionCols):
            plt.subplot(maxA, 1, i+1)
            plt.ylabel(f'{name}')
            plt.plot(seq[0, :horizon+h, plotActionCols[name]])

        #plt.tight_layout()
        if dir is not None:
            name = os.path.join(dir, f"{file_name}-actions.png")
            plt.savefig(name)
            plt.close()
    
    plt.show()


def rand_roll(models, histories, plotStateCols, plotActionCols, horizon, dir, device):
    trajs = {}
    seq = 5. * torch.normal(
        mean=torch.zeros(1, horizon+10, 6),
        std=torch.ones(1, horizon+10, 6)).to(device)

    for model, h in zip(models, histories):
        init = np.zeros((1, h, 13))
        init[:, :, 6] = 1.
        #rot = np.eye(3)
        #init[:, :, 3:3+9] = np.reshape(rot, (9,))
        init = init.astype(np.float32)
        init = torch.from_numpy(init).to(device)
        pred = rollout(model, init, seq, h, device, horizon)
        trajs[model.name + "_rand"] = traj_to_euler(pred.cpu().numpy(), rep="quat")
        print(pred.isnan().any())
    plot_traj(trajs, seq.cpu(), histories, plotStateCols, plotActionCols, horizon, dir)


def traj_to_euler(traj, rep="rot"):
    if rep == "rot":
        rot = traj[:, 3:3+9].reshape((-1, 3, 3))
        r = R.from_matrix(rot)
    elif rep == "quat":
        quat = traj[:, 3:3+4]
        r = R.from_quat(quat)
    else:
        raise NotImplementedError
    pos = traj[:, :3]
    euler = r.as_euler('XYZ', degrees=True)
    vel = traj[:, -6:]

    traj = np.concatenate([pos, euler, vel], axis=-1)
    return traj
