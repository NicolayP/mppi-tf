import torch
import pypose as pp
import numpy as np
import pandas as pd
import os

from scipy.spatial.transform import Rotation as R
from utile import tdtype, npdtype, to_euler, gen_imgs_3D
import random

import matplotlib.pyplot as plt

from tabulate import tabulate
from tqdm import tqdm



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



############################################
#                                          #
#              LOSS FUNCTIONS              #
#                                          #
############################################

'''
    Compute the Left-Geodesic loss between two SE(3) poses.
'''
class GeodesicLoss(torch.nn.Module):
    '''
        GeodesicLoss constructor
    '''
    def __init__(self):
        super(GeodesicLoss, self).__init__()

    '''
        inputs:
        -------
            - X1 pypose.SE3. The first pose.
            - X2 pypose.SE3. The second pose.

        outputs:
        --------
            - Log(X1 + X2^{-1})^{2}
    '''
    def forward(self, X1, X2):
        d = (X1 * X2.Inv()).Log()
        square = torch.pow(d, 2)
        return square


'''
    Trajectory loss object. This module can compute loss between two
    trajectories represented by a sequence of SE3 Poses, Velocities and
    Velocities Delta.
'''
class TrajLoss(torch.nn.Module):
    '''
        Trajectory loss consstructor.

        inputs:
        -------
            - alpha: float, weight for trajectory loss.
            - beta: float, weight for velocity loss.
            - gamma: float, weight for $\delta V$ loss.
    '''
    def __init__(self, alpha=1., beta=0., gamma=0.):
        super(TrajLoss, self).__init__()
        self.l2 = torch.nn.MSELoss()
        self.geodesic = GeodesicLoss()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        pass

    '''
        Returns true if beta > 0.
    '''
    def has_v(self):
        return self.beta > 0.

    '''
        Returns true if gamma > 0.
    '''
    def has_dv(self):
        return self.gamma > 0.

    '''
        Computes loss on an entire trajectory. Optionally if
        dv is passed, it computes the loss on the velocity delta.

        inputs:
        -------
            traj1: pypose SE(3) elements sequence representing first trajectory
                shape [k, tau]
            traj2: pypose SE(3) elements sequence representing second trajectory
                shape [k, tau]
            v1: pytorch Tensor. velocity profiles
                shape [k, tau, 6]
            v2: pytorch Tensor. velocity profiles
                shape [k, tau, 6]
            dv1: pytorch Tensor. Delta velocities profiles
                shape [k, tau, 6]
            dv2: pytorch Tensor. Delta velocities profiles
                shape [k, tau, 6]
            split: bool (default = False), if true, returns the loss function
                splitted across each controlled dimension
    '''
    def forward(self, traj1, traj2, v1=None, v2=None, dv1=None, dv2=None, split=False):
        if split:
            return self.split_loss(traj1, traj2, v1, v2, dv1, dv2)
        return self.loss(traj1, traj2, v1, v2, dv1, dv2)

    '''
        Computes trajectory, velocity and $\Delta V$ loss split accross each dimesnions.

        inputs:
        -------
            traj1: pypose SE(3) elements sequence representing first trajectory
                shape [k, tau]
            traj2: pypose SE(3) elements sequence representing second trajectory
                shape [k, tau]
            v1: pytorch Tensor. velocity profiles
                shape [k, tau, 6]
            v2: pytorch Tensor. velocity profiles
                shape [k, tau, 6]
            dv1: pytorch Tensor. Delta velocities profiles
                shape [k, tau, 6]
            dv2: pytorch Tensor. Delta velocities profiles
                shape [k, tau, 6]

        outputs:
        --------
            t_l: torch.tensor, trajectory loss
                shape [6]
            v_l: torch.tensor, velocity loss
                shape [6]
            dv_l: torch.tensor, delta velocity loss
                shape [6]
    '''
    def split_loss(self, t1, t2, v1, v2, dv1, dv2):
        # only used for logging and evaluating the performances.
        t_l = self.geodesic(t1, t2).mean((0, 1))
        v_l = torch.pow(v1 - v2, 2).mean((0, 1))
        dv_l = torch.pow(dv1 - dv2, 2).mean((0, 1))
        return t_l, v_l, dv_l

    '''
        Computes trajectory, velocity and $\Delta V$ loss.

        inputs:
        -------
            traj1: pypose SE(3) elements sequence representing first trajectory
                shape [k, tau]
            traj2: pypose SE(3) elements sequence representing second trajectory
                shape [k, tau]
            v1: pytorch Tensor. velocity profiles
                shape [k, tau, 6]
            v2: pytorch Tensor. velocity profiles
                shape [k, tau, 6]
            dv1: pytorch Tensor. Delta velocities profiles
                shape [k, tau, 6]
            dv2: pytorch Tensor. Delta velocities profiles
                shape [k, tau, 6]

        outputs:
        --------
            loss: the full trajectory loss.
    '''
    def loss(self, t1, t2, v1, v2, dv1, dv2):
        t_l = self.geodesic(t1, t2).mean()
        v_l = self.l2(v1, v2).mean()
        dv_l = self.l2(dv1, dv2).mean()
        return self.alpha*t_l + self.beta*v_l + self.gamma*dv_l



############################################
#                                          #
#          DATASET DEFINITIONS             #
#                                          #
############################################

'''
    Dataset for trajectories.
'''
class DatasetList3D(torch.utils.data.Dataset):
    '''
        Dataset Constructor.

        inputs:
        -------
            - data_list: List, A list of pandas dataframe representing trajectories.
            - steps: Int, The number of steps to use for prediction.
            - v_frame: String, The frame in which the velocity is represented ('world' or 'body'), default 'body'
            - dv_frame: String, The frame in which the velocity delta is represented ('world' or 'body'), default 'body'
            - rot: String, the representation used for rotations. (only 'quat' supported at the moment.)
            - act_normed: Bool, whether or not to normalize the action before feeing them to the network.
            - se3: Bool, whether or not to use pypose as underlying library for the pose representation.
            - out_normed: Bool, whether or not to normalize the targets.
            - stats: dict with entries:
                - std:
                    - world_norm: list of floats. Shape (6)
                    - body_norm: list of floats. Shape (6)
                - mean:
                    - world_norm: list of floats. Shape (6)
                    - body_norm: list of floats. Shape (6)
    '''
    def __init__(self, data_list, steps=1,
                 v_frame="body", dv_frame="body", rot="quat",
                 act_normed=False, se3=False, out_normed=True, stats=None):
        super(DatasetList3D, self).__init__()
        self.data_list = data_list
        self.s = steps
        if v_frame == "body":
            v_prefix = "B"
        elif v_frame == "world":
            v_prefix = "I"

        if dv_frame == "body":
            dv_prefix = "B"
        elif dv_frame == "world":
            dv_prefix = "I"

        self.pos = ['x', 'y', "z"]
        # used for our SE3 implementation.
        if rot == "rot":
            self.rot = ['r00', 'r01', 'r02',
                        'r10', 'r11', 'r12',
                        'r20', 'r21', 'r22']
        # Used in pypose implementation.
        elif rot == "quat":
            self.rot = ['qx', 'qy', 'qz', 'qw']

        self.lin_vel = [f'{v_prefix}u', f'{v_prefix}v', f'{v_prefix}w']
        self.ang_vel = [f'{v_prefix}p', f'{v_prefix}q', f'{v_prefix}r']

        self.x_labels = self.pos + self.rot + self.lin_vel + self.ang_vel

        self.traj_labels = self.pos + self.rot
        self.vel_labels = self.lin_vel + self.ang_vel
        self.dv_labels = [
            f'{dv_prefix}du', f'{dv_prefix}dv', f'{dv_prefix}dw',
            f'{dv_prefix}dp', f'{dv_prefix}dq', f'{dv_prefix}dr'
        ]

        if act_normed:
            self.u_labels = ['Ux', 'Uy', 'Uz', 'Vx', 'Vy', 'Vz']
        else:
            self.u_labels = ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']

        self.samples = [traj.shape[0] - self.s for traj in data_list]
        self.len = sum(self.samples)
        self.bins = self.create_bins()
        self.se3 = se3
        
        if out_normed:
            self.std = np.array(stats["std"][f'{dv_prefix}_norm'], dtype=npdtype)
            self.mean = np.array(stats["mean"][f'{dv_prefix}_norm'], dtype=npdtype)
        else:
            self.std = 1.
            self.mean = 0.

    '''
        returns the number of samples in the dataset.
    '''
    def __len__(self):
        return self.len

    '''
        get a sample at a specific index.

        inputs:
        -------
            - idx, int < self.__len__().

        outputs:
        --------
            - x, the state of the vehicle (pose and velocity)
                shape (1, 7+6)
            - u, The actions applied to the vehicle. Shape (steps, 6)
            - traj, The resulting trajectory. Shape (steps, 7)
            - vel, The resulting velocity profiles, shape (steps, 6)
            - dv, The normalized velocity delta prrofiles, shape (steps, 6)
    '''
    def __getitem__(self, idx):
        i = (np.digitize([idx], self.bins)-1)[0]
        traj = self.data_list[i]
        j = idx - self.bins[i]
        sub_frame = traj.iloc[j:j+self.s+1]
        x = sub_frame[self.x_labels].to_numpy()
        x = x[:1]

        u = sub_frame[self.u_labels].to_numpy()
        u = u[:self.s]

        traj = sub_frame[self.traj_labels].to_numpy()[1:1+self.s]
        vel = sub_frame[self.vel_labels].to_numpy()[1:1+self.s]
        dv = sub_frame[self.dv_labels].to_numpy()[1:1+self.s]

        dv = (dv-self.mean)/self.std

        traj = pp.SE3(traj)

        return x, u, traj, vel, dv

    '''
        Returns the number of trajectories in the dataset.
    '''
    @property
    def nb_trajs(self):
        return len(self.data_list)
    
    '''
        Get the traj at a specific index ind the dataset.

        inputs:
        -------
            - idx, int, the trajectory index.

        outputs:
        --------
            - trajectory, shape (tau, 7+6)
    '''
    def get_traj(self, idx):
        if idx >= self.nb_trajs:
            raise IndexError
        return self.data_list[idx][self.x_labels].to_numpy()
    
    '''
        internal function that creats bins to compute the number
        of samples in the dataset.
    '''
    def create_bins(self):
        bins = [0]
        cummul = 0
        for s in self.samples:
            cummul += s
            bins.append(cummul)
        return bins

    '''
        get all the trajectories from the dataset. Only works if all
        the trajs in the dataset have the same length.

        inputs:
        -------
            - None

        outputs:
        --------
            - trajs, shape (nb_traj, tau, se3_rep)
            - vels, shape (nb_traj, tau, 6)
            - dvs, shape (nb_traj, tau, 6)
            - actions, shape (nb_traj, tau, 6)
    '''
    def get_trajs(self):
        traj_list = []
        vel_list = []
        dv_list = []
        action_seq_list = []
        for data in self.data_list:
            traj = data[self.traj_labels].to_numpy()[None]
            traj_list.append(traj)

            vel = data[self.vel_labels].to_numpy()[None]
            vel_list.append(vel)

            dv = data[self.dv_labels].to_numpy()[None]
            dv_list.append(dv)

            action_seq = data[self.u_labels].to_numpy()[None]
            action_seq_list.append(action_seq)

        trajs = torch.Tensor(np.concatenate(traj_list, axis=0))
        vels = torch.Tensor(np.concatenate(vel_list, axis=0))
        dvs = torch.Tensor(np.concatenate(dv_list, axis=0))
        actions = torch.Tensor(np.concatenate(action_seq_list, axis=0))

        dvs = (dvs-self.mean)/self.std

        if self.se3:
            trajs = pp.SE3(trajs)

        return trajs, vels, dvs, actions

    '''
        Get the mean and std of the velocity delta.

        outputs:
        --------
            - mean, torch.tensor, shape [6]
            - std, torch.tensor, shape [6]
    '''
    def get_stats(self):
        return self.mean, self.std



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
    t = tqdm(enumerate(dataloader), desc=f"Val: {epoch}", ncols=200, colour="red", leave=False)
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
    t = tqdm(enumerate(dataloader), desc=f"Epoch: {epoch}", ncols=200, colour="red", leave=False)
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
     postfix={"loss": f"Loss: {l:>7f} [{cur:>5d}/{size:>5d}]"})
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
