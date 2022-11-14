from tkinter import image_names
from typing import ForwardRef
import torch
from torch.nn.functional import normalize
import numpy as np
from scipy.spatial.transform import Rotation as R
import pandas as pd
import random

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from tabulate import tabulate
from tqdm import tqdm
import os
import onnx
from onnx_tf.backend import prepare


dtype=torch.float32
#import lietorch as lie

class ListDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, steps=1, history=1, rot='rot'):
        self.data_list = data_list
        self.s = steps
        self.h = history
        self.pos = ["x", "y", "z"]
        self.lin_vel = ["u", "v", "w"]
        self.ang_vel = ["p", "q", "r"]
        self.parse_rot(rot)
        self.x_labels = self.pos + self.rot + self.lin_vel + self.ang_vel
        self.y_labels = self.x_labels.copy()
        self.u_labels = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]
        
        self.nb_traj = len(data_list)
        
        self.samples = [traj.shape[0] - self.h - self.s + 1 for traj in data_list]
        
        self.len = sum(self.samples)
        
        self.bins = self.create_bins()

    def parse_rot(self, rot):
        if rot == "rot":
            self.rot = ["r00", "r01", "r02", "r10", "r11", "r12", "r20", "r21", "r22"]
        elif rot == "quat":
            self.rot = ["qw", "qx", "qy", "qz"]
        elif rot == "euler":
            self.rot = ["roll", "pitch", "yaw"]
        else:
            raise TypeError

    def set_rot(self, rot):
        self.parse_rot(rot)
        self.x_labels = self.pos + self.rot + self.lin_vel + self.ang_vel
        self.y_labels = self.x_labels.copy()

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


class SE3enc(torch.nn.Module):
    '''
        prepares the data to be fed in a velocity/state predictor

        x: the state of the system, shape [k, 13, 1]
        u: the action, shape [k, 6, 1]

        euler: bool, default false, if true represents the 
        orientationn as euler angles.
        sinCos: bool, default true, represents the orientation 
        with sin and cos of the eurler angles.
    '''
    def __init__(self, rot=True, normV=[0., 1.], maxU=10.):
        super(SE3enc, self).__init__()
        self.rot = rot
        self.normV = normV
        self.maxU = maxU
        if self.rot:
            self.b2i = Body2Inertial()

    def forward(self, x, u, norm):
        pose = x[:, :7]
        vel = x[:, 7:]
        if norm:
            vel = (vel - self.normV[0]) / self.normV[1]
            u = u / self.maxnU
        
        if self.rot:
            rot, _ = self.b2i(pose)
            rot = torch.flatten(rot, start_dim=1)
            return torch.concat([rot, vel, u], dim=1)
        else:
            return torch.concat([x[:, 3:], u], dim=1)

    def __len__(self):
        if self.rot:
            return 9 + 6 + 6
        else:
            return 4 + 6 + 6


class SE3integ(torch.nn.Module):
    '''
        Computes x_{t+1} given x_{t} and \delta_{t}
        x: the state of the system composed of pose and velocites.
            shape [k, 13, 1]
        delta: the velocity delta. 
            shape [k, 6, 1]
    '''
    def __init__(self):
        super(SE3integ, self).__init__()
        self.jac = Jacobian()
        self.norm_quat = NormQuat()

    def forward(self, x, delta, dt=0.1):
        pose = x[:, :7]
        vel = x[:, 7:]
        jac = self.jac(pose)
        pDot = torch.matmul(
            jac,
            torch.unsqueeze(vel, dim=-1))
        pDot = torch.squeeze(pDot,dim=-1)
        nextPose = self.norm_quat(pose + pDot*dt)
        nextVel = vel + delta
        return torch.concat([nextPose, nextVel], dim=-1)


class Jacobian(torch.nn.Module):
    def __init__(self):
        super(Jacobian, self).__init__()
        self.pad3x3 = torch.zeros(1, 3, 3)
        self.pad4x3 = torch.zeros(1, 4, 3)
        self.b2i = Body2Inertial()

    def forward(self, pose):
        rotBtoI, tBtoI = self.b2i(pose)
        k = rotBtoI.shape[0]
        pad3x3 = torch.broadcast_to(self.pad3x3, (k, 3, 3))
        pad4x3 = torch.broadcast_to(self.pad4x3, (k, 4, 3))
        jacR1 = torch.concat([rotBtoI, pad3x3], dim=-1)
        jacR2 = torch.concat([pad4x3, tBtoI], dim=-1)
        return torch.concat([jacR1, jacR2], dim=-2)


class Body2Inertial(torch.nn.Module):
    def __init__(self):
        super(Body2Inertial, self).__init__()

    def forward(self, pose):
        quat = torch.unsqueeze(pose[:, 3:7], dim=-1)

        x = quat[:, 0]
        y = quat[:, 1]
        z = quat[:, 2]
        w = quat[:, 3]

        r1 = torch.unsqueeze(
                torch.concat([1 - 2 * (y**2 + z**2),
                              2 * (x * y - z * w),
                              2 * (x * z + y * w)], dim=-1),
                dim=-2)
        r2 = torch.unsqueeze(
                torch.concat([2 * (x * y + z * w),
                              1 - 2 * (x**2 + z**2),
                              2 * (y * z - x * w)], dim=-1),
                dim=-2)
        r3 = torch.unsqueeze(
                torch.concat([2 * (x * z - y * w),
                              2 * (y * z + x * w),
                              1 - 2 * (x**2 + y**2)], dim=-1),
                dim=-2)

        rotBtoI = torch.concat([r1, r2, r3], dim=-2)

        r1t = torch.unsqueeze(
                torch.concat([-x, -y, -z], dim=-1),
                dim=-2)
        r2t = torch.unsqueeze(
                torch.concat([w, -z, y], dim=-1),
                dim=-2)
        r3t = torch.unsqueeze(
                torch.concat([z, w, -x], dim=-1),
                dim=-2)
        r4t = torch.unsqueeze(
                torch.concat([-y, x, w], dim=-1),
                dim=-2)

        tBtoI = 0.5 * torch.concat([r1t, r2t, r3t, r4t], dim=-2)
        return rotBtoI, tBtoI


class NormQuat(torch.nn.Module):
    def __init__(self):
        super(NormQuat, self).__init__()

    def forward(self, pose):
        quat = pose[:, 3:7].clone()
        norm = torch.unsqueeze(torch.linalg.norm(quat, dim=-1), dim=-1)
        quat = quat/norm
        pose[:, 3:7] = quat.clone()
        return pose


class ToSE3Mat(torch.nn.Module):
    def __init__(self):
        super(ToSE3Mat, self).__init__()
        pad = torch.Tensor([[[0., 0., 0., 1.]]])
        self.register_buffer('pad_const', pad)

    def forward(self,x):
        '''
            input:
            ------
                x flatten state shape [k, 18] or [18]
                x[:, 0:3] = [x, y, z]
                x[:, 3:12] = [r00, r01, r02, r10, r11, r12, r20, r21, r22]
                x[:, 12:15] = [u, v, w]
                x[:, 15:18] = [p, q, r]
            
            output:
            -------
                M a Lie Group element. Shape [k, 4, 4] or [4, 4]
        '''
        k = x.shape[0]
        p = x[:, :3].unsqueeze(dim=-1)
        r = x[:, 3:3+9].reshape((-1, 3, 3))

        noHomo = torch.concat([r, p], dim=-1)
        homo = torch.concat([noHomo, self.pad_const.broadcast_to((k, 1, 4))], dim=-2)

        return homo


class SO3Exp(torch.nn.Module):
    def __init__(self):
        super(SO3Exp, self).__init__()
        self.skew = Skew()
        a = torch.eye(3)
        self.register_buffer('a', a)

    def forward(self, tau):
        '''
            Computes the exponential map of Tau in SO(3).

            input:
            ------
                - tau: perturbation in so(3). shape [k, 3] or [3]

            output:
            -------
                - Exp(Tau). shape [k, 3, 3] or [3, 3]
        '''
        theta = torch.linalg.norm(tau, dim=1)
        u = normalize(tau, dim=-1)

        skewU = self.skew(u)
        b = torch.sin(theta)[:, None, None]*skewU
        c = (1-torch.cos(theta))[:, None, None]*torch.pow(skewU, 2)

        res = self.a + b + c
        
        return res


class SO3int(torch.nn.Module):
    def __init__(self):
        super(SO3int, self).__init__()
        self.so3exp = SO3Exp()

    def forward(self, R, tau):
        '''
            Applies the perturbation Tau on M using the exponential mapping and the right plus operator.
            input:
            ------
                - M Element of SO(3), shape [k, 3, 3] or [3, 3]
                - Tau perturbation vector in R^3 ~ so(3) shape [k, 3] or [3].
                
            
            output:
            -------
                - M (+) Exp(Tau)
        '''
        return R @ self.so3exp(tau)


class SE3V(torch.nn.Module):
    def __init__(self):
        super(SE3V, self).__init__()
        self.eps = 1e-10
        a = torch.eye(3)
        self.register_buffer("a", a)
        self.skew = Skew()

    def forward(self, theta_vec):
        '''
            Compute V(\theta) used in the exponential mapping. See 

            input:
            ------
                - theta_vec. Rotation vector \theta * u. Where theta is the
                rotation angle around the unit vector u. Shape [k, 3] or [3]
        '''
        k = theta_vec.shape[0]
        theta = torch.linalg.norm(theta_vec, dim=-1) # [k,]
        result = torch.zeros((k, 3, 3)).to(theta_vec.device)
        non_zero_theta = theta[theta > self.eps] # [k - zeros, ]
        non_zero_vec = theta_vec[theta > self.eps] # [k - zeros, 3]
        if non_zero_theta.shape[0] > 0:
            skewT = self.skew(non_zero_vec) # [k - zeros, 3, 3]
            b = ((1-torch.cos(non_zero_theta))/torch.pow(non_zero_theta, 2))[:, None, None] * skewT
            c = ((non_zero_theta - torch.sin(non_zero_theta))/torch.pow(non_zero_theta, 3))[:, None, None] * torch.pow(skewT, 2)
            non_zero_tmp = b + c
            result[theta > self.eps] = non_zero_tmp
        result = result + self.a[None, ...]
        return result


class SE3Exp(torch.nn.Module):
    def __init__(self):
        super(SE3Exp, self).__init__()
        self.se3v = SE3V()
        self.so3exp = SO3Exp()
        pad = torch.Tensor([[[0., 0., 0., 1.]]])
        self.register_buffer('pad_const', pad)

    def forward(self, tau):
        k = tau.shape[0]
        rho_vec = tau[:, :3]
        theta_vec = tau[:, 3:]

        r = self.so3exp(theta_vec)
        p = self.se3v(theta_vec) @ rho_vec.unsqueeze(dim=-1)

        noHomo = torch.concat([r, p], dim=-1)
        pad = self.pad_const.broadcast_to((k, 1, 4))
        homo = torch.concat([noHomo, pad], dim=-2)
        return homo


class SE3int(torch.nn.Module):
    def __init__(self):
        super(SE3int, self).__init__()
        self.se3exp = SE3Exp()

    def forward(self, M, tau):
        '''
            Applies the perturbation Tau on M (in SE(3)) using the exponential mapping and the right plus operator.
            input:
            ------
                - M Element of SE(3), shape [k, 4, 4] or [4, 4]
                - Tau perturbation vector in R^6 ~ se(3) shape [k, 6] or [6].

            output:
            -------
                - M (+) Exp(Tau)
        '''
        exp = self.se3exp(tau)
        return M @ exp


class SE3inv(torch.nn.Module):
    def __init__(self):
        super(SE3inv, self).__init__()

    def forward(self, M):
        '''
            Inputs:
            -------
                - M. SE(3) element. Shape [k, 4, 4]
        '''
        R = M[:, 0:3, 0:3]
        t = M[:, 0:3, 3:]
        Rt = torch.transpose(R, dim0=-1, dim1=-2)
        M[:, 0:3, 0:3] = Rt
        M[:, 0:3, 3:] = - Rt @ t
        return M


class Adjoint(torch.nn.Module):
    def __init__(self):
        super(Adjoint, self).__init__()
        pad = torch.zeros(1, 3, 3)
        self.register_buffer('pad_const', pad)
        self.skew = Skew()

    def forward(self, M):
        k = M.shape[0]
        pad3x3 = self.pad_const.broadcast_to((k, 3, 3))
        R = M[:, 0:3, 0:3]
        t = M[:, 0:3, 3]
        r1 = torch.cat([R, self.skew(t) @ R], axis=-1)
        r2 = torch.cat([pad3x3, R], axis=-1)
        return torch.cat([r1, r2], axis=-2)


class Skew(torch.nn.Module):
    def __init__(self):
        super(Skew, self).__init__()
        e1 = torch.Tensor([
                           [0., 0., 0.],
                           [0., 0., -1.],
                           [0., 1., 0.]
                          ])
        self.register_buffer('e1_const', e1)

        e2 = torch.Tensor([
                           [0., 0., 1.],
                           [0., 0., 0.],
                           [-1., 0., 0.]
                          ])
        self.register_buffer('e2_const', e2)

        e3 = torch.Tensor([
                           [0., -1., 0.],
                           [1., 0., 0.],
                           [0., 0., 0.]
                          ])
        self.register_buffer('e3_const', e3)

    def forward(self, vec):
        '''
            Computes the skew-symetric matrix of vector vec

            input:
            ------
                - vec. A 3D vector or batch of vector. Shape [k, 3] or [3]

            output:
            -------
                - skew(vec) a skew symetric matrix. Shape [k, 3, 3] or [3, 3]
        '''
        batch = True
        if vec.dim() < 2:
            batch = False
            vec = vec.unsqueeze(dim=0)
        a = self.e1_const * vec[:, 0, None, None]
        b = self.e2_const * vec[:, 1, None, None]
        c = self.e3_const * vec[:, 2, None, None]
        if batch:
            return a + b + c
        else:
            return (a + b + c).squeeze()


class FlattenSE3(torch.nn.Module):
    def __init__(self):
        super(FlattenSE3, self).__init__()

    def forward(self, M, vel):
        '''
            Flattens out a SE(3) elements to it's core components

            input:
            ------
                - M in SE(3). Shape [k, 4, 4] or [4, 4]
                - vel a perturbation vector, usually representing the velocity. [k, 6] or [6]
            
            output:
            ------- 
                - The flattend vector [x, y, z, r00, r01, ..., r22, u, v, w, p, q, r]. Shape [k, 18] or [18]
        '''
        p = M[:, 0:3, 3]
        r = M[:, 0:3, 0:3].reshape((-1, 9))
        x = torch.concat([p, r, vel], dim=1)
        return x


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


def train_step(model, loss, opti, s, A, gt):
    pred = model(s, A)
    opti.zero_grad()
    l = loss(gt, pred)
    l.backward()
    opti.step()


def log_step():
    pass


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


def train_lie(dataloader, model, loss, opti, writer=None, epoch=None, device="cpu", verbose=True):
    torch.autograd.set_detect_anomaly(True)
    size = len(dataloader.dataset)
    vel_loss = loss.l2
    model.train()
    t = tqdm(enumerate(dataloader), desc=f"Epoch: {epoch}", ncols=150, colour="red", leave=False)
    for batch, data in t:
        X, U, Y = data
        X, U, Y = X.to(device), U.to(device), Y.to(device)
        Y_M = (lie.SE3.InitFromVec(Y[:, :, :7]), Y[:, :, 7:])

        pred = model(X, U)
        opti.zero_grad()
        l = loss(pred, Y_M)
        opti.step()

        if writer is not None:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    writer.add_histogram("train/" + name, param, epoch*size + batch)
            for dim in range(6):
                lossDim = vel_loss(pred[1][:, dim], Y_M[1][:, dim])
                writer.add_scalar("loss/"+ str(dim), lossDim, epoch*size + batch)
    return l.item(), batch*len(X)


def save_model(model, dir, tf=True, dummy_input=None, input_names=[], output_names=[], dynamic_axes={}):
    torch_filename = os.path.join(dir, f"{model.name}.pth")
    onnx_filename = os.path.join(dir, f"{model.name}.onnx")
    tf_filename = os.path.join(dir, f"{model.name}.pb")
    torch.save(model.state_dict(), torch_filename)
    if tf:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_filename,
            verbose=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes
        )

        onnx_model = onnx.load(onnx_filename)
        onnx.checker.check_model(onnx_model)
        print(onnx.helper.printable_graph(onnx_model.graph))

        tf_rep = prepare(onnx_model)
        tf_rep.export_graph(tf_filename)


def learn(dataLoaders, model, loss, opti, writer=None, maxEpochs=1, device="cpu", encoding="lie", ckpt_dir=None):
    if encoding == "lie":
        train_fct = train_lie
    else:
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

        l_test = test(
            dataLoader=dls[1],
            model=model,
            loss=loss,
            writer=writer,
            step=e,
            device=device)
        if (e % 2 == 0) and ckpt_dir is not None:
            tmp_path = os.path.join(ckpt_dir, f"e_{e}.pth")
            torch.save(model.state_dict(), tmp_path)
        t.set_postfix({"loss": f"Loss: {l:>7f} [{current:>5d}/{size:>5d}]"})
    print("Done!\n")


def test(dataLoader, model, loss, writer=None, step=None, device="cpu"):
    pass


def val(dataset, models, metric, histories=None, device="cpu",
        plotStateCols=None, plotActionCols=None, horizon=50, dir=".",
        plot=True, img_name_prefix="foo"):
    dataset.set_rot("quat")
    gtTrajsQuat, actionSeqsQuat = dataset.getTrajs()
    dataset.set_rot("rot")
    gtTrajsRot, actionSeqsRot = dataset.getTrajs()
    histories.append(1)
    errs = {}
    for j, (trajQuat, seqQuat, trajRot, seqRot) in enumerate(zip(gtTrajsQuat, actionSeqsQuat, gtTrajsRot, actionSeqsRot)):
        trajs = {}
        trajEuler = traj_to_euler(trajQuat.to_numpy(), rep="quat")
        trajEulerTorch = torch.from_numpy(trajEuler).to(device)
        for model, h in zip(models, histories):
            if model.sDim == 13:
                traj, seq = trajQuat.copy(), seqQuat.copy()
            if model.sDim == 18:
                traj, seq = trajRot.copy(), seqRot.copy()
            traj = torch.from_numpy(traj.to_numpy()).to(device)
            seq = torch.from_numpy(seq.to_numpy()).to(device)
            init = traj[0:h][None, ...]
            pred = rollout(model, init, seq[None, ...], h, device, horizon)
            if model.sDim == 13:
                quat = pred[:, 3:3+4]
                predTraj = traj_to_euler(pred.cpu().numpy(), rep="quat")
            if model.sDim == 18:
                predTraj = traj_to_euler(pred.cpu().numpy(), rep="rot")
                # BUG FIX DUE TO ROT TRANSFORMATIONS AND REPRESENTATIONS ??
                predTraj[:, 5] += 180
            trajs[model.name + f"_traj_{j}"] = predTraj
            
            predTrajTorch = torch.from_numpy(predTraj).to(device)
            errs[model.name + f"_traj_{j}"] = [metric(predTrajTorch, trajEulerTorch[h:horizon+h])]
        trajs["gt"] = trajEuler 
        print(trajs.keys())
        if plot:
            names = [n.name for n in models]
            plot_traj(trajs, seq[None, ...].cpu(), histories, names, plotStateCols, plotActionCols, horizon, dir, f"traj_{j}_{img_name_prefix}")
    headers = ["name", "l2"]
    print(tabulate([[k,] + v for k, v in errs.items()], headers=headers))


def rollout(model, init, seq, h=1, device="cpu", horizon=50):
    state = init
    with torch.no_grad():
        pred = []
        #print("Init vel: ", init[:, :, -6:])
        for i in range(h, horizon+h):
            #print("State vel: ", state[:, :, -6:])
            nextState = model(state, seq[:, i-h:i])
            pred.append(nextState)
            state = push_to_tensor(state, nextState)
        traj = torch.concat(pred, dim=0)
        return traj


def plot_traj(
    trajs, seq=None, histories=None, names=None,
    plotStateCols=None, plotActionCols=None, horizon=50, stateLim=None,
    dir=".", file_name="foo", to_numpy=False):
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
    fig_3d = plt.figure(figsize=(50, 50))
    axs_3d = fig_3d.add_subplot(111, projection='3d')

    fig_state = plt.figure(figsize=(10, 10))

    axs_states = {}
    for i, name in enumerate(plotStateCols):
        m, n = np.unravel_index(i, (2, 6))
        idx = 1*m + 2*n + 1
        axs_states[name] = fig_state.add_subplot(6, 2, idx)

    for k, h in zip(trajs, histories):
        t = trajs[k]
        x, y, z = t[:horizon, 0], t[:horizon, 1], t[:horizon, 2]
        if k == "gt":
            axs_3d.plot(x, y, z, alpha=1)
        else:
            axs_3d.plot(x, y, z, alpha=1)

        for i, name in enumerate(plotStateCols):
            axs_states[name].set_ylabel(f'{name}', fontsize=10)
            if k == "gt":
                if i == 0:
                    axs_states[name].plot(t[:horizon+1, i], marker='.', zorder=-10, label=k)
                else:
                    axs_states[name].plot(t[:horizon+1, i], marker='.', zorder=-10)
                if stateLim is not None:
                    axs_states[name].set_ylim(stateLim[i])
                axs_states[name].set_xlim([0, horizon+1])

            else:
                if i == 0:
                    axs_states[name].scatter(
                        np.arange(h, horizon+h), t[:horizon, plotStateCols[name]],
                        marker='X', edgecolors='k', s=32, label=k
                    )
                else:
                    axs_states[name].scatter(
                        np.arange(h, horizon+h), t[:horizon, plotStateCols[name]],
                        marker='X', edgecolors='k', s=32
                    )
    fig_state.text(x=0.5, y=0.03, s="steps", fontsize=10)
    fig_state.suptitle("State evolution", fontsize=10)
    fig_state.legend(fontsize=5)

    fig_state.tight_layout(rect=[0, 0.05, 1, 0.98])

    if to_numpy:
        canvas = FigureCanvas(fig_state)
        canvas.draw()
        img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        img = img.reshape(fig_state.canvas.get_width_height()[::-1] + (3,))
        plt.close("all")
        return img


    if dir is not None:
        dir_state = os.path.join(dir, "state")
        if not os.path.exists(dir_state):
            os.makedirs(dir_state)
        name = os.path.join(dir_state, f"{file_name}.png")
        fig_state.savefig(name)

        dir_3d = os.path.join(dir, "3d")
        if not os.path.exists(dir_3d):
            os.makedirs(dir_3d)
        name = os.path.join(dir_3d, f"{file_name}_3d.png")
        fig_3d.savefig(name)

    if seq is not None:
        maxA = len(plotActionCols)
        fig_act = plt.figure(figsize=(30, 30))
        for i, name in enumerate(plotActionCols):
            ax = fig_act.add_subplot(maxA, 1, i+1)
            ax.set_ylabel(f'{name}')
            ax.plot(seq[0, :horizon+h, plotActionCols[name]])

        #plt.tight_layout()
        if dir is not None:
            dir_act = os.path.join(dir, "act")
            if not os.path.exists(dir_act):
                os.makedirs(dir_act)
            name = os.path.join(dir, "act", f"{file_name}-actions.png")
            fig_act.savefig(name)
    plt.close("all")


def gen_imgs(dict, tau):
    plotStateCols={
        "x (m)":0 , "y (m)": 1, "z (m)": 2,
        "roll (deg)": 3, "pitch (deg)": 4, "yaw (deg)": 5,
        "u (m/s)": 6, "v (m/s)": 7, "w (m/s)": 8,
        "p (deg/s)": 9, "q (deg/s)": 10, "r (deg/s)": 11
    }

    stateLim = [
        [-1., 1.], [-1., 1.], [-1., 1.], [-180., 180.], [-180., 180.], [-180., 180.],
        [-1., 1.], [-1., 1.], [-1., 1.], [-10., 10.], [-10., 10.], [-10., 10.]
    ]

    plotActionCols={
        "Fx (N)": 0, "Fy (N)": 1, "Fz (N)": 2,
        "Tx (N.m)": 3, "Ty (N.m)": 4, "Tz (N.m)": 5
    }
    imgs = []
    for t in tau:
        img = plot_traj(
            trajs=dict,
            histories=[1, 0],
            plotStateCols=plotStateCols,
            stateLim=stateLim,
            horizon=t,
            to_numpy=True)
        imgs.append(img)
    return imgs


def axs_roll(models, histories, axis,
              plotStateCols, plotActionCols, horizon,
              dir, device, img_name_prefix="foo"):
    trajs = {}
    seq = torch.zeros(1, horizon+10, 6).to(device)
    seq[:, :, axis] = 100.
    for model, h in zip(models, histories):
        if model.sDim == 13:
            init = np.zeros((1, h, 13))
            init[:, :, 6] = 1.
            rep = "quat"
        elif model.sDim == 18:
            init = np.zeros((1, h, 18))
            rot = np.eye(3)
            init[:, :, 3:3+9] = np.reshape(rot, (9,))
            rep = "rot"

        init = init.astype(np.float32)
        init = torch.from_numpy(init).to(device)
        pred = rollout(model, init, seq, h, device, horizon)

        trajs[model.name + "_rand"] = traj_to_euler(pred.cpu().numpy(), rep=rep)
        print(pred.isnan().any())
    names = [n.name for n in models]
    plot_traj(trajs, seq.cpu(), histories, names, plotStateCols, plotActionCols, horizon, dir, file_name=img_name_prefix)


def zero_roll(models, histories,
              plotStateCols, plotActionCols, horizon,
              dir, device, img_name_prefix="foo"):
    trajs = {}
    seq = torch.zeros(1, horizon+10, 6).to(device)

    for model, h in zip(models, histories):
        if model.sDim == 13:
            init = np.zeros((1, h, 13))
            init[:, :, 6] = 1.
            rep = "quat"
        elif model.sDim == 18:
            init = np.zeros((1, h, 18))
            rot = np.eye(3)
            init[:, :, 3:3+9] = np.reshape(rot, (9,))
            rep = "rot"

        init = init.astype(np.float32)
        init = torch.from_numpy(init).to(device)
        pred = rollout(model, init, seq, h, device, horizon)

        trajs[model.name + "_rand"] = traj_to_euler(pred.cpu().numpy(), rep=rep)
        print(pred.isnan().any())
    names = [n.name for n in models]
    plot_traj(trajs, seq.cpu(), histories, names, plotStateCols, plotActionCols, horizon, dir, file_name=img_name_prefix)


def rand_roll(models, histories, 
              plotStateCols, plotActionCols, horizon,
              dir, device, img_name_prefix="foo"):
    trajs = {}
    seq = 5. * torch.normal(
        mean=torch.zeros(1, horizon+10, 6),
        std=torch.ones(1, horizon+10, 6)).to(device)

    for model, h in zip(models, histories):
        if model.sDim == 13:
            init = np.zeros((1, h, 13))
            init[:, :, 6] = 1.
            rep = "quat"
        elif model.sDim == 18:
            init = np.zeros((1, h, 18))
            rot = np.eye(3)
            init[:, :, 3:3+9] = np.reshape(rot, (9,))
            rep = "rot"

        init = init.astype(np.float32)
        init = torch.from_numpy(init).to(device)
        pred = rollout(model, init, seq, h, device, horizon)

        trajs[model.name + "_rand"] = traj_to_euler(pred.cpu().numpy(), rep=rep)
        print(pred.isnan().any())
    names = [n.name for n in models]
    plot_traj(trajs, seq.cpu(), histories, names, plotStateCols, plotActionCols, horizon, dir, file_name=img_name_prefix)


def traj_to_euler(traj, rep="rot", deg=True):
    if rep == "rot":
        rot = traj[:, 3:3+9].reshape((-1, 3, 3))
        r = R.from_matrix(rot)
    elif rep == "quat":
        quat = traj[:, 3:3+4]
        r = R.from_quat(quat)
    else:
        raise NotImplementedError
    pos = traj[:, :3]
    euler = r.as_euler('XYZ', degrees=deg)
    
    vel = traj[:, -6:]
    if deg:
        vel[:, -3:] = vel[:, -3:]*180./np.pi

    traj = np.concatenate([pos, euler, vel], axis=-1)
    return traj
