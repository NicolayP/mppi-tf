import torch
import numpy as np
import pandas as pd
import random

import matplotlib.pyplot as plt

from tabulate import tabulate
from tqdm import tqdm
import os
import onnx
from onnx_tf.backend import prepare


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
        traj_samples = self.samples[i]
        j = idx - self.bins[i]
        sub_frame = traj.iloc[j:j+self.s+self.h]
        s = sub_frame[self.x_labels].to_numpy()
        u = sub_frame[self.u_labels].to_numpy()
        x = s[:self.h]
        u = u[:self.h+self.s-1]
        y = s[self.h:self.h+self.s]
        return x, u, y


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, steps=1, history=1):
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
        batch = True
        if x.dim() < 2:
            x = x.unsqueeze(dim=0)
            batch = False
        k = x.shape[0]
        p = x[:, :3].unsqueeze(dim=-1)
        r = x[:, 3:3+9].reshape((-1, 3, 3))

        noHomo = torch.concat([r, p], dim=-1)
        homo = torch.concat([noHomo, self.pad_const.broadcast_to((k, 1, 4))], dim=-2)
        if batch:
            return homo
        else:
            return homo.squeeze()


class SE3int(torch.nn.Module):
    def __init__(self):
        super(SE3int, self).__init__()
        self.skew = Skew()
        self.so3int = SO3int(self.skew)
        pad = torch.Tensor([[[0., 0., 0., 1.]]])
        self.register_buffer('pad_const', pad)
    
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
        return M @ self.exp(tau)

    def exp(self, tau):
        '''
            Computes the exponential map of Tau.

            input:
            ------
                - tau: perturbation in se(3). shape [k, 6] or [6]
            
            output:
            -------
                - Exp(Tau). shape [k, 4, 4] or [4, 4]
        '''
        batch = True
        if tau.dim() < 2:
            batch = False
            tau = tau.unsqueeze(dim=0)
        k = tau.shape[0]
        rho_vec = tau[:, :3]
        theta_vec = tau[:, 3:]

        r = self.so3int.exp(theta_vec)
        p = self.v(theta_vec) @ rho_vec.unsqueeze(dim=-1)
        
        noHomo = torch.concat([r, p], dim=-1)
        homo = torch.concat([noHomo, self.pad_const.broadcast_to((k, 1, 4))], dim=-2)
        if batch:
            return homo
        else:
            return homo.squeeze()

    def v(self, theta_vec):
        '''
            Compute V(\theta) used in the exponential mapping. See 

            input:
            ------
                - theta_vec. Rotation vector \theta * u. Where theta is the
                rotation angle around the unit vector u. Shape [k, 3] or [3]
        '''
        theta = torch.linalg.norm(theta_vec)
        skewT = self.skew(theta_vec)
        a = torch.eye(3)
        b = (1-torch.cos(theta))/torch.pow(theta, 2) * skewT
        c = (theta - torch.sin(theta))/torch.pow(theta, 3) * torch.pow(skewT, 2)
        return a + b + c


class SO3int(torch.nn.Module):
    def __init__(self, skew=None):
        super(SO3int, self).__init__()
        if skew is None:
            self.skew = Skew()
        else:
            self.skew = skew

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
        return R @ self.exp(tau)

    def exp(self, tau):
        '''
            Computes the exponential map of Tau in SO(3).

            input:
            ------
                - tau: perturbation in so(3). shape [k, 3] or [3]

            output:
            -------
                - Exp(Tau). shape [k, 3, 3] or [3, 3]
        '''
        batch = True
        if tau.dim() < 2:
            batch = False
            tau = tau.unsqueeze(dim=0)

        theta = torch.linalg.norm(tau, dim=1)
        u = tau/theta[:, None]

        skewU = self.skew(u)
        a = torch.eye(3)
        b = torch.sin(theta)[:, None, None]*skewU
        c = (1-torch.cos(theta))[:, None, None]*torch.pow(skewU, 2)

        res = a + b + c
        if batch:
            return res
        else:
            return res.squeeze()


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
        batch = True
        if M.dim() < 3:
            M = M.unsqueeze(dim=0)
            vel = vel.unsqueeze(dim=0)
            batch = False
        p = M[:, 0:3, 3]
        r = M[:, 0:3, 0:3].reshape((-1, 9))
        x = torch.concat([p, r, vel], dim=1)
        if batch:
            return x
        else:
            return x.squeeze(dim=0)


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
    t = tqdm(
        enumerate(dataloader),
        desc=f"Epoch: {epoch}",
        ncols=150,
        colour="red",
        leave=False
    )
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
        #print("*"*5, " Prediction ", "*"*5)
        #print(pred)
        l = loss(pred, y)
        l.backward()
        #for name, param in model.named_parameters():
        #    if param.requires_grad:
        #        print("*"*5, f" Param {name} ", "*"*5)
        #        print(param.grad)
        opti.step()

        if writer is not None:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    writer.add_histogram("train/" + name, param, epoch*size + batch)
            for dim in range(6):
                lossDim = loss(pred[:, dim], y[:, dim])
                writer.add_scalar("loss/"+ str(dim), lossDim, epoch*size + batch)
    return l.item(), batch*len(X)


def save_model(model, dir, tf=True, dummy_input=None, input_names=[], output_names=[]):
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
            output_names=output_names
        )
        onnx_model = onnx.load(onnx_filename)
        onnx.checker.check_model(onnx_model)
        print(onnx.helper.printable_graph(onnx_model.graph))

        tf_rep = prepare(onnx_model)
        tf_rep.export_graph(tf_filename)


def learn(dataLoaders, model, loss, opti, writer=None, maxEpochs=1, device="cpu"):
    dls = dataLoaders
    size = len(dls[0].dataset)
    l = np.nan
    current = 0
    t = tqdm(range(maxEpochs), desc="Training", ncols=150, colour="blue", postfix={"loss": f"Loss: {l:>7f} [{current:>5d}/{size:>5d}]"})
    for e in t:
        l, current = train(dataloader=dls[0],
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


def val(dataLoader, models, metric, device="cpu", plot=False, plotCols=None, horizon=50, filename="foo.png"):
    w = 1
    gtTrajs, actionSeqs = dataLoader.dataset.getTrajs()
    gtTrajs = torch.from_numpy(gtTrajs).to(device)
    actionSeqs = torch.from_numpy(actionSeqs).to(device)

    tau = actionSeqs.shape[1]
    if tau > horizon:
        tau = horizon
    errs = {}
    trajs = {}

    samples = gtTrajs.shape[0]
    samples = 10

    with torch.no_grad():
        for m in models:
            h = m.history
            initState = gtTrajs[:, :h]
            pred = []
            state = initState
            for i in range(h, tau):
                nextState = m(state, actionSeqs[:, i-h:i])
                pred.append(nextState.unsqueeze(dim=1))
                state = push_to_tensor(state, nextState)
            pred = torch.concat(pred, dim=1)
            err = metric(pred, gtTrajs[:, h:tau])
        
            errs[m.name] = [err]
            trajs[m.name] = pred

        headers = ["name", "l2"]
        
        print(tabulate([[k,] + v for k, v in errs.items()], headers=headers))

        if plot:
            maxN = len(plotCols)
            for j in tqdm(range(samples)):
                fig = plt.figure(figsize=(60, 60))            
                for i, name in enumerate(plotCols):
                    plt.subplot(maxN, 1, i+1)
                    plt.ylabel(f'{name} [normed]')
                
                    plt.plot(gtTrajs[j, :tau, plotCols[name]], label='gt', marker='.', zorder=-10)
                
                    for m in models:
                        plt.scatter(np.arange(m.history, tau), trajs[m.name][j, :, plotCols[name]],
                                    marker='X', edgecolors='k', label=m.name, s=64)
                    
                fig.legend()
                plt.tight_layout()
                foo = os.path.split(filename)
                name = os.path.join(foo[0], f"traj-{j}-"+foo[1])
                plt.savefig(name)
                plt.close()
    pass
