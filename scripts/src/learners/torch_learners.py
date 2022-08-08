from enum import auto
import torch
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt

class OneStepDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        '''
            Constructor:
            input:
            ------
                - data (x, u) pair of trajectories.
                    x is of shape [h, Tau, 13]
                    u is of shape [h, Tau, 6]
        '''
        self.x = data[:, :, :-6]
        self.u = data[:, :, -6:]
        self.y = self.x[:, 1:]
        self.x = self.x[:, :-1]
        self.u = self.u[:, :-1]

        xs = self.x.shape
        ys = self.y.shape
        us = self.u.shape

        self.x = np.reshape(self.x, (xs[0]*xs[1], xs[2]))
        self.u = np.reshape(self.u, (us[0]*us[1], us[2]))
        self.y = np.reshape(self.y, (ys[0]*ys[1], ys[2]))

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.u[idx], self.y[idx]


class MultiStepDataset(torch.utils.data.Dataset):
    def __init__(self, data, winSize):
        '''
            Constructor:
            input:
            ------
                - data (x, u) pair of trajectories.
                    x is of shape [h, Tau, 13]
                    u is of shape [h, Tau, 6]
        '''
        self.x = data[:, :, :-6]
        self.u = data[:, :, -6:]
        xs = self.x.shape
        self.Tau = xs[1]
        self.w = winSize
        self.h = xs[0]
        self.trajSampl = (self.Tau - self.w + 1)
        pass

    def __len__(self):
        return self.h*self.trajSampl

    def __getitem__(self, idx):
        ij = np.unravel_index([idx], (self.h, self.trajSampl))
        i = ij[0][0]
        j = ij[1][0]
        x = self.x[i, j:j+self.w]
        u = self.u[i, j:j+self.w-1]
        return x, u


def train(dataloader, model, lossFn, optimizer, writer=None, epoch=None, device="cpu", multi=False, verbose=False):
    size = len(dataloader.dataset)
    model.train()
    for batch, data in enumerate(dataloader):
        if multi:
            X, U = data
            X, U = X.to(device), U.to(device)
            y = X[:, 1:, -6:]
            pred = model(X[:, :1], U)[:, :, -6:]
        else:
            X, u, y = data
            X, u, y = X.to(device), u.to(device), y.to(device)
            y = y[:, -6:]
            pred = model(X, u)[:, -6:]
        
        optimizer.zero_grad()
        loss = lossFn(pred, y)
        loss.backward()
        optimizer.step()

        if writer is not None:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    writer.add_histogram("train/" + name, param, epoch*size + batch)
            for dim in range(6):
                if multi:
                    lossDim = lossFn(pred[:, :, dim], y[:, :, dim])
                else:
                    lossDim = lossFn(pred[:, dim], y[:, dim])
                writer.add_scalar("loss/"+ str(dim), lossDim, epoch*size + batch)
                    

    if epoch % 10 == 0 and verbose:
        loss, current = loss.item(), batch * len(X)
        print(f'Loss: {loss:>7f} [{current:>5d}/{size:>5d}]')

def test(osDataloader, msDataloader, model, lossFn, writer, step, device="cpu"):
    lossMulti = testMulti(msDataloader, model, lossFn, device)
    lossStep = testStep(osDataloader, model, lossFn, device)
    
    for dim in range(6):
        writer.add_scalar("stepTest/"+ str(dim), lossStep[dim], step)
        writer.add_scalar("multiTest/"+ str(dim), lossMulti[dim], step)

def testMulti(msDataloader, model, lossFn, device="cpu", split=True):
    numBatches = len(msDataloader)
    if split:
        testLoss = torch.zeros(6)
    with torch.no_grad():

        for data in msDataloader:
            X, U = data
            X, U = X.to(device), U.to(device)
            y = X[:, 1:, -6:]
            # Wrap the model around the multi.
            if not model.multi:

                state = torch.squeeze(X[:, :1], dim=1)
                pred = X[:, :1]
                tau = U.shape[1]
                for i in range(tau):
                    u = U[:, i]
                    nextState = model(state, u)
                    state = nextState
                    pred = torch.concat(
                        [pred, torch.unsqueeze(nextState, dim=1)],
                    dim=1)
                pred = pred[:, 1:, -6:]
            else:
                pred = model(X[:, :1], U)[:, :, -6:]
            if split:
                for i in range(6):
                    testLoss[i] += lossFn(pred[:, :, i], y[:, :, i]).item()
            else:
                testLoss += lossFn(pred, y).item()

    testLoss /= numBatches
    return testLoss

def testStep(osDataloader, model, lossFn, device="cpu", split=True):
    numBatches = len(osDataloader)
    testLoss = 0.
    if split:
        testLoss = torch.zeros(6)
    with torch.no_grad():
        for data in osDataloader:
            X, u, y = data
            X, u, y = X.to(device), u.to(device), y.to(device)
            y = y[:, -6:]
            # Wrap the model around the multi.
            if model.multi:
                pred = model(torch.unsqueeze(X, dim=1),
                            torch.unsqueeze(u, dim=1))
                pred = pred[:, 0, -6:]
            else:
                pred = model(X, u)[:, -6:]
            if split:
                for i in range(6):
                    testLoss[i] += lossFn(pred[:, i], y[:, i]).item()
            else:
                testLoss += lossFn(pred, y).item()
    testLoss /= numBatches
    return testLoss

def val(models, gtTraj, actionSeq, metric, device="cpu", plot=False, plotCols=None, autoregressive=False):
    gtTraj = torch.from_numpy(gtTraj).to(device)
    actionSeq = torch.from_numpy(actionSeq).to(device)
    initState = gtTraj[:, :1]
    tau = actionSeq.shape[1]
    errs = {}
    traj = {}
    with torch.no_grad():
        for m in models:
            if m.multi:
                pred = m(initState, actionSeq)
            else:
                pred = []
                state = torch.squeeze(initState, dim=1)
                for i in range(tau):
                    if autoregressive:
                        nextState = m(state, actionSeq[:, i])
                    else:
                        nextState = m(gtTraj[:, i], actionSeq[:, i])
                    pred.append(torch.unsqueeze(nextState, dim=1))
                    state = nextState
                pred = torch.concat(pred, dim=1)
            if autoregressive:
                err = metric(pred, gtTraj[:, 1:])
            else:
                err = metric(pred, gtTraj[:, 1:, 9:9+6])
            errs[m.name] = [err]
            traj[m.name] = pred

        headers = ["name", "l2"]
        
        print(tabulate([[k,] + v for k, v in errs.items()], headers=headers))

        if plot:
            maxN = len(plotCols)
            plt.figure(figsize=(30, 20))
            for i, name in enumerate(plotCols.keys()):
                plt.subplot(maxN, 1, i+1)
                plt.ylabel(f'{name} [normed]')
            
                plt.plot(gtTraj[0, :, plotCols[name]], label='gt', marker='.', zorder=-10)
            
                for m in models:
                    plt.scatter(np.arange(1, tau+1), traj[m.name][0, :, plotCols[name]],
                                marker='X', edgecolors='k', label=m.name, s=64)
                if i==0:
                    plt.legend()

def val_step(models, gtTraj, acitonSeq, metric, device="cpu", plot=False, plotCols=None):
    gtTraj = torch.from_numpy(gtTraj).to(device)
    acitonSeq = torch.from_numpy(acitonSeq).to(device)
    xShape = gtTraj.shape
    print(xShape)
    x = gtTraj[:, 0:-1]
    u = acitonSeq
    y = gtTraj[:, 1:]
    pass