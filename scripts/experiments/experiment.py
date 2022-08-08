import os
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from src_torch.models.auv_torch import VelPred, StatePredictorHistory
from src_torch.models.torch_utils import ToSE3Mat, SE3int, FlattenSE3, get_dataloader, train, val, learn, save_model

params = {'batch_size': 1024,
          'shuffle': True,
          'num_workers': 1}
s=1

dls = {1: get_dataloader("transitions.csv", params, steps=s, history=1),
       2: get_dataloader("transitions.csv", params, steps=s, history=2),
       3: get_dataloader("transitions.csv", params, steps=s, history=3),
       4: get_dataloader("transitions.csv", params, steps=s, history=4),
       5: get_dataloader("transitions.csv", params, steps=s, history=5)
      }
dim = 18 + 6 - 3 #State + aciton - position
max_epoch = 100

loss_fn = torch.nn.MSELoss()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True
#device = "cpu"

histories=[1, 2, 3, 4, 5]
topologies = [
    #[32],
    #[32, 32, 32],
    #[32, 32, 32, 32],
    #[64],
    #[64, 64, 64],
    #[64, 64, 64, 64],
    #[128],
    #[128, 128, 128],
    [128, 128, 128, 128]
]

nns = []
predictors = []
dir = "./run_hist/"
for t in topologies:
    for h in histories:
        print("*"*5, " Training ", "*"*5)
        print("   - h: ", h, "  ; t: ", t)
        nn = VelPred(in_size=h*dim, topology=t).to(device)
        opti = torch.optim.Adam(nn.parameters(), lr=1e-4)
        writer = SummaryWriter(dir + nn.name)
        learn(dls[h], nn, loss_fn, opti, writer, max_epoch, device)
        predictors.append(StatePredictorHistory(nn, 0.1, h))
        save_model(nn, os.path.join(dir, f"{nn.name}.pt"))
        nns.append(nn)

plotCols = {'x': 0, 'y': 1, 'z': 2,
            'r00': 3, 'r01': 4, 'r02': 5,
            'r10': 6, 'r11': 7, 'r12': 8,
            'r20': 9, 'r21': 10, 'r22': 11,
            'vx': 12, 'vy': 13, 'vz': 14,
            'vp': 15, 'vq': 16, 'vr': 17}

for pred in predictors:
    val(dls[1][1], [pred], loss_fn, plot=True, plotCols=plotCols, horizon=50, filename=os.path.join(dir, f"{pred.name}.png"))

val(dls[1][1], predictors, loss_fn, plot=True, plotCols=plotCols, horizon=50, filename=os.path.join(dir, "all.png"))
