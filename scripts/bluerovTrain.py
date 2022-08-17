import os
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from src_torch.models.auv_torch import VelPred, StatePredictorHistory
from src_torch.models.torch_utils import ToSE3Mat, SE3int, FlattenSE3, get_dataloader, train, val, learn, save_model, ListDataset
import glob
import numpy as np
import pandas as pd

params = {'batch_size': 1024,
          'shuffle': True,
          'num_workers': 1}
s=1


csvs = glob.glob('/home/pierre/workspace/uuv_ws/src/mppi_ros/data/bluerovData/clean-*.csv')

dfs = []
for csv in csvs:
    df = pd.read_csv(csv)
    df = df.drop(['Time', 'header.seq', 'header.stamp.secs', 'header.stamp.nsecs', 'child_frame_id'], axis=1)
    df = df.astype(np.float32)
    dfs.append(df)

dataset = ListDataset(dfs, steps=1, history=5)
ds = (torch.utils.data.DataLoader(
        dataset,
        **params), None)

dim = 18 + 6 - 3 #State + aciton - position
max_epoch = 1

loss_fn = torch.nn.MSELoss()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True
#device = "cpu"

histories=[5]
topologies = [
    [32],
    #[32, 32],
    #[32, 32, 32],
    #[32, 32, 32, 32],
    #[64],
    #[64, 64, 64],
    #[64, 64, 64, 64],
    #[128],
    #[128, 128, 128],
    #[128, 128, 128, 128],
    #[128, 128, 128, 128, 128],
]

nns = []
predictors = []
dir = "./bluerov_hist/"
for t in topologies:
    for h in histories:
        print("*"*5, " Training ", "*"*5)
        print("   - h: ", h, "  ; t: ", t)
        nn = VelPred(in_size=h*dim, topology=t).to(device)
        opti = torch.optim.Adam(nn.parameters(), lr=1e-4)
        writer = SummaryWriter(dir + nn.name)
        learn(ds, nn, loss_fn, opti, writer, max_epoch, device)
        predictors.append(StatePredictorHistory(nn, 0.1, h))
        dummy_state, dummy_action = torch.zeros((1, h*(18-3))).to(device), torch.zeros((1, h*6)).to(device)
        save_model(nn, dir, True, (dummy_state, dummy_action), ['x', 'u'], ['vel'])
        nns.append(nn)

