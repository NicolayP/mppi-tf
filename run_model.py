# General program to train a model using torch.
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        prog="run-models-torch",
        description="General program to run and plot models using torch."
    )

    parser.add_argument(
        'nn', metavar='n', type=str,
        help='Yaml file containing the configuration for neural netowrk.'
    )

    parser.add_argument(
        'fossen', metavar='f', type=str,
        help='Yaml file containing the configuration for the AUV model.'        
    )

    parser.add_argument(
        '-g', '--gpu', action='store_true',
        help='Wether to train on gpu device or cpu')

    parser.add_argument(
        '-l', '--log', type=str, default='.',
        help='Log directory for the training.'
    )

    args = parser.parse_args()

    return args

args = parse_args()

import torch
torch.manual_seed(1)
import numpy as np
import pandas as pd
from scripts.src_torch.models.auv_torch import AUVFossen, AUVFossenWrapper, VelPred, StatePredictorHistory
from scripts.src_torch.models.torch_utils import rand_roll, zero_roll, ListDataset, val, axs_roll
from scripts.src.misc.utile import parse_config, npdtype, dtype

import os

import warnings
from tqdm import tqdm
from datetime import datetime


def get_model(config, device):
    type = config['type']
    if type == 'velPred':
        h = config['history']
        sDim = config['sDim']
        aDim = config['aDim']
        t = config['topology']    
        step_model =  VelPred(in_size=h*(sDim-3+aDim), topology=t)
        path = config['weights']
        file = os.path.join(path, f"e_{e}.pth")
        step_model.load_state_dict(torch.load(file))
        step_model.eval()
        step_model.to(device)
        model = StatePredictorHistory(step_model, dt=0.1, h=h).to(device)
        return model
    
    if type == "fossen":
        return AUVFossenWrapper(AUVFossen(config))

def get_dataset(dir, config):
    data_dir = dir
    dir_name = os.path.basename(data_dir)
    dfs = []
    files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
    for f in tqdm(files, desc=f"Dir {dir_name}", ncols=100, colour="blue"):
        csv = os.path.join(data_dir, f)
        df = pd.read_csv(csv)
        # TEMPORARY: used for current bluerov dataset that have those entries for some reason
        #df = df.drop(['Time', 'header.seq', 'header.stamp.secs', 'header.stamp.nsecs', 'child_frame_id'], axis=1)
        if 'x' not in df.columns:
            print('\n' + csv)
        df = df.astype(npdtype)
        dfs.append(df)
    dataset = ListDataset(dfs, steps=config['steps'], history=config['history'], rot=config['rot'])
    return dataset

use_cuda = False
if args.gpu:
    use_cuda = torch.cuda.is_available()
    if not use_cuda:
        warnings.warn("Asked for GPU but torch couldn't find a Cuda capable device")

device = torch.device("cuda:0" if use_cuda else "cpu")

config_nn = parse_config(args.nn)
config_fossen = parse_config(args.fossen)


model_fossen = get_model(config_fossen, device)

dir = "/home/pierre/workspace/uuv_ws/src/mppi_ros/scripts/mppi_tf/data_train/val_rexrov/"
ds = get_dataset(dir, config_nn)

plotStateCols={
    "x":0 , "y": 1, "z": 2,
    "roll": 3, "pitch": 4, "yaw": 5,
    "u": 6, "v": 7, "w": 8,
    "p": 9, "q": 10, "r": 11
}

plotActionCols={
    "Fx": 0, "Fy": 1, "Fz": 2,
    "Tx": 3, "Ty": 4, "Tz": 5
}

stamp = datetime.now().strftime("%Y.%m.%d-%H:%M:%S")
dir_val = os.path.join("comparaison-rexrov/val")
dir_rand = os.path.join("comparaison-rexrov/rand")
dir_zeros = os.path.join("comparaison-rexrov/zeros")
dir_x = os.path.join("comparaison-rexrov/x")
dir_y = os.path.join("comparaison-rexrov/y")
dir_z = os.path.join("comparaison-rexrov/z")
dir_yaw = os.path.join("comparaison-rexrov/yaw")
if not os.path.exists(dir_val):
        os.makedirs(dir_val)
if not os.path.exists(dir_rand):
        os.makedirs(dir_rand)
if not os.path.exists(dir_zeros):
        os.makedirs(dir_zeros)
if not os.path.exists(dir_x):
        os.makedirs(dir_x)
if not os.path.exists(dir_y):
        os.makedirs(dir_y)
if not os.path.exists(dir_z):
        os.makedirs(dir_z)
if not os.path.exists(dir_yaw):
        os.makedirs(dir_yaw)

nn_iter = os.path.basename(config_nn["weights"])
for e in range(2, 20, 2):
    model_nn = get_model(config_nn, device)
    nn_iter = e
    val(
        ds, models=[model_nn, model_fossen], metric=torch.nn.MSELoss().to(device), device=device,
        histories=[config_nn['history'], 1], horizon=30, plot=True,
        plotStateCols=plotStateCols, plotActionCols=plotActionCols, dir=dir_val, img_name_prefix=nn_iter
    )

    axs_roll(
        models=[model_nn, model_fossen], histories=[config_nn['history'], 1], axis=0,
        plotStateCols=plotStateCols, plotActionCols=plotActionCols,
        horizon=50, dir=dir_x, device=device, img_name_prefix=nn_iter
    )
    axs_roll(
        models=[model_nn, model_fossen], histories=[config_nn['history'], 1], axis=1,
        plotStateCols=plotStateCols, plotActionCols=plotActionCols,
        horizon=50, dir=dir_y, device=device, img_name_prefix=nn_iter
    )
    axs_roll(
        models=[model_nn, model_fossen], histories=[config_nn['history'], 1], axis=2,
        plotStateCols=plotStateCols, plotActionCols=plotActionCols,
        horizon=50, dir=dir_z, device=device, img_name_prefix=nn_iter
    )

    axs_roll(
        models=[model_nn, model_fossen], histories=[config_nn['history'], 1], axis=5,
        plotStateCols=plotStateCols, plotActionCols=plotActionCols,
        horizon=50, dir=dir_yaw, device=device, img_name_prefix=nn_iter
    )

    zero_roll(
        models=[model_nn, model_fossen], histories=[config_nn['history'], 1],
        plotStateCols=plotStateCols, plotActionCols=plotActionCols,
        horizon=50, dir=dir_zeros, device=device, img_name_prefix=nn_iter
    )

    rand_roll(
        models=[model_nn, model_fossen], histories=[config_nn['history'], 1],
        plotStateCols=plotStateCols, plotActionCols=plotActionCols,
        horizon=50, dir=dir_rand, device=device, img_name_prefix=nn_iter
    )
