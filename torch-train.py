# General program to train a model using torch.
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        prog="train-network-torch",
        description="General program to train a network using torch."
    )

    parser.add_argument(
        'params', metavar='p', type=str,
        help='Yaml file containing the configuration for training.'
    )

    parser.add_argument(
        '-t', '--tf', action='store_true',
        help='save the model as a tensorflow model (using onnx)'
    )

    parser.add_argument(
        '--save_dir', type=str, default="torch-training",
        help="saving directory for the model in it's different formats"
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
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import warnings
import os
from scripts.src_torch.models.auv_torch import VelPred, StatePredictorHistory

from scripts.src_torch.models.torch_utils import ListDataset, learn, rand_roll, save_model, val

# TODO: SEPARATE FROM TF IMPLEMENTATION
from scripts.src.misc.utile import parse_config, npdtype, dtype
from tqdm import tqdm
from datetime import datetime

def get_model(config, device):
    type = config['model']['type']
    h = config['history']

    if type == 'velPred':
        sDim = config['model']['sDim']
        aDim = config['model']['aDim']
        t = config['model']['topology']    
        return VelPred(in_size=h*(sDim-3+aDim), topology=t).to(device)
    if type == "liePred":
        return LieAUVNN(h=h).to(device)

def get_optimizer(model, config):
    type = config['optimizer']['type']
    params = model.parameters()
    lr = config['optimizer']['lr']
    if type == 'adam':
        return torch.optim.Adam(params, lr=lr)

def get_loss(config, device):
    type = config['loss']['type']
    if type == "l2":
        return torch.nn.MSELoss().to(device)
    elif type == "geodesic":
        return GeodesicLoss().to(device)

def get_train_params(config, device):
    return config['training_params']

def get_dataset(config, device):
    type = config['dataset']['type']
    data_dir = config['dataset']['dir']
    dir_name = os.path.basename(data_dir)
    multi_dir = config['dataset']['multi_dir']
    multi_file = config['dataset']['multi_file']

    dfs = []
    if multi_dir:
        dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        for d in tqdm(dirs, desc="Directories", ncols=100, colour="green"):
            sub_dir = os.path.join(data_dir, d)
            files = [f for f in os.listdir(sub_dir) if os.path.isfile(os.path.join(sub_dir, f))]
            for f in tqdm(files, leave=False, desc=f"Dir {d}", ncols=100, colour="blue"):
                csv = os.path.join(sub_dir, f)
                df = pd.read_csv(csv)
                if 'x' not in df.columns:
                    print('\n' + csv)
                df = df.astype(npdtype)
                dfs.append(df)
    else:
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
    if type == "ListDataset":
        dataset = ListDataset(dfs, steps=config['steps'], history=config['history'], rot=config['dataset']['rot'])
    elif type == "ListDatasetLie":
        dataset = ListDataset(dfs, steps=config['steps'], history=config['history'], rot="quat")
    return dataset

def get_device(gpu=False):
    use_cuda = False
    if gpu:
        use_cuda = torch.cuda.is_available()
        if not use_cuda:
            warnings.warn("Asked for GPU but torch couldn't find a Cuda capable device")
    return torch.device("cuda:0" if use_cuda else "cpu")

def make_dirs(save_dir, model_name):
    '''
        Creates a set of directories where the training information
        and validation will be logged.

        inputs:
        -------
            - save_dir, string: the path to the saving directory.
            - model_name, string: the name of the model being trained.
    '''
    stamp = datetime.now().strftime("%Y.%m.%d-%H:%M:%S")
    dir = os.path.join(save_dir, model_name, stamp)
    dir_rand = os.path.join(dir, "rand")
    dir_ckpt = os.path.join(dir, "ckpt")

    if not os.path.exists(dir):
        os.makedirs(dir)
    if not os.path.exists(dir_rand):
        os.makedirs(dir_rand)
    if not os.path.exists(dir_ckpt):
        os.makedirs(dir_ckpt)

def main():
    device = get_device(args.gpu)
    config = parse_config(args.params)
    model = get_model(config, device)
    optim = get_optimizer(model, config)
    loss_fn = get_loss(config, device)
    train_params = get_train_params(config, device)
    dataset = get_dataset(config, device)
    epochs = config['epochs']
    h = config['history']
    steps = config['steps']

    make_dirs(config['log_dir'], model.name)


    ds = (
        torch.utils.data.DataLoader(
            dataset,
            **train_params), 
        None
    )

    writer = None
    if args.log is not None:
        path = os.path.join(dir, args.log)
        writer = SummaryWriter(path)
    learn(ds, model, loss_fn, optim, writer, epochs, device, "foo", ckpt_dir)

    samples = 1

    dummy_state = torch.zeros((samples, h, 13)).to(device)
    dummy_action = torch.zeros((samples, h, 6)).to(device)
    dummy_inputs = (dummy_state, dummy_action)
    input_names = ["x", "u"]
    output_names = ["x_next"]
    dynamic_axes = {
        "x": {0: "kx"},
        "u": {0: "ku"},
        "x_next": {0: "kv"}
    }

    #state_model = LieAUVWrapper(model.step_nn).to(device)

    state_model = StatePredictorHistory(model, dt=0.1, h=h).to(device)

    plotStateCols={
        "x":0 , "y": 1, "z": 2,
        "qx": 3, "qy": 4, "qz": 5, "qw": 6,
        "u": 7, "v": 8, "w": 9,
        "p": 10, "q": 11, "r": 12
    }

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

    #rand_roll(
    #    models=[state_model], histories=[h],
    #    plotStateCols=plotStateCols, plotActionCols=plotActionCols,
    #    horizon=50, dir=dir_rand, device=device
    #)

    val(
        ds[0], models=[state_model], metric=torch.nn.MSELoss().to(device), device=device,
        histories=[h], horizon=5, plot=True,
        plotStateCols=plotStateCols, plotActionCols=plotActionCols, dir=dir
    )

    #save_model(
    #    model, dir=dir, tf=args.tf,
    #    dummy_input=dummy_inputs, input_names=input_names,
    #    output_names=output_names, dynamic_axes=dynamic_axes
    #)

    # Lie Part
    #save_model(
    #    model.step_nn, dir=dir, tf=args.tf,
    #    dummy_input=dummy_inputs, input_names=input_names,
    #    output_names=output_names, dynamic_axes=dynamic_axes
    #)

    return

if __name__ == "__main__":
    main()