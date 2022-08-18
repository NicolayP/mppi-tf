# General program to train a model using torch.
import argparse
import warnings
import yaml
import os


def parse_args():
    parser = argparse.ArgumentParser(
        prog="train-network-torch",
        description="General program to train a network using torch."
    )
    parser.add_argument(
        'topology', metavar='t', type=str,
        help='Yaml file with the topology configuraiton.'
    )
    parser.add_argument(
        'hyper', metavar='h', type=str,
        help='Yaml file with the hyperparameters of the optimizer.'
    )
    parser.add_argument(
        'data', metavar='d', type=str,
        help='Yaml file containing the training data information.'
    )
    parser.add_argument(
        'train_param', metavar='p', type=str,
        help='Yaml file containing the training parameters.'
    )
    parser.add_argument(
        '-e', '--epoch', type=int, default=20,
        help='number of epoch used for training.'
    )
    parser.add_argument(
        '-o', '--onnx', action=argparse.BooleanOptionalAction,
        help='save the model as onnx model'
    )
    parser.add_argument(
        '--onnx_dir', type=str, default=None,
        help='if the onnx flag is activated, \
            this flag indicates the saving directory \
            for the onnx model'
    )
    parser.add_argument(
        '-g', '--gpu', action=argparse.BooleanOptionalAction,
        help='Wether to train on gpu device or cpu')

    parser.add_argument(
        '-l', '--log', type=str,
        help='Log directory for the training.'
    )

    args = parser.parse_args()

    if args.onnx and (args.onnx_dir is None):
        parser.error("--onnx requires --onnx_dir.")

    return args

args = parse_args()

import torch
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from scripts.src_torch.models.torch_utils import learn, save_model

def get_model(topology):
    pass

def get_optimizer(hyperparams):
    pass

def get_train_params(trainparams):
    pass

def get_dataset(data):
    pass


def main():
    model = get_model(args.topology)
    optim, loss_fn = get_optimizer(args.hyper)
    train_params = get_train_params(args.train_param)
    dataset = get_dataset(args.data)

    ds = (
        torch.utils.data.DataLoader(
            dataset,
            **train_params), 
        None
    )
    
    use_cuda = False
    if args.gpu:
        use_cuda = torch.cuda.is_available()
        if not use_cuda:
            warnings.warn("Asked for GPU but torch couldn't find a Cuda capable device")

    device = torch.device("cuda:0" if use_cuda else "cpu")

    writer = None
    if args.log is not None:
        writer = SummaryWriter(args.log)

    learn(ds, model, loss_fn, optim, writer, args.epoch, device)



if __name__ == "__main__":
    main()