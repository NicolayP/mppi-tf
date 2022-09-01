import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        prog="run-torch-network",
        description="Runs a torch network with random inputs drawn from a normal\
            law with std defined in a yaml file."
    )

    parser.add_argument(
        'conf', metavar='c', type=str,
        help='Config file containing loading information for the network'
    )

    parser.add_argument(
        '--std', type=str, default=None,
        help='yaml file containing std values for the noise generation'
    )

    parser.add_argument(
        '-g', '--gpu', action=argparse.BooleanOptionalAction,
        help='Wether to train on gpu device or cpu'
    )

    args = parser.parse_args()
    return args

args = parse_args()

import torch
import warnings
from scripts.src_torch.models.auv_torch import VelPred, StatePredictorHistory
from scripts.src_torch.models.torch_utils import rand_roll
from scripts.src.misc.utile import parse_config


def get_model(config, device):
    sDim = config['sDim']
    aDim = config['aDim']
    h = config['history']
    t = config['topology']
    w = config['trainedFileTorch']

    pred = VelPred(in_size=h*(sDim-3+aDim), topology=t)
    pred.load_state_dict(torch.load(w))
    pred.eval()
    return pred.to(device)

def main():
    use_cuda = False
    if args.gpu:
        use_cuda = torch.cuda.is_available()
        if not use_cuda:
            warnings.warn("Asked for GPU but torch couldn't find a Cuda capable device")

    device = torch.device("cuda:0" if use_cuda else "cpu")

    config = parse_config(args.conf)
    h = config['history']

    model = get_model(config, device)
    state_model = StatePredictorHistory(model, 0.1, h).to(device)

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

    rand_roll(
        models=[state_model],
        histories=[h],
        plotStateCols=plotStateCols,
        plotActionCols=plotActionCols,    
        horizon=100,
        dir=None,
        device=device
    )
    pass

if __name__ == "__main__":
    main()