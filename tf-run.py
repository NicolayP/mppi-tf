import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        prog="run-tensorflow-network",
        description="Runs a tensorflow network with random inputs drawn from a normal\
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

import tensorflow as tf
import warnings
from scripts.src.models.nn_model import LaggedNNAUVSpeed
from scripts.src.models.model_utils import rand_rollout
from scripts.src.misc.utile import parse_config, npdtype, dtype
from tqdm import tqdm
from datetime import datetime


def get_model(config):
    sDim = config['sDim']
    aDim = config['aDim']
    h = config['history']
    t = config['topology']
    w = config['trainedFileTorch']

    internal = tf.saved_model.load(config['trainedFile']).signatures['serving_default']
    return internal

def main():
    config = parse_config(args.conf)
    h = config['history']

    model = get_model(config)
    state_model = LaggedNNAUVSpeed(
        k=1,
        h=config['history'],
        dt=0.1,
        velPred=model
    )

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

    rand_rollout(
        models=[state_model],
        histories=[h],
        plotStateCols=plotStateCols,
        plotActionCols=plotActionCols,    
        horizon=50,
        dir=None,
    )
    pass

if __name__ == "__main__":
    main()