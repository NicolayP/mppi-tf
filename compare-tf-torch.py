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

import numpy as np
import warnings

import tensorflow as tf
from scripts.src.models.nn_model import LaggedNNAUVSpeed
import scripts.src.models.model_utils as tf_utils

import torch
from scripts.src_torch.models.auv_torch import VelPred, StatePredictorHistory
import scripts.src_torch.models.torch_utils as torch_utils

from scripts.src.misc.utile import parse_config, npdtype, dtype


def get_tensorflow_model(config):
    sDim = config['sDim']
    aDim = config['aDim']
    h = config['history']
    t = config['topology']
    w = config['trainedFileTorch']

    internal = tf.saved_model.load(config['trainedFile']).signatures['serving_default']
    return internal

def get_torch_model(config, device):
    sDim = config['sDim']
    aDim = config['aDim']
    h = config['history']
    t = config['topology']
    w = config['trainedFileTorch']

    pred = VelPred(in_size=h*(sDim-3+aDim), topology=t)
    pred.load_state_dict(torch.load(w))
    pred.eval()
    return pred.to(device)

def comp_rollout(models, histories, plotStateCols, plotActionCols, horizon, dir, device):
    trajs = {}
    seq = 5. * np.random.normal(
        size=(1, horizon+10, 6)
    )
    seq = seq.astype(np.float32)
    for k, h in zip(models, histories):
        model = models[k]
        init = np.zeros(shape=(1, h, 18))
        rot = np.eye(3)
        init[:, :, 3:3+9] = np.reshape(rot, (9,))
        init = init.astype(np.float32)
        if k == "tf":
            tf_init = tf.convert_to_tensor(init, dtype=dtype)
            tf_seq = tf.convert_to_tensor(seq, dtype=dtype)
            traj = tf_utils.rollout(model, tf_init[..., None], tf_seq[..., None], h, horizon=horizon).numpy()
            traj = np.squeeze(traj, axis=-1)
            traj = np.concatenate([init[0], traj], axis=0)

        elif k == "torch":
            torch_init = torch.from_numpy(init).to(device)
            torch_seq = torch.from_numpy(seq).to(device)
            traj = torch_utils.rollout(model, torch_init, torch_seq, h, horizon=horizon).cpu().numpy()
            traj = np.concatenate([init[0], traj], axis=0)
        trajs[k + "_" +  model.name] = tf_utils.traj_to_euler(traj, rep="rot")

    tf_utils.plot_traj(trajs, seq, histories, plotStateCols, plotActionCols, horizon+4, dir)

def main():
    use_cuda = False
    if args.gpu:
        use_cuda = torch.cuda.is_available()
        if not use_cuda:
            warnings.warn("Asked for GPU but torch couldn't find a Cuda capable device")

    device = torch.device("cuda:0" if use_cuda else "cpu")
    
    config = parse_config(args.conf)
    h = config['history']

    torch_model = get_torch_model(config, device)
    tf_model = get_tensorflow_model(config)
    
    tf_state_model = LaggedNNAUVSpeed(
        k=1,
        h=config['history'],
        dt=0.1,
        velPred=tf_model
    )

    torch_state_model = StatePredictorHistory(torch_model, 0.1, h).to(device)

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

    comp_rollout(
        models={"tf": tf_state_model, "torch": torch_state_model},
        histories=[h, h],
        plotStateCols=plotStateCols,
        plotActionCols=plotActionCols,    
        horizon=50,
        dir=None,
        device=device
    )
    pass

if __name__ == "__main__":
    main()