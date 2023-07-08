import torch
import numpy as np
import yaml

import warnings


npdtype= np.double
dtype = torch.double

def load_param(yaml_file):
    with open(yaml_file, "r") as stream:
        dict = yaml.safe_load(stream)
    return dict

def get_device(gpu=0, cpu=False):
    if not cpu:
        cpu = not torch.cuda.is_available()
        if cpu:
            warnings.warn("Asked for GPU but torch couldn't find a Cuda capable device")

    device = torch.device(f"cuda:{gpu}" if not cpu else "cpu")
    return device