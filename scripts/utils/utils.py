import torch
import numpy as np
import yaml

import warnings


# tdtype = torch.double
# npdtype = np.double

tdtype = torch.float32
npdtype = np.float32

torch.set_default_dtype(tdtype)

def diag(tensor):
    diag_matrix = tensor.unsqueeze(1) * torch.eye(len(tensor), device=tensor.device)
    return diag_matrix


def diag_embed(tensor):
    return torch.stack([diag(s_) for s_ in tensor]) if tensor.dim() > 1 else diag(tensor)


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