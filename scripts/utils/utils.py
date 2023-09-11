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


def rollout(model, init, seq, h=1, device="cpu", horizon=50):
    state = init
    with torch.no_grad():
        pred = []
        for i in range(h, horizon+h):
            nextState = model(state, seq[:, i-h:i])
            pred.append(nextState)
            state = push_to_tensor(state, nextState)
        traj = torch.concat(pred, dim=0)
        return traj


def plot_traj(trajs, seq=None, histories=None, plotStateCols=None, plotActionCols=None, horizon=50, dir=".", file_name="foo"):
    '''
        Plot trajectories and action sequence.
        inputs:
        -------
            - trajs: dict with model name as key and trajectories entry. If key is "gt" then it is assumed to be
                the ground truth trajectory.
            - seq: Action Sequence associated to the generated trajectoires. If not None, plots the 
                action seqence.
            - h: list of history used for the different models, ignored when model entry is "gt".
            - plotStateCols: Dict containing the state axis name as key and index as entry
            - plotAcitonCols: Dict containing the action axis name as key and index as entry.
            - horizon: The horizon of the trajectory to plot.
            - dir: The saving directory for the generated images.
    '''
    maxS = len(plotStateCols)
    maxA = len(plotActionCols)
    fig_state = plt.figure(figsize=(50, 50))
    for k, h in zip(trajs, histories):
        t = trajs[k]
        for i, name in enumerate(plotStateCols):
            m, n = np.unravel_index(i, (2, 6))
            idx = 1*m + 2*n + 1
            plt.subplot(6, 2, idx)
            plt.ylabel(f'{name}')
            if k == "gt":
                plt.plot(t[:horizon, i], marker='.', zorder=-10)
            else:
                plt.scatter(
                    np.arange(h, horizon+h), t[:, plotStateCols[name]],
                    marker='X', edgecolors='k', s=64
                )
    #plt.tight_layout()
    if dir is not None:
        name = os.path.join(dir, f"{file_name}.png")
        plt.savefig(name)
        plt.close()

    if seq is not None:
        fig_act = plt.figure(figsize=(30, 30))
        for i, name in enumerate(plotActionCols):
            plt.subplot(maxA, 1, i+1)
            plt.ylabel(f'{name}')
            plt.plot(seq[0, :horizon+h, plotActionCols[name]])

        #plt.tight_layout()
        if dir is not None:
            name = os.path.join(dir, f"{file_name}-actions.png")
            plt.savefig(name)
            plt.close()
    
    plt.show()


def rand_roll(models, histories, plotStateCols, plotActionCols, horizon, dir, device):
    trajs = {}
    seq = 5. * torch.normal(
        mean=torch.zeros(1, horizon+10, 6),
        std=torch.ones(1, horizon+10, 6)).to(device)

    for model, h in zip(models, histories):
        init = np.zeros((1, h, 13))
        init[:, :, 6] = 1.
        #rot = np.eye(3)
        #init[:, :, 3:3+9] = np.reshape(rot, (9,))
        init = init.astype(np.float32)
        init = torch.from_numpy(init).to(device)
        pred = rollout(model, init, seq, h, device, horizon)
        trajs[model.name + "_rand"] = traj_to_euler(pred.cpu().numpy(), rep="quat")
        print(pred.isnan().any())
    plot_traj(trajs, seq.cpu(), histories, plotStateCols, plotActionCols, horizon, dir)


def traj_to_euler(traj, rep="rot"):
    if rep == "rot":
        rot = traj[:, 3:3+9].reshape((-1, 3, 3))
        r = R.from_matrix(rot)
    elif rep == "quat":
        quat = traj[:, 3:3+4]
        r = R.from_quat(quat)
    else:
        raise NotImplementedError
    pos = traj[:, :3]
    euler = r.as_euler('XYZ', degrees=True)
    vel = traj[:, -6:]

    traj = np.concatenate([pos, euler, vel], axis=-1)
    return traj
