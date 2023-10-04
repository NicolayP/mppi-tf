import torch
import numpy as np
import pandas as pd

import yaml
import os

import warnings
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from scipy.spatial.transform import Rotation as R
# tdtype = torch.double
# npdtype = np.double

tdtype = torch.float32
npdtype = np.float32
torch.set_default_dtype(tdtype)


import sys

disable_tqdm = False
if 'unittest' in sys.modules.keys():
    disable_tqdm = True
    

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


def push_to_tensor(tensor, x):
    return torch.cat((tensor[:, 1:], x.unsqueeze(dim=1)), dim=1)

# MISC FILES
'''
    Reads csv files from a directory.
    
    input:
    ------
        - data_dir, string. relative or absoulte path to the directory
        containing the csv files.
        - files, list of string. List containing the names of all the files
        the need to be loaded.
        - type, string. Decorator for tqdm.

    output:
    -------
        - dfs, list of dataframe containing the loaded csv files.
'''
def read_files(data_dir, files, type="train"):
    dfs = []
    for f in tqdm(files, desc=f"Dir {type}", ncols=100, colour="blue", disable=disable_tqdm):
        csv_file = os.path.join(data_dir, f)
        df = pd.read_csv(csv_file)
        df = df.astype(npdtype)
        dfs.append(df)
    return dfs

'''
    Plots trajectories with euler representation and velocity profiles from 
    the trajectory and velocity dictionaries respectively.
    The trajectories are plotted with length tau.

    input:
    ------
        - t_dict, dictionnary. Entries are "plotting-label": trajectory. The key will be used as
        label for the plots. The trajectory need to have the following entries [x, y, z, roll, pitch, yaw].
        - v_dict, dictionnary. Entries are "plotting-label": velocities. The key will be used as
        label for the plots. The veloties need to have the following entries [u, v, w, p, q, r].
        - dv_dict, dictionnary (default, None). Entries are "plotting-label": \detla V. The key will be used as
        label for the plots. The \delta V need to have the following entries [\delta u, \delta v, \delta w, \delta p, \delta q, \delta r].
        - tau, int. The number of points to plot

    output:
    -------
        - image that can be plotted or send to tensorboard. Returns tuple (trajectory_img, velocity_img).
        if dv_dict is not None, returns (trajectory_img, velocity_img, delta_v_img)
'''
def gen_imgs_3D(t_dict, v_dict, dv_dict=None, tau=100):
    plotState={"x(m)":0, "y(m)": 1, "z(m)": 2, "roll(rad)": 3, "pitch(rad)":4, "yaw(rad)": 5}
    plotVels={"u(m/s)":0, "v(m/s)": 1, "w(m/s)": 2, "p(rad/s)": 3, "q(rad/s)": 4, "r(rad/s)": 5}
    plotDVels={"du(m/s)":0, "dv(m/s)": 1, "dw(m/s)": 2, "dp(rad/s)": 3, "dq(rad/s)": 4, "dr(rad/s)": 5}
    t_imgs = []
    v_imgs = []
    if dv_dict is not None:
        dv_imgs = []

    t_imgs.append(plot_traj(t_dict, plotState, tau, title="State evolution"))
    v_imgs.append(plot_traj(v_dict, plotVels, tau, title="Velcoity Profiles"))
    if dv_dict is not None:
        dv_imgs.append(plot_traj(dv_dict, plotDVels, tau, title="Delta V"))

    if dv_dict is not None:
        return t_imgs, v_imgs, dv_imgs

    return t_imgs, v_imgs

'''
    Plots trajectories from a dictionnary.

    input:
    ------
        - traj_dict, dict. Entries are "plotting-label": trajectory. The key will be used as
        label for the plots.
        - plot_cols, dict. Entires are "axis-name": index-in-trajectory. This matches the trajectory
        from traj_dict.
        - tau, int. The number of steps to plot.
        - fig, bool (default false). If true, returns the matplotlib figure that can be shown with plt.show().
        - title, string. The title of the graph.
        - save, bool. (default false) If true, save the image in a dir called Img.

    output:
    -------
        - if fig == True, returns the matplotlib figure.
        - if fig == False, returns a np.array containing the RBG values of the image.
'''
def plot_traj(traj_dict, plot_cols, tau, fig=False, title="State Evolution", save=False):
    fig_state = plt.figure(figsize=(10, 10))
    axs_states = {}
    for i, name in enumerate(plot_cols):
        m, n = np.unravel_index(i, (2, 3))
        idx = 1*m + 2*n + 1
        axs_states[name] = fig_state.add_subplot(3, 2, idx)
    
    for k in traj_dict:
        t = traj_dict[k]
        for i, name in enumerate(plot_cols):
            axs_states[name].set_ylabel(f'{name}', fontsize=10)
            if k == 'gt':
                if i == 0:
                    axs_states[name].plot(t[:tau+1, i], marker='.', zorder=-10, label=k)
                else:
                    axs_states[name].plot(t[:tau+1, i], marker='.', zorder=-10)
                axs_states[name].set_xlim([0, tau+1])
            
            else:
                if i == 0:
                    axs_states[name].plot(np.arange(0, tau), t[:tau, plot_cols[name]],
                        marker='.', label=k)
                else:
                    axs_states[name].plot(np.arange(0, tau), t[:tau, plot_cols[name]],
                        marker='.')
    fig_state.text(x=0.5, y=0.03, s="steps", fontsize=10)
    fig_state.suptitle(title, fontsize=10)
    fig_state.legend(fontsize=5)
    fig_state.tight_layout(rect=[0, 0.05, 1, 0.98])

    if save:
        fig_state.savefig("img/" + title + ".png")

    if fig:
        return fig_state

    canvas = FigureCanvas(fig_state)
    canvas.draw()
    img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    img = img.reshape(fig_state.canvas.get_width_height()[::-1] + (3,))
    plt.close('all')
    return img


# def plot_traj(trajs, seq=None, histories=None, plotStateCols=None, plotActionCols=None, horizon=50, dir=".", file_name="foo"):
#     '''
#         Plot trajectories and action sequence.
#         inputs:
#         -------
#             - trajs: dict with model name as key and trajectories entry. If key is "gt" then it is assumed to be
#                 the ground truth trajectory.
#             - seq: Action Sequence associated to the generated trajectoires. If not None, plots the 
#                 action seqence.
#             - h: list of history used for the different models, ignored when model entry is "gt".
#             - plotStateCols: Dict containing the state axis name as key and index as entry
#             - plotAcitonCols: Dict containing the action axis name as key and index as entry.
#             - horizon: The horizon of the trajectory to plot.
#             - dir: The saving directory for the generated images.
#     '''
#     maxS = len(plotStateCols)
#     maxA = len(plotActionCols)
#     fig_state = plt.figure(figsize=(50, 50))
#     for k, h in zip(trajs, histories):
#         t = trajs[k]
#         for i, name in enumerate(plotStateCols):
#             m, n = np.unravel_index(i, (2, 6))
#             idx = 1*m + 2*n + 1
#             plt.subplot(6, 2, idx)
#             plt.ylabel(f'{name}')
#             if k == "gt":
#                 plt.plot(t[:horizon, i], marker='.', zorder=-10)
#             else:
#                 plt.scatter(
#                     np.arange(h, horizon+h), t[:, plotStateCols[name]],
#                     marker='X', edgecolors='k', s=64
#                 )
#     #plt.tight_layout()
#     if dir is not None:
#         name = os.path.join(dir, f"{file_name}.png")
#         plt.savefig(name)
#         plt.close()

#     if seq is not None:
#         fig_act = plt.figure(figsize=(30, 30))
#         for i, name in enumerate(plotActionCols):
#             plt.subplot(maxA, 1, i+1)
#             plt.ylabel(f'{name}')
#             plt.plot(seq[0, :horizon+h, plotActionCols[name]])

#         #plt.tight_layout()
#         if dir is not None:
#             name = os.path.join(dir, f"{file_name}-actions.png")
#             plt.savefig(name)
#             plt.close()
    
#     plt.show()

'''
    Converts a trajectory using quaternion representation to euler 'xyz' angle representation.
    
    input:
    ------
        - traj, numpy array. The trajectory with quaternion representation. It assumes that the quaternion
        is represented with entry index 3-7.

    output:
    -------
        - traj_euler, numpy array. The same trajectory with euler representation.
'''
def to_euler(traj):
    # assume quaternion representation
    p = traj[..., :3]
    q = traj[..., 3:]
    r = R.from_quat(q)
    e = r.as_euler('xyz')
    return np.concatenate([p, e], axis=-1)

'''
    Reads a yaml file and returns the matching dictionnary.
    
    input:
    ------
        - file, string. String to the yaml file.

    output:
    -------
        - dict, the associated dictionnary.
'''
def parse_param(file):
    with open(file) as file:
        conf = yaml.load(file, Loader=yaml.FullLoader)
    return conf

'''
    Saves a dictionnary to a yaml file.

    input:
    ------
        - path, string. Filename.
        - params, dict. The dictionnary to be saved.
'''
def save_param(path, params):
    with open(path, "w") as stream:
        yaml.dump(params, stream)