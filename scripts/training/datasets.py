import torch
import pypose as pp
import numpy as np

from scripts.utils.utils import tdtype, npdtype
from scripts.inputs.ModelInput import ModelInput, ModelInputPypose

############################################
#                                          #
#          DATASET DEFINITIONS             #
#                                          #
############################################

'''
    Dataset for trajectories.
'''
class DatasetList3D(torch.utils.data.Dataset):
    '''
        Dataset Constructor.

        inputs:
        -------
            - data_list: List, A list of pandas dataframe representing trajectories.
            - steps: Int, The number of steps to use for prediction.
            - history: Int, the number of previous steps that should be used.
            - v_frame: String, The frame in which the velocity is represented ('world' or 'body'), default 'body'
            - dv_frame: String, The frame in which the velocity delta is represented ('world' or 'body'), default 'body'
            - rot: String, the representation used for rotations. (only 'quat' supported at the moment.)
            - act_normed: Bool, whether or not to normalize the action before feeing them to the network.
            - se3: Bool, whether or not to use pypose as underlying library for the pose representation.
            - out_normed: Bool, whether or not to normalize the targets.
            - stats: dict with entries:
                - std:
                    - world_norm: list of floats. Shape (6)
                    - body_norm: list of floats. Shape (6)
                - mean:
                    - world_norm: list of floats. Shape (6)
                    - body_norm: list of floats. Shape (6)
    '''
    def __init__(self, data_list, steps=1, history=1,
                 v_frame="body", dv_frame="body", rot="quat",
                 act_normed=False, se3=False, out_normed=True, stats=None):
        super(DatasetList3D, self).__init__()
        self.data_list = data_list
        self.s = steps
        self.h = history
        if v_frame == "body":
            v_prefix = "B"
        elif v_frame == "world":
            v_prefix = "I"

        if dv_frame == "body":
            dv_prefix = "B"
        elif dv_frame == "world":
            dv_prefix = "I"

        self.pos = ['x', 'y', "z"]
        # used for our SE3 implementation.
        if rot == "rot":
            self.rot = ['r00', 'r01', 'r02',
                        'r10', 'r11', 'r12',
                        'r20', 'r21', 'r22']
        # Used in pypose implementation.
        elif rot == "quat":
            self.rot = ['qx', 'qy', 'qz', 'qw']

        self.lin_vel = [f'{v_prefix}u', f'{v_prefix}v', f'{v_prefix}w']
        self.ang_vel = [f'{v_prefix}p', f'{v_prefix}q', f'{v_prefix}r']

        self.x_labels = self.pos + self.rot + self.lin_vel + self.ang_vel

        self.traj_labels = self.pos + self.rot
        self.vel_labels = self.lin_vel + self.ang_vel
        self.dv_labels = [
            f'{dv_prefix}du', f'{dv_prefix}dv', f'{dv_prefix}dw',
            f'{dv_prefix}dp', f'{dv_prefix}dq', f'{dv_prefix}dr'
        ]

        if act_normed:
            self.u_labels = ['Ux', 'Uy', 'Uz', 'Vx', 'Vy', 'Vz']
        else:
            self.u_labels = ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']

        self.samples = [traj.shape[0] - self.s for traj in data_list]
        self.len = sum(self.samples)
        self.bins = self.create_bins()
        self.se3 = se3
        
        if out_normed:
            self.std = np.array(stats["std"][f'{dv_prefix}_norm'], dtype=npdtype)
            self.mean = np.array(stats["mean"][f'{dv_prefix}_norm'], dtype=npdtype)
        else:
            self.std = 1.
            self.mean = 0.

    '''
        returns the number of samples in the dataset.
    '''
    def __len__(self):
        return self.len

    '''
        get a sample at a specific index.

        inputs:
        -------
            - idx, int < self.__len__().

        outputs:
        --------
            - x, the state of the vehicle (pose and velocity)
                shape (1, 7+6)
            - u, The actions applied to the vehicle. Shape (steps, 6)
            - traj, The resulting trajectory. Shape (steps, 7)
            - vel, The resulting velocity profiles, shape (steps, 6)
            - dv, The normalized velocity delta prrofiles, shape (steps, 6)
    '''
    def __getitem__(self, idx):
        i = (np.digitize([idx], self.bins)-1)[0]
        traj = self.data_list[i]
        j = idx - self.bins[i]
        sub_frame = traj.iloc[j:j+self.s+1]
        x = sub_frame[self.x_labels].to_numpy()
        x = x[:1]

        u = sub_frame[self.u_labels].to_numpy()
        u = u[:self.s]

        traj = sub_frame[self.traj_labels].to_numpy()[1:1+self.s]
        vel = sub_frame[self.vel_labels].to_numpy()[1:1+self.s]
        dv = sub_frame[self.dv_labels].to_numpy()[1:1+self.s]

        dv = (dv-self.mean)/self.std

        traj = pp.SE3(traj)

        return x, u, traj, vel, dv

    '''
        Returns the number of trajectories in the dataset.
    '''
    @property
    def nb_trajs(self):
        return len(self.data_list)
    
    '''
        Get the traj at a specific index ind the dataset.
        Raises IndexError if the index is out of bound.
        inputs:
        -------
            - idx, int, the trajectory index.

        outputs:
        --------
            - trajectory, shape (tau, 7+6)
    '''
    def get_traj(self, idx):
        if idx >= self.nb_trajs:
            raise IndexError
        return self.data_list[idx][self.x_labels].to_numpy()
    
    '''
        internal function that creats bins to compute the number
        of samples in the dataset.
    '''
    def create_bins(self):
        bins = [0]
        cummul = 0
        for s in self.samples:
            cummul += s
            bins.append(cummul)
        return bins

    '''
        get all the trajectories from the dataset. Only works if all
        the trajs in the dataset have the same length.

        inputs:
        -------
            - None

        outputs:
        --------
            - trajs, shape (nb_traj, tau, se3_rep)
            - vels, shape (nb_traj, tau, 6)
            - dvs, shape (nb_traj, tau, 6)
            - actions, shape (nb_traj, tau, 6)
    '''
    def get_trajs(self):
        traj_list = []
        vel_list = []
        dv_list = []
        action_seq_list = []
        for data in self.data_list:
            traj = data[self.traj_labels].to_numpy()[None]
            traj_list.append(traj)

            vel = data[self.vel_labels].to_numpy()[None]
            vel_list.append(vel)

            dv = data[self.dv_labels].to_numpy()[None]
            dv_list.append(dv)

            action_seq = data[self.u_labels].to_numpy()[None]
            action_seq_list.append(action_seq)

        trajs = torch.Tensor(np.concatenate(traj_list, axis=0))
        vels = torch.Tensor(np.concatenate(vel_list, axis=0))
        dvs = torch.Tensor(np.concatenate(dv_list, axis=0))
        actions = torch.Tensor(np.concatenate(action_seq_list, axis=0))

        dvs = (dvs-self.mean)/self.std

        if self.se3:
            trajs = pp.SE3(trajs)

        return trajs, vels, dvs, actions

    '''
        Get the mean and std of the velocity delta.

        outputs:
        --------
            - mean, torch.tensor, shape [6]
            - std, torch.tensor, shape [6]
    '''
    def get_stats(self):
        return self.mean, self.std

'''
    Dataset for trajectories that returns ModelInput variables
'''
class DatasetListModelInput(torch.utils.data.Dataset):
    def __init__(self, data_list, steps=1, history=1,
                 v_frame="body", dv_frame="body",
                 act_normed=False, out_normed=True, stats=None, se3=False):
        super(DatasetListModelInput, self).__init__()
        self.data_list = data_list
        self.s = steps
        self.h = history

        v_prefix = "B"
        if v_frame == "world":
            v_prefix = "I"

        dv_prefix = "B"
        if dv_frame == "world":
            dv_prefix = "I"            

        self.pos = ['x', 'y', "z"]
        # used for our SE3 implementation.
        self.rot = ['qx', 'qy', 'qz', 'qw']

        self.lin_vel = [f'{v_prefix}u', f'{v_prefix}v', f'{v_prefix}w']
        self.ang_vel = [f'{v_prefix}p', f'{v_prefix}q', f'{v_prefix}r']

        self.x_labels = self.pos + self.rot + self.lin_vel + self.ang_vel

        self.traj_labels = self.pos + self.rot
        self.vel_labels = self.lin_vel + self.ang_vel

        self.dv_labels = [
            f'{dv_prefix}du', f'{dv_prefix}dv', f'{dv_prefix}dw',
            f'{dv_prefix}dp', f'{dv_prefix}dq', f'{dv_prefix}dr'
        ]

        if act_normed:
            self.u_labels = ['Ux', 'Uy', 'Uz', 'Vx', 'Vy', 'Vz']
        else:
            self.u_labels = ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']

        self.samples = [traj.shape[0] - self.s - (self.h - 1) for traj in data_list]
        self.len = sum(self.samples)
        
        self.bins = self.create_bins()
        self.se3 = se3

        if out_normed:
            self.std = np.array(stats["std"][f'{dv_prefix}_norm'], dtype=npdtype)
            self.mean = np.array(stats["mean"][f'{dv_prefix}_norm'], dtype=npdtype)
        else:
            self.std = 1.
            self.mean = 0.

    '''
        returns the number of samples in the dataset.
    '''
    def __len__(self):
        return self.len

    '''
        get a sample at a specific index.

        inputs:
        -------
            - idx, int < self.__len__().

        outputs:
        --------
            - x, ModelInput (or ModelInputPypose if SE3 is true). With "shape"
                [history]
            - u, The action sequence to applied to the model. Shape [steps, 6]
            - traj, The target trajectory. Shape [steps, 7]
            - vel, The target velocities. Shape [steps, 6]
            - dv, The target velocity deltas. Shape (steps, 6)
    '''
    def __getitem__(self, idx):
        i = (np.digitize([idx], self.bins)-1)[0]
        traj = self.data_list[i]
        j = idx - self.bins[i]
        sub_frame = traj.iloc[j:j+self.s+self.h]
        X = sub_frame[self.x_labels].to_numpy()
        x_past = X[:self.h]

        U = sub_frame[self.u_labels].to_numpy()
        u_past = U[:self.h-1]
        u = U[self.h-1:self.h - 1 + self.s]

        target_traj = sub_frame[self.traj_labels].to_numpy()[self.h:self.h+self.s]
        target_vel = sub_frame[self.vel_labels].to_numpy()[self.h:self.h+self.s]
        target_dv = sub_frame[self.dv_labels].to_numpy()[self.h:self.h+self.s]

        target_dv = (target_dv-self.mean)/self.std

        target_traj = pp.SE3(target_traj)
        pose_past = pp.SE3(x_past[:, :7])
        vel_past = x_past[:, 7:]

        x = ModelInputPypose(1, self.h, 6)
        x.init_form_state(pose_past, torch.tensor(vel_past), torch.tensor(u_past))

        return x, u, target_traj, target_vel, target_dv

    '''
        internal function that creats bins to compute the number
        of samples in the dataset.
    '''
    def create_bins(self):
        bins = [0]
        cummul = 0
        for s in self.samples:
            cummul += s
            bins.append(cummul)
        return bins
