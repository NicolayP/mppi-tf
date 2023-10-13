import torch
import pypose as pp

import numpy as np

class DatasetTensor(torch.utils.data.Dataset):
    def __init__(self, data_list, tau=1, frame="body",
                 act_normed=False, out_normed=True, stats=None):
        # Input data_list is a set of pandas Dataframes
        self.prediction_steps = tau
        frame_prefix = "B"
        if frame == "world":
            frame_prefix = "I"

        position_labels = ["x", "y", "z"]
        rot_labels = ["qx", "qy", "qz", "qw"]

        lin_vel_labels = [f"{frame_prefix}u", f"{frame_prefix}v", f"{frame_prefix}w"]
        lin_dv_labels = [f"{frame_prefix}du", f"{frame_prefix}dv", f"{frame_prefix}dw"]

        ang_vel_labels = [f"{frame_prefix}p", f"{frame_prefix}q", f"{frame_prefix}r"]
        ang_dv_labels = [f"{frame_prefix}dp", f"{frame_prefix}dq", f"{frame_prefix}dr"]

        act_labels = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]
        if act_normed:
            act_labels = ["Ux", "Uy", "Uz", "Vx", "Vy", "Vz"]

        pose_trajs, velocity_trajs, delta_v_trajs, action_sequences = \
            self.extract_trajs(data_list,
                               position_labels + rot_labels,
                               lin_vel_labels + ang_vel_labels,
                               lin_dv_labels + ang_dv_labels,
                               act_labels)
        # Extract the interesting subparts of the Dataframe files and
        # load them into a numpy array.
        # Create the different elements: targets and inputs.
        self.se3_trajs = pp.SE3(pose_trajs)
        self.vel_trajs = torch.tensor(velocity_trajs)
        self.dv_trajs = torch.tensor(delta_v_trajs)
        self.act_seqs = torch.tensor(action_sequences)

        self.nb_trajs = len(data_list)
        self.samples_per_traj = len(data_list[0] - tau)

        pass

    def extract_trajs(self, dataframe_list, pose_labels, vel_labels, dv_labels, act_labels):
        pose_trajs = []
        vel_trajs = []
        dv_trajs = []
        act_sequence = []

        for dataframe in dataframe_list:
            pose_trajs.append(dataframe[pose_labels].to_numpy()[None])
            vel_trajs.append(dataframe[vel_labels].to_numpy()[None])
            dv_trajs.append(dataframe[dv_labels].to_numpy()[None])
            act_sequence.append(dataframe[act_labels].to_numpy()[None])

        pose_trajs = np.concatenate(pose_trajs, axis=0)
        vel_trajs = np.concatenate(vel_trajs, axis=0)
        dv_trajs = np.concatenate(dv_trajs, axis=0)
        act_sequence = np.concatenate(act_sequence, axis=0)

        return pose_trajs, vel_trajs, dv_trajs, act_sequence

    def get_traj(self, idx, tau=None):
        if tau is None:
            tau = len(self.se3_trajs.shape[1]) - 1
        pose_init = self.se3_trajs[idx, 0:1].clone()[None]
        vel_init = self.vel_trajs[idx, 0:1].clone()[None]
        seq = self.act_seqs[idx, :tau].clone()[None]

        pose_target = self.se3_trajs[idx, 1:tau+1].clone()[None]
        vel_target = self.vel_trajs[idx, 1:tau+1].clone()[None]
        dv_target = self.dv_trajs[idx, 1:tau+1].clone()[None]

        input = (pose_init, vel_init, seq)
        target = (pose_target, vel_target, dv_target)
        return input, target

    def get_trajs(self, tau=None):
        if tau is None:
            tau = len(self.se3_trajs.shape[1]) - 1
        pose_init = self.se3_trajs[:, 0:1].clone()
        vel_init = self.vel_trajs[:, 0:1].clone()
        seq = self.act_seqs[:, :tau].clone()

        pose_target = self.se3_trajs[:, 1:tau+1].clone()
        vel_target = self.vel_trajs[:, 1:tau+1].clone()
        dv_target = self.dv_trajs[:, 1:tau+1].clone()

        input = (pose_init, vel_init, seq)
        target = (pose_target, vel_target, dv_target)
        return input, target

    def __len__(self):
        return self.nb_trajs * self.samples_per_traj

    def __getitem__(self, idx):
        i, j = np.unravel_index(idx, (self.nb_trajs, self.samples_per_traj))

        # Get the designed trajectory
        se3_traj = self.se3_trajs[i]
        vel_traj = self.vel_trajs[i]
        dv_traj = self.dv_trajs[i]
        act_seq = self.act_seq[i]

        # Get the inputs
        pose_input = se3_traj[j:j+1]
        vel_input = vel_traj[j:j+1]
        action_seq = act_seq[j:j+self.prediction_steps]

        # Get the target
        pose_target = se3_traj[j+1, j+self.prediction_steps+1]
        vel_target = vel_traj[j+1, j+self.prediction_steps+1]
        dv_target = dv_traj[j+1, j+self.prediction_steps+1]
        X = (pose_input, vel_input, action_seq)
        Y = (pose_target, vel_target, dv_target)

        return X, Y