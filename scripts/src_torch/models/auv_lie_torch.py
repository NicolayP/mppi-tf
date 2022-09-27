import numpy as np
import torch
import lietorch as lie
from ..utils import dtype


class LieAUVStep(torch.nn.Module):
    def __init__(self, h=1, dt=0.1, bias=False, internal=None):
        super(LieAUVStep, self).__init__()
        self.name = "LieAUVNNStep"
        if internal is None:
            self.nn = torch.nn.Sequential(
                torch.nn.Linear(h*(3*6), 128, bias=bias),
                torch.nn.LeakyReLU(negative_slope=0.1),
                torch.nn.Linear(128, 128, bias=bias),
                torch.nn.LeakyReLU(negative_slope=0.1),
                torch.nn.Linear(128, 128, bias=bias),
                torch.nn.LeakyReLU(negative_slope=0.1),
                torch.nn.Linear(128, 6, bias=bias)
            )
        else:
            self.nn = internal
        self.dt = dt
    
    def forward(self, M, v, u):
        '''
            Computes a single step prediciton using LieTorch
            as workframe.

            inputs:
            -------
                - M. The pose of the AUV with h past steps
                    shape [k, h]
                - v. The velocities of the AUV with h past steps.
                    shape [k, h, 6]
                - u. The actions applied to the AUV with h past steps.
                    shape [k, h, 6]

            outputs:
            --------
                - M_next. The next pose of the AUV.
                    shape [k, 1]
                - v_next. The next velocity of the AUV.
                    shape [k, 6]
        '''
        M_prev = M[:, -1]
        v_prev = v[:, -1]
        v_act = lie.SE3.exp(v_prev*self.dt)
        M_next = M_prev.mul(v_act)
        m_inp = M.log().flatten(1)
        v_inp = v.flatten(1)
        u_inp = u.flatten(1)
        inp = torch.concat([m_inp, v_inp, u_inp], dim=-1)
        v_next = self.nn(inp)
        return M_next, v_next


class LieAUVNN(torch.nn.Module):
    def __init__(self, h=1, internal=None):
        super(LieAUVNN, self).__init__()
        self.h = h
        if internal is None:
            self.step_nn = LieAUVStep(h=h)
        else:
            self.step_nn = internal
    
    def forward(self, x, U):
        # Assumes quaternion representation.
        # x.shape = [k, h, 13]
        # U.shape = [k, tau+h, 6]
        tau = U.shape[1] - self.h + 1
        pose = x[:, :, :7]
        v = x[:, :, 7:]
        M = lie.SE3.InitFromVec(pose)
        pose_traj = []
        vel_traj = []
        for t in range(tau):
            u = U[:, t:t+self.h]
            M_next, v_next = self.step_nn(M, v, u)
            pose_traj.append(M_next[:, None, ...])
            vel_traj.append(v_next[:, None, ...])
            M = self.push_to_lie(M, M_next)
            v = self.push_to_torch(v, v_next)

        pose_traj = lie.cat(pose_traj, dim=1)
        vel_traj = torch.cat(vel_traj, dim=1)
        return pose_traj, vel_traj

    def push_to_lie(self, poses, new_pose):
        tmp = poses[:, 1:]
        return lie.cat([tmp, new_pose[:, None]], dim=1)

    def push_to_torch(self, vels, new_vel):
        tmp = vels[:, 1:]
        return torch.cat([tmp, new_vel[:, None]], dim=1)


class LieAUVWrapper(torch.nn.Module):
    def __init__(self, internal):
        super(LieAUVWrapper, self).__init__()
        self.internal = internal
        self.name = "LieAUVWrapper"

    def forward(self, x, u):
        '''
            Wrapper for the LieAUVStep model. Converts the pose
            to SE(3) element. And feed the input to the step model
            in the right format.

            inputs:
            -------
                - x. The state of the AUV in quaterion format.
                    shape [k, h, 13]
                - u. The action applied to the AUV.
                    shape [k, h, 6]
            
            outputs:
            --------
                - x_next. The next state of the AUV in quaterion format.
                    shape [k, 13]
        '''
        M = lie.SE3.InitFromVec(x[:, :, :7])
        v = x[:, :, 7:]
        M_next, v_next = self.internal(M, v, u)
        pose_next = M_next.vec()
        x_next = torch.concat([pose_next, v_next], axis=-1)
        return x_next


class GeodesicLoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(GeodesicLoss, self).__init__()
        self.l2 = torch.nn.MSELoss()

    def forward(self, gt_traj, pred_traj):
        gt_poses, gt_vels = gt_traj
        pred_poses, pred_vels = pred_traj
        
        gt_poses_inv = gt_poses.inv()
        Poses_loss = pred_poses.mul(gt_poses_inv)
        poses_loss = Poses_loss.log()

        poses_norm = poses_loss.norm(dim=-1)
        vels_norm = self.l2(gt_vels, pred_vels)
        loss = (poses_norm + vels_norm).sum()
        return loss

