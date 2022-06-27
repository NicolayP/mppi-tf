from turtle import forward
import torch


class SE3enc(torch.nn.Module):
    '''
        prepares the data to be fed in a velocity/state predictor

        x: the state of the system, shape [k, 13, 1]
        u: the action, shape [k, 6, 1]

        euler: bool, default false, if true represents the 
        orientationn as euler angles.
        sinCos: bool, default true, represents the orientation 
        with sin and cos of the eurler angles.
    '''
    def __init__(self, rot=True, normV=[0., 1.], maxU=10.):
        super(SE3enc, self).__init__()
        self.rot = rot
        self.normV = normV
        self.maxU = maxU
        if self.rot:
            self.b2i = Body2Inertial()

    def forward(self, x, u, norm):
        pose = x[:, :7]
        vel = x[:, 7:]
        if norm:
            vel = (vel - self.normV[0]) / self.normV[1]
            u = u / self.maxnU
        
        if self.rot:
            rot, _ = self.b2i(pose)
            rot = torch.flatten(rot, start_dim=1)
            return torch.concat([rot, vel, u], dim=1)
        else:
            return torch.concat([x[:, 3:], u], dim=1)

    def __len__(self):
        if self.rot:
            return 9 + 6 + 6
        else:
            return 4 + 6 + 6


class SE3integ(torch.nn.Module):
    '''
        Computes x_{t+1} given x_{t} and \delta_{t}
        x: the state of the system composed of pose and velocites.
            shape [k, 13, 1]
        delta: the velocity delta. 
            shape [k, 6, 1]
    '''
    def __init__(self):
        super(SE3integ, self).__init__()
        self.jac = Jacobian()
        self.norm_quat = NormQuat()

    def forward(self, x, delta, dt=0.1):
        pose = x[:, :7]
        vel = x[:, 7:]
        jac = self.jac(pose)
        pDot = torch.matmul(
            jac,
            torch.unsqueeze(vel, dim=-1))
        pDot = torch.squeeze(pDot,dim=-1)
        nextPose = self.norm_quat(pose + pDot*dt)
        nextVel = vel + delta
        return torch.concat([nextPose, nextVel], dim=-1)


class Jacobian(torch.nn.Module):
    def __init__(self):
        super(Jacobian, self).__init__()
        self.pad3x3 = torch.zeros(1, 3, 3)
        self.pad4x3 = torch.zeros(1, 4, 3)
        self.b2i = Body2Inertial()

    def forward(self, pose):
        rotBtoI, tBtoI = self.b2i(pose)
        k = rotBtoI.shape[0]
        pad3x3 = torch.broadcast_to(self.pad3x3, (k, 3, 3))
        pad4x3 = torch.broadcast_to(self.pad4x3, (k, 4, 3))
        jacR1 = torch.concat([rotBtoI, pad3x3], dim=-1)
        jacR2 = torch.concat([pad4x3, tBtoI], dim=-1)
        return torch.concat([jacR1, jacR2], dim=-2)


class Body2Inertial(torch.nn.Module):
    def __init__(self):
        super(Body2Inertial, self).__init__()

    def forward(self, pose):
        quat = torch.unsqueeze(pose[:, 3:7], dim=-1)

        x = quat[:, 0]
        y = quat[:, 1]
        z = quat[:, 2]
        w = quat[:, 3]

        r1 = torch.unsqueeze(
                torch.concat([1 - 2 * (y**2 + z**2),
                              2 * (x * y - z * w),
                              2 * (x * z + y * w)], dim=-1),
                dim=-2)
        r2 = torch.unsqueeze(
                torch.concat([2 * (x * y + z * w),
                              1 - 2 * (x**2 + z**2),
                              2 * (y * z - x * w)], dim=-1),
                dim=-2)
        r3 = torch.unsqueeze(
                torch.concat([2 * (x * z - y * w),
                              2 * (y * z + x * w),
                              1 - 2 * (x**2 + y**2)], dim=-1),
                dim=-2)

        rotBtoI = torch.concat([r1, r2, r3], dim=-2)

        r1t = torch.unsqueeze(
                torch.concat([-x, -y, -z], dim=-1),
                dim=-2)
        r2t = torch.unsqueeze(
                torch.concat([w, -z, y], dim=-1),
                dim=-2)
        r3t = torch.unsqueeze(
                torch.concat([z, w, -x], dim=-1),
                dim=-2)
        r4t = torch.unsqueeze(
                torch.concat([-y, x, w], dim=-1),
                dim=-2)

        tBtoI = 0.5 * torch.concat([r1t, r2t, r3t, r4t], dim=-2)
        return rotBtoI, tBtoI


class NormQuat(torch.nn.Module):
    def __init__(self):
        super(NormQuat, self).__init__()

    def forward(self, pose):
        quat = pose[:, 3:7].clone()
        norm = torch.unsqueeze(torch.linalg.norm(quat, dim=-1), dim=-1)
        quat = quat/norm
        pose[:, 3:7] = quat.clone()
        return pose
