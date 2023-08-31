import yaml
import torch
import numpy as np

from scripts.utils.utils import tdtype, diag, diag_embed
from scripts.models.model_base import ModelBase
from scripts.inputs.ModelInput import ModelInput


class AUVFossen(ModelBase):
    def __init__(self, dt=0.1, config=None, yaml_config=None):
        super(AUVFossen, self).__init__(dt, config, yaml_config)

        self.register_buffer("gravity", torch.tensor(9.81, dtype=tdtype))
        self.register_buffer("density", torch.tensor(1028., dtype=tdtype))

        # masks/pads
        self.register_buffer("z", torch.tensor([0., 0., 1.], dtype=tdtype))
        self.register_buffer("pad3x3", torch.zeros(1, 3, 3, dtype=tdtype))
        self.register_buffer("pad4x3", torch.zeros(1, 4, 3, dtype=tdtype))

        ## Skew matrix masks
        self.register_buffer("A", torch.tensor([[[0., 0., 0.], [0., 0., 1.], [0., -1., 0.]]], dtype=tdtype))
        self.register_buffer("B", torch.tensor([[[0., 0., -1.], [0., 0., 0.], [1., 0., 0.]]], dtype=tdtype))
        self.register_buffer("C", torch.tensor([[[0., 1., 0.], [-1., 0., 0.], [0., 0., 0.]]], dtype=tdtype))

        self.init_param()

    def init_param(self):
        mass = 100.
        if "mass" in self.config:
            mass = self.config["mass"]

        self.mass = torch.nn.Parameter(
            torch.tensor(mass, dtype=tdtype),
            requires_grad=False)

        volume = 1.
        if "volume" in self.config:
            volume = self.config["volume"]

        self.volume = torch.nn.Parameter(
            torch.tensor(volume, dtype=tdtype),
            requires_grad=False)


        cog = [0., 0., 0.]
        if "cog" in self.config:
            cog = self.config["cog"]

        self.cog = torch.nn.Parameter(
            torch.tensor([cog], dtype=tdtype),
            requires_grad=False)


        cob = [0., 0., 0.2]
        if "cob" in self.config:
            cob = self.config["cob"]

        self.cob = torch.nn.Parameter(
            torch.tensor([cob], dtype=tdtype),
            requires_grad=False)


        inertialKey = ["ixx", "iyy", "izz", "ixy", "ixz", "iyz"]
        inertialArg = dict(ixx=200, iyy=300, izz=250, ixy=110, ixz=120, iyz=130)
        if "inertial" in self.config:
            inertialArg = self.config["inertial"]
            for key in inertialKey:
                if key not in inertialArg:
                    raise AssertionError('Invalid moments of inertia')
        self.inertial = self.get_inertial(inertialArg)

        addedMass = np.zeros((6, 6))
        if "Ma" in self.config:
            addedMass = np.array(self.config["Ma"])
            assert (addedMass.shape == (6, 6)), "Invalid add mass matrix"
        self.addedMass = torch.tensor(addedMass, dtype=tdtype)


        massEye = self.mass * torch.eye(3, dtype=tdtype)
        massLower = self.mass * self.skew_sym(self.cog[..., None])[0]

        upper = torch.concat([massEye, -massLower], dim=1)
        lower = torch.concat([massLower, self.inertial], dim=1)
        self.rbMass = torch.concat([upper, lower], dim=0)

        self.mTot = torch.unsqueeze(self.rbMass + self.addedMass, dim=0)
        self.invMtot = torch.nn.Parameter(
            torch.linalg.inv(self.mTot),
            requires_grad=False
        )


        linDamp = [-70., -70., -700., -300., -300., -100.]
        if "linear_damping" in self.config:
            linDamp = self.config["linear_damping"]

        self.linDamp = torch.nn.Parameter(
            torch.unsqueeze(
                diag_embed(
                    torch.tensor(linDamp, dtype=tdtype),
                    ),
            dim=0),
            requires_grad=False)


        linDampFow = [0., 0., 0., 0., 0., 0.]
        if "linear_damping_forward" in self.config:
            linDampFow = self.config["linear_damping_forward"]

        self.linDampFow = torch.nn.Parameter(
            torch.unsqueeze(
                diag_embed(
                    torch.tensor(linDampFow, dtype=tdtype)),
                dim=0),
            requires_grad=False)


        quadDam = [-740., -990., -1800., -670., -770., -520.]
        if "quad_damping" in self.config:
            quadDam = self.config["quad_damping"]

        self.quadDamp = torch.nn.Parameter(
            torch.unsqueeze(
                diag_embed(
                    torch.tensor(quadDam, dtype=tdtype)),
                dim=0),
            requires_grad=False)

    def forward(self, x: torch.tensor, u: torch.tensor, rk: int=2) -> torch.tensor:
        # Rk2 integration.
        # self.k = x.shape[0]
        k1 = self.x_dot(x, u)
        tmp = k1*self.dt
        # if rk == 1:
        #     tmp = k1*self.dt

        if rk == 2:
            k2 = self.x_dot(x + k1*self.dt, u)
            tmp = (self.dt/2.)*(k1 + k2)

        return self.norm_quat(x+tmp)

    def x_dot(self, x: torch.tensor, u: torch.tensor) -> torch.tensor:
        p, v = torch.split(x, [7, 6], dim=1)
        rotBtoI, tBtoI = self.body2inertial(p)
        jac = self.jacobian(rotBtoI, tBtoI)
        pDot = torch.bmm(jac, v)
        vDot = self.acc(v, u, rotBtoI)
        return torch.concat([pDot, vDot], dim=-2)

    def norm_quat(self, quatState: torch.tensor) -> torch.tensor:
        quat = quatState[:, 3:7].clone()
        norm = torch.linalg.norm(quat, dim=-2)[..., None]
        quat = quat/norm
        quatState[:, 3:7] = quat.clone()
        return quatState

    def body2inertial(self, pose: torch.tensor):
        quat = pose[:, 3:7]
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

    def jacobian(self, rotBtoI: torch.tensor, tBtoI: torch.tensor) -> torch.tensor:
        k = rotBtoI.shape[0]
        pad3x3 = torch.broadcast_to(self.pad3x3, (k, 3, 3))
        pad4x3 = torch.broadcast_to(self.pad4x3, (k, 4, 3))
        jacR1 = torch.concat([rotBtoI, pad3x3], dim=-1)
        jacR2 = torch.concat([pad4x3, tBtoI], dim=-1)

        return torch.concat([jacR1, jacR2], dim=-2)
    
    def acc(self, v: torch.tensor, u: torch.tensor, rotBtoI: torch.tensor) -> torch.tensor:
        Dv = torch.bmm(self.damping(v), v)
        Cv = torch.bmm(self.coriolis(v), v)
        g = torch.unsqueeze(self.restoring(rotBtoI), dim=-1)
        rhs = u - Cv - Dv - g
        acc = torch.matmul(self.invMtot, rhs)
        # acc = torch.linalg.solve(self.mTot, rhs)
        return acc

    def restoring(self, rotBtoI: torch.tensor) -> torch.tensor:
        fng = -self.mass * self.gravity * self.z
        fnb = self.volume * self.density * self.gravity * self.z

        rotItoB = torch.transpose(rotBtoI, -1, -2)

        fbg = torch.matmul(rotItoB, fng)
        fbb = torch.matmul(rotItoB, fnb)

        mbg = torch.linalg.cross(self.cog, fbg)
        mbb = torch.linalg.cross(self.cob, fbb)

        return -torch.concat([fbg+fbb, mbg+mbb], dim=-1)

    def damping(self, v:torch.tensor) -> torch.tensor:
        D = - self.linDamp - (v * self.linDampFow)
        tmp = - torch.mul(self.quadDamp, 
                          torch.abs(
            diag_embed(torch.squeeze(v, dim=-1))))

        return D + tmp

    def coriolis(self, v: torch.tensor) -> torch.tensor:
        k = v.shape[0]
        skewCori = torch.matmul(self.mTot[:, 0:3, 0:3].clone(), v[:, 0:3]) + \
                   torch.matmul(self.mTot[:, 0:3, 3:6].clone(), v[:, 3:6])
        s12 = - self.skew_sym(skewCori)

        skewCoriDiag = torch.matmul(self.mTot[:, 3:6, 0:3].clone(), v[:, 0:3]) + \
                       torch.matmul(self.mTot[:, 3:6, 3:6].clone(), v[:, 3:6])
        s22 = - self.skew_sym(skewCoriDiag)
        
        pad3x3 = torch.broadcast_to(self.pad3x3, (k, 3, 3))
        r1 = torch.concat([pad3x3, s12], dim=-1)
        r2 = torch.concat([s12, s22], dim=-1)
        return torch.concat([r1, r2], dim=-2)

    def skew_sym(self, vec: torch.tensor) -> torch.tensor:
        k = vec.shape[0]
        A = torch.broadcast_to(self.A, (k, 3, 3))
        B = torch.broadcast_to(self.B, (k, 3, 3))
        C = torch.broadcast_to(self.C, (k, 3, 3))
        c1 = torch.matmul(A, vec)
        c2 = torch.matmul(B, vec)
        c3 = torch.matmul(C, vec)
        S = torch.concat([c1, c2, c3], dim=-1)

        return S

    def print_info(self):
        """Print the vehicle's parameters."""
        print("="*5, " Model Info ", "="*5)
        print('Mass: {} kg, Trainable: {}\n'.format(self.mass.detach().cpu().numpy(),
                                                    self.mass.requires_grad))
        print('Volume: {} m^3, Trainable: {}\n'.format(self.volume.detach().cpu().numpy(),
                                                       self.volume.requires_grad))
        print('M:\n{}\nTrainable: {}'.format(self.mTot.detach().cpu().numpy(),
                                             self.mTot.requires_grad))
        print('Linear damping:\n{}\nTrainable: {}\n'.format(self.linDamp.detach().cpu().numpy(),
                                                            self.linDamp.requires_grad))
        print('Quad. damping:\n{}\nTrainable: {}\n'.format(self.quadDamp.detach().cpu().numpy(),
                                                           self.quadDamp.requires_grad))
        print('Center of gravity:\n{}\nTrainable: {}\n'.format(self.cog.detach().cpu().numpy(),
                                                               self.cog.requires_grad))
        print('Center of buoyancy:\n{}\nTrainable: {}\n'.format(self.cob.detach().cpu().numpy(),
                                                                self.cob.requires_grad))

    def reset(self):
        pass

    def get_inertial(self, dict):
        # buid the inertial matrix
        ixx = dict['ixx']
        ixy = dict['ixy']
        ixz = dict['ixz']
        iyy = dict['iyy']
        iyz = dict['iyz']
        izz = dict['izz']

        row0 = torch.tensor([[ixx], [ixy], [ixz]], dtype=tdtype)
        row1 = torch.tensor([[ixy], [iyy], [iyz]], dtype=tdtype)
        row2 = torch.tensor([[ixz], [iyz], [izz]], dtype=tdtype)

        inertial = torch.concat([row0, row1, row2], dim=-1)

        return inertial