import yaml
import torch
import numpy as np
from .torch_utils import SE3enc, SE3integ, ToSE3Mat, SE3int, FlattenSE3, dtype
from torch.nn.utils.parametrizations import spectral_norm

class AUVFossen(torch.nn.Module):
    '''
        Constructor of AUV model using fossen equation,
        can be initalized using either a dict or a file containing
        the dict entries (yaml)
    '''
    def __init__(self, dict={}, file=None):
        super(AUVFossen, self).__init__()
        self.name = "AUV Fossen"
        self.dt = 0.1

        # masks/pads
        self.register_buffer("z", torch.tensor([0., 0., 1.]))
        self.register_buffer("gravity", torch.tensor(9.81))
        self.register_buffer("density", torch.tensor(1028.))
        
        self.register_buffer("pad3x3", torch.zeros(1, 3, 3))
        self.register_buffer("pad4x3", torch.zeros(1, 4, 3))
        
        ## Skew matrix masks
        self.register_buffer("A", torch.tensor([[[0., 0., 0.], [0., 0., 1.], [0., -1., 0.]]]))
        self.register_buffer("B", torch.tensor([[[0., 0., -1.], [0., 0., 0.], [1., 0., 0.]]]))
        self.register_buffer("C", torch.tensor([[[0., 1., 0.], [-1., 0., 0.], [0., 0., 0.]]]))
        self.init_param(dict, file)

    def init_param(self, param, file=None):
        
        if file is not None:
            with open(file, "r") as stream:
                param = yaml.safe_load(stream)

        if "mass" in param:
            mass = param["mass"]
            grads = False
        else:
            mass = 100.
            grads = True

        self.mass = torch.nn.Parameter(
            torch.tensor(mass, dtype=dtype),
            requires_grad=grads)

        if "volume" in param:
            volume = param["volume"]
            grads = False
        else:
            volume = 1.
            grads = True

        self.volume = torch.nn.Parameter(
            torch.tensor(volume, dtype=dtype),
            requires_grad=grads)

        if "cog" in param:
            cog = param["cog"]
            grads = False
        else:
            cog = [0., 0., 0.]
            grads = True

        self.cog = torch.nn.Parameter(
            torch.tensor([cog], dtype=dtype),
            requires_grad=grads)

        if "cob" in param:
            cob = param["cob"]
            grads = False
        else:
            cob = [0., 0., 0.2]
            grads = True

        self.cob = torch.nn.Parameter(
            torch.tensor([cob], dtype=dtype),
            requires_grad=grads)

        if "Ma" in param:
            ma = np.array(param["Ma"])
            grads = False
        else:
            ma = np.array([[100., 0., 0., 0., 0., 0.],
                             [0., 100., 0., 0., 0., 0.],
                             [0., 0., 100., 0., 0., 0.],
                             [0., 0., 0., 100., 0., 0.],
                             [0., 0., 0., 0., 100., 0.],
                             [0., 0., 0., 0., 0., 100.]]) + 1e-7
            grads = True
        
        inertial = dict(ixx=0, iyy=0, izz=0, ixy=0, ixz=0, iyz=0)
        if "inertial" in param:
            inertialArg = param["inertial"]
            for key in inertial:
                if key not in inertialArg:
                    raise AssertionError('Invalid moments of inertia')
        
        inertial = self.get_inertial(inertialArg)

        massEye = self.mass * torch.eye(3, dtype=dtype)
        massLower = (self.mass * self.skew_sym(self.cog[..., None]))[0]
        self.rbMass = self.rigid_body_mass(massEye, massLower, inertial)
        mtot = self.total_mass(self.rbMass, ma)

        self.mTot = torch.nn.Parameter(
            torch.unsqueeze(
                torch.tensor(mtot, dtype=dtype),
                dim=0),
            requires_grad=grads)
            

        if "linear_damping" in param:
            linDamp = param["linear_damping"]
            grads = False
        else:
            linDamp = [-70., -70., -700., -300., -300., -100.]
            grads = True

        self.linDamp = torch.nn.Parameter(
            torch.unsqueeze(
                torch.diag_embed(
                    torch.tensor(linDamp, dtype=dtype),
                    ),
            dim=0),
            requires_grad=grads)

        if "linear_damping_forward" in param:
            linDampFow = param["linear_damping_forward"]
            grads = False
        else:
            linDampFow = [0., 0., 0., 0., 0., 0.]
            grads = True

        self.linDampFow = torch.nn.Parameter(
            torch.unsqueeze(
                torch.diag_embed(
                    torch.tensor(linDampFow, dtype=dtype)),
                dim=0),
            requires_grad=grads)
            
        if "quad_damping" in param:
            quadDam = param["quad_damping"]
            grads = False
        else:
            quadDam = [-740., -990., -1800., -670., -770., -520.]
            grads = True

        self.quadDamp = torch.nn.Parameter(
            torch.unsqueeze(
                torch.diag_embed(
                    torch.tensor(quadDam, dtype=dtype)),
                dim=0),
            requires_grad=grads)

    def get_inertial(self, param):
        # buid the inertial matrix
        ixx = param['ixx']
        ixy = param['ixy']
        ixz = param['ixz']
        iyy = param['iyy']
        iyz = param['iyz']
        izz = param['izz']

        row0 = torch.Tensor([[ixx, ixy, ixz]])
        row1 = torch.Tensor([[ixy, iyy, iyz]])
        row2 = torch.Tensor([[ixz, iyz, izz]])

        inertial = torch.concat([row0, row1, row2], dim=0)

        return inertial

    def rigid_body_mass(self, massEye, massLower, inertial):
        upper = torch.concat([massEye, -massLower], dim=1)
        lower = torch.concat([massLower, inertial], dim=1)
        return torch.concat([upper, lower], dim=0)

    def total_mass(self, rbMass, addedMass):
        return rbMass + addedMass

    def forward(self, x, u, rk=2):
        # Rk2 integration.
        self.k = x.shape[0]
        k1 = self.dot(x, u)
        if rk == 1:
            tmp = k1*self.dt

        elif rk == 2:
            k2 = self.dot(x + k1*self.dt, u)
            tmp = (self.dt/2.)*(k1 + k2)
        return self.norm_quat(x+tmp)

    def dot(self, x, u):
        p, v = x[:, 0:7], x[:, 7:]
        rotBtoI, tBtoI = self.body2inertial(p)
        jac = self.jacobian(rotBtoI, tBtoI)
        pDot = torch.bmm(jac, torch.unsqueeze(v, dim=-1))
        vDot = self.acc(v, u, rotBtoI)
        return torch.squeeze(torch.concat([pDot, vDot], dim=-2), dim=-1)

    def norm_quat(self, quatState):
        quat = quatState[:, 3:7].clone()
        norm = torch.unsqueeze(torch.linalg.norm(quat, dim=-1), dim=-1)
        quat = quat/norm
        quatState[:, 3:7] = quat.clone()
        return quatState

    def body2inertial(self, pose):
        quat = torch.unsqueeze(pose[:, 3:7], dim=-1)
        w = quat[:, 3]
        x = quat[:, 0]
        y = quat[:, 1]
        z = quat[:, 2]

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

    def jacobian(self, rotBtoI, tBtoI):
        k = rotBtoI.shape[0]
        pad3x3 = torch.broadcast_to(self.pad3x3, (k, 3, 3))
        pad4x3 = torch.broadcast_to(self.pad4x3, (k, 4, 3))
        jacR1 = torch.concat([rotBtoI, pad3x3], dim=-1)
        jacR2 = torch.concat([pad4x3, tBtoI], dim=-1)

        return torch.concat([jacR1, jacR2], dim=-2)
    
    def acc(self, v, u, rotBtoI):
        Dv = torch.bmm(self.damping(v), torch.unsqueeze(v, dim=-1))
        Cv = torch.bmm(self.coriolis(v), torch.unsqueeze(v, dim=-1))
        g = torch.unsqueeze(self.restoring(rotBtoI), dim=-1)

        uExt = torch.unsqueeze(u, dim=-1)
        rhs = uExt - Cv - Dv - g
        acc = torch.linalg.solve(self.mTot, rhs)
        
        #print("*"*5, " C ", "*"*5)
        #print(self.coriolis(v))
        #print("*"*5, " g ", "*"*5)
        #print(self.restoring(rotBtoI))
        return acc

    def restoring(self, rotBtoI):
        fng = -self.mass * self.gravity * self.z
        fnb = self.volume * self.density * self.gravity * self.z

        rotItoB = torch.transpose(rotBtoI, -1, -2)

        fbg = torch.matmul(rotItoB, fng)
        fbb = torch.matmul(rotItoB, fnb)

        mbg = torch.linalg.cross(self.cog, fbg)
        mbb = torch.linalg.cross(self.cob, fbb)

        return -torch.concat([fbg+fbb, mbg+mbb], dim=-1)

    def damping(self, v):
        vExt = torch.unsqueeze(v, dim=-1)
        D = - self.linDamp - (vExt * self.linDampFow)
        tmp = - torch.mul(self.quadDamp, torch.abs(torch.diag_embed(v)))
        return D + tmp

    def coriolis(self, v):

        k = v.shape[0]
        vExt = torch.unsqueeze(v, dim=-1)
        #print(vExt.dtype)
        #print(self.mTot.dtype)

        skewCori = torch.matmul(self.mTot[:, 0:3, 0:3].clone(), vExt[:, 0:3]) + \
                   torch.matmul(self.mTot[:, 0:3, 3:6].clone(), vExt[:, 3:6])
        s12 = - self.skew_sym(skewCori)

        skewCoriDiag = torch.matmul(self.mTot[:, 3:6, 0:3].clone(), vExt[:, 0:3]) + \
                       torch.matmul(self.mTot[:, 3:6, 3:6].clone(), vExt[:, 3:6])
        s22 = - self.skew_sym(skewCoriDiag)
        
        pad3x3 = torch.broadcast_to(self.pad3x3, (k, 3, 3))
        r1 = torch.concat([pad3x3, s12], dim=-1)
        r2 = torch.concat([s12, s22], dim=-1)
        return torch.concat([r1, r2], dim=-2)

    def skew_sym(self, vec):
        # TODO: Define A B and C in the constructor.
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

    @property
    def multi(self):
        return False


class AUVFossenWrapper(torch.nn.Module):
    def __init__(self, fossenModel):
        super(AUVFossenWrapper, self).__init__()
        self.fossen = fossenModel
        self.sDim = 13
        self.name = fossenModel.name
    
    def forward(self, x, u):
        '''
            Input is of shape [k, 1, 13] and [k, 1, 6]
            Output [k, 1, 13]
        '''
        x_next = self.fossen(
            torch.squeeze(x, dim=1),
            torch.squeeze(u, dim=1)
        )
        return x_next


class AUVNN(torch.nn.Module):
    def __init__(self, sn=True, norm=None, maxU=10., bias=True, topology=[64]):
        super(AUVNN, self).__init__()
        self.sn = sn
        
        self.name = "AUV_NN"
        if sn:
            self.name += "_sn"

        if norm is not None:
            self.name += "_norm"
        else:
            norm=[0., 1.]

        if bias:
            self.name += "_bias"

        self.encoder = SE3enc(normV=norm, maxU=maxU)
        layers = []

        for i, s in enumerate(topology):
            if i == 0:
                self.name += f"_{len(self.encoder)}"
                layer = torch.nn.Linear(len(self.encoder), s, bias=bias)
            else:
                self.name += f"x{topology[i-1]}"
                layer = torch.nn.Linear(topology[i-1], s, bias=bias)
            
            if sn:
                layer = spectral_norm(layer)
            layers.append(layer)
            layers.append(torch.nn.LeakyReLU(negative_slope=0.1))
        
        self.name += f"x{topology[-1]}x6"
        layer = torch.nn.Linear(topology[-1], 6, bias=bias)
        if sn:
            layer = spectral_norm(layer)
        layers.append(layer)

        self.nn = torch.nn.Sequential(*layers)
        self.integrator = SE3integ()

    def forward(self, x, u, norm=False):
        '''
            x: inital state, torch tensor, shape [k, 13, 1].
            u: action, torch tensor, shape [k, 6, 1]

            out: trajectories, torch tensor, shape [k, 13, 1]
        '''
        enc = self.encoder(x, u, norm)
        delta = self.nn(enc)
        return self.integrator(x, delta)

    @property
    def multi(self):
        return False


class AUVODE(torch.nn.Module):
    def __init__(self):
        super(AUVODE, self).__init__()
        self.name = "AUV ODE"

    def forward(self, x, u):
        '''
            x: inital state, torch tensor, shape [k, 13, 1].
            u: action, torch tensor, shape [k, 6, 1]

            out: trajectories, torch tensor, shape [k, 13, 1]
        '''
        pass

    @property
    def multi(self):
        return False


class AUVMulti(torch.nn.Module):
    def __init__(self, stepModel):
        super(AUVMulti, self).__init__()
        self.stepModel = stepModel
        self.name = "Multi " + self.stepModel.name

    def forward(self, x, U):
        '''
            x: inital state, torch tensor, shape [k, 1, 13, 1].
            U: action sequence, torch tensor, shape [k, tau, 6, 1]

            out: trajectories, torch tensor, shape [k, tau, 13, 1]
        '''
        state = torch.squeeze(x, dim=1)
        traj = []
        tau = U.shape[1]
        for i in range(tau):
            u = U[:, i]
            nextState = self.stepModel(state, u)
            state = nextState
            traj.append(torch.unsqueeze(nextState, dim=1))

        traj = torch.concat(traj, dim=1)
        return traj

    @property
    def multi(self):
        return True


class StatePredictor(torch.nn.Module):
    def __init__(self, internal, dt):
        super(StatePredictor, self).__init__()
        self.internal = internal
        self.int = SE3int()
        self.toMat = ToSE3Mat()
        self.flat = FlattenSE3()
        self.dt = dt
        self.name = f"Predictor_{self.internal.name}"

    def forward(self, x, u):
        if x.dim() < 2:
            x = x.unsqueeze(dim=0)
            u = u.unsqueeze(dim=0)
        # Remove x, y, z
        x_red = x[:, 3:]

        v_next = self.internal(x_red, u)
        v = x[:, 12:]
        tau = v*self.dt

        M = self.toMat(x)
        M = self.int(M, tau)

        return self.flat(M, v_next).squeeze()

    @property
    def multi(self):
        return False


class StatePredictorHistory(torch.nn.Module):
    def __init__(self, internal, dt, h=1):
        super(StatePredictorHistory, self).__init__()
        self.internal = internal
        self.int = SE3int()
        self.toMat = ToSE3Mat()
        self.flat = FlattenSE3()
        self.dt = dt
        self.name = f"Predictor_{self.internal.name}"
        self.h = h
        self.sDim = 18

    def forward(self, x, u):
        '''
            input:
            ------
                - x the state vector with shape [batch, history, sDim] or [history, sDim]
                - u the action vecotr with shape [batch, history, aDim] or [history, aDim]
        '''
        if x.dim() < 3:
            x = x.unsqueeze(dim=0)
            u = u.unsqueeze(dim=0)
        # Remove x, y, z
        x_red = x[:, :, 3:]
        v_next = self.internal(x_red.flatten(1), u.flatten(1))
        x_last = x[:, -1]
        v = x[:, -1, 12:]
        tau = v*self.dt
        M = self.toMat(x_last)
        M = self.int(M, tau)
        nextState = self.flat(M, v_next)
        return nextState

    @property
    def history(self):
        return self.h


class VelPred(torch.nn.Module):
    def __init__(self, in_size=21, topology=[64]):
        super(VelPred, self).__init__()
        self.name = "nn-velocity"
        bias = False
        layers = []
        for i, s in enumerate(topology):
            if i == 0:
                self.name += f"_{in_size}"
                layer = torch.nn.Linear(in_size, s, bias=bias)
            else:
                self.name += f"x{topology[i-1]}"
                layer = torch.nn.Linear(topology[i-1], s, bias=bias)
            layers.append(layer)
            layers.append(torch.nn.LeakyReLU(negative_slope=0.1))

        self.name += f"x{topology[-1]}x6"
        layer = torch.nn.Linear(topology[-1], 6, bias=bias)
        layers.append(layer)

        self.nn = torch.nn.Sequential(*layers)
        self.nn.apply(init_weights)


    def forward(self, x, u):
        #x = x[:, 3:] # Remove x, y, z for prediciton.
        return self.nn(torch.cat((x, u), dim=1))

    @property
    def multi(self):
        return False


def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        #m.bias.data.fill_(0.01)
    pass
