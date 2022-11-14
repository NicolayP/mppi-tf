import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from scripts.src_torch.models.torch_utils import Adjoint, ToSE3Mat, SE3int, FlattenSE3, SE3inv
dtype = torch.float32
from tqdm import tqdm


def fake_state(k, tau, f_s):
    rot = np.eye(3)
    rot = rot.reshape((9,))
    s = np.zeros(shape=(k, tau, f_s))
    s[:, :, 3:3+9] = rot
    return torch.Tensor(s)


def fake_act(k, tau, f_a):
    return torch.Tensor(np.zeros(shape=(k, tau, f_a)))


class GeodesicLoss(torch.nn.Module):
    def __init__(self):
        super(GeodesicLoss, self).__init__()
        self.l2 = torch.nn.MSELoss()
        self.l2_axis = torch.nn.MSELoss(reduction='none')
    
    def forward(self, s_gt, s):
        lp = self.l_pos(s_gt, s)
        lv = self.l_vel(s_gt, s)       
        lt = self.l_theta(s_gt, s)

        return (lp + lv + lt.mean())

    def l_pos(self, s_gt, s, dim=None):
        pos_gt, pos = s_gt[..., :3], s[..., :3]
        if dim is None:
            return self.l2(pos_gt, pos)
        return self.l2_axis(pos_gt, pos).reshape((-1, 3)).mean(dim)

    def l_vel(self, s_gt, s, dim=None):
        v_gt, v = s_gt[..., -6:], s[..., -6:]
        if dim is None:
            return self.l2(v_gt, v)
        return self.l2_axis(v_gt, v).reshape((-1, 6)).mean(dim)

    def l_dv(self):
        pass

    def l_theta(self, s_gt, s):
        ss = s_gt.shape

        rot_gt, rot = s_gt[..., 3:3+9].reshape((*ss[:-1], 3, 3)), s[..., 3:3+9].reshape((*ss[:-1], 3, 3))
        rot_gt = torch.transpose(rot_gt, -1, -2)
        rot_mul = torch.mul(rot_gt, rot)

        trace = rot_mul.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
        theta = torch.cos(
            torch.clip(
                ((trace -1.) / 2.),
            -1, 1)
        )
        return theta


class AUVRNNStepModel(torch.nn.Module):
    def __init__(self, bias=False):
        super(AUVRNNStepModel, self).__init__()
        self.tau, self.f_s, self.f_a = 1, 18, 6
        self.input_size = self.f_s -3 + self.f_a
        self.hidden_size = 12
        self.num_layers = 1

        self.dt = 0.1

        self.rnn = torch.nn.RNN(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bias=bias)

        layers = [
            torch.nn.Linear(self.hidden_size, 128, bias=bias),
            torch.nn.LeakyReLU(negative_slope=0.1),
            torch.nn.Linear(128, 128, bias=bias),
            torch.nn.LeakyReLU(negative_slope=0.1),
            torch.nn.Linear(128, 6, bias=bias)
        ]

        self.fc = torch.nn.Sequential(*layers)
        self.to_mat = ToSE3Mat()
        self.int = SE3int()
        self.adj = Adjoint()
        self.inv = SE3inv()
        self.flat = FlattenSE3()

    def forward(self, s, a, h0=None):
        print(s.shape)
        print(a.shape)
        k = s.shape[0]
        input = torch.concat([s[:, :, 3:], a], dim=-1)
        if h0 is None:
            h0 = self.init_hidden(k)
        out, hN = self.rnn(input, h0)
        dv = self.fc(out)[:, -1, :, None]
        v = s[:, -1, -6:]
        t = v*self.dt

        M_current = self.to_mat(s[:, -1])
        M_next = self.int(M_current, t)
        M_next_inv = self.inv(M_next)

        v_current = v[..., None]

        v_I_current = self.adj(M_current) @ v_current

        v_next = self.adj(M_next_inv) @ (v_I_current + dv)
        s_next = self.flat(M_next, v_next[..., 0])[:, None, :]
        return s_next, hN, dv[..., 0]

    def init_hidden(self, k):
        return torch.Tensor(np.zeros(shape=(self.num_layers, k, self.hidden_size)))        


class AUVRNNModel(torch.nn.Module):
    def __init__(self, bias=False):
        super(AUVRNNModel, self).__init__()
        self.step = AUVRNNStepModel(bias=bias)

    def forward(self, s, A):
        k = s.shape[0]
        tau = A.shape[1]
        h = self.step.init_hidden(k)
        traj = torch.zeros(size=(k, tau, 18))
        traj_dv = torch.zeros(size=(k, tau, 6))
        for i in range(tau):
            s_next, h_next, dv = self.step(s, A[:, i:i+1], h)
            s, h = s_next, h_next
            traj[:, i:i+1] = s
            traj_dv[:, i] = dv
        return traj



# Create model, loss and optimizer.
model = AUVRNNModel()
opti = torch.optim.Adam(model.parameters(), lr=1e-4)
loss = GeodesicLoss()

# Trace the model graph.
k, tau, f_s, f_a = 2, 1, 18, 6
dt = 0.1
s = fake_state(k, tau, f_s)
A = fake_act(k, tau+2, f_a)
traj = model(s, A)

writer = SummaryWriter("rnn-auv-log")
writer.add_graph(model, (s, A))
writer.close()


traj_gt = torch.cat([s, s, s], dim=1)
traj_gt_plot = fake_state(1, 51, f_s)
a_plot = fake_act(1, 50, f_a)

for e in tqdm(range(100)):
    traj_pred = model(s, A)
    opti.zero_grad()
    l = loss(traj_gt, traj_pred)
    l.backward()
    opti.step()

    ########## LOGGING ##########
    lp = loss.l_pos(traj_gt, traj_pred, dim=0)
    lv = loss.l_vel(traj_gt, traj_pred, dim=0)
    lt = loss.l_theta(traj_gt, traj_pred).mean()

    t = model(traj_gt_plot[:, 0:1], a_plot)
    t_dict = {"model": traj_to_euler(t[0].detach(), "rot"), "gt": traj_to_euler(traj_gt_plot[0], "rot")}
    horizons = [10, 20, 30, 40, 50]
    t10, t20, t30, t40, t50 = gen_imgs(t_dict, horizons)
    t10_l, t20_l, t30_l, t40_l, t50_l = [loss(t[:, :h], traj_gt_plot[:, :h]) for h in horizons]
    t1_l = loss(t[:, :1], traj_gt_plot[:, :1])

    for name, param in model.named_parameters():
        grad = param.grad
        max_grad = torch.max(grad)
        mean_grad = torch.mean(grad)
        writer.add_histogram(f"grad/{name}", grad, e)
        writer.add_scalar(f"max_grad/{name}", max_grad, e)
        writer.add_scalar(f"mean_grad/{name}", mean_grad, e)
        writer.add_histogram(f"weights/{name}", param, e)
    writer.add_scalar("Loss", l, e)
    writer.add_scalar("Loss-split/x", lp[0], e)
    writer.add_scalar("Loss-split/y", lp[1], e)
    writer.add_scalar("Loss-split/z", lp[2], e)
    writer.add_scalar("Loss-split/angle", lt, e)
    writer.add_scalar("Loss-split/u", lv[0], e)
    writer.add_scalar("Loss-split/v", lv[1], e)
    writer.add_scalar("Loss-split/w", lv[2], e)
    writer.add_scalar("Loss-split/p", lv[3], e)
    writer.add_scalar("Loss-split/q", lv[4], e)
    writer.add_scalar("Loss-split/r", lv[5], e)
    writer.add_scalar("multi_loss/t1", t1_l, e)
    writer.add_image("traj-10", t10, e, dataformats="HWC")
    writer.add_scalar("multi_loss/t10", t10_l, e)
    writer.add_image("traj-20", t20, e, dataformats="HWC")
    writer.add_scalar("multi_loss/t20", t20_l, e)
    writer.add_image("traj-30", t30, e, dataformats="HWC")
    writer.add_scalar("multi_loss/t30", t30_l, e)
    writer.add_image("traj-40", t40, e, dataformats="HWC")
    writer.add_scalar("multi_loss/t40", t40_l, e)
    writer.add_image("traj-50", t50, e, dataformats="HWC")
    writer.add_scalar("multi_loss/t50", t50_l, e)
    writer.flush()
