import torch
import pypose as pp

'''
    Compute the Left-Geodesic loss between two SE(3) poses.
'''
class GeodesicLoss(torch.nn.Module):
    '''
        GeodesicLoss constructor
    '''
    def __init__(self):
        super(GeodesicLoss, self).__init__()

    '''
        inputs:
        -------
            - X1 pypose.SE3. The first pose. Shape [batch, steps]
            - X2 pypose.SE3. The second pose. Shape [batch, steps]

        outputs:
        --------
            - Log(X1 + X2^{-1})^{2}. Shape [batch, steps, 6]
    '''
    def forward(self, X1, X2):
        d = (X1 * X2.Inv()).Log()
        square = torch.pow(d, 2)
        return square


'''
    Trajectory loss object. This module can compute loss between two
    trajectories represented by a sequence of SE3 Poses, Velocities and
    Velocities Delta.
'''
class TrajLoss(torch.nn.Module):
    '''
        Trajectory loss consstructor.

        inputs:
        -------
            - alpha: float, weight for trajectory loss.
            - beta: float, weight for velocity loss.
            - gamma: float, weight for $\delta V$ loss.
    '''
    def __init__(self, alpha=1., beta=0., gamma=0.):
        super(TrajLoss, self).__init__()
        self.l2 = torch.nn.MSELoss()
        self.geodesic = GeodesicLoss()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        pass

    '''
        Returns true if beta > 0.
    '''
    def has_v(self):
        return self.beta > 0.

    '''
        Returns true if gamma > 0.
    '''
    def has_dv(self):
        return self.gamma > 0.

    '''
        Computes loss on an entire trajectory. Optionally if
        dv is passed, it computes the loss on the velocity delta.

        inputs:
        -------
            traj1: pypose SE(3) elements sequence representing first trajectory
                shape [k, tau]
            traj2: pypose SE(3) elements sequence representing second trajectory
                shape [k, tau]
            v1: pytorch Tensor. velocity profiles
                shape [k, tau, 6]
            v2: pytorch Tensor. velocity profiles
                shape [k, tau, 6]
            dv1: pytorch Tensor. Delta velocities profiles
                shape [k, tau, 6]
            dv2: pytorch Tensor. Delta velocities profiles
                shape [k, tau, 6]
            split: bool (default = False), if true, returns the loss function
                splitted across each controlled dimension
    '''
    def forward(self, traj1, traj2, v1=None, v2=None, dv1=None, dv2=None, split=False):
        if split:
            return self.split_loss(traj1, traj2, v1, v2, dv1, dv2)
        return self.loss(traj1, traj2, v1, v2, dv1, dv2)

    '''
        Computes trajectory, velocity and $\Delta V$ loss split accross each dimesnions.

        inputs:
        -------
            traj1: pypose SE(3) elements sequence representing first trajectory
                shape [k, tau]
            traj2: pypose SE(3) elements sequence representing second trajectory
                shape [k, tau]
            v1: pytorch Tensor. velocity profiles
                shape [k, tau, 6]
            v2: pytorch Tensor. velocity profiles
                shape [k, tau, 6]
            dv1: pytorch Tensor. Delta velocities profiles
                shape [k, tau, 6]
            dv2: pytorch Tensor. Delta velocities profiles
                shape [k, tau, 6]

        outputs:
        --------
            t_l: torch.tensor, trajectory loss
                shape [6]
            v_l: torch.tensor, velocity loss
                shape [6]
            dv_l: torch.tensor, delta velocity loss
                shape [6]
    '''
    def split_loss(self, t1, t2, v1, v2, dv1, dv2):
        # only used for logging and evaluating the performances.
        t_l = self.geodesic(t1, t2).mean((0, 1))
        v_l = torch.pow(v1 - v2, 2).mean((0, 1))
        dv_l = torch.pow(dv1 - dv2, 2).mean((0, 1))
        return t_l, v_l, dv_l

    '''
        Computes trajectory, velocity and $\Delta V$ loss.

        inputs:
        -------
            traj1: pypose SE(3) elements sequence representing first trajectory
                shape [k, tau]
            traj2: pypose SE(3) elements sequence representing second trajectory
                shape [k, tau]
            v1: pytorch Tensor. velocity profiles
                shape [k, tau, 6]
            v2: pytorch Tensor. velocity profiles
                shape [k, tau, 6]
            dv1: pytorch Tensor. Delta velocities profiles
                shape [k, tau, 6]
            dv2: pytorch Tensor. Delta velocities profiles
                shape [k, tau, 6]

        outputs:
        --------
            loss: the full trajectory loss.
    '''
    def loss(self, t1, t2, v1, v2, dv1, dv2):
        t_l = self.geodesic(t1, t2).mean()
        v_l = self.l2(v1, v2).mean()
        dv_l = self.l2(dv1, dv2).mean()
        return self.alpha*t_l + self.beta*v_l + self.gamma*dv_l
