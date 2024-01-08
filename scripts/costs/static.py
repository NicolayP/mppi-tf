import torch
from .cost_base import CostBase
from scripts.utils.utils import tdtype
import pypose as pp

# TODO: compute all constants without tensorflow. Out of the graph computation.
class Static(CostBase):
    '''
        Compute the cost for a static point.

        - input:
        --------
            - lam (lambda) the inverse temperature. 
            - gamma: decoupling parameter between action and noise.
            - upsilon: covariance augmentation for noise generation.
            - sigma: the noise covariance matrix. shape [aDim, aDim].
            - goal_pose: target goal pose. shape [pDim].
            - goal_vel: target goal velocity. shape [vDim].
            - Q: weight matrix for the different part of the cost function. shape: [sDim, sDim]
    '''
    def __init__(self, lam, gamma, upsilon, sigma, goal, Q, diag=False):
        super(Static, self).__init__(lam, gamma, upsilon, sigma)
        # TODO register buffer for those parameters.)
        self.register_buffer("Q", torch.diag(torch.tensor(Q, dtype=tdtype)))
        self.register_buffer("goal", torch.tensor(goal, dtype=tdtype))

    '''
        Set the goals of the cost function.

        input:
        ------
            - goal, torch.tensor. The goal of the cost function. 
                [x, y, z, qx, qy, qz, qw, u, v, w, p, q, r]
                Shape: [pDim+vDim]
    '''
    def setGoal(self, goal):
        self.goal = goal

    '''
        Computes state cost for the static point.

        - input:
        --------
            - pose: current pose. Shape: [k/1, pDim]
            - velocity: current velocity. Shape: [k/1, vDim]

        - output:
        ---------
            - (state-goal)^T Q (state-goal): scalar
    '''
    def state_cost(self, pose, velocity):
        state = torch.concat([pose, velocity], dim=-1)
        diff = torch.subtract(state, self.goal)[..., None]
        stateCost = torch.matmul(torch.transpose(diff, -1, -2), torch.matmul(self.Q, diff))[:, 0, 0]
        return stateCost

    '''
        Computes the final state cost.

        input:
        ------
            - pose, torch.tensor, the pose of the vehicle.
                Shape [k, pDim]
            - velocity, torch.tensor, the velocity.
                Shape [k, vDim]
    '''
    def final_cost(self, pose, velocity):
        return self.state_cost(pose, velocity)


class StaticPypose(CostBase):
    '''
        Compute the cost for a static point.

        - input:
        --------
            - lam (lambda) the inverse temperature. 
            - gamma: decoupling parameter between action and noise.
            - upsilon: covariance augmentation for noise generation.
            - sigma: the noise covariance matrix. shape [aDim, aDim].
            - goal_pose: target goal pose. pypose.SE3 shape [pDim].
            - goal_vel: target goal velocity. shape [vDim].
            - Q: weight matrix for the different part of the cost function. shape: [sDim, sDim]
    '''
    def __init__(self, lam, gamma, upsilon, sigma, goal_pose: pp.SE3, goal_vel: torch.tensor, Q, diag=False):
        super(StaticPypose, self).__init__(lam, gamma, upsilon, sigma)
        # TODO register buffer for those parameters.
        self.register_buffer("Q", torch.diag(torch.tensor(Q, dtype=tdtype)))
        self.register_buffer("goal_pose", pp.SE3(goal_pose))
        self.register_buffer("goal_vel", torch.tensor(goal_vel, dtype=tdtype))

        
    def setGoal(self, goal_pose: pp.SE3, goal_vel: torch.tensor):
        self.goal_pose = goal_pose
        self.goal_vel = goal_vel

    '''
        Computes state cost for the static point.

        - input:
        --------
            - state: current state. Shape: [k/1, sDim]

        - output:
        ---------
            - (state-goal)^T Q (state-goal): scalar
    '''
    def state_cost(self, pose, velocity):
        dist = self.dist(pose, velocity)
        c = torch.sqrt(torch.matmul(torch.transpose(dist, -1, -2), torch.matmul(self.Q, dist)))[..., 0, 0]
        return c


    def final_cost(self, pose, velocity):
        return self.state_cost(pose, velocity)


    def dist(self, pose: pp.SE3, vel: torch.tensor) -> torch.tensor:
        r1 = pose.rotation()
        r2 = self.goal_pose.rotation()
        t1 = pose.translation()
        t2 = self.goal_pose.translation()
        d_ang = (r1.Inv() * r2).Log().data # Equivalent to Log(R1.T * R2)
        d_lin = (t1 - t2)
        d_vel = vel - self.goal_vel
        dist = torch.concat([d_lin, d_ang, d_vel], dim=-1)[..., None]
        return dist