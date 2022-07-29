import torch
from .cost_base import CostBase

# TODO: compute all constants without tensorflow. Out of the graph computation.
class Static(CostBase):
    def __init__(self, lam, gamma, upsilon, sigma, goal, Q, diag=False):
        '''
            Compute the cost for a static point.

            - input:
            --------
                - lam (lambda) the inverse temperature. 
                - gamma: decoupling parameter between action and noise.
                - upsilon: covariance augmentation for noise generation.
                - sigma: the noise covariance matrix. shape [aDim, aDim].
                - goal: target goal (psition; speed). shape [sDim, 1].
                - Q: weight matrix for the different part of the cost function. shape: [sDim, sDim]
        '''

        super(Static, self).__init__(lam, gamma, upsilon, sigma)
        # TODO register buffer for those parameters.
        self.register_buffer("Q", torch.diag(torch.tensor(Q)))
        self.register_buffer("goal", torch.tensor(goal))
        
    def setGoal(self, goal):
        self.goal = goal

    def state_cost(self, state):
        '''
            Computes state cost for the static point.

            - input:
            --------
                - state: current state. Shape: [k/1, sDim, 1]

            - output:
            ---------
                - dict with entries:
                    "state_cost" = (state-goal)^T Q (state-goal)
        '''
        diff = torch.unsqueeze(torch.subtract(state, self.goal), dim=-1)
        stateCost = torch.matmul(torch.transpose(diff, -1, -2), torch.matmul(self.Q, diff))
        return stateCost

    def final_cost(self, state):
        return self.state_cost(state)