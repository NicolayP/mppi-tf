import torch
from utils import dtype

# TODO: compute all constants without tensorflow. Out of the graph
# computation.
class CostBase(torch.nn.Module):
    '''
        Cost base class for the mppi controller.
        This is an abstract class and should be heritated by every
        specific cost.
    '''
    def __init__(self, lam, gamma, upsilon, sigma):
        '''
            Abstract class for the cost function.
            - input:
            --------
                - lam (lambda) the inverse temperature.
                - gamma: decoupling parameter between action and noise.
                - upsilon: covariance augmentation for noise generation.
                - sigma: the noise covariance matrix. shape [aDim, aDim]
                - TODO: add diag arg to feed sigma as the diag element if
                prefered.

            - output:
            ---------
                the new cost object.
        '''
        super(CostBase, self).__init__()
        self._observer = None
        self.lam = lam
        self.gamma = gamma
        self.upsilon = upsilon
        self.register_buffer("invSig", torch.linalg.inv(torch.tensor(sigma, dtype=dtype)))

    def forward(self, state, action=None, noise=None, final: bool =False):
        '''
            Computes the cost of a sample at a given time.
            - input:
            --------
                - state: The current state of the system.
                    shape: [k/1, sDim, 1]
                - action: The action applied to reach the current state.
                    shape: [aDim, 1]
                - noise: The noise applied to the sample.
                    shape: [k/1, aDim, 1]
                - final: Bool, if true it computes the final state cost.

            - output:
            ---------
                - cost for a given step,
                    shape: [k/1]
        '''

        if final:
            return torch.squeeze(self.final_cost(state))

        s_cost = torch.squeeze(self.state_cost(state))
        a_cost = torch.squeeze(self.action_cost(action, noise))
        return torch.add(s_cost, a_cost)

    def final_cost(self, state):
        raise NotImplementedError
    
    def state_cost(self, state):
        raise NotImplementedError

    def action_cost(self, action, noise):
        '''
            action related cost part.

            - input: 
            --------
                - action: the action applied to reach the current state.
                    shape: [k/1, aDim, 1]
                - noise: the noise applied to the sample.
                    shape: [k/1, aDim, 1]

            - output:
            ---------
                - The cost associated with the current action.
        '''
        # Right hand side noise
        rhsNcost = torch.matmul(self.invSig, noise)
        
        # Right hand side action
        rhsAcost = torch.matmul(self.invSig, action)

        # \u^{T}_t \Sigma^{-1} \epsilon_t, Mix cost
        mixCost = torch.matmul(torch.transpose(action, -1, -2), rhsNcost)
        mixCost = torch.multiply(mixCost, 2.)

        # \epsilon^{T}_t \Sigma^{-1} \epsilon_t
        nCost = torch.matmul(torch.transpose(noise, -1, -2), rhsNcost)

        # \u^{T}_t \Sigma^{-1} \u_t
        aCost = torch.matmul(torch.transpose(action, -1, -2), rhsAcost)

        aCost = torch.multiply(aCost, self.gamma)

        mixCost = torch.multiply(mixCost, self.gamma)

        nCost = torch.multiply(nCost, self.lam*(1.-1./self.upsilon))

        # \gamma [action_cost + 2mix_cost]
        controlCost = torch.add(aCost, mixCost)
        actionCost = torch.multiply(torch.add(controlCost, nCost), 0.5)
        return actionCost

    def set_observer(self, observer):
        self._observer = observer