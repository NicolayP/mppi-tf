import torch
from utils import dtype
import numpy as np
import scipy.signal
import yaml

import time

class ControllerBase(torch.nn.Module):
    '''
        Mppi controller base class constructor.

        - Input:
        --------
            - model: a model object heritated from the model_base class.
            - Cost: a cost object heritated from the cost_base class.
            - k: Int, The number of samples used inside the controller.
            - tau: Int, The number of prediction timesteps.
            - sDim: Int, The state space dimension.
            - aDim: Int, The action space dimension.
            - lam: Float, The inverse temperature.
            - upsilon: Float, The augmented covariance on the noise term.
            - sigma: The noise of the system.
                Array with shape [aDim, aDim]
            - initSeq: The inital action sequence.
                Array of shape [tau, aDim, 1]
            - normalizeCost: Bool, wether or not normalizin the cost,
                simplifies tuning of lambda
            - filterSeq: Bool, wether or not to filter the input
                sequence after each optimization.
            - log: Bool, if true logs controller info in tensorboard.
            - logPath: String, the path where the info will be logged.
            - gif: Bool, if true generates an animated gif of the
                controller execution (not tested in ros).
            - configDict: Environment config dict.
            - taskDict: Task config dict, the cost hyper parameter.
            - modelDict: Model config dict, the model parameters.
            - debug: Bool, if true, the controller goes in debug mode
                and logs more information.

    '''
    def __init__(self,
                 model,
                 cost,
                 observer,
                 k=1,
                 tau=1,
                 lam=1.,
                 upsilon=1.,
                 sigma=0.):
        # TODO: Check parameters and make the tensors.
        super(ControllerBase, self).__init__()
        # This is needed to create a correct trace.
        self.k = k
        self.tau = tau

        self.aDim = 6
        self.sDim = 13

        self.register_buffer("sigma", torch.tensor(sigma))
        self.register_buffer("upsilon", torch.tensor(upsilon))
        self.register_buffer("lam", torch.tensor(lam))
        self.register_buffer("A", torch.zeros(tau, self.aDim, 1))

        # Shift_init.
        self.register_buffer("init", torch.zeros(self.aDim, 1))

        # TODO: Create observer.
        self.obs = observer
        self.model = model
        self.cost = cost

        self.update = Update(self.lam)
        self.rest_timer()

    '''
        Computes the next action with MPPI.
        input:
        ------
            - state: The current observed state of the system.
                shape: [StateDim, 1]
        output:
        -------
            - action: the next optimal aciton.
                shape: [ActionDim, 1]
    '''    
    def forward(self, state):
        start = time.perf_counter()

        action, self.A = self.control(self.k, state, self.A)
        end = time.perf_counter()
        self.timeDict['total'] += end-start
        self.timeDict['calls'] += 1
        return action

    '''
        Computes the optimal action sequence with MPPI.

        input:
        ------
            - k: the number of samples to use.
            - s: the state of the system.
                shape: [StateDim, 1]
            - A: the action sequence to optimize.
                shape: [tau, ActionDim, 1]
    '''
    def control(self, k, s, A):
        # Compute random noise.
        noises = self.noise(k)
        # Rollout the model and compute the cost of every sample.

        start = time.perf_counter()
        costs = self.rollout_cost(k, s, noises, A)
        end = time.perf_counter()
        self.timeDict['rollout'] += end-start

        # Compute the update of the action sequence.
        start = time.perf_counter()
        weighted_noises, nabla = self.update(costs, noises)
        end = time.perf_counter()
        self.timeDict['update'] += end-start
        # print("costs:          ", costs)
        # print("weighted_noise: ", weighted_noises)
        # print("A:              ", A)
        A = torch.add(A, weighted_noises)
        # print("A_update:       ", A)
        # Get next action.
        next = A[0].clone()
        # Shift and Update the Action Sequence.
        A[0] = self.init
        A_next = torch.roll(A, -1, 0)

        # Log the percent of samples contributing to the decision makeing.
        # self.obs.write_control("state", s)
        # self.obs.write_control("nabla", nabla)
        # self.obs.write_control("action", next)
        # self.obs.write_control("sample_cost", costs)
        # self.obs.write_control("sample_weight", weights)
        # self.obs.advance()
        # return next action and updated action sequence.
        return next, A_next

    def noise(self, k):
        n = torch.normal(mean=torch.zeros(k, self.tau, self.aDim, 1, dtype=dtype),
                         std=torch.ones(k, self.tau, self.aDim, 1, dtype=dtype)).to(self.upsilon.device)
        noise = torch.matmul(self.upsilon*self.sigma, n)
        return noise

    '''
        Computes the rollout of samples and it's associated cost.

        input:
        ------
            - s: the inital state of the system.
                Shape: [sDim, 1]
            - noise: The noise generated for each sample and applied for the rollout.
                Shape: [k, tau, aDim, 1]
            - A: the action sequence. The mean to apply.
                Shape: [tau, aDim, 1]

        output:
        -------
            - costs: Cost tensor of each rollout. 
                Shape: [k/1]
    '''
    def rollout_cost(self, k, s, noise, A):
        s = torch.unsqueeze(s, dim=0)
        cost = torch.zeros(self.k).to(s.device)
        s = torch.broadcast_to(s, (self.k, self.sDim, 1))

        for t in range(self.tau):
            a = A[t]
            n = noise[:, t]
            act = torch.add(a, n)

            start = time.perf_counter()
            next_s = self.model(s, act)
            end = time.perf_counter()
            self.timeDict['model'] += end-start
            
            start = time.perf_counter()
            tmp = self.cost(next_s, a, n)
            end = time.perf_counter()
            self.timeDict['cost'] += end-start

            cost = torch.add(cost, tmp)
            s = next_s

        f_cost = self.cost(s, final=True)
        cost = torch.add(cost, f_cost)

        #self.obs.write_control("sample_costs", cost)

        return cost

    def stats(self):
        print("*"*5, " Mean execution time ", "*"*5)
        print("* Next step : {:.4f} (sec)".format(self.timeDict["total"]/self.timeDict["calls"]))

    def rest_timer(self):
        self.timeDict = {}
        self.timeDict["total"] = 0.
        self.timeDict["rollout"] = 0.
        self.timeDict["update"] = 0.
        self.timeDict["model"] = 0.
        self.timeDict["cost"] = 0.
        self.timeDict["calls"] = 0

class Update(torch.nn.Module):
    def __init__(self, lam):
        super(Update, self).__init__()
        self.lam = lam

    def forward(self, costs, noise):
        beta = self.beta(costs)
        arg = self.arg(costs, beta)
        exp_arg = self.exp_arg(arg)
        exp = self.exp(exp_arg)
        nabla = self.nabla(exp)
        weights = self.weights(exp, nabla)
        weighted_noise = self.weighted_noise(weights, noise)
        return weighted_noise, nabla

    def beta(self, costs):
        return torch.min(costs)
    
    def arg(self, costs, beta):
        return torch.sub(costs, beta)

    def exp_arg(self, arg):
        return torch.mul((-1/self.lam), arg)

    def exp(self, arg):
        return torch.exp(arg)

    def nabla(self, exp):
        return torch.sum(exp)

    def weights(self, exp, nabla):
        return torch.div(exp, nabla)

    def weighted_noise(self, weights, noise):
        w = torch.unsqueeze(
                torch.unsqueeze(
                    torch.unsqueeze(weights, -1), -1), -1)

        return torch.sum(torch.mul(w, noise), 0)
