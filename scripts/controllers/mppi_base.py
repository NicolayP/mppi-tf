import torch
from utils import dtype

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

        self.register_buffer("sigma", torch.tensor(sigma, dtype=dtype))
        self.register_buffer("upsilon", torch.tensor(upsilon, dtype=dtype))
        self.register_buffer("lam", torch.tensor(lam))
        self.register_buffer("A", torch.zeros(tau, self.aDim, 1, dtype=dtype))

        # Shift_init.
        self.register_buffer("init", torch.zeros(self.aDim, 1))

        # TODO: Create observer.
        self.obs = observer
        self.model = model
        self.cost = cost

        self.update = Update(self.lam)

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
    def forward(self, state) -> torch.Tensor:
        action, self.A = self.control(state, self.A)
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
    def control(self, s, A):
        # Compute random noise.
        noises = self.noise()

        # Rollout the model and compute the cost of every sample.
        costs = self.rollout_cost(s, noises, A)
        # Compute the update of the action sequence.
        weighted_noises, eta = self.update(costs, noises)
        A = torch.add(A, weighted_noises)

        # Get next action.
        next = A[0].clone()

        # Shift and Update the Action Sequence.
        A[0] = self.init
        A_next = torch.roll(A, -1, 0)

        # Log the percent of samples contributing to the decision makeing.
        # self.obs.write_control("state", s)
        # self.obs.write_control("eta", eta)
        # self.obs.write_control("action", next)
        # self.obs.write_control("sample_cost", costs)
        # self.obs.write_control("sample_weight", weights)
        # self.obs.advance()
        # return next action and updated action sequence.

        return next, A_next

    '''
        Noise generator for the samples.

        input:
        ------
            - k: the number of samples
        
        output:
        -------
            - the noise associated with each samples ~ \mathcal{N}(\mu, \Sigma)
                Shape, [k, tau, aDim, 1]
    '''
    def noise(self):
        n = torch.normal(mean=torch.zeros(self.k, self.tau, self.aDim, 1, dtype=torch.double),
                         std=torch.ones(self.k, self.tau, self.aDim, 1, dtype=torch.double)).to(self.upsilon.device)
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
    def rollout_cost(self, s, noise, A) -> torch.Tensor:
        s = torch.unsqueeze(s, dim=0)
        cost = torch.zeros(self.k).to(s.device)
        s = torch.broadcast_to(s, (self.k, self.sDim, 1))

        for t in range(self.tau):
            a = A[t]
            n = noise[:, t]
            act = torch.add(a, n)

            next_s = self.model(s, act)
            tmp = self.cost(next_s, a, n)

            cost = torch.add(cost, tmp)
            s = next_s

        f_cost = self.cost(s, A[-1], noise[:, -1], final=True)
        cost = torch.add(cost, f_cost)
        return cost

class Update(torch.nn.Module):
    '''
        Update Module.

        input:
        ------
            - lam: float, the inverse temperature \lambda
    '''
    def __init__(self, lam):
        super(Update, self).__init__()
        self.lam = lam

    '''
        Compute the weights update according to the MPPI algorithm.

        input:
        ------
            - costs: torch.tensor, the costs associated with each sample rollout. 
                Shape, [k]
            - noise: torch.tensor, the noise associated with each sample rollout.
                Shape, [k, tau, aDim, 1]

        output:
        -------
            - weighted_noise: torch.tensor, the noise reweighted according to the
                importance sampling procedure. Shape, [k, tau, aDim, 1]
            - eta: float, the normalization term, indicator of MPPI's behavior.
    '''
    def forward(self, costs, noise):
        beta = self.beta(costs)
        arg = self.arg(costs, beta)
        exp_arg = self.exp_arg(arg)
        exp = self.exp(exp_arg)
        eta = self.eta(exp)
        weights = self.weights(exp, eta)
        weighted_noise = self.weighted_noise(weights, noise)
        return weighted_noise, eta

    '''
        Finds the cost with the smallest value. Alows to shift the
        samples costs so that at least 1 sample has a non-zeros weight.

        input:
        ------
            - costs: torch.tensor, the cost tensor. Shape [k]

        output:
        -------
            - min(costs), float, the minimal cost value.
    '''
    def beta(self, costs):
        return torch.min(costs)

    '''
        Shifts the costs by beta. And normalize the cost so that every
        value is in the interval [0, 1] if norm is true.

        input:
        ------
            - costs: torch.tensor, the costs tensor. Shape [k]
            - beta: float, the min value of the costs.
            - norm: bool, if true the costs will be normalized. Default: False

        output:
        -------
            - normalize(costs - beta)
    '''
    def arg(self, costs, beta, norm:bool=False):
        shift = torch.sub(costs, beta)
        if norm:
            max = torch.max(shift)
            return torch.div(shift, max)
        return shift

    '''
        Computes the exponential.

        input:
        ------
            - arg, the shifted (and normalized) costs.
                Shape [k]

        output:
        -------
            - (-\frac{1}{\lambda} * arg).
                Shape [k]
    '''
    def exp_arg(self, arg):
        return torch.mul((-1/self.lam), arg)

    '''
        Computes the exponential.

        input:
        ------
            - arg, the exponential argument.
                shape [k]

        output:
        -------
            - exp(arg), shape [k]
    '''
    def exp(self, arg):
        return torch.exp(arg)

    '''
        Computes the normalization term eta to make the exponential
        represent a probability distribution.

        input:
        ------
            - exp, the exponential, shape [k]

        output:
        -------
            - sum(exp), float. The normalization term
    '''
    def eta(self, exp):
        return torch.sum(exp)

    '''
        Normalize the exponential to get the sample weights.

        input:
        ------
            - exp, torch.tensor. the exponential, shape [k]
            - eta, float, the normalization term.
        
        output:
        -------
            - \frac{exp}{eta}, torch.Tensor. The sample weights.
                Shape [k]
    '''
    def weights(self, exp, eta):
        return torch.div(exp, eta)

    '''
        compute the weighted noise.

        input:
        ------
            - weights, torch.Tensor, the sample weights. Shape [k]
            - noise, torch.Tensor, the noise samples.
                Shape [k, tau, aDim, 1]

        output:
        -------
            - Weighted_noise, torch.Tensor.
                Shape [tau, aDim, 1]
    '''
    def weighted_noise(self, weights, noise):
        w = torch.unsqueeze(
                torch.unsqueeze(
                    torch.unsqueeze(weights, -1), -1), -1)

        return torch.sum(torch.mul(w, noise), 0)
