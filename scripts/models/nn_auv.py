import torch
#torch.autograd.set_detect_anomaly(True)
import pypose as pp
from scripts.utils.utils import tdtype, npdtype
from scripts.models.model_base import ModelBase
from scripts.inputs.ModelInput import ModelInputPypose

###########################################
#                                         #
#      NETWORKS AND PROXY DEFINITIONS     #
#                                         #
###########################################
'''
    RNN predictor for $\delta v$.

    parameters:
    -----------
        - rnn:
            - rnn_layer: int, The number of rnn layers.
            - rnn_hidden_size: int, Number of hidden units
            - bias: Whether bool, or not to apply bias to the RNN units. (default False)
            - activation: string, The activation function used (tanh or relu)
        - fc:
            - topology: array of ints, each entry indicates the number of hidden units on that
                corresponding layer.
            - bias: bool, Whether or not to apply bias to the FC units. (default False)
            - batch_norm: bool, Whether or not to apply batch normalization. (default False)
            - relu_neg_slope: float, the negative slope of the relu activation.
'''
class AUVRNNDeltaV(torch.nn.Module):
    '''
        RNN network predictinfg the next velocity delta.

        inputs:
        -------
            params: dict, the parameters that define the topology of the network.
            see in class definition.
    '''
    def __init__(self, params=None):
        super(AUVRNNDeltaV, self).__init__()

        self.input_size = 9 + 6 + 6 # rotation matrix + velocities + action. I.E 21
        self.output_size = 6

        # RNN part
        self.rnn_layers = 5
        self.rnn_hidden_size = 1
        rnn_bias = False
        nonlinearity = "tanh"

        # FC part
        topology = [32, 32]
        fc_bias = False
        bn = True
        relu_neg_slope = 0.1

        if params is not None:
            if "rnn" in params:
                if "rnn_layer" in params["rnn"]:
                    self.rnn_layers = params["rnn"]["rnn_layer"]
                if "rnn_hidden_size" in params["rnn"]:
                    self.rnn_hidden_size = params["rnn"]["rnn_hidden_size"]
                if "bias" in params["rnn"]:
                    rnn_bias = params["rnn"]["bias"]
                if "activation" in params["rnn"]:
                    nonlinearity = params["rnn"]["activation"]

            if "fc" in params:
                if "topology" in params["fc"]:
                    topology = params["fc"]["topology"]
                if "bias" in params["fc"]:
                    fc_bias = params["fc"]["bias"]
                if "batch_norm" in params["fc"]:
                    bn = params["fc"]["batch_norm"]
                if "relu_neg_slope" in params["fc"]:
                    relu_neg_slope = params["fc"]["relu_neg_slope"]

        self.rnn = torch.nn.RNN(
            input_size=self.input_size,
            hidden_size=self.rnn_hidden_size,
            num_layers=self.rnn_layers,
            batch_first=True,
            bias=rnn_bias,
            nonlinearity=nonlinearity
        )

        fc_layers = []
        for i, s in enumerate(topology):
            if i == 0:
                layer = torch.nn.Linear(self.rnn_hidden_size, s, bias=fc_bias)
            else:
                layer = torch.nn.Linear(topology[i-1], s, bias=fc_bias)
            fc_layers.append(layer)

            # TODO try batch norm.
            if bn:
                fc_layers.append(torch.nn.BatchNorm1d(s))

            fc_layers.append(torch.nn.LeakyReLU(negative_slope=relu_neg_slope))

        layer = torch.nn.Linear(topology[-1], 6, bias=fc_bias)
        fc_layers.append(layer)

        self.fc = torch.nn.Sequential(*fc_layers)
        self.fc.apply(init_weights)

    '''
        Forward function of the velocity delta predictor.

        inputs:
        -------
            - x, the state of the vehicle. Pypose element. Shape (k, se3_rep)
            - v, the velocity of the vehicle. Pytorch tensor, Shape (k, 6)
            - u, the action applied to the vehicle. Pytorch tensor, Shape (k, 6)
            - h0, the internal state of the rnn unit. Shape (rnn_layers, k, rnn_hiden_size)
                if None, the object will create a new one.

        outputs:
        --------
            - dv, the next velocity delta. Tensor, shape (k, 6, 1)
            - hN, the next rnn internal state. Shape (rnn_layers, k, rnn_hidden_size)
    '''
    def forward(self, x, v, u, hidden=None):
        k = x.shape[0]
        r = x.rotation().matrix().flatten(start_dim=-2)
        input_seq = torch.concat([r, v, u], dim=-1)

        if hidden is None:
            hidden = self.init_hidden(k, x.device)

        out, h = self.rnn(input_seq, hidden)
        dv = self.fc(out[:, 0])
        return dv[:, None], h

    '''
        Helper function to create the rnn internal layer.

        inputs:
        -------
            - k, int, the batch size.
            - device, the device on which to load the tensor.

        outputs:
        --------
            - h0, shape (rnn_layers, k, rnn_hidden_size)
    '''
    def init_hidden(self, k, device):
        return torch.zeros(self.rnn_layers, k, self.rnn_hidden_size, device=device)


class AUVLSTMDeltaV(torch.nn.Module):
    def __init__(self, params=None):
        super(AUVLSTMDeltaV, self).__init__()

        self.input_size = 9 + 6 + 6  # rotation matrix + velocities + action. I.E 21
        self.output_size = 6

        self.lstm_layers = 5
        self.lstm_hidden_size = 1
        lstm_bias = False

        topology = [32, 32]
        fc_bias = False
        bn = True
        relu_neg_slope = 0.1

        if params is not None:
            if "lstm" in params:
                if "lstm_layer" in params["lstm"]:
                    self.lstm_layers = params["lstm"]["lstm_layer"]
                if "lstm_hidden_size" in params["lstm"]:
                    self.lstm_hidden_size = params["lstm"]["lstm_hidden_size"]
                if "bias" in params["lstm"]:
                    lstm_bias = params["lstm"]["bias"]

            if "fc" in params:
                if "topology" in params["fc"]:
                    topology = params["fc"]["topology"]
                if "bias" in params["fc"]:
                    fc_bias = params["fc"]["bias"]
                if "batch_norm" in params["fc"]:
                    bn = params["fc"]["batch_norm"]
                if "relu_neg_slope" in params["fc"]:
                    relu_neg_slope = params["fc"]["relu_neg_slope"]

        self.lstm = torch.nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_layers,
            batch_first=True,
            bias=lstm_bias,
        )

        fc_layers = []
        for i, s in enumerate(topology):
            if i == 0:
                layer = torch.nn.Linear(self.lstm_hidden_size, s, bias=fc_bias)
            else:
                layer = torch.nn.Linear(topology[i - 1], s, bias=fc_bias)
            fc_layers.append(layer)

            if bn:
                fc_layers.append(torch.nn.BatchNorm1d(s))

            fc_layers.append(torch.nn.LeakyReLU(negative_slope=relu_neg_slope))

        layer = torch.nn.Linear(topology[-1], 6, bias=fc_bias)
        fc_layers.append(layer)

        self.fc = torch.nn.Sequential(*fc_layers)

    def forward(self, x, v, u, hidden=None):
        k = x.shape[0]
        r = x.rotation().matrix().flatten(start_dim=-2)
        input_seq = torch.cat([r, v, u], dim=-1)

        if hidden is None:
            hidden = self.init_hidden(k, x.device)

        out, h = self.lstm(input_seq, hidden)
        dv = self.fc(out[:, 0])

        return dv[:, None], h

    def init_hidden(self, k, device):
        return (torch.zeros(self.lstm_layers, k, self.lstm_hidden_size, device=device),
                torch.zeros(self.lstm_layers, k, self.lstm_hidden_size, device=device))


class AUVNNDeltaV(torch.nn.Module):
    def __init__(self, params=None):
        super(AUVNNDeltaV, self).__init__()
        
        self.input_size = (9 + 6 + 6) # rotation matrix + velocities + action. I.E 21
        self.output_size = 6

        # FC part
        history = 4
        topology = [32, 32]
        fc_bias = False
        bn = True
        relu_neg_slope = 0.1

        if params is not None:
            if "topology" in params:
                topology = params["topology"]
            if "bias" in params:
                fc_bias = params["bias"]
            if "batch_norm" in params:
                bn = params["batch_norm"]
            if "relu_neg_slope" in params:
                relu_neg_slope = params["relu_neg_slope"]
            if "history" in params:
                history = params["history"]

        self.input_size *= history

        fc_layers = []
        for i, s in enumerate(topology):
            if i == 0:
                layer = torch.nn.Linear(self.input_size, s, bias=fc_bias)
            else:
                layer = torch.nn.Linear(topology[i-1], s, bias=fc_bias)
            fc_layers.append(layer)

            # TODO try batch norm.
            if bn:
                fc_layers.append(torch.nn.BatchNorm1d(s))

            fc_layers.append(torch.nn.LeakyReLU(negative_slope=relu_neg_slope))

        layer = torch.nn.Linear(topology[-1], 6, bias=fc_bias)
        fc_layers.append(layer)

        self.fc = torch.nn.Sequential(*fc_layers)

    '''
        Forward function of the velocity delta predictor.

        inputs:
        -------
            - x, the state of the vehicle. Pypose element. Shape (k, steps, se3_rep)
            - v, the velocity of the vehicle. Pytorch tensor, Shape (k, steps, 6)
            - u, the action applied to the vehicle. Pytorch tensor, Shape (k, steps, 6)
            - h, fake hidden to match prototype of RNN and LSTM networks

        outputs:
        --------
            - dv, the next velocity delta. Tensor, shape (k, 6, 1)
    '''
    def forward(self, x, v, u, h=None):
        k = x.shape[0]
        r = x.rotation().matrix().flatten(-2) # shape [k, steps, 9]
        input_seq = torch.concat([r, v, u], dim=-1).view(k, -1)

        dv = self.fc(input_seq)
        return dv[:, None], None


'''
    Proxy for the RNN part of the network. Used to ensure that
    the integration using PyPose is correct.
'''
class AUVRNNDeltaVProxy(torch.nn.Module):
    '''
        Delta veloctiy proxy constructor.

        input:
        ------
            - dv, pytorch tensor. The ground truth velocity delta.
            Shape (k, tau, 6)
    '''
    def __init__(self, dv):
        super(AUVRNNDeltaVProxy, self).__init__()
        self._dv = dv
        self.i = 0

    '''
        Forward function.
        Returns the next ground truth entry. Inputs are only there
        to match the function prototype.

        inputs:
        -------
            - x, that state of the vehicle (not used)
            - v, the velocity of the vehicle (not used)
            - u, the action applied to the vehicle (not used)
            - h0, the vecotr representing the last steps (used for rnn but not here)

        outputs:
        --------
            - dv[:, current-1:current, ...], the current velocity delta.
                shape [k, 1, 6]
            - hNext: set to None.
    '''
    def forward(self, x, v, a, h0=None):
        self.i += 1
        return self._dv[:, self.i-1:self.i], None


'''
    Performs a single integration step using pypose and velocity delta.

    parameters:
    -----------
        - model: dict, entry that contains the NN model definition. See AUVRNNDeltaV.
        - dataset_params:
            - v_frame: string, The frame in which the velocity is expressed, world or body
                default: body
            - dv_frame: string, The frame in which the velocity delta is expressed, world or body
                default: body
'''
class AUVStep(ModelBase):
    '''
        AUVStep Constructor.

        inputs:
        -------
            - params, dict. See object definition above.
            - dt, the integration time.
    '''
    def __init__(self, params=None, dt=0.1, dv_pred=None):
        super(AUVStep, self).__init__()
        if params is not None:
            if params["model"]["type"] == "rnn":
                self.dv_pred = AUVRNNDeltaV(params["model"])
            elif params["model"]["type"] == "lstm":
                self.dv_pred = AUVLSTMDeltaV(params["model"])
            elif params["model"]["type"] == "nn":
                self.dv_pred = AUVNNDeltaV(params["model"])

            if "dataset_params" in params:
                if "v_frame" in params["dataset_params"]:
                    self.v_frame = params["dataset_params"]["v_frame"]
                if "dv_frame" in params["dataset_params"]:
                    self.dv_frame = params["dataset_params"]["dv_frame"]

        else:
            self.dv_pred = AUVRNNDeltaV()
            self.v_frame = "body"
            self.dv_frame = "body"

        self.dt = dt
        self.std = 1.
        self.mean = 0.

    '''
        Predicts next state (x_next, v_next) from (x, v, a, h0)

        inputs:
        -------
            - x, pypose.SE3. The current pose on the SE3 manifold.
                shape [k, n, 7] (pypose uses quaternion representation)
            - v, torch.Tensor \in \mathbb{R}^{6} \simeq pypose.se3.
                The current velocity. shape [k, n, 6]
            - u, torch.Tensor the current applied forces.
                shape [k, n, 6]
            - h0, the hidden state of the RNN network.

        outputs:
        --------
            - x_next, pypose.SE3. The next pose on the SE3 manifold.
                shape [k, 1, 7]
            - v_next, torch.Tensor \in \mathbb{R}^{6} \simeq pypose.se3.
                The next velocity. Shape [k, 1, 6]
    '''
    def forward(self, x, v, u, h0=None, train=False):
        dv, h = self.dv_pred(x, v, u, h0)
        dv_unnormed = dv*self.std + self.mean
        v_next = v[:, -1:] + dv_unnormed
        t = pp.se3(self.dt*v_next).Exp()
        x_next = x[:, -1:] * t
        if not train:
            return x_next, v_next, h
        return x_next, v_next, dv_unnormed, h

    '''
        Set the mean and variance of the input data.
        This will be used to normalize the input data.

        inputs:
        -------
            - mean, tensor, shape (6)
            - std, tensor, shape (6)
    '''
    def set_stats(self, mean, std):
        self.mean = mean
        self.std = std

    '''

    '''
    def update_model(self, dv_pred):
        self.dv_pred = dv_pred


'''
    Performs full trajectory integration.

    params:
    -------
        - model:
            - se3: bool, whether or not to use pypose.
            - for other entries look at AUVStep and AUVRNNDeltaV.
'''
class AUVTraj(torch.nn.Module):
    '''
        Trajectory generator objects.

        inputs:
        -------
            - params: see definnition above.
    '''
    def __init__(self, params=None):
        super(AUVTraj, self).__init__()
        self.step = AUVStep(params)
        if params is not None:
            self.se3 = params["model"]["se3"]
        else:
            self.se3 = True

    '''
        Generates a trajectory using a Inital State (pose + velocity) combined
        with an action sequence.

        inputs:
        -------
            - x, ModelInputPypose, the model input containing and managing the previous steps.
            - U, torch.Tensor. The action sequence appliedto the system.
                shape [k, Tau, 6]

        outputs:
        --------
            - traj, torch.Tensor. The generated trajectory.
                shape [k, tau, 7]
            - traj_v, torch.Tensor. The generated velocity profiles.
                shape [k, tau, 6]
            - traj_dv, torch.Tensor. The predicted velocity delta. Used for
                training. shape [k, tau, 6]
    '''
    def forward(self, x: ModelInputPypose, U):
        k = U.shape[0]
        tau = U.shape[1]
        h = None
        traj = torch.zeros(size=(k, tau, 7)).to(U.device)
        traj = pp.SE3(traj)
        traj_v = torch.zeros(size=(k, tau, 6)).to(U.device)
        traj_dv = torch.zeros(size=(k, tau, 6)).to(U.device)

        for i in range(tau):
            poses, vels, actions = x(U[:, i])
            next_pose, next_vel, dv, h_next = self.step(poses, vels, actions, h, train=True)
            traj[:, i:i+1] = next_pose.clone()
            traj_v[:, i:i+1] = next_vel.clone()
            traj_dv[:, i:i+1] = dv.clone()
            h = h_next
            x.update(next_pose, next_vel)
        return traj, traj_v, traj_dv


'''
    Inits the weights of the neural network.

    inputs:
    -------
        - m the neural network layer.
'''
def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
