import torch
import pypose as pp


class AUVPROXYDeltaV(torch.nn.Module):
    def __init__(self, dv=None):
        super(AUVPROXYDeltaV, self).__init__()
        self._dv = dv
        self.i = 0

    def forward(self, x, v, u, h0=None):
        self.i += 1
        return self._dv[:, self.i-1:self.i], None


class AUVRNNDeltaV(torch.nn.Module):
    def __init__(self, params=None):
        super(AUVRNNDeltaV, self).__init__()
        self.input_size = 21 # 9 for rotation matrix + 6 for velocities + 6 for actions
        self.output_size = 6

        # lstm default config
        self.rnn_layers = 1
        self.rnn_hidden_size = 64
        self.rnn_bias = False

        # FC default config
        self.topology = [32, 32]
        self.fc_bias = False
        self.bn = True
        self.relu_neg_slope = 0.1

        # Parse parameters
        self.parse_params(params)

        self.rnn = build_rnn(self.input_size, self.rnn_layers, self.rnn_hidden_size, self.rnn_bias)
        
        self.fc = build_fc(self.rnn_hidden_size, self.output_size, # input/output size
                           self.topology, self.fc_bias, self.bn, self.relu_neg_slope) #config

    def forward(self, x, v, u, h0=None):
        if h0 is None:
            k = x.shape[0]
            h0 = self.init_hidden(k, x.device)
        r = x.rotation().matrix().flatten(start_dim=-2)
        input_seq = torch.concat([r, v, u], dim=-1)
        out, hN = self.rnn(input_seq, h0)
        dv = self.fc(out)
        return dv, hN

    def init_hidden(self, k, device):
        return torch.zeros(self.rnn_layers, k, self.rnn_hidden_size, device=device)

    def parse_params(self, params):
        if params is not None:
            if "rnn" in params:
                if "rnn_layer" in params["rnn"]:
                    self.rnn_layers = params["rnn"]["rnn_layer"]
                if "rnn_hidden_size" in params["rnn"]:
                    self.rnn_hidden_size = params["rnn"]["rnn_hidden_size"]
                if "bias" in params["rnn"]:
                    self.rnn_bias = params["rnn"]["bias"]

            if "fc" in params:
                if "topology" in params["fc"]:
                    self.topology = params["fc"]["topology"]
                if "bias" in params["fc"]:
                    self.fc_bias = params["fc"]["bias"]
                if "batch_norm" in params["fc"]:
                    self.bn = params["fc"]["batch_norm"]
                if "relu_neg_slope" in params["fc"]:
                    self.relu_neg_slope = params["fc"]["relu_neg_slope"]


class AUVLSTMDeltaV(torch.nn.Module):
    def __init__(self, params=None):
        super(AUVLSTMDeltaV, self).__init__()
        self.input_size = 21 # 9 for rotation matrix + 6 for velocities + 6 for actions
        self.output_size = 6

        # lstm default config
        self.lstm_layers = 1
        self.lstm_hidden_size = 64
        self.lstm_bias = False

        # FC default config
        self.topology = [32, 32]
        self.fc_bias = False
        self.bn = True
        self.relu_neg_slope = 0.1

        # Parse parameters
        self.parse_params(params)

        self.lstm = build_lstm(self.input_size, self.lstm_layers, self.lstm_hidden_size, self.lstm_bias)
        
        self.fc = build_fc(self.lstm_hidden_size, self.output_size, # input/output size
                           self.topology, self.fc_bias, self.bn, self.relu_neg_slope) #config

    def forward(self, x, v, u, h0=None):
        if h0 is None:
            k = x.shape[0]
            h0 = self.init_hidden(k, x.device)
        
        r = x.rotation().matrix().flatten(start_dim=-2)
        input_seq = torch.concat([r, v, u], dim=-1)
        out, hN = self.lstm(input_seq, h0)
        dv = self.fc(out)
        return dv, hN

    def init_hidden(self, k, device):
        return (torch.zeros(self.lstm_layers, k, self.lstm_hidden_size, device=device),
                torch.zeros(self.lstm_layers, k, self.lstm_hidden_size, device=device))

    def parse_params(self, params):
        if params is not None:
            if "lstm" in params:
                if "lstm_layer" in params["lstm"]:
                    self.lstm_layers = params["lstm"]["lstm_layer"]
                if "lstm_hidden_size" in params["lstm"]:
                    self.lstm_hidden_size = params["lstm"]["lstm_hidden_size"]
                if "bias" in params["lstm"]:
                    self.lstm_bias = params["lstm"]["bias"]

            if "fc" in params:
                if "topology" in params["fc"]:
                    self.topology = params["fc"]["topology"]
                if "bias" in params["fc"]:
                    self.fc_bias = params["fc"]["bias"]
                if "batch_norm" in params["fc"]:
                    self.bn = params["fc"]["batch_norm"]
                if "relu_neg_slope" in params["fc"]:
                    self.relu_neg_slope = params["fc"]["relu_neg_slope"]


class AUVStep(torch.nn.Module):
    def __init__(self, params=None, dt=0.1):
        super(AUVStep, self).__init__()
        self.dv_pred = AUVLSTMDeltaV()
        self.dt = dt
        if params:
            if "rnn" in params:
                self.dv_pred = AUVRNNDeltaV(params)
            elif "lstm" in params:
                self.dv_pred = AUVLSTMDeltaV(params)
        self.std = 1.
        self.mean = 0.

    def forward(self, x, v, u, h0=None):
        normed_dv, next_h = self.dv_pred(x, v, u, h0)
        next_dv = normed_dv*self.std + self.mean
        next_v = v + next_dv
        t = pp.se3(self.dt*next_v).Exp()
        next_x = x * t
        return next_x, next_v, next_dv, next_h


class AUVTraj(torch.nn.Module):
    def __init__(self, params=None, dt=0.1, limMax=None, limMin=None):
        super(AUVTraj, self).__init__()
        self.auv_step = AUVStep(params, dt)

    '''
    '''
    def forward(self, pose: pp.SE3, velocity: torch.tensor, actions: torch.tensor):
        k = actions.shape[0]
        tau = actions.shape[1]
        h = None # rnn or lstm hidden state
        pose_traj = pp.SE3(torch.zeros(size=(k, tau, 7)).to(pose.device))
        vel_traj = torch.zeros(size=(k, tau, 6)).to(pose.device)
        dv_traj = torch.zeros(size=(k, tau, 6)).to(pose.device)

        for i in range(tau):
            u = actions[:, i:i+1]
            next_pose, next_vel, next_dv, next_h = self.auv_step(pose, velocity, u, h)
            pose_traj[:, i:i+1] = next_pose.clone()
            vel_traj[:, i:i+1] = next_vel.clone()
            dv_traj[:, i:i+1] = next_dv.clone()
            h = next_h
            pose, velocity = next_pose, next_vel
        return pose_traj, vel_traj, dv_traj


class AUVTrajV2(torch.nn.Module):
    def __init__(self, params=None, dt=0.1, limMax=None, limMin=None):
        super(AUVTrajV2, self).__init__()
        self.auv_step = AUVStep(params, dt)

    '''
    '''
    def forward(self, state: torch.tensor, actions: torch.tensor):
        k = actions.shape[0]
        tau = actions.shape[1]
        h = None # rnn or lstm hidden state
        pose = pp.SE3(state[..., :7]).to(state.device)
        velocity = state[..., 7:]
        pose_traj = pp.SE3(torch.zeros(size=(k, tau, 7)).to(pose.device))
        vel_traj = torch.zeros(size=(k, tau, 6)).to(pose.device)
        dv_traj = torch.zeros(size=(k, tau, 6)).to(pose.device)

        for i in range(tau):
            u = actions[:, i:i+1]
            next_pose, next_vel, next_dv, next_h = self.auv_step(pose, velocity, u, h)
            pose_traj[:, i:i+1] = next_pose.clone()
            vel_traj[:, i:i+1] = next_vel.clone()
            dv_traj[:, i:i+1] = next_dv.clone()
            h = next_h
            pose, velocity = next_pose, next_vel
        return pose_traj, vel_traj, dv_traj


def build_rnn(input_size, layers, hidden_size, bias=False):
    return torch.nn.RNN(input_size=input_size,
                        hidden_size=hidden_size,
                        num_layers=layers,
                        batch_first=True,
                        bias=bias,
                        nonlinearity="tanh")


def build_lstm(input_size, layers, hidden_size, bias=False):
    return torch.nn.LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=layers,
        batch_first=True,
        bias=bias
    )


def build_fc(input_size, output_size, topology, bias=False, batch_norm=False, relu_slope=0.1):
    fc_layers = []
    for i, s in enumerate(topology):
        if i == 0:
            fc_layers.append(torch.nn.Linear(input_size, s, bias=bias))
        else:
            fc_layers.append(torch.nn.Linear(topology[i-1], s, bias=bias))

        if batch_norm:
            fc_layers.append(torch.nn.BatchNorm1d(s))

        fc_layers.append(torch.nn.LeakyReLU(negative_slope=relu_slope))
    fc_layers.append(torch.nn.Linear(topology[-1], output_size, bias=bias))
    fc = torch.nn.Sequential(*fc_layers)
    fc.apply(init_weights)
    return fc


def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)