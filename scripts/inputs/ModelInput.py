import torch
import pypose as pp
from scripts.utils.utils import tdtype
from scripts.inputs.ControllerInput import ControllerInput, ControllerInputPypose


'''
    Model Input class. This class is responsible to handle the data that will be fed to the 
    model. Depening on the model it can maintain up to "steps" last states and actions. 
'''
class ModelInput(torch.nn.Module):
    '''
        Constructor.

        inputs:
        -------
            - k: the number of samples.
            - steps: the number of previous states and action to maintain.
            - sDim: the state dimension.
            - aDim: the action dimesnion.
    '''
    def __init__(self, k, steps, sDim=13, aDim=6):
        super(ModelInput, self).__init__()
        self.register_buffer("states", torch.zeros(k, steps, sDim, dtype=tdtype))
        self.register_buffer("actions", torch.zeros(k, steps, aDim, dtype=tdtype))
        self.k = k

    '''
        Init function from the ControllerInput class. This will broadcast the
        input to have k samples.

        inputs:
        -------
            - input: ControllerInput. The input should have the "steps" last 
                states and actions.
    '''
    def init(self, input: ControllerInput):
        self.states, self.actions[:, :-1] = input.get(self.k)

    '''
        Given a new action it returns the states and actions that will be applied
        to the model.

        inputs:
        -------
            - action: the latest action to apply to the model.
    '''
    def forward(self, action: torch.tensor):
        self.actions[:, -1] = action
        return self.states.clone(), self.actions.clone()

    '''
        Update the ModelInput with the newest state.
        
        inputs:
        -------
            - state: torch.tensor with shape (k, sDim, 1). The newest
                state of the system.
    '''
    def update(self, state: torch.tensor):
        # add the new state and action to the buffers at the location of
        # the oldest element.

        # TODO: FIGURE OUT WHAT IS HAPPENING HERE 
        self.states = self.states.clone()

        self.states[:, 0] = state
        # roll the tensor to the "left" by one step.
        self.states = torch.roll(self.states, -1, 1)


class ModelInputPypose(torch.nn.Module):
    def __init__(self, k, steps, aDim=6):
        super(ModelInputPypose, self).__init__()
        self.register_buffer("poses", pp.identity_SE3(k, steps, dtype=tdtype))
        self.register_buffer("vels", torch.zeros(k, steps, 6, dtype=tdtype))
        self.register_buffer("actions", torch.zeros(k, steps, aDim, dtype=tdtype))
        self.k = k

    def init(self, input: ControllerInputPypose):
        self.poses, self.vels, self.actions[:, :-1] = input.get(self.k)

    def forward(self, action: torch.tensor):
        self.actions[:, -1] = action
        return self.poses.clone(), self.vels.clone(), self.actions.clone()

    def update(self, pose, vel):
        # add the new state and action to the buffers at the location of
        # the oldest element.

        # TODO: FIGURE OUT WHAT IS HAPPENING HERE 
        self.poses = self.poses.clone()
        self.vels = self.vels.clone()

        self.poses[:, 0] = pose
        self.vels[:, 0] = vel
        # roll the tensor to the "left" by one step.
        self.poses = torch.roll(self.poses, -1, 1)
        self.vels = torch.roll(self.vels, -1, 1)
        self.actions = torch.roll(self.actions, -1, 1)