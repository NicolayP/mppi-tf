import torch
import pypose as pp
from scripts.utils.utils import tdtype
from scripts.inputs.ControllerInput import ControllerInput, ControllerInputPypose


'''
    Model Input class. This class is responsible to handle the data that will be fed to the 
    model. Depening on the model it can maintain up to "history" last states and actions. 
'''
class ModelInput(torch.nn.Module):
    '''
        Constructor.

        inputs:
        -------
            - k: the number of samples.
            - history: the number of previous states and action to maintain.
            - pDim: the pose dimension.
            - vDim: the velocity dimension.
            - aDim: the action dimesnion.
    '''
    def __init__(self, k, history, pDim=7, vDim=6, aDim=6):
        super(ModelInput, self).__init__()
        self.register_buffer("poses", torch.zeros(k, history, pDim, dtype=tdtype))
        self.register_buffer("vels", torch.zeros(k, history, vDim, dtype=tdtype))
        self.register_buffer("actions", torch.zeros(k, history, aDim, dtype=tdtype))
        self.k = k

    '''
        Init function from the ControllerInput class. This will broadcast the
        input to have k samples.

        inputs:
        -------
            - input: ControllerInput. The input should have the "history" last 
                states and actions.
    '''
    def init(self, input: ControllerInput):
        poses, vels, actions = input.get(self.k)
        self.poses, self.vels, self.actions[:, :-1] = poses, vels, actions

    '''
        Init the model input from data immidiatly.

        inputs:
        -------
            - poses: torch.tensor. Poses of the vehicle.
                shape [k, history, 7]
            - vels: torch.tensor. Velocities of the vehicle.
                shape [k, history, 6]
            - actions: torch.tensor. Actions applied on the vehicle.

    '''
    def init_from_states(self, poses, vels, actions: torch.tensor):
        self.poses = poses
        self.vels = vels
        self.actions[:, :-1] = actions

    '''
        Given a new action it returns the states and actions that will be applied
        to the model.

        inputs:
        -------
            - action: the latest action to apply to the model.
                shape [k, 1, aDim]

        output:
        -------
            - poses, torch.tensor, the poses, shape [k, history, pDim]
            - vels, torch.tensor, the velocities, shape [k, history, vDim]
            - actions, torch.tensor, the actions, shape [k, history, aDim]
    '''
    def forward(self, action: torch.tensor):
        self.actions[:, -1] = action
        return self.poses.clone(), self.vels.clone(), self.actions.clone()

    '''
        Update the ModelInput with the newest state.
        
        inputs:
        -------
            - pose: torch.tensor with shape (k, 1, pDim). The newest
                state of the system.
    '''
    def update(self, pose: torch.tensor, vel: torch.tensor):
        # add the new state and action to the buffers at the location of
        # the oldest element.

        # TODO: FIGURE OUT WHAT IS HAPPENING HERE 
        self.poses = self.poses.clone()
        self.vels = self.vels.clone()

        self.poses[:, 0:1] = pose
        self.vels[:, 0:1] = vel
        # roll the tensor to the "left" by one step.
        self.poses = torch.roll(self.poses, -1, 1)
        self.vels = torch.roll(self.vels, -1, 1)


class ModelInputPypose(torch.nn.Module):
    def __init__(self, k, history, aDim=6):
        super(ModelInputPypose, self).__init__()
        self.register_buffer("poses", pp.identity_SE3(k, history, dtype=tdtype))
        self.register_buffer("vels", torch.zeros(k, history, 6, dtype=tdtype))
        self.register_buffer("actions", torch.zeros(k, history, aDim, dtype=tdtype))
        self.k = k


    def init(self, input: ControllerInputPypose):
        self.poses, self.vels, self.actions[:, :-1] = input.get(self.k)

    '''
        Init the model input from data immidiatly.

        inputs:
        -------
            - poses: pypose.SE3. Poses of the vehicle.
                shape [k, history, 7]
            - vels: torch.tensor. Velocities of the vehicle.
                shape [k, history, 6]
            - actions: torch.tensor. Actions applied on the vehicle.

    '''
    def init_from_states(self, poses, vels, actions):
        self.poses = poses
        self.vels = vels
        self.actions[:, :-1] = actions


    def forward(self, action: torch.tensor):
        self.actions[:, -1] = action
        return self.poses.clone(), self.vels.clone(), self.actions.clone()


    def update(self, pose, vel):
        # add the new state and action to the buffers at the location of
        # the oldest element.

        # TODO: FIGURE OUT WHAT IS HAPPENING HERE 
        self.poses = self.poses.clone()
        self.vels = self.vels.clone()
        self.poses[:, 0:1] = pose
        self.vels[:, 0:1] = vel
        # roll the tensor to the "left" by one step.
        # TODO This is ugly as we need to call a constructor at every step. need to figure
        # out if we can create a pypose.roll method
        self.poses = pp.SE3(torch.roll(self.poses, -1, 1))
        self.vels = torch.roll(self.vels, -1, 1)
        self.actions = torch.roll(self.actions, -1, 1)