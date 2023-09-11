import torch
import pypose as pp
from scripts.utils.utils import tdtype


'''
    ControllerInput Class. This class maintains the information about the
    previous states and actions applied to the "real" system.
    The class keeps "steps" state and "steps-1" actions.
'''
class ControllerInput(torch.nn.Module):
    '''
        Contructor

        inputs:
        -------
            - steps: the number of past states and acitons to maintain.
            - pDim: int, the pose dimension.
            - vDim: int, the velocity dimension.
            - aDim: int, the action dimension.
    '''
    def __init__(self, steps, pDim=7, vDim=6, aDim=6):
        super(ControllerInput, self).__init__()
        self.register_buffer("poses", torch.zeros(steps, pDim, dtype=tdtype))
        self.register_buffer("vels", torch.zeros(steps, vDim, dtype=tdtype))
        self.register_buffer("actions", torch.zeros(steps-1, aDim, dtype=tdtype))

        self.steps = steps
        self.cur = [0, 0]
        self.pDim = pDim
        self.vDim = vDim
        self.aDim = aDim

    '''
        Add state method. This adds a state (pose and velocity) to the history buffer.

        inputs:
        -------
            - pose: torch.tensor with shape (pDim) the pose to be saved.
            - vel: torch.tensor with shape (vDim) the velocity to be saved.
    '''
    def add_state(self, pose: torch.tensor, vel: torch.tensor):
        self.poses[0] = pose
        self.poses = torch.roll(self.poses, -1, 0)

        self.vels[0] = vel
        self.vels = torch.roll(self.vels, -1, 0)

        if not self.is_init:
            self.cur[0] += 1

    '''
        Add act methods. Add an action to the action history buffer.

        inputs:
        -------
            - action: torch.tensor with shape (aDim, 1) the action to be saved.
    '''
    def add_act(self, action: torch.tensor):
        # Do nothing when we only need the last state
        if self.steps - 1 == 0:
            return

        self.actions[0] = action
        self.actions = torch.roll(self.actions, -1, 0)

        if not self.is_init:
            self.cur[1] += 1

    '''
        Add state and action to the state and action history buffer.

        inputs:
        -------
            - state: torch.tensor with shape (sDim, 1) the state to be saved.
            - action: torch.tensor with shape (aDim, 1) the action to be saved.
    '''
    def add(self, pose: torch.tensor, vel: torch.tensor, action: torch.tensor):
        # add the new state and action to the buffers at the location of
        # the oldest element.
        self.add_state(pose, vel)
        self.add_act(action)

    '''
        Getter for the history buffers. It also broadcasts the bufferes to match
        the number of samples k.

        inputs:
        -------
            - k: the number of samples.

        outputs:
        --------
            - poses, the poses history buffer with shape (k, steps, pDim)
            - vels, the velocities history buffer with shape (k, steps, vDim)
            - actions, the action history buffer with shape (k, steps-1, aDim)
    '''
    def get(self, k):
        return torch.broadcast_to(self.poses.clone(), (k, self.steps, self.pDim)), \
               torch.broadcast_to(self.vels.clone(), (k, self.steps, self.vDim)), \
               torch.broadcast_to(self.actions.clone(), (k, self.steps-1, self.aDim))

    '''
        Returns the number of steps to maintain.

        outputs:
        --------
            - steps, the number of steps.
    '''
    def get_steps(self):
        return self.steps

    '''
        Returns True if the bufferes have been filled.
    '''
    @property
    def is_init(self):
        if self.cur[0] == self.steps and self.cur[1] == self.steps-1:
            return True
        return False
    

'''
    ControllerInputPypose Class. This class maintains the information about the
    previous pose, velocity and actions applied to the "real" system.
    The class keeps "steps" poses and vels and "steps-1" actions.
'''
class ControllerInputPypose(torch.nn.Module):
    '''
        Contructor

        inputs:
        -------
            - steps: the number of past states and acitons to maintain.
            - sDim: int, the state dimension.
            - aDim: int, the action dimension.
    '''
    def __init__(self, steps, aDim=6):
        super(ControllerInputPypose, self).__init__()
        self.register_buffer("poses", pp.identity_SE3(steps, dtype=tdtype))
        self.register_buffer("vels", torch.zeros(steps, 6, dtype=tdtype))
        self.register_buffer("actions", torch.zeros(steps-1, aDim, dtype=tdtype))

        self.steps = steps
        self.cur = [0, 0]
        self.aDim = aDim

    '''
        Add state method. This adds a state to the state history buffer.

        inputs:
        -------
            - state: torch.tensor with shape (sDim, 1) the state to be saved.
    '''
    def add_state(self, state: torch.tensor):
        self.poses[0] = pp.SE3(state[:7])
        self.vels[0] = state[7:]
        self.poses = torch.roll(self.poses, -1, 0)
        self.vels = torch.roll(self.vels, -1, 0)

        if not self.is_init:
            self.cur[0] += 1

    '''
        Add act methods. Add an action to the action history buffer.

        inputs:
        -------
            - action: torch.tensor with shape (aDim, 1) the action to be saved.
    '''
    def add_act(self, action: torch.tensor):
        # Do nothing when we only need the last state
        if self.steps - 1 == 0:
            return

        self.actions[0] = action
        self.actions = torch.roll(self.actions, -1, 0)

        if not self.is_init:
            self.cur[1] += 1

    '''
        Add state and action to the state and action history buffer.

        inputs:
        -------
            - state: torch.tensor with shape (sDim, 1) the state to be saved.
            - action: torch.tensor with shape (aDim, 1) the action to be saved.
    '''
    def add(self, state: torch.tensor, action: torch.tensor):
        # add the new state and action to the buffers at the location of
        # the oldest element.
        self.add_state(state)
        self.add_act(action)

    '''
        Getter for the history buffers. It also broadcasts the bufferes to match
        the number of samples k.

        inputs:
        -------
            - k: the number of samples.

        outputs:
        --------
            - poses, the pose history buffer with shape [k, steps, pDim]
            - vels, the velocity history buffer with shape [k, steps, vDim]
            - actions, the action history buffer with shape [k, steps-1, aDim]
    '''
    def get(self, k):
        # Broadcasting through multiplication as pypose doesn't have a method for broadcasting
        # and torch.broadcast_to returns a tensor
        i = pp.identity_SE3(k, self.steps, dtype=tdtype)
        return i + self.poses.clone(), \
               torch.broadcast_to(self.vels.clone(), (k, self.steps, 6)), \
               torch.broadcast_to(self.actions.clone(), (k, self.steps-1, self.aDim))

    '''
        Returns the number of steps to maintain.

        outputs:
        --------
            - steps, the number of steps.
    '''
    def get_steps(self):
        return self.steps

    '''
        Returns True if the bufferes have been filled.
    '''
    @property
    def is_init(self):
        if self.cur[0] == self.steps and self.cur[1] == self.steps-1:
            return True
        return False