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
            - sDim: int, the state dimension.
            - aDim: int, the action dimension.
    '''
    def __init__(self, steps, sDim=13, aDim=6):
        super(ControllerInput, self).__init__()
        self.register_buffer("states", torch.zeros(steps, sDim, dtype=tdtype))
        self.register_buffer("actions", torch.zeros(steps-1, aDim, dtype=tdtype))

        self.steps = steps
        self.cur = [0, 0]
        self.sDim = sDim
        self.aDim = aDim

    '''
        Add state method. This adds a state to the state history buffer.

        inputs:
        -------
            - state: torch.tensor with shape (sDim, 1) the state to be saved.
    '''
    def add_state(self, state: torch.tensor):
        self.states[0] = state
        self.states = torch.roll(self.states, -1, 0)

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
            - states, the state history buffer with shape (k, steps, sDim, 1)
            - actions, the action history buffer with shape (k, steps-1, aDim, 1)
    '''
    def get(self, k):
        return torch.broadcast_to(self.states.clone()[None], (k, self.steps, self.sDim)), \
               torch.broadcast_to(self.actions.clone()[None], (k, self.steps-1, self.aDim))

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
            - states, the state history buffer with shape (k, steps, sDim, 1)
            - actions, the action history buffer with shape (k, steps-1, aDim, 1)
    '''
    def get(self, k):
        # Broadcasting through multiplication as pypose doesn't have a method for broadcasting
        # and torch.broadcast_to returns a tensor
        i = pp.identity_SE3(k, self.steps, dtype=tdtype)
        return i + self.poses.clone()[None], \
               torch.broadcast_to(self.vels.clone()[None], (k, self.steps, 6)), \
               torch.broadcast_to(self.actions.clone()[None], (k, self.steps-1, self.aDim))

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