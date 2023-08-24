import torch
from scripts.utils.utils import tdtype

class ControllerInput(torch.nn.Module):
    def __init__(self, steps, sDim=13, aDim=6):
        super(ControllerInput, self).__init__()
        self.register_buffer("states", torch.zeros(steps, sDim, 1, dtype=tdtype))
        self.register_buffer("actions", torch.zeros(steps-1, aDim, 1, dtype=tdtype))

        self.steps = steps
        self.cur = [0, 0]
        self.sDim = sDim
        self.aDim = aDim

    def add_state(self, state):
        self.states[0] = state
        self.states = torch.roll(self.states, -1, 0)

        if not self.is_init:
            self.cur[0] += 1

    def add_act(self, action):
        # Do nothing when we only need the last state
        if self.steps - 1 == 0:
            return

        self.actions[0] = action
        self.actions = torch.roll(self.actions, -1, 0)

        if not self.is_init:
            self.cur[1] += 1

    def add(self, state, action):
        # add the new state and action to the buffers at the location of
        # the oldest element.
        self.add_state(state)
        self.add_act(action)

    def get(self, k):
        return torch.broadcast_to(self.states.clone()[None], (k, self.steps, self.sDim, 1)), \
               torch.broadcast_to(self.actions.clone()[None], (k, self.steps-1, self.aDim, 1))

    def get_steps(self):
        return self.steps

    @property
    def is_init(self):
        if self.cur[0] == self.steps and self.cur[1] == self.steps-1:
            return True
        return False