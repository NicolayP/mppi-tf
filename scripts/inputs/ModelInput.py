import torch
from scripts.utils.utils import tdtype
from scripts.inputs.ControllerInput import ControllerInput

class ModelInput(torch.nn.Module):
    def __init__(self, k, steps, sDim=13, aDim=6):
        super(ModelInput, self).__init__()
        self.register_buffer("states", torch.zeros(k, steps, sDim, 1, dtype=tdtype))
        self.register_buffer("actions", torch.zeros(k, steps, aDim, 1, dtype=tdtype))
        self.k = k

    def init(self, input: ControllerInput):
        self.states, self.actions[:, :-1] = input.get(self.k)

    def forward(self, action: torch.tensor):
        self.actions[:, -1] = action
        return self.states.clone(), self.actions.clone()

    def update(self, state: torch.tensor, action: torch.tensor):
        # add the new state and action to the buffers at the location of
        # the oldest element.

        # TODO: FIGURE OUT WHAT IS HAPPENING HERE 
        self.states = self.states.clone()

        self.states[:, 0] = state
        self.actions[:, 0] = action
        # roll the tensor to the "left" by one step.
        self.states = torch.roll(self.states, -1, 1)
        self.actions = torch.roll(self.actions, -1, 1)