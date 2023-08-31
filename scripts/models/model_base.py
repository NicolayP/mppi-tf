import torch
from scripts.inputs.ModelInput import ModelInput
from scripts.utils.utils import load_param


'''
    Abstract class for every Model that can run with MPPI.
'''
class ModelBase(torch.nn.Module):
    '''
        Constructor.

        inputs:
        -------
            - dt, the timestep inbetween two integration steps.
            - config, a dictionnary containing model relative parameters.
            - yaml_config, contains the same information than the config.
                and can be used instead of the config dictionnary.
    '''
    def __init__(self, dt=0.1, config=None, yaml_config=None):
        super(ModelBase, self).__init__()
        self.dt = dt

        if yaml_config is not None and config is None:
            self.config = load_param(yaml_config)
        elif yaml_config is None and config is not None:
            self.config = config
        else:
            self.config = {}
            self.config['name'] = 'default'
        self.name = self.config["name"]


    '''
        Forward function for the model. Performs a single step integration.

        inputs:
        -------
            - input: ModelInput.
    '''
    def forward(self, x, u: torch.tensor) -> torch.tensor:
        raise NotImplementedError

    '''
        Initalize the model parameters from the configuration.
    '''
    def init_param(self):
        raise NotImplementedError

    '''
        Used to reset internal parameters of the model
        (i.e internal state of a RNN)
    '''
    def reset(self):
        raise NotImplementedError