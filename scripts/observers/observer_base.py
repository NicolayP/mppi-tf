import torch
from torch.utils.tensorboard import SummaryWriter
import yaml
from datetime import datetime
import os


class ObserverBase(object):
    def __init__(self,
                 log=True, logpath=None, k=1,
                 sName=["x", "y", "z", "qw", "qx", "qy", "qz", "u", "v", "w", "p", "q", "r"],
                 aName=["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"],
                 angId=3,
                 angFormat="quat",
                 configDict=None,
                 taskDict=None,
                 modelDict=None):
        '''
            observer for the controller.
            input:
            ------
                - log: bool, weather or not to activate the observer.
                - logpath: string, the path where to log the data.
                - sName: list, the names of the different states.
                - aName: list, the names of the inputs.
                - posId: list, the indicies of the positions,
                - angId: list, the indicies of the angles, can be empty.
                - angFormat: string, the angular representation used. 
                    "quat", quaternion represntation/
                    "euler", euler representation XYZ.
                    "rot", rotatianl representation.
                - linVelId: list, the indicies of the linear velocities.
                - angVelId: list, the indicies of the angular velocities.
        '''
        self.log = log
        self.k = k
        self.step=0 # log step

        self.sDim, self.sName = len(sName), sName # state dimension and Names.
        self.aDim, self.aName = len(aName), aName # Action dimension and Name.

        #self.posId, self.angId = posId, angId # Pose id in the pose vector and angular id.
        self.andId = angId # inital index of the angle
        self.angFormat = angFormat # format used for angles to convert to eular angles.
        #self.linVelId, self.anvVelId = linVelId, angVelId # Velocites id and the angular velocity.

        if log:
            stamp = datetime.now().strftime("%Y.%m.%d-%H:%M:%S")
            self.logdir = os.path.join(logpath, stamp, "controller")
            os.makedirs(self.logdir)
            self.writer = SummaryWriter(self.logdir)

            if configDict is not None:
                confDest = os.path.join(self.logdir, "config.yaml")
                with open(confDest, "w") as stream:
                    yaml.dump(configDict, stream)

            if taskDict is not None:
                taskDest = os.path.join(self.logdir, "task.yaml")
                with open(taskDest, "w") as stream:
                    yaml.dump(taskDict, stream)

            if modelDict is not None:
                modelDest = os.path.join(self.logdir, "model.yaml")
                with open(modelDest, "w") as stream:
                    yaml.dump(modelDict, stream)

    def advance(self):
        if not self.log:
            return
        self.step += 1

    def save_graph(self, function):
        data = torch.zeros(self.sDim)
        oldk = function.k
        function.k = 1
        self.writer.add_graph(function, input_to_model=data, verbose=False)
        function.k = oldk

    def write_control(self, name, tensor):
        if not self.log:
            return

        if name == "nabla":
            self.writer.add_scalar("Controller/Nabla_percent",
                                   tensor/self.k, self.step)

        elif name == "state":
            # TODO: ANGLE FORMAT
            #if self.angFormat is not None or self.angFormat == "euler":
            #    if self.angFormat == "quat":
            #        angs = tensor[self.angId:self.angId+4]
            #    elif self.angFormat == "rot":
            #        angs = tensor[self.angId:self.angId+9]

            for i, n in enumerate(self.sName):
                self.writer.add_scalar(f"State/{n}",
                                       torch.squeeze(tensor[i]), self.step)

        elif name == "action":
            for i, n in enumerate(self.aName):
                self.writer.add_scalar(f"Action/{n}",
                                       torch.squeeze(tensor[i]), self.step)

        elif name == "sample_cost":
            self.writer.add_histogram("Cost/sample_cost",
                                      tensor, self.step)
            self.writer.add_scalar("Cost/mean_cost",
                                   torch.mean(tensor), self.step)
        
        elif name == "sample_weight":
            self.writer.add_histogram("Cost/samples_weights",
                                      tensor, self.step)

    def write_predict(self, name, tensor):
        pass