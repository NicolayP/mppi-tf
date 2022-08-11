from dataclasses import dataclass
from distutils.log import error
import tensorflow as tf
from datetime import datetime
import os
from shutil import copyfile
from tensorflow.python.ops import summary_ops_v2
import yaml

class ObserverBase(tf.Module):

    def __init__(self,
                 logPath,
                 log,
                 debug,
                 k,
                 tau,
                 lam,
                 configDict,
                 taskDict,
                 modelDict,
                 aDim,
                 sDim,
                 modelName):

        self._aDim = aDim
        self._sDim = sDim
        self._writer = None
        self._k = k
        self._log = log
        self._stateName = ["x", "y", "z",
                            "theta",
                            "u", "v", "w",
                            "p", "q", "r"]
        self._poseId = [0, 1, 2, 3, 4, 5, 6]
        self._velId = [7, 8, 9, 10, 11, 12]
        self._tau = tau

        if log or debug:
            self.logdir = os.path.join(logPath, "controller")
            os.makedirs(self.logdir)
            self._writer = tf.summary.create_file_writer(self.logdir)

            self._summary_name = ["x", "y", "z"]

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

            self.predNextState = None       # DONE
            self.predState = None           # DONE
            self.predError = None           # DONE
            self.predDist = None            # DONE
            self.predCost = None            # DONE
            self.predSampleCost = None      # DONE
            self.update = None              # DONE
            self.next = None                # DONE
            self.actionSeq = None           #
            self.path = None                #
            self.applied = None             #
            self.sampleCosts = None         # DONE
            self.stateCost = None           # DONE
            self.actionCost = None          # DONE
            self.weights = None             # DONE
            self.nabla = None               # DONE
            self.arg = None                 # DONE
            self.weightedNoise = None       # DONE
            self.step = tf.Variable(0, dtype=tf.int64)

    def get_logdir(self):
        return self.logdir

    def save_graph(self, function, graphMode=True):
        state = tf.zeros((self._sDim, 1), dtype=tf.float64)
        seq = tf.zeros((self._tau, self._aDim, 1), dtype=tf.float64)
        goal = tf.zeros((self._sDim, 1), dtype=tf.float64)
        with self._writer.as_default():
            if graphMode:
                graph = function.get_concrete_function(1, state, seq, goal).graph
            else:
                graph = tf.function(function).get_concrete_function(1, state, seq, goal).graph
            # visualize
            summary_ops_v2.graph(graph.as_graph_def())

    def advance(self):
        if not self._log:
            return
        self._writer.flush()
        self.step.assign_add(1)

    def write_control(self, name, tensor):
        if not self._log:
            return

        with self._writer.as_default():
            if name == "update":
                tf.summary.scalar("Controller/update",
                                tensor,
                                step=self.step)

            elif name == "noises":
                for i in range(6):
                    tf.summary.histogram(f"Controller/noises_{i}",
                                         tensor[:, :, i],
                                         step=self.step)

            elif name == "next":
                action = tensor[0]
                for i in range(self._aDim):
                    tf.summary.scalar("input/{}".format(i),
                                    action[i, 0],
                                    step=self.step)

            elif name == "sample_costs":
                self.best_id = tf.squeeze(tf.argmin(tensor, axis=0))
                tf.summary.histogram("Cost/All/All_cost",
                                    tensor,
                                    step=self.step)
                tf.summary.scalar("Cost/All/Average_cost",
                                tf.math.reduce_mean(tensor),
                                step=self.step)
                tf.summary.scalar("Cost/All/Best_cost",
                                tf.squeeze(tensor[self.best_id, :]),
                                step=self.step)

            elif name == "state_cost":
                tf.summary.histogram("Cost/State/Cost",
                                tensor,
                                step=self.step)
                tf.summary.scalar("Cost/State/Average_cost",
                                tf.math.reduce_mean(tensor),
                                step=self.step)
                tf.summary.scalar("Cost/State/Best_cost",
                                tf.squeeze(tensor[self.best_id, :]),
                                step=self.step)

            elif name == "action_cos":
                tf.summary.histogram("Cost/Action/Cost",
                                tensor,
                                step=self.step)
                tf.summary.scalar("Cost/Action/Average_cost",
                                tf.math.reduce_mean(tensor),
                                step=self.step)
                tf.summary.scalar("Cost/Action/Best_cost",
                                tf.squeeze(tensor[self.best_id, :]),
                                step=self.step)

            elif name == "weights":
                tf.summary.histogram("Controller/Weights",
                                    tensor,
                                    step=self.step)

            elif name == "nabla":
                tf.summary.scalar("Controller/Nabla_percent",
                                tf.squeeze(tensor/tf.cast(self._k,
                                                            dtype=tf.float64)),
                                step=self.step)

            elif name == "arg":
                tf.summary.histogram("Controller/exponenetial_argument",
                                tensor,
                                step=self.step)

            elif name == "weighted_noise":
                tf.summary.histogram("Controller/Weighted_noises",
                                    tensor,
                                    step=self.step)

            elif name == "state":
                for i in self._poseId:
                    tf.summary.scalar("State/position_{}".format(i),
                                      tf.squeeze(tensor[i, :]),
                                      step=self.step)
                for i in self._velId:
                    tf.summary.scalar("State/velocity_{}".format(i),
                                      tf.squeeze(tensor[i, :]),
                                      step=self.step)

    def write_predict(self, name, tensor):
        if not self._log:
            return

        with self._writer.as_default():
            if name == "predicted/next_state":
                for i in range(self._sDim):
                    tf.summary.scalar("Predicted/Next_state_{}".format(self._stateName[i]),
                                    tf.squeeze(tensor[i, :]),
                                    step=self.step)

            elif name == "predicted/state":
                for i in range(self._sDim):
                    tf.summary.scalar("Predicted/State_{}".format(self._stateName[i]),
                                        tf.squeeze(tensor[i, :]),
                                        step=self.step)

            elif name == "predicted/step_cost":
                tf.summary.scalar(f"Predicted/Step_cost",
                                    tf.squeeze(tensor),
                                    step=self.step)

            elif name == "predicted/error":
                error_pos, error_rot, error_vel, error_vel_dec = tensor
                tf.summary.scalar("Predicted/error_position",
                                tf.squeeze(error_pos),
                                step=self.step)

                tf.summary.scalar("Predicted/error_orientation",
                                tf.squeeze(error_rot),
                                step=self.step)

                tf.summary.scalar("Predicted/error_velocities",
                                tf.squeeze(error_vel),
                                step=self.step)
                
                for i in range(6):
                    tf.summary.scalar(f"Predicted/error_vel_{i}",
                                      error_vel_dec[i],
                                      step=self.step)

            elif name == "predicted/dist":
                for i in range(len(self._stateName)):
                    tf.summary.scalar("Predicted/goal_distance_{}".format(self._stateName[i]),
                                    tf.squeeze(tensor[i, :]),
                                    step=self.step)

            elif name == "predicted/sample_cost":
                tf.summary.histogram("Predicted/Sample_cost",
                                    tf.squeeze(tensor),
                                    step=self.step)
