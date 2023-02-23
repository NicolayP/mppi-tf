from dataclasses import dataclass
import tensorflow as tf
import tensorflow_graphics as tfg
from datetime import datetime
import os
from shutil import copyfile
from tensorflow.python.ops import summary_ops_v2
import yaml

from ..misc.utile import assert_shape, dtype


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

        self._poseId = [0, 1, 2, 3, 4, 5]
        self._velId = [6, 7, 8, 9, 10, 11]
        self._tau = tau

        self._aName = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]
        self._sName = ["x", "y", "z", "roll", "pitch", "yaw", "u", "v", "w", "p", "q", "r"]


        if log or debug:
            self.logdir = os.path.join(logPath, "controller")
            if not os.path.exists(self.logdir):
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
        state = tf.zeros((self._sDim, 1), dtype=dtype)
        seq = tf.zeros((self._tau, self._aDim, 1), dtype=dtype)
        with self._writer.as_default():
            if graphMode:
                graph = function.get_concrete_function(1, state, seq).graph
            else:
                graph = tf.function(function).get_concrete_function(1, state, seq).graph
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
                print("Update: ", tensor.shape)
                for i in range(self._aDim):
                    mean = tf.math.reduce_mean(tensor[:, i])
                    tf.summary.scalar("Controller/update_mean/{}".format(i),
                        mean, step=self.step)
                    tf.summary.histogram("Controller/update/{}".format(i),
                        tensor[:, i], step=self.step)

            elif name == "actionSeq":
                for i in range(self._aDim):
                    mean = tf.math.reduce_mean(tensor[:, i])
                    tf.summary.scalar("action_seq_mean/{}".format(i),
                        mean, step=self.step)
                    tf.summary.histogram("action_seq/{}".format(i),
                        tensor[:, i], step=self.step)

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
                                                            dtype=dtype)),
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
                tensor_euler = self.to_euler(tensor)
                for i, n in enumerate(self._sName):
                    tf.summary.scalar(f"Predicted/Next_state_{n}",
                                    tf.squeeze(tensor_euler[i, :]),
                                    step=self.step)

            elif name == "predicted/state":
                tensor_euler = self.to_euler(tensor)
                for i, n in enumerate(self._sName):
                    tf.summary.scalar(f"Predicted/State_{n}",
                                        tf.squeeze(tensor_euler[i, :]),
                                        step=self.step)

            elif name == "predicted/error":
                tf.summary.scalar("Predicted/error",
                                tf.squeeze(tensor),
                                step=self.step)

            elif name == "predicted/dist_to_goal":
                for i in range(tf.shape(tensor)[0]):
                    tf.summary.scalar(f"Predicted/goal_distance_{i}",
                                    tf.squeeze(tensor[i, :]),
                                    step=self.step)

            elif name == "predicted/sample_cost":
                tf.summary.histogram("Predicted/Sample_cost",
                                    tf.squeeze(tensor),
                                    step=self.step)

    def to_euler(self, state):
        raise NotImplementedError


class ObserverLagged(ObserverBase):
    def __init__(
        self,
        logPath, log, debug,
        k, tau, lam,
        configDict, taskDict, modelDict,
        h, aDim, sDim, modelName):
        super().__init__(
            logPath, log, debug,
            k, tau, lam,
            configDict, taskDict, modelDict,
            aDim, sDim, modelName)
        self._h = h

    def save_graph(self, function, graphMode=True):
        if self._h > 1:
            state = (
                tf.zeros((self._h, self._sDim, 1), dtype=dtype), 
                tf.zeros((self._h-1, self._aDim, 1), dtype=dtype)
            )
        else:
            state = (
                tf.zeros((self._h, self._sDim, 1), dtype=dtype), 
                None
            )
        seq = tf.zeros((self._tau, self._aDim, 1), dtype=dtype)
        with self._writer.as_default():
            if graphMode:
                graph = function.get_concrete_function(tf.Variable(1, dtype=tf.int32), state, seq).graph
            else:
                graph = tf.function(function).get_concrete_function(tf.Variable(1, dtype=tf.int32), state, seq).graph
            # visualize
            summary_ops_v2.graph(graph.as_graph_def())

    def write_control(self, name, tensor):
        if not self._log:
            return

        with self._writer.as_default():
            if name == "update":
                for i, n in enumerate(self._aName):
                    mean = tf.math.reduce_mean(tensor[:, i])
                    tf.summary.scalar(f"Controller/update_mean/{n}",
                        mean, step=self.step)
                    tf.summary.histogram(f"Controller/update/{n}",
                        tensor[:, i], step=self.step)

            elif name == "actionSeq":
                for i, n in enumerate(self._aName):
                    mean = tf.math.reduce_mean(tensor[:, i])
                    tf.summary.scalar(f"action_seq_mean/{n}",
                        mean, step=self.step)
                    tf.summary.histogram(f"action_seq/{n}",
                        tensor[:, i], step=self.step)

            elif name == "next":
                action = tensor[0]
                for i, n in enumerate(self._aName):
                    tf.summary.scalar(f"input/{n}",
                                    action[i, 0],
                                    step=self.step)

            elif name == "samples_cost":
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

            elif name == "states_cost":
                tf.summary.histogram("Cost/State/Cost",
                                tensor,
                                step=self.step)
                tf.summary.scalar("Cost/State/Average_cost",
                                tf.math.reduce_mean(tensor),
                                step=self.step)
                tf.summary.scalar("Cost/State/Best_cost",
                                tf.squeeze(tensor[self.best_id, :]),
                                step=self.step)

            elif name == "actions_cost":
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
                                                            dtype=dtype)),
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
                tensor_euler = self.to_euler(tensor[0])

                for i in self._poseId:
                    n = self._sName[i]
                    tf.summary.scalar(f"State/pose_{n}",
                                      tf.squeeze(tensor_euler[-1, i, :]),
                                      step=self.step)
                for i in self._velId:
                    n = self._sName[i]
                    tf.summary.scalar(f"State/vel_{n}",
                                      tf.squeeze(tensor_euler[-1, i, :]),
                                      step=self.step)
            
            elif name == "goal":
                pass

    def to_euler(self, state):
        raise NotImplemented


class ObserverLaggedQuat(ObserverLagged):
    def __init__(
        self,
        logPath, log, debug,
        k, tau, lam,
        configDict, taskDict, modelDict,
        h, aDim, sDim, modelName):
        super().__init__(
            logPath=logPath,
            log=log,
            debug=debug,
            k=k, tau=tau, lam=lam,
            configDict=configDict,
            taskDict=taskDict,
            modelDict=modelDict,
            h=h,
            aDim=aDim,
            sDim=sDim,
            modelName=modelName)

    def to_euler(self, state):
        position = state[..., 0:3, :]
        quat = state[..., 3:7, :]
        vel = state[..., 7:, :]

        euler = tf.expand_dims(
            tfg.geometry.transformation.euler.from_quaternion(quat[..., 0]),
            axis=-1
        )
        return tf.concat([position, euler, vel], axis=-2)
