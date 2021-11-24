import tensorflow as tf
from datetime import datetime
import os
from shutil import copyfile
from tensorflow.python.ops import summary_ops_v2

class ObserverBase(tf.Module):

    def __init__(self,
                 logPath,
                 log,
                 debug,
                 k,
                 tau,
                 lam,
                 configFile,
                 taskFile,
                 aDim,
                 sDim,
                 modelName):

        self._aDim = aDim
        self._sDim = sDim
        self._writer = None
        self._k = k

        if log or debug:
            stamp = datetime.now().strftime("%Y.%m.%d-%H:%M:%S")
            path = 'graphs/python/'
            if debug:
                path = os.path.join(path, 'debug')
            logdir = os.path.join(logPath,
                                  path,
                                  modelName,
                                  "k" + str(k.numpy()),
                                  "T" + str(tau),
                                  "L" + str(lam),
                                  stamp)

            print(logdir)
            os.makedirs(logdir)

            self._writer = tf.summary.create_file_writer(logdir)

            self._summary_name = ["x", "y", "z"]

            if configFile is not None:
                conf_dest = os.path.join(logdir, "config.yaml")
                task_dest = os.path.join(logdir, "task.yaml")
                copyfile(configFile, conf_dest)
                copyfile(taskFile, task_dest)

            self.predNextState = None       #
            self.predState = None           #
            self.predError = None           #
            self.predDist = None            #
            self.predCost = None            #
            self.predSampleCost = None      #
            self.update = None              # DONE
            self.next = None                #
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
            self.step = 0

            with tf.name_scope("summary") as self.scope:
                pass

    def save_graph(self, function, graphMode=True):
        state = tf.zeros((self._sDim, 1), dtype=tf.float64)
        seq = tf.zeros((5, self._aDim, 1), dtype=tf.float64)
        with self._writer.as_default():
            if graphMode:
                graph = function.get_concrete_function(1, state, seq).graph
            else:
                graph = tf.function(function).get_concrete_function(1, state, seq).graph
            # visualize
            summary_ops_v2.graph(graph.as_graph_def())

    def advance(self):
        self.step += 1

    def write_control(self, name, tensor):
        with tf.name_scope(self.scope):
            if name == "update":
                tf.summary.scalar("Controller/update",
                                tensor,
                                step=self.step)

            elif name == "next":
                action = tensor[0]
                for i in range(self._aDim):
                    tf.summary.scalar("input/input_{}".format(i),
                                    action[i, 0],
                                    step=self.step)

            elif name == "sample_costs":
                self.best_id = tf.squeeze(tf.argmax(tensor, axis=0))
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

    def write_predict(self, name, tensor):
        with tf.name_scope(self.scope):
            if name == "predicted/next_state":
                tf.summary.scalar("Predictied/Next_state",
                                tf.squeeze(tensor),
                                step=self.step)

            elif name == "predicted/state":
                tf.summary.scalar("Predicted/State",
                                tf.squeeze(tensor),
                                step=self.step)

            elif name == "predicted/error":
                tf.summary.scalar("Predicted/error",
                                tf.squeeze(tensor),
                                step=self.step)

            elif name == "predicted/dist":
                tf.summary.scalar("Predicted/goal_distance",
                                tf.squeeze(tensor),
                                step=self.step)

            elif name == "predicted/cost":
                tf.summary.scalar("Predicted/cost",
                                tf.squeeze(tensor),
                                step=self.step)

            elif name == "predicted/sample_cost":
                tf.summary.scalar("Predicted/Sample_cost",
                                tf.squeeze(tensor),
                                step=self.step)
