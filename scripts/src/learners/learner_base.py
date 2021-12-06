from cpprb import ReplayBuffer
import tensorflow as tf
from tensorflow.python.ops import summary_ops_v2
from datetime import datetime
import os

class LearnerBase(tf.Module):
    def __init__(self,
                 model,
                 filename=None,
                 bufferSize=264,
                 numEpochs=100,
                 batchSize=30,
                 log=False,
                 logPath=None):
        self.model = model
        self.sDim = model.get_state_dim()
        self.aDim = model.get_action_dim()
        self.rb = ReplayBuffer(bufferSize,
                               env_dict={"obs": {"shape": (self.sDim, 1)},
                               "act": {"shape": (self.aDim, 1)},
                               "next_obs": {"shape": (self.sDim, 1)}
                               }
                              )
        self.numEpochs = numEpochs
        self.batchSize = batchSize

        if filename is not None:
            self.load_rb(filename)

        self.log = log
        self.step = 0

        if self.log:
            self.logdir = os.path.join(logPath, "learner")
            self.writer = tf.summary.create_file_writer(self.logdir)
            self.save_graph()

    def load_rb(self, filename):
        self.rb.load_transitions(filename)

    def add_rb(self, x, u, xNext):
        self.rb.add(obs=x, act=u, next_obs=xNext)

    def train_epoch(self):
        for e in range(self.numEpochs):
            samples = self.rb.sample(self.batchSize)
            batchLoss = self.model.train_step(samples["next_obs"],
                                              samples["obs"],
                                              samples["act"])
            if self.log:
                with self.writer.as_default():
                    tf.summary.scalar("Loss", batchLoss, self.step)
                    self.step += 1

    def get_batch(self, batchSize):
        return self.rb.sample(batchSize)

    def rb_trans(self):
        return self.rb.get_all_transitions().copy()

    def save_graph(self):
        state = tf.zeros((1, self.model.get_state_dim(), 1), dtype=tf.float64)
        action = tf.zeros((1, self.model.get_action_dim(), 1), dtype=tf.float64)
        with self.writer.as_default():
            graph = tf.function(
                        self.model.build_step_graph
                    ).get_concrete_function(
                      "graph", state, action  
                    ).graph
            # visualize
            summary_ops_v2.graph(graph.as_graph_def())

    def save_transitions(self, filename):
        self.rb.save_transitions(filename)
 