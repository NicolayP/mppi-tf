from cpprb import ReplayBuffer
import tensorflow as tf

class LearnerBase(tf.Module):
    def __init__(self, model, filename=None, bufferSize=264, numEpochs=100, batchSize=30):
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

    def load_rb(self, filename):
        self.rb.load_transitions(filename)

    def add_rb(self, x, u, xNext):
        self.rb.add(obs=x, act=u, next_obs=xNext)

    def train_epoch(self):
        for e in range(self.numEpochs):
            sample = self.rb.sample(self.batchSize)
            self.model.train_step(sample["next_obs"],
                                  sample["obs"],
                                  sample["act"])
        pass

    def get_batch(self, batchSize):
        return self.rb.sample(batchSize)

    def rb_trans(self):
        return self.rb.get_all_transitions().copy()
