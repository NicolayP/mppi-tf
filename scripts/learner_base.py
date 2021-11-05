import numpy as np
from cpprb import ReplayBuffer
from nn_model import NNAUVModel

class Learner_base(object):
    def __init__(self, model, filename=None):
        self.model = model
        self.s_dim = model.get_state_dim()
        self.a_dim = model.get_action_dim()
        self.rb = ReplayBuffer(264,
                               env_dict={"obs": {"shape": (self.s_dim, 1)},
                               "act": {"shape": (self.a_dim, 1)},
                               "next_obs": {"shape": (self.s_dim, 1)}
                               }
                              )

        if filename is not None:
            self.load_rb(filename)

    def load_rb(self, filename):
        self.rb.load_transitions(filename)

    def add_rb(self, msg):
        x = msg.x
        u = msg.u
        x_next = msg.next
        self.rb.add(obs=x, act=u, next_obs=x_next)

    def train(self):
        
        pass

    def print_rb(self):
        print(self.rb.get_all_transitions())

def main():
    model = NNAUVModel()
    learner = Learner_base(model, "/home/pierre/workspace/uuv_ws/src/mppi-ros/log/transitons.npz")
    learner.print_rb()




if __name__ == "__main__":
    main()