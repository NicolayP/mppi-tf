from cpprb import ReplayBuffer
from mppi_tf.scripts.simulation import Simulation
import numpy as np
from mppi_tf.scripts.nn_model import NNModel
import tensorflow as tf
from datetime import datetime

def main():
    s_dim = 4
    a_dim = 2
    batch_size = 64

    env = "../envs/point_mass2d.xml"
    sim = Simulation(env, s_dim, a_dim, None, False)

    length = 500
    rb = ReplayBuffer(length,
                      env_dict={"obs": {"shape": (s_dim, 1)},
                      "act": {"shape": (a_dim, 1)},
                      "rew": {},
                      "next_obs": {"shape": (s_dim, 1)},
                      "done": {}})

    x = sim.getState()
    for _ in range(length):
        u = np.random.rand(1, a_dim, 1)
        x_next = sim.step(u)

        rb.add(obs=x, act=u, rew=0, next_obs=x_next, done=False)
        x = x_next


    model = NNModel(dt=0.1, state_dim=s_dim, action_dim=a_dim, name="nn_model")

    stamp = datetime.now().strftime("%Y.%m.%d-%H:%M:%S")
    logdir = "../graphs/test_training/{}".format(stamp)

    writer = tf.summary.create_file_writer(logdir)
    log = True

    epochs = 1000
    for e in range(epochs):
        sample = rb.sample(batch_size)
        gt = sample['next_obs']
        x = sample['obs']
        u = sample['act']
        model.train_step(gt, x, u, e, writer, log)


if __name__ == "__main__":
    main()
