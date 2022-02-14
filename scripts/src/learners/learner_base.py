from tabnanny import verbose
from cpprb import ReplayBuffer
from sklearn.model_selection import KFold
import tensorflow as tf
from tensorflow.python.ops import summary_ops_v2
from datetime import datetime
import os
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt


class LearnerBase(tf.Module):
    # PUBLIC
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
        self.optimizer = tf.optimizers.Adam(learning_rate=0.5)
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
            stamp = datetime.now().strftime("%Y.%m.%d-%H:%M:%S")
            self.logdir = os.path.join(logPath, "learner", stamp)
            self.writer = tf.summary.create_file_writer(self.logdir)
            self._save_graph()

    def load_rb(self, filename):
        self.rb.load_transitions(filename)

    def add_rb(self, x, u, xNext):
        self.rb.add(obs=x, act=u, next_obs=xNext)

    def train(self, X, y, batchSize=-1, epoch=1, learninRate=0.1, kfold=None):
        for e in range(epoch):
            if batchSize == -1:
                batchLoss = self._train_step(X,
                                             y)
                if self.log:
                    with self.writer.as_default():
                        if kfold is not None:
                            scope = "epoch{}/batch{}/lr{}/loss".format(epoch, batchSize, learninRate)
                        else:
                            scope = "Loss"
                        tf.summary.scalar(scope, batchLoss, self.step)
                        self.step += 1
                pass
            for i in range(0, X.shape[0], batchSize):
                batchLoss = self._train_step(X[i:i+batchSize],
                                             y[i:i+batchSize])
                if self.log:
                    with self.writer.as_default():
                        if kfold is not None:
                            scope = "epoch{}/batch{}/lr{}/loss_fold{}".format(epoch, batchSize, learninRate, kfold)
                        else:
                            scope = "Loss"
                        tf.summary.scalar(scope, batchLoss, self.step)
                        self.step += 1

    def rb_trans(self):
        return self.rb.get_all_transitions().copy()

    def save_rb(self, filename):
        self.rb.save_transitions(filename)
 
    def save_params(self, step):
        self.model.save_params(self.logdir, step)

    def grid_search(self, trajs, actionSeqs):
        init_weights = self.model.get_weights()
        
        learningRate = np.linspace(0.0001, 0.1, 10)
        batchSize = np.array([-1])
        epoch = np.array([100, 500, 1000])

        mean = []
        for lr in learningRate:
            for bs in batchSize:
                for e in epoch:
                    fold =self.k_fold_validation(learningRate=lr,
                                                 batchSize=bs,
                                                 epoch=e, k=10)
                    mean.append(np.mean(fold))
                    print("*"*5, " Grid ", 5*"*")
                    print("lr: ", lr)
                    print("bs: ", bs)
                    print("e: ", e)
                    print("fold: ", fold)
                    print("mean: ", np.mean(fold))

                    self.train_all(learningRate=lr, batchSize=bs, epoch=e)
                    err = self.validate(actionSeqs, trajs)
                    print("validation error: ", err.numpy())
                    self.model.update_weights(init_weights, msg=False)
        print("Best mean:", np.max(mean))

    def train_all(self, learningRate=0.1, batchSize=32, epoch=100):
        self.optimizer = tf.optimizers.Adam(learning_rate=learningRate)
        data = self.rb_trans()
        (X, y) = self.model.prepare_training_data(data['obs'],
                                                  data['next_obs'],
                                                  data['act'])
        self.step = 0
        self.train(X, y, batchSize=batchSize, epoch=epoch, learninRate=learningRate)

    def k_fold_validation(self, k=10, learningRate=0.1, batchSize=32, epoch=100):
        # First get all the data
        self.optimizer = tf.optimizers.Adam(learning_rate=learningRate)
        data = self.rb_trans()
        (X, y) = self.model.prepare_training_data(data['obs'],
                                                  data['next_obs'],
                                                  data['act'])
        kfold = KFold(n_splits=k, shuffle=True)

        init_weights = self.model.get_weights()
        fold = []
        X = X.numpy()
        y = y.numpy()
        i = 0
        for train, test in kfold.split(X, y):
            self.step = 0
            self.train(X[train], y[train], batchSize=batchSize, epoch=epoch, learninRate=learningRate, kfold=i)
            self.model.update_weights(init_weights, msg=False)
            lossFold = self.evaluate(X[test], y[test])
            fold.append(lossFold.numpy())
            i += 1

        self.model.update_weights(init_weights, msg=False)
        return fold

    def evaluate(self, X, y):
        pred = self.model._predict_nn("Eval", np.squeeze(X, axis=-1))
        loss = tf.reduce_mean(tf.math.squared_difference(pred, y),
                              name="loss")
        return loss

    def plot_seq(self, traj, gtTraj):
        fig, axs = plt.subplots(figsize=(20, 10),nrows=2, ncols=8)
        # Position
        axs[0, 0].plot(traj[:, 0])
        axs[0, 0].plot(gtTraj[:, 0])

        axs[0, 1].plot(traj[:, 1])
        axs[0, 1].plot(gtTraj[:, 1])

        axs[0, 2].plot(traj[:, 2])
        axs[0, 2].plot(gtTraj[:, 2])

        # Quaternion
        axs[0, 3].plot(traj[:, 3])
        axs[0, 3].plot(gtTraj[:, 3])

        axs[0, 4].plot(traj[:, 4])
        axs[0, 4].plot(gtTraj[:, 4])

        axs[0, 5].plot(traj[:, 5])
        axs[0, 5].plot(gtTraj[:, 5])

        axs[0, 6].plot(traj[:, 6])
        axs[0, 6].plot(gtTraj[:, 6])

        # Lin Vel
        axs[1, 0].plot(traj[:, 0])
        axs[1, 0].plot(gtTraj[:, 0])

        axs[1, 1].plot(traj[:, 1])
        axs[1, 1].plot(gtTraj[:, 1])

        axs[1, 2].plot(traj[:, 2])
        axs[1, 2].plot(gtTraj[:, 2])

        # Ang vel
        axs[1, 3].plot(traj[:, 3])
        axs[1, 3].plot(gtTraj[:, 3])

        axs[1, 4].plot(traj[:, 4])
        axs[1, 4].plot(gtTraj[:, 4])

        axs[1, 5].plot(traj[:, 5])
        axs[1, 5].plot(gtTraj[:, 5])
        plt.show()

    def validate(self, actionSeqs, gtTrajs):
        '''
            computes the error of the model for a number of trajectories with
            the matching action sequences.

            - input:
            --------
                - acitonSeqs: Tensor of the action sequences.
                    Shape [k, tau, 6, 1]
                
                - gtTrajs: Tensor of the ground truth trajectories.
                    Shape [k, tau, 13, 1]

            - output:
            ---------
                - L(nn(actionSeqs), trajs), the loss between the predicted trajectory
                and the ground truth trajectory.
        '''
        tau = actionSeqs.shape[1]
        k = actionSeqs.shape[0]
        state = np.expand_dims(gtTrajs[:, 0], axis=-1)
        trajs = [np.expand_dims(state, axis=1)]
        # PAY ATTENTION TO THE FOR LOOPS WITH @tf.function.
        for i in range(tau-1):
            with tf.name_scope("Rollout_" + str(i)):
                with tf.name_scope("Prepare_data_" + str(i)) as pd:
                    # make the action a [1, 6, 1] tensor
                    action = np.expand_dims(actionSeqs[:, i], axis=-1)
                with tf.name_scope("Step_" + str(i)) as s:
                    nextState = self.model.build_step_graph(s,
                                                            state,
                                                            action)
            state = nextState
            trajs.append(np.expand_dims(state, axis=1))
        
        trajs = np.squeeze(np.concatenate(trajs, axis=1), axis=-1)
        err = tf.linalg.norm(tf.subtract(trajs, gtTrajs))/k

        self.plot_seq(trajs[0], gtTrajs[0])
        return err

    # PRIVATE
    def _train_step(self, X, y):
        # If batchSize = -1, feed in the entire batch
        with tf.GradientTape() as tape:
            pred = self.model._predict_nn("train", np.squeeze(X, axis=-1))
            loss = tf.reduce_mean(tf.math.squared_difference(pred, y),
                                  name="loss")
            grads = tape.gradient(loss, self.model.weights())
            self.optimizer.apply_gradients(zip(grads,
                                               self.model.weights()))
            return loss

    def _save_graph(self):
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
