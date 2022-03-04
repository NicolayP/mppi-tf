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

import copy

from ..model import copy_model

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

    def rb_trans(self):
        return self.rb.get_all_transitions().copy()

    def save_rb(self, filename):
        self.rb.save_transitions(filename)
 
    def save_params(self, step):
        self.model.save_params(self.logdir, step)

    def stats(self):
        data = self.rb_trans()
        (X, y) = self.model.prepare_training_data(data['obs'],
                                                  data['next_obs'],
                                                  data['act'], norm=False)
        Xmean = np.mean(X, axis=0)
        Xstd = np.std(X, axis=0)
        
        Ymean = np.mean(y, axis=0)
        Ystd = np.std(y, axis=0)

        self.model.set_Xmean_Xstd(Xmean, Xstd)
        self.model.set_Ymean_Ystd(Ymean, Ystd)

    def grid_search(self, trajs, actionSeqs):
        init_weights = self.model.get_weights()
        
        learningRate = np.linspace(0.0001, 0.1, 10)
        #learningRate = np.linspace(0.0001, 0.1, 2)
        batchSize = np.array([-1])
        epoch = np.array([50, 100, 500])
        #epoch = np.array([2])
        
        writers = []
        for lr in learningRate:
            lr_writers = []
            for bs in batchSize:
                logdir = os.path.join(self.logdir, "-lr-{:4f}".format(lr), "-bs-{}".format(bs))
                lr_writers.append(tf.summary.create_file_writer(logdir))
            writers.append(lr_writers)
        
        k = 10
        k = 2
        self.stats()

        valErr = []
        valTransErr = []
        for e in epoch:
            for lr, lr_writers in zip(learningRate, writers):
                for bs, writer in zip(batchSize, lr_writers):
                    self.k_fold_validation(learningRate=lr,
                                           batchSize=bs,
                                           epoch=e, k=k,
                                           val=(trajs, actionSeqs),
                                           writer=writer)

                    self.train_all(learningRate=lr, batchSize=bs,
                                   epoch=e, val=(trajs, actionSeqs),
                                   writer=writer)
                    err, transErr = self.validate(self.model, actionSeqs, trajs, plot=True, transition=True, split=False)
                    valErr.append(np.expand_dims(err, axis=0))
                    valTransErr.append(np.expand_dims(transErr, axis=0))
                    print("lr: {}, e: {}, validation error: {}, transition error: {}".format(lr, e, err.numpy(), transErr.numpy()))
                    self.model.update_weights(init_weights, msg=False)
        valErr = np.concatenate(valErr, axis=0)
        print("Best mean:", np.max(valErr))

    def train_all(self, learningRate=0.1, batchSize=32,
                  epoch=100, val=None, writer=None):
        self.optimizer = tf.optimizers.Adam(learning_rate=learningRate)
        data = self.rb_trans()
        (X, y) = self.model.prepare_training_data(data['obs'],
                                                  data['next_obs'],
                                                  data['act'])
        self.train(X, y, epoch=epoch, learningRate=learningRate, val=val, writer=writer)

    def k_fold_validation(self, k=10, learningRate=0.1, batchSize=32,
                          epoch=100, val=None, writer=None):

        data = self.rb_trans()
        (X, y) = self.model.prepare_training_data(data['obs'],
                                                  data['next_obs'],
                                                  data['act'])
        X = X.numpy()
        y = y.numpy()

        # Prepare fold algorithm
        kfold = KFold(n_splits=k, shuffle=True)
        # Extract data for folds to run them concurently (easier log).
        kFoldData = [(X[train], y[train], X[test], y[test]) for train, test in kfold.split(X, y)]

        # prepare models copy
        m = copy_model(self.model)
        

        models = [copy_model(self.model) for i in range(k)]
        init_weights = self.model.get_weights()
        for m in models:
            m.update_weights(init_weights, msg=False) 
        # perpare optimizers (not necessary I think but not sure if adam hasn't a form of internal memory about gradients.)
        optimizers = [tf.optimizers.Adam(learning_rate=learningRate) for i in range(k)]

        for e in range(epoch):
            lossesTrain, lossesTest = self.kfold_step(models=models, optimizers=optimizers, data=kFoldData)
            logScope="kfold/e{}".format(epoch)
            self._log_hist(writer, "train/" + logScope, lossesTrain, e)
            self._log_hist(writer, "test/" + logScope, lossesTest, e)
            if e % 10 == 0 and (val is not None):
                lossFold = self.validate_fold(models, val)
                self._log_hist(writer, "val/" + logScope, lossFold, e)

    def kfold_step(self, models, optimizers, data):
        k = len(data)
        lossesTrain = []
        lossesTest = []
        for i in range(k):
            loss, _ = self._train_step(models[i], optimizers[i], data[i][0], data[i][1])
            lossesTrain.append(loss)

            loss = self.evaluate(models[i], data[i][2], data[i][3])
            lossesTest.append(loss)

        lossesTrain=tf.concat(lossesTrain, axis=0)
        lossesTest=tf.concat(lossesTest, axis=0)
        return lossesTrain, lossesTest

    def evaluate(self, model, X, y):
        pred = model._predict_nn("Eval", X)
        loss = tf.reduce_mean(tf.math.squared_difference(pred, y),
                              name="loss")
        return loss

    def validate_fold(self, models, val):
        losses = []
        for m in models:
            losses.append(self.validate(m, val[1], val[0]))
        
        return tf.concat(losses, axis=0)

    def validate(self, model, actionSeqs, gtTrajs, plot=False, transition=False, split=False):
        '''
            computes the error of the model for a number of trajectories with
            the matching action sequences.

            - input:
            --------
                - acitonSeqs: Tensor of the action sequences.
                    Shape [k, tau, 6]

                - gtTrajs: Tensor of the ground truth trajectories.
                    Shape [k, tau, 13]

                - plot: bool (default:False) if true, also plots
                    the first predicted trajectory id:0

                - transition: bool (default:False) if true, also
                    computes the individual prediction error.

                - split: bool (default:False) if true, also computes
                    the error for every output dimension individually
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
                    nextState = model.build_step_graph(s, state, action)
            state = nextState
            trajs.append(np.expand_dims(state, axis=1))
        
        trajs = np.squeeze(np.concatenate(trajs, axis=1), axis=-1)
        errSplit = tf.reduce_mean(tf.math.squared_difference(trajs, gtTrajs),
                                  name="loss", axis=[0, 1])
        err = tf.reduce_mean(errSplit)

        if transition:
            actions = actionSeqs[:, :-1, :]
            states = gtTrajs[:, :-1, :]
            nextStates = gtTrajs[:, 1:, :]

            actions = tf.reshape(actions, shape=((tau-1)*k, actionSeqs.shape[2], 1))
            states = tf.reshape(states, shape=((tau-1)*k, gtTrajs.shape[2], 1))
            nextStates = tf.reshape(nextStates, shape=((tau-1)*k, gtTrajs.shape[2], 1))

            (X, y) = model.prepare_training_data(states, nextStates, actions)
            pred = model._predict_nn("transition", X)
            transErrSplit = tf.reduce_mean(tf.math.squared_difference(pred, y),
                                      name="transition-loss", axis=0)
            transErr = tf.reduce_mean(transErrSplit)

        if plot:
            self.plot_seq(trajs[0], gtTrajs[0])

        if transition and split:
            return err, errSplit, transErr, transErrSplit
        if transition:
            return err, transErr
        if split:
            return err, errSplit
        return err

    def train(self, X, y, epoch=1, learningRate=0.1, val=None, writer=None):
        for e in range(epoch):
            scope = "e{}".format(epoch)
            loss, lossSplit, grad = self._train_step(self.model, self.optimizer, X, y, split=True)
            self._log_scalar(writer, "train/" + scope, loss, e)
            #self._log_hist(writer, "grad/" + scope, grad, e)
            self._log_split(writer, "trainSplit/" + scope, lossSplit, e)

            if (val is not None) and (e % 10 == 0):
                valLoss, valLossSplit, transValLoss, transValLossSplit = self.validate(self.model, val[1], val[0], transition=True, split=True)
                self._log_scalar(writer, "valTotal/" + scope, valLoss, e)
                self._log_scalar(writer, "valTransition/" + scope, transValLoss, e)
                self._log_split(writer, "valSplit/" + scope, valLossSplit, e)
                self._log_split(writer, "valTransitionSplit/" + scope, transValLossSplit, e)

    def plot_seq(self, traj, gtTraj):
        fig, axs = plt.subplots(figsize=(20, 10), nrows=2, ncols=7)
        dim = traj.shape[1]
        # Position
        axs[0, 0].plot(traj[:, 0])
        axs[0, 0].plot(gtTraj[:, 0])
        axs[0, 0].set_title("x")

        axs[0, 1].plot(traj[:, 1])
        axs[0, 1].plot(gtTraj[:, 1])
        axs[0, 1].set_title("y")

        axs[0, 2].plot(traj[:, 2])
        axs[0, 2].plot(gtTraj[:, 2])
        axs[0, 2].set_title("z")

        # Rotation
        axs[0, 3].plot(traj[:, 3])
        axs[0, 3].plot(gtTraj[:, 3])

        axs[0, 4].plot(traj[:, 4])
        axs[0, 4].plot(gtTraj[:, 4])

        axs[0, 5].plot(traj[:, 5])
        axs[0, 5].plot(gtTraj[:, 5])

        if dim==12: # euler
            off=1
            axs[0, 3].set_title("Roll")
            axs[0, 4].set_title("Pitch")
            axs[0, 5].set_title("Yaw")
        elif dim==13: #quaternion
            off = 0
            axs[0, 6].plot(traj[:, 6])
            axs[0, 6].plot(gtTraj[:, 6])
            
            axs[0, 3].set_title("Qx")
            axs[0, 4].set_title("Qy")
            axs[0, 5].set_title("Qz")
            axs[0, 6].set_title("Qz")
            
        # Lin Vel
        axs[1, 0].plot(traj[:, 7-off])
        axs[1, 0].plot(gtTraj[:, 7-off])
        axs[1, 0].set_title("V_x")

        axs[1, 1].plot(traj[:, 8-off])
        axs[1, 1].plot(gtTraj[:, 8-off])
        axs[1, 1].set_title("V_y")

        axs[1, 2].plot(traj[:, 9-off])
        axs[1, 2].plot(gtTraj[:, 9-off])
        axs[1, 2].set_title("V_z")

        # Ang vel
        axs[1, 3].plot(traj[:, 10-off])
        axs[1, 3].plot(gtTraj[:, 10-off])
        axs[1, 3].set_title("p")

        axs[1, 4].plot(traj[:, 11-off])
        axs[1, 4].plot(gtTraj[:, 11-off])
        axs[1, 4].set_title("q")

        axs[1, 5].plot(traj[:, 12-off])
        axs[1, 5].plot(gtTraj[:, 12-off])
        axs[1, 5].set_title("r")
        
        bb = (fig.subplotpars.left, fig.subplotpars.top+0.02, 
              fig.subplotpars.right-fig.subplotpars.left,.1)
    
        fig.legend(bbox_to_anchor=bb, loc="lower left",
                   ncol=2, borderaxespad=0., bbox_transform=fig.transFigure)
        plt.show()

    # PRIVATE
    def _train_step(self, model, optimizer, X, y, split=False):
        # If batchSize = -1, feed in the entire batch
        with tf.GradientTape() as tape:
            pred = model._predict_nn("train", X)
            lossSplit = tf.reduce_mean(tf.math.squared_difference(pred, y), axis=0)
            loss = tf.reduce_mean(tf.math.squared_difference(pred, y),
                                  name="loss")
            grads = tape.gradient(loss, model.weights())
            optimizer.apply_gradients(zip(grads,
                                          model.weights()))

            if split:
                return loss, lossSplit, grads
            return loss, grads

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

    def _log_hist(self, writer, scope, tensor, step):
        if self.log:
            with writer.as_default():
                tf.summary.histogram(scope, tensor, step)

    def _log_scalar(self, writer, scope, tensor, step):
        if self.log:
            with writer.as_default():
                tf.summary.scalar(scope, tensor, step)

    def _log_split(self, writer, scope, tensor, step):
        '''
            Tensor shape [6/12/13]
        '''
        if self.log:
            dim =  tensor.shape[0]
            if dim == 6:
                names = ["vx", "vy", "vz", "p", "q", "r"]
            elif dim == 12:
                names = ["x", "y", "z", "roll", "pitch", "yaw", "vx", "vy", "vz", "p", "q", "r"]
            elif dim == 13:
                names = ["x", "y", "z", "qx", "qy", "qz", "qw", "vx", "vy", "vz", "p", "q", "r"]
            else:
                print("Error, accepted dimension are 6, 12, 13 got {}", dim)
                return

            with writer.as_default():
                for i, axs in enumerate(names):
                    tf.summary.scalar(scope + "-" + axs, tensor[i], step)