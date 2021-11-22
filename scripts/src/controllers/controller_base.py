import tensorflow as tf
from tensorflow.python.ops import summary_ops_v2
from cpprb import ReplayBuffer

import numpy as np

from datetime import datetime
import matplotlib.pyplot as plt
import os
from shutil import copyfile
import scipy.signal

from ..misc.utile import log_control, assert_shape
import time as t

import warnings

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu_devices[0], True)


class ControllerBase(tf.Module):

    def __init__(self,
                 model,
                 cost,
                 k=1,
                 tau=1,
                 dt=0.01,
                 sDim=1,
                 aDim=1,
                 lam=1.,
                 upsilon=1.,
                 sigma=np.array([]),
                 initSeq=np.array([]),
                 normalizeCost=False,
                 filterSeq=False,
                 log=False,
                 logPath=None,
                 gif=False,
                 graphMode=False,
                 configFile=None,
                 taskFile=None,
                 debug=False):

        '''
            Mppi controller base class constructor.

            - Input:
            --------
                - model: a model object heritated from the model_base class.
                - Cost: a cost object heritated from the cost_base class.
                - k: Int, The number of samples used inside the controller.
                - tau: Int, The number of prediction timesteps.
                - dt: Float, The timestep size.
                - sDim: Int, The state space dimension.
                - aDim: Int, The action space dimension.
                - lam: Float, The inverse temperature.
                - upsilon: Float, The augmented covariance on the noise term.
                - sigma: The noise of the system.
                    Array with shape [aDim, aDim]
                - initSeq: The inital action sequence.
                    Array of shape [tau, aDim, 1]
                - normalizeCost: Bool, wether or not normalizin the cost,
                    simplifies tuning of lambda
                - filterSeq: Bool, wether or not to filter the input
                    sequence after each optimization.
                - log: Bool, if true logs controller info in tensorboard.
                - logPath: String, the path where the info will be logged.
                - gif: Bool, if true generates an animated gif of the
                    controller execution (not tested in ros).
                - configFile: Environment config file.
                - taskFile: Task config file, the cost hyper parameter.
                - debug: Bool, if true, the controller goes in debug mode
                    and logs more information.

        '''
        # TODO: Check parameters and make the tensors.
        # This is needed to create a correct trace.
        self._k = tf.Variable(k, trainable=False, dtype=tf.int32)
        self._tau = tau
        self._sDim = sDim
        self._aDim = aDim
        self._lam = tf.Variable(lam, trainable=False, dtype=tf.float64)
        self._upsilon = tf.Variable(upsilon, trainable=False, dtype=tf.float64)
        self._normalizeCost = normalizeCost
        self._filterSeq = filterSeq
        self._log = log
        self._debug = debug
        self._graphMode = graphMode

        if self._graphMode:
            self._next_graph = tf.function(self._next)

        self._sigma = tf.convert_to_tensor(sigma, dtype=tf.float64)

        self._logDict = {}

        self._steps = 0

        if initSeq.size == 0:
            self._actionSeq = tf.zeros((tau, aDim, 1), dtype=tf.float64)
        else:
            if assert_shape(initSeq, (tau, aDim, 1)):
                self._actionSeq = initSeq
            else:
                raise AssertionError

        self._model = model
        self._cost = cost

        self._bufferSize = 264
        self._batchSize = 32

        self._rb = ReplayBuffer(self._bufferSize,
                                env_dict={"obs": {"shape": (self._sDim, 1)},
                                          "act": {"shape": (self._aDim, 1)},
                                          "next_obs": {"shape":
                                                       (self._sDim, 1)}})

        self._trainStep = 0
        self._writer = None

        if self._log or self._debug:
            stamp = datetime.now().strftime("%Y.%m.%d-%H:%M:%S")
            path = 'graphs/python/'
            if self._debug:
                path = os.path.join(path, 'debug')
            logdir = os.path.join(logPath,
                                  path,
                                  model.get_name(),
                                  "k" + str(k),
                                  "T" + str(tau),
                                  "L" + str(lam),
                                  stamp)

            print(logdir)
            os.makedirs(logdir)

            self._writer = tf.summary.create_file_writer(logdir)
            # self.save_graph()

            self._summary_name = ["x", "y", "z"]

            if configFile is not None:
                conf_dest = os.path.join(logdir, "config.yaml")
                task_dest = os.path.join(logdir, "task.yaml")
                copyfile(configFile, conf_dest)
                copyfile(taskFile, task_dest)

    def save_graph(self):
        '''
            Saves the graph in tensorboard.

        '''
        state = tf.zeros((self._sDim, 1))
        seq = tf.zeros((5, self._aDim, 1))
        with self._writer.as_default():
            # get graph from function
            graph = tf.function(
                self._next).get_concrete_function(1, state, seq).graph
            # visualize
            summary_ops_v2.graph(graph.as_graph_def())

    def save(self, x, u, xNext):
        '''
            Saves the transitions to the replay buffer.
            If log is True, saves controller info to tensorboard.

            - input:
            --------
                - x: the current state. Shape [sDim, 1]
                - u: the current action. Shape [aDim, 1]
                - xNext: the next state. Shape [sDim, 1]
        '''

        if not (assert_shape(x, (self._sDim, 1)) and
                assert_shape(u, (self._aDim, 1)) and
                assert_shape(xNext, (self._sDim, 1))):
            raise AssertionError("Input shape missmatch, \
                                 x.shape = {}; expected {}, u.shape = {}; \
                                 expected {}, xNext.shape = {}; expected {}".
                                 format(x.shape, (self._sDim, 1), u.shape,
                                        (self._aDim, 1), xNext.shape,
                                        (self._sDim, 1)))

        self._rb.add(obs=x, act=u, next_obs=xNext)

        if self._log:
            returnDict = self.predict(x, u, self._actionSeq, xNext)
            self._logDict = {**self._logDict, **returnDict}
            log_control(self._writer,
                        self._logDict,
                        [0, 1, 2, 3, 4, 5, 6],
                        [7, 8, 9, 10, 11, 12])

    def save_rp(self, filename):
        '''
            Save the replay buffer transitions to a file.

            - input:
            --------
                - filename, string.
        '''
        self._rb.save_transitions(filename)

    def ch_dict_prefix(self, prefix, cur_dict):
        returnDict = {}
        for item in cur_dict:
            returnDict[prefix+"_"+item] = cur_dict[item]
        return returnDict

    def state_error(self, state_gt, state_pred):
        returnDict = {}
        shape = state_gt.shape[0]/2
        xGt = state_gt[0]
        xPred = state_pred[0]
        vGt = state_gt[1]
        vPred = state_pred[1]
        for dim in range(1, int(shape)):
            xGt = np.hstack([xGt, state_gt[2*dim]])
            xPred = np.hstack([xPred, state_pred[2*dim]])
            vGt = np.hstack([vGt, state_gt[2*dim+1]])
            vPred = np.hstack([vPred, state_pred[2*dim+1]])
        returnDict["error_pos"] = np.linalg.norm(xGt - xPred)
        returnDict["error_vel"] = np.linalg.norm(vGt - vPred)

        return returnDict

    def predict(self, x, u, actionSeq, xNext):
        # TODO: get first predicted next state, predict action sequence,
        # compute diff between first next state and next state
        # log error on prediction, predicted cost, action input, state input.
        returnDict = {}
        nextState = self._model.predict(np.expand_dims(x, axis=0),
                                        np.expand_dims(u, axis=0))
        errorDict = self.state_error(xNext, nextState.numpy()[0, :, :])
        returnDict["next_state"] = xNext
        returnDict["state"] = x
        distDict = self._cost.dist(x)

        costDict = self._cost.state_cost("predict", nextState)

        state = nextState

        for i in range(actionSeq.shape[0]):
            with tf.name_scope("Prepare_data_" + str(i)) as pd:
                action = self.prepare_action(pd, actionSeq, i)
            with tf.name_scope("Step_" + str(i)) as s:
                nextState = self._model.build_step_graph(s, state, action)
            with tf.name_scope("Cost_" + str(i)) as c:
                tmpDict = self._cost.state_cost(c, nextState)
                costDict = self._cost.add_cost(c, tmpDict, costDict)
            state = nextState

        with tf.name_scope("terminal_cost") as s:
            fCostDict = self._cost.build_final_step_cost_graph(s, nextState)
        with tf.name_scope("Rollout_cost"):
            sampleCostsDict = self._cost.add_cost(c, fCostDict, costDict)

        sampleCostsDict = self.ch_dict_prefix("predicted", sampleCostsDict)
        returnDict = {**returnDict,
                      **sampleCostsDict,
                      **errorDict,
                      **distDict}

        return returnDict

    def train(self):
        if self._rb.get_stored_size() < 32 or self._model.is_trained():
            return

        epochs = 500
        for e in range(epochs):
            sample = self._rb.sample(self._batchSize)
            gt = sample['next_obs']
            x = sample['obs']
            u = sample['act']
            self._model.train_step(gt, x, u, self._trainStep*epochs + e,
                                   self._writer, self._log)

        self._trainStep += 1

    def _next(self, k, state, actionSeq, normalizeCost=False):
        '''
            Internal tensorflow part of the controller.
            Computes the next action based on the number of samples,
            the state, the current action sequence.
            Should not be called by user directly.

            - input:
            --------
                - k: tf.Variable, dtype=int, the number of samples to use.
                - state: tf.Placeholder, the current state of the system.
                    Shape: [sDim, 1]
                - actionSeq: tf.Placeholder, the current optimized action
                sequence.
                    Shape: [tau, aDim ,1]
                - normalizeCost: Bool, if true the cost are normalized
                    before the expodential.

            - output:
            ---------
                - dictionnary with entries:
                    'next' the next action to be applied. Shape: [aDim, 1]

        '''
        print("Tracing with {}".format(state))
        self._model.set_k(k)
        # every input has already been check in parent function calls
        with tf.name_scope("Controller") as cont:
            return self.build_graph(cont, k, state, actionSeq,
                                    normalize=normalizeCost)

    def next(self, state):
        '''
            Computes the next action from MPPI controller.
            input:
            ------
                -state. The Current observed state of the system.
                    Shape: [sDim, 1]

            output:
            -------
                - next_action. The next action to be applied to the system.
                    Shape: [aDim, 1]
        '''

        # if not tf.ensure_shape(state, [self._sDim, 1]):
        #    raise AssertionError("State tensor doesn't have the expected \
        #                         shape.\n Expected [{}, 1], got {}".
        # format(self._sDim, state.shape))
        start = t.perf_counter()
        if self._graphMode:
            returnDict = self._next_graph(self._k,
                                          state,
                                          self._actionSeq,
                                          self._normalizeCost)
        else:
            returnDict = self._next(self._k,
                                    state,
                                    self._actionSeq,
                                    self._normalizeCost)

        # FIRST GUESS window_length = 5, we don't want to filter out to much
        # since we expect smooth inputs, need to be played around with.
        # FIRST GUESS polyorder = 3, think that a 3rd degree polynome is
        # enough for this.
        if self._filterSeq:
            returnDict["actionSeq"] = np.expand_dims(
                    scipy.signal.savgol_filter(
                            returnDict["actionSeq"].numpy()[:, :, 0],
                            29, 9, deriv=0, delta=1.0, axis=0),
                    axis=-1)

        end = t.perf_counter()

        self._timingDict['total'] += end-start
        self._timingDict['calls'] += 1

        self._actionSeq = returnDict["actionSeq"]
        self._logDict = returnDict
        self._logDict["time_steps"] = self._elapsed["steps"]/self._steps

        return np.squeeze(returnDict["next"].numpy())

    def get_paths(self):
        return self._logDict["paths"].numpy()

    def get_applied(self):
        return self._logDict["applied"].numpy()

    def get_log(self):
        return self._logDict.copy()

    def beta(self, scope, cost):
        # shapes: in [k, 1, 1]; out [1, 1]
        return tf.reduce_min(cost, 0)

    def norm_arg(self, scope, cost, beta, normalize=False):
        shift = tf.math.subtract(cost, beta)

        if normalize:
            max = tf.reduce_max(shift, 0)
            return tf.divide(shift, max)
        return shift

    def exp_arg(self, scope, arg):
        # shapes: in [k, 1, 1], [1, 1]; out [k, 1, 1]
        return tf.math.multiply(-1./self._lam, arg)

    def exp(self, scope, arg):
        # shapes: in [k, 1, 1]; out [k, 1, 1]
        return tf.math.exp(arg)

    def nabla(self, scope, arg):
        # shapes: in [k, 1, 1]; out [k, 1, 1]
        return tf.math.reduce_sum(arg, 0)

    def weights(self, scope, arg, nabla):
        # shapes: in [k, 1, 1], [1, 1]; out [k, 1, 1]
        return tf.realdiv(arg, nabla)

    def weighted_noise(self, scope, weights, noises):
        # shapes: in [k, 1, 1], [k, tau, aDim, 1]; out [tau, aDim, 1]
        return tf.math.reduce_sum(
                tf.math.multiply(
                    tf.expand_dims(weights, -1),
                    noises),
                0)

    def prepare_action(self, scope, actions, timestep):
        '''
            Prepares the next action to be applied during the rollouts.

            - input:
            --------
                - scope: string, the tensorflow scope name.
                - actions: the action sequence. Shape [tau, aDim, 1]
                - timestep: the current timestep in the rollout.

            - output:
            ---------
                - the action to apply:
                    tf.Tensor. Shape [aDim, 1], dtype=tf.float64
        '''

        return tf.squeeze(tf.slice(actions, [timestep, 0, 0], [1, -1, -1]), 0)

    def prepare_noise(self, scope, noises, timestep):
        '''
            Prepares the noise to be applied at the current timestep
            of the rollouts.

            - input:
            --------
                - scope: string, the tensorflow scope name.
                - noises: The noise tensor.
                    tf.Tensor shape [k, tau, aDim, 1], dtype=tf.float64
                - timestep: the current timestep of the rollout.

            - output:
            ---------
                - noise at time timestep.
                    tf.Tensor Shape [k, aDim, 1]
        '''

        return tf.squeeze(tf.slice(noises,
                                   [0, timestep, 0, 0],
                                   [-1, 1, -1, -1]),
                          1)

    def update(self, scope, cost, noises, normalize=False):
        returnDict = {}
        # shapes: in [k, 1, 1], [k, tau, aDim, 1]; out [tau, aDim, 1]
        with tf.name_scope("Beta"):
            beta = self.beta(scope, cost)
        with tf.name_scope("Expodential_arg"):
            arg = self.norm_arg(scope, cost, beta, normalize=normalize)
            exp_arg = self.exp_arg(scope, arg)
        with tf.name_scope("Expodential"):
            exp = self.exp(scope, exp_arg)
        with tf.name_scope("Nabla"):
            nabla = self.nabla(scope, exp)
        with tf.name_scope("Weights"):
            weights = self.weights(scope, exp, nabla)
        with tf.name_scope("Weighted_Noise"):
            weighted_noises = self.weighted_noise(scope, weights, noises)
        with tf.name_scope("Sequence_update"):
            update = tf.add(self._actionSeq, weighted_noises)

        returnDict["weights"] = weights
        returnDict["nabla"] = nabla
        returnDict["arg"] = arg
        returnDict["weighted_noises"] = weighted_noises
        returnDict["update"] = update
        return returnDict

    def shift(self, scope, actionSeq, init, length):
        # shapes: in [tau, aDim, 1], [x, aDim, 1], scalar;
        # out [tau-len + x, aDim, 1]
        remain = tf.slice(actionSeq, [length, 0, 0],
                          [self._tau-length, -1, -1])
        return tf.concat([remain, init], 0)

    def get_next(self, scope, current, length):
        # shapes: in [tau, aDim, 1], scalar; out [scalar, aDim, 1]
        return tf.slice(current, [0, 0, 0], [length, -1, -1])

    def advance_goal(self, next):
        self._cost.setGoal(next)

    def build_graph(self, scope, k, state, actionSeq, normalize=False):
        '''
            Builds the tensorflow computational graph for the
            controller update. First it generates the noise used
            for the samples. Then it computes the cost of every
            sample's rollout and then performes the update of the
            action sequence (compute weighted average). Finally it
            gets the next element to apply and shifts the action
            sequence.

            - input:
            --------
                - k: tf.Variable, dtype=int, the number of samples to use.
                - state: tf.Placeholder, the current state of the system.
                    Shape: [sDim, 1]
                - actionSeq: tf.Variable, the current optimized
                action sequence.
                    Shape: [tau, aDim ,1]
                - normalizeCost: Bool, if true the cost are normalized
                    before the expodential.

            - output:
            ---------
                - dictionnary with entries:
                    'noises' the noise tensor used for the current step.
                        Shape [k, tau, aDim, 1]
                    'actionSeq' the action sequence at the current step.
                        Shape [tau, aDim, 1]
                    'next' the next action to be applied.
                        Shape: [aDim, 1]
        '''
        returnDict = {}
        with tf.name_scope(scope) as scope:
            with tf.name_scope("random") as rand:
                noises = self.build_noise(rand, k)
            with tf.name_scope("Rollout") as roll:
                costDict = self.build_model(roll, k, state, noises, actionSeq)
            with tf.name_scope("Update") as up:
                update_dict = self.update(up, costDict["cost"], noises,
                                          normalize=normalize)
            with tf.name_scope("Next") as n:
                next = self.get_next(n, update_dict["update"], 1)
            with tf.name_scope("shift_and_init") as si:
                init = self.init_zeros(si, 1)
                actionSeq = self.shift(si, update_dict["update"], init, 1)

        returnDict["noises"] = noises
        returnDict["actionSeq"] = actionSeq
        returnDict["next"] = next
        returnDict = {**returnDict, **costDict, **update_dict}

        return returnDict

    def build_model(self, scope, k, state, noises, actionSeq):
        '''
            Builds the rollout graph and computes the cost for every sample.

            - input:
            --------
                - scope: string, the tensorflow scope name.
                - k: the number of samples. tf.Variable shape [1]
                - state: the current state of the system.
                    tf.Tensor Shape [sDim, 1]
                - noises: the noise tensor to use for the rollouts.
                    tf.tensor Shape [k, tau, aDim, 1]
                - actionSeq: the current action sequence.
                    tf.tensor Shape [tau, aDim, 1]

            - output:
            ---------
                - dictionnary with entries:
                    'paths' the generated paths for every samples.
                        Shape [k, tau, sDim, 1]

        '''
        state = tf.expand_dims(state, axis=0)
        nextState = tf.zeros(shape=state.shape, dtype=tf.float64)
        sshape = state.shape
        returnDict = {}
        costDict = {}
        c = ""
        paths = [tf.broadcast_to(state, [k, sshape[0], sshape[1], sshape[2]])]
        applied = []

        # PAY ATTENTION TO THE FOR LOOPS WITH @tf.function.
        for i in range(self._tau):
            with tf.name_scope("Prepare_data_" + str(i)) as pd:
                action = self.prepare_action(pd, actionSeq, i)
                noise = self.prepare_noise(pd, noises, i)
                toApply = tf.add(action, noise, name="toApply")
            with tf.name_scope("Step_" + str(i)) as s:
                nextState = self._model.build_step_graph(s, state, toApply)
            with tf.name_scope("Cost_" + str(i)) as c:
                tmpDict = self._cost.build_step_cost_graph(c, nextState,
                                                           action, noise)
                costDict = self._cost.add_cost(c, tmpDict, returnDict)

            state = nextState

            paths.append(tf.expand_dims(state, 1))
            applied.append(tf.expand_dims(toApply, 1))
        paths = tf.concat(paths, 1)
        applied = tf.concat(applied, 1)

        with tf.name_scope("terminal_cost") as s:
            fCostDict = self._cost.build_final_step_cost_graph(s, nextState)
        with tf.name_scope("Rollout_cost"):
            sampleCostsDict = self._cost.add_cost(c, fCostDict, costDict)

        returnDict["paths"] = paths
        returnDict["applied"] = applied
        returnDict = {**returnDict, **sampleCostsDict}

        return returnDict

    def build_noise(self, scope, k):
        '''
            Buids the tensorflow ops to generate the random noise
            for the controller.

            - input:
            --------
                - Scope: String, name of the current scope.
                - k: the number of samples to use.

            - output:
            ---------
                - Noise tensor: float tensor. Shape [k, tau, aDim, 1]
        '''
        rng = tf.random.normal(shape=(k, self._tau, self._aDim, 1),
                               stddev=1.,
                               mean=0.,
                               dtype=tf.float64,
                               seed=1)

        return tf.linalg.matmul(self._upsilon*self._sigma, rng)

    def init_zeros(self, scope, size):
        # shape: out [size, aDim, 1]
        return tf.zeros([size, self._aDim, 1], dtype=tf.float64)

    def get_profile(self):
        profile = self._timingDict.copy()
        # Update the model profile first
        model_rollout = self._model.get_profile()
        model_rollout['total'] = self._rolloutTimingDict['model_t']

        profile['rollout'] = self._rolloutTimingDict.copy()
        profile['rollout']['model'] = model_rollout
        profile['rollout']['horizon'] = self._tau
        del profile['rollout']['model_t']
        return profile

    def trace(self):
        '''
            Runs the controller "a blanc" to build the tensorflow
            computational graph. After that it resets the controller
            internal variables for a fresh start.

            inputs:
            -------
                None.

            outputs:
            --------
                None.
        '''
        fake_state = np.zeros((self._sDim, 1))
        fake_state[6] = 1.
        fake_sequence = np.zeros((self._tau, self._aDim, 1))

        if self._graphMode:
            _ = self._next_graph(tf.Variable(1, dtype=tf.int32),
                                 fake_state,
                                 fake_sequence,
                                 self._normalizeCost)
        else:
            _ = self._next(tf.Variable(1, dtype=tf.int32),
                           fake_state,
                           fake_sequence,
                           self._normalizeCost)
            warnings.warn("Not using graph mode, no trace to generate.")
        pass
