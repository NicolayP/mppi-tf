import tensorflow as tf
from cpprb import ReplayBuffer

import numpy as np
import scipy.signal

from ..misc.utile import assert_shape, push_to_tensor, dtype
from ..observer.observer_base import ObserverBase
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
                 graphMode=False,
                 configDict=None,
                 taskDict=None,
                 modelDict=None,
                 debug=False):

        '''
            Mppi controller base class constructor.

            - Input:
            --------
                - model: a model object heritated from the model_base class.
                - Cost: a cost object heritated from the cost_base class.
                - k: Int, The number of samples used inside the controller.
                - tau: Int, The number of prediction timesteps.
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
                - configDict: Environment config dict.
                - taskDict: Task config dict, the cost hyper parameter.
                - modelDict: Model config dict, the model parameters.
                - debug: Bool, if true, the controller goes in debug mode
                    and logs more information.

        '''
        # TODO: Check parameters and make the tensors.
        # This is needed to create a correct trace.
        self._model = model
        self._cost = cost

        self._k = tf.Variable(k,
                              trainable=False,
                              dtype=tf.int32,
                              name="samples")

        self._lam = tf.Variable(lam,
                                trainable=False,
                                dtype=dtype,
                                name="lambda")

        self._upsilon = tf.Variable(upsilon,
                                    trainable=False,
                                    dtype=dtype,
                                    name="upsilon")

        self._tau = tau
        self._sDim = sDim
        self._aDim = aDim
        self._normalizeCost = normalizeCost
        self._filterSeq = filterSeq
        self._log = log
        self._debug = debug
        self._graphMode = graphMode

        self._observer = ObserverBase(logPath=logPath,
                                      log=log,
                                      debug=debug,
                                      k=k,
                                      tau=tau,
                                      lam=lam,
                                      configDict=configDict,
                                      taskDict=taskDict,
                                      modelDict=modelDict,
                                      aDim=aDim,
                                      sDim=sDim,
                                      modelName=self._model.get_name())

        self._model.set_observer(self._observer)
        self._cost.set_observer(self._observer)

        self._sigma = tf.convert_to_tensor(sigma,
                                           dtype=dtype,
                                           name="Sigma")

        if self._graphMode:
            self._next_fct = tf.function(self._next)
        else:
            self._next_fct = self._next

        self._steps = 0

        if initSeq.size == 0:
            self._actionSeq = tf.zeros((tau, aDim, 1),
                                       dtype=dtype,
                                       name="Action_sequence_init")
        else:
            if assert_shape(initSeq, (tau, aDim, 1)):
                self._actionSeq = initSeq
            else:
                raise AssertionError

        self._trainStep = 0

        self._timingDict = {}
        self._timingDict['total'] = 0.
        self._timingDict['calls'] = 0

        if self._log:
            self._observer.save_graph(self._next_fct, self._graphMode)

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
        if self._log:
            pred = self.predict(x, u, self._actionSeq, xNext)
            self._observer.advance()

    def state_error(self, stateGt, statePred):
        error = tf.linalg.norm(tf.subtract(stateGt, statePred))
        return error

    # @tf.function
    def predict(self, model_input, actionSeq, xNext):
        raise NotImplementedError

    def print_concrete(self):
        print(self._next_fct.pretty_printed_concrete_signatures())

    def next(self, model_input):
        '''
            Computes the next action from MPPI controller.
            input:
            ------
                - laggedState. The Current and previous observed states of the system.
                    Shape: [history, sDim, 1]
                - laggedAction: The previous actions applied to the system
                    Shape: [history-1, aDim, 1]

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

        # tf.profiler.experimental.start(self._observer.get_logdir())
        next = self._next_fct(self._k,
                              model_input,
                              self._actionSeq,
                              self._normalizeCost)
        # tf.profiler.experimental.stop()

        # FIRST GUESS window_length = 5, we don't want to filter out to much
        # since we expect smooth inputs, need to be played around with.
        # FIRST GUESS polyorder = 3, think that a 3rd degree polynome is
        # enough for this.
        if self._filterSeq:
            self._actionSeqNp = tf.convert_to_tensor(
                                    np.expand_dims(
                                        scipy.signal.savgol_filter(
                                            self._actionSeq.numpy()[:, :, 0],
                                            29,
                                            9,
                                            deriv=0,
                                            delta=1.0,
                                            axis=0),
                                        axis=-1))

        end = t.perf_counter()

        self._timingDict['total'] += end-start
        self._timingDict['calls'] += 1
        return np.squeeze(next.numpy())

    def _next(self, k, model_input, actionSeq, normalizeCost=False, profile=False):
        '''
            Internal tensorflow part of the controller.
            Computes the next action based on the number of samples,
            the state, the current action sequence.
            Should not be called by user directly.

            - input:
            --------
                - k: tf.Variable, dtype=int, the number of samples to use.
                - laggedState: tf.Placeholder, the current state of the system.
                    Shape: [history, sDim, 1]
                - laggedAction: tf.Placeholder, the previous acitons applied to the system.
                    Shape: [history-1, aDim, 1]
                - actionSeq: tf.Placeholder, the current optimized action
                sequence.
                    Shape: [tau, aDim ,1]
                - normalizeCost: Bool, if true the cost are normalized
                    before the expodential.

            - output:
            ---------
                - next the next action to be applied. Shape: [aDim, 1]

        '''
        #print("Tracing with {}".format(model_input))
        self._model.set_k(k)
        # every input has already been check in parent function calls
        if profile:
            tf.profiler.experimental.start(self._observer.get_logdir())

        with tf.name_scope("Controller") as cont:
            action = self.build_graph(cont, k, model_input, actionSeq,
                                    normalize=normalizeCost)
        if profile:
            tf.profiler.experimental.stop()
        return action

    def build_graph(self, scope, k, model_input, actionSeq, normalize=False):
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
                - laggedState: tf.Placeholder, the current and previous states of the system.
                    Shape: [history, sDim, 1]
                - laggedAction: tf.Placeholder, the previous actions applied on the system.
                    Shape: [history-1, aDim, 1]
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
        with tf.name_scope(scope) as s:
            with tf.name_scope("random") as rand:
                noises = self.build_noise(rand, k)
            with tf.name_scope("Rollout") as roll:
                cost = self.build_model(roll, k, model_input, noises, actionSeq)
            with tf.name_scope("Update") as up:
                update = self.update(up, cost, noises, normalize=normalize)
            with tf.name_scope("Next") as n:
                next = self.get_next(n, update, 1)
            with tf.name_scope("shift_and_init") as si:
                init = self.init_zeros(si, 1)
                actionSeq = self.shift(si, update, init, 1)

        self._observer.write_control("state", model_input)
        self._observer.write_control("next", next)
        return next

    def build_model(self, scope, k, model_input, noises, actionsSeq):
        raise NotImplementedError

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
                               dtype=dtype,
                               seed=1)

        return tf.linalg.matmul(self._upsilon*self._sigma, rng)

    def update(self, scope, cost, noises, normalize=False):
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
            rawUpdate = tf.add(self._actionSeq, weighted_noises)
            #update = self.clip_act("clipping", rawUpdate)
            update = rawUpdate

        self._observer.write_control("weights", weights)
        self._observer.write_control("nabla", nabla)
        self._observer.write_control("arg", arg)
        self._observer.write_control("weighted_noise", weighted_noises)
        # self._observer.write_control("update", update)

        return update

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
                    weights[..., None],
                    noises),
                0)

    def clip_act(self, scope, update):
        maxInp = self._model.max_act()
        minInp = self._model.min_act()
        updateClipped = tf.clip_by_value(update, minInp, maxInp, scope)
        return updateClipped

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
                    tf.Tensor. Shape [aDim, 1], dtype=dtype
        '''

        return actions[timestep]

    def prepare_noise(self, scope, noises, timestep):
        '''
            Prepares the noise to be applied at the current timestep
            of the rollouts.

            - input:
            --------
                - scope: string, the tensorflow scope name.
                - noises: The noise tensor.
                    tf.Tensor shape [k, tau, aDim, 1], dtype=dtype
                - timestep: the current timestep of the rollout.

            - output:
            ---------
                - noise at time timestep.
                    tf.Tensor Shape [k, aDim, 1]
        '''
        return noises[:, timestep]

    def shift(self, scope, actionSeq, init, length):
        # shapes: in [tau, aDim, 1], [x, aDim, 1], scalar;
        # out [tau-len + x, aDim, 1]
        remain = actionSeq[length:]
        return tf.concat([remain, init], axis=0)

    def get_next(self, scope, current, length):
        # shapes: in [tau, aDim, 1], out [length, aDim, 1]
        return current[:length]

    def init_zeros(self, scope, size):
        # shape: out [size, aDim, 1]
        return tf.zeros([size, self._aDim, 1], dtype=dtype)

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
        fake_input = self._model.fake_input()
        
        fake_sequence = np.zeros((self._tau, self._aDim, 1))

        _ = self._next_fct(tf.Variable(1, dtype=tf.int32),
                           fake_input,
                           fake_sequence,
                           self._normalizeCost)
        if not self._graphMode:
            warnings.warn("Not using graph mode, no trace to generate.")

    def profile(self):
        fake_state = np.zeros((self._sDim, 1))
        fake_state[6] = 1.
        fake_sequence = np.zeros((self._tau, self._aDim, 1))
        _ = self._next_fct(tf.Variable(1, dtype=tf.int32),
                           fake_state,
                           fake_sequence,
                           self._normalizeCost,
                           profile=True)
