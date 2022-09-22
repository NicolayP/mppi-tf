import tensorflow as tf
from utile import dtype, npdtype, push_to_tensor

import numpy as np
import scipy.signal

import time as t
import warnings

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu_devices[0], True)


class MPPIBase(tf.module):
    def __init__(
        self, model, cost, data,
        k=1, tau=1, lam=1., upsilon=1.,
        sigma=None, initSeq=None,
        norm=False, filter=False,
        graph=False,
        log=False, logPath=None,
        debug=False
    ):
        '''
        MPPI Controller constructor.

        inputs:
        -------
            - model: the predictive model. Must herit ModelBase.
            - cost: the cost funciton. Must herit from CostBase.
            - data: the data type used by the model. Must herit fomr DataType.
            - k: int, the number of samples.
            - tau: int, the prediction horizon.
            - lam: float, the inverse temperature (lambda).
            - upsilon: float, the augmented covariance for the noise.
            - sigma: np array of floats (defualt None). Noise covariance. Shape [aDim, aDim]
            - initSeq: np array of floats (dfault None). the inital action sequece.
            - norm: bool (default False). If true, normalizes the cost during the update.
            - filter: bool (default False). If true, filters the actoin sequence to have a
                smoother control signal.
            - graph: bool (default False). If true, the computational graph is traced and
                stored in GPU memory for faster inference.
            - log: bool (default False). If true, logs intermediate results in tensorboard.
            - logPath: string (default None). If log is true, then this indicates the log directory.
            - debug: bool (default False). If true, the controller is executed in debug mode.
        '''

        super(MPPIBase, self).__init__(name="MPPI_CONTROLLER")
        self._model, self._cost, self._data = model, cost, data
        self._k = tf.Variable(k, trainable=False, dtype=tf.int32, name="samples")
        self._tau = tau
        self._lam = tf.Variable(lam, trainable=False, dtype=dtype, name="lambda")
        self._ups = tf.Variable(upsilon, trainable=False, dtype=dtype, name="upsilon")
        
        if sigma is None:
            sigma = np.eye(data.aDim)
        self._sig = tf.Variable(sigma, trainable=False, dtype=dtype, name="sigma")
        
        if initSeq is None:
            initSeq = np.zeros(shape=(tau, data.aDim, 1))
        self._seq = tf.Variable(initSeq, trainable=False, dtype=dtype, name="sequence")

        self._norm, self._filt = norm, filter
        self._log, self._logPaht = log, logPath
        self._graph = graph
        self._debug = debug

        if graph:
            self._next_fct = tf.function(self.__internal__)
        else:
            self._next_fct = self.__internal__

        self._timingDict = {}
        self._timingDict['total'] = 0.
        self._timingDict['calls'] = 0

    def __call__(self, x):
        '''
        Calls the controller.

        inputs:
        -------
            - x: DataType, the data type used by the predictive model.
        
        outputs:
        --------
            - act: np.array of floats. The action to be applied to the plant.
                shape: [aDim, 1]
            - trajs: np.array of floats. The generated trajectories.
                shape: [k, tau, sDim, 1]
            - weights: np.array of floats. The weights of every samples.
                shape: [k,]
        '''
        start = t.perf_counter()
        act, trajs, weights = self._next_fct(k=self._k, x=x, seq=self._seq, norm=self._norm)
        if self._filt:
            filteredSeq = self.filter_seq(self._seq.numpy())
            self._seq.assigne(filteredSeq)
        end = t.perf_counter()
        
        self._timingDict['total'] += end-start
        self._timingDict['calls'] += 1
        
        return np.squeeze(act.numpy()), np.squeeze(trajs.numpy()), np.squeeze(weights.numpy())

    def __internal__(self, k, x, seq, norm=False):
        '''
        Internal function used by the call method. Entry to algorithm 2.
        Can be traced and uploaded to the GPU for faster inference.

        inputs:
        -------
            - k: tf.Variable, the number of samples. Shape [1]
            - x: Datatype, the data input used by the predictive model.
            - seq: tf.Variable. The inital control sequence. Shape [Tau, aDim, 1]
            - norm: bool (default False). If true, the cost are normalized during the 
                computation of the update.

        outputs:
        --------
            - action: tensor. the next actoin to apply. Shape [aDim, 1]
            - trajs: tensor. The generated trajectories. Shape [k, tau, aDim, 1].
            - weights: tensor. The weights associated with every traj.
            - seq: tf.Variable. The updated action sequence.
        '''
        with tf.name_scope("Mppi") as mppi:
            with tf.name_scope("noise") as n:
                noises = self.noise(n, k=k)
            with tf.name_scope("rollout") as r:
                costs, trajs = self.rollout(r, k=k, x=x, noises=noises, seq=seq)
            with tf.name_scope("update") as u:
                update, weights = self.update(u, costs=costs, noises=noises, seq=seq norm=norm)
            with tf.name_scope("next") as n:
                next = update[:1]
            with tf.name_scope("shift") as s:
                init = tf.zeros([1, self._aDim, 1], dtype=dtype)
                seq.assign(push_to_tensor(update, init, 1))
        
        #self._observer.write_control("state", model_input)
        #self._observer.write_control("next", next)
        
        return next, trajs, weights

    def noise(self, scope, k):
        '''
        Generates the noise tensor used for the rollout.

        inputs:
        -------
            - scope: string, the tensorflow scope.
            - k: tf.Variables. The number of samples. Shape [1].
        
        outputs:
        --------
            - the noise tensor. Shape [k, tau, aDim, 1]
        '''
        rng = tf.random.normal(
            shape=(k, self._tau, self._aDim, 1),
            mean=0., stddev=1.,
            dtype=dtype, seed=1
        )

        return tf.linalg.matmul(self._ups*self._sig, rng)

    def rollout(self, scope, k, x, noises, seq):
        raise NotImplementedError

    def update(self, scope, costs, noises, seq, norm=False):
        '''
        Computed the importance sampling update. Called after rollout in algorithm 2.

        inputs:
        -------
            - scope: string, the tensorflow scope.
            - cost: tensor, the cost associated to every samples. Shape [k, 1, 1].
            - noises: tensor, the noise tensor used to generate the samples.
                Shape [k, tau, aDim, 1].
            - norm: bool (default false). If enabled, computes the norm of the cost.
        
        outputs:
        --------
            - update: tensor, the update to apply to the action sequence. shape [tau, aDim, 1]
            - weights. The weights of every samples. shape [k, 1, 1]
        '''
        with tf.name_scope("Beta"):
            beta = self.beta(scope, costs)
        with tf.name_scope("Expodential_arg"):
            arg = self.norm_arg(scope, costs, beta, norm=norm)
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
            rawUpdate = tf.add(seq, weighted_noises)
            #update = self.clip_act("clipping", rawUpdate)
            update = rawUpdate

        #self._observer.write_control("weights", weights)
        #self._observer.write_control("nabla", nabla)
        #self._observer.write_control("arg", arg)
        #self._observer.write_control("weighted_noise", weighted_noises)
        #self._observer.write_control("update", update)

        return update, weights

    def beta(self, scope, cost):
        '''
        Computes the min value of the cost array.

        inputs:
        -------
            - scope: string, the tensorlfow scope.
            - cost: tensor with the samples costs. Shape [k, 1, 1]
        outputs:
        --------
            - Min of cost. Shape [1, 1]
        '''
        return tf.reduce_min(cost, 0)

    def norm_arg(self, scope, cost, beta, norm=False):
        '''
        Sets the lowest cost to 0 and subtract that value from all other costs.
        if norm is true, also normalized the cost between [0, 1]

        inputs:
        -------
           - scope: string, the tensorflow scope.
           - cost: tensor containing the samples cost. Shape [k, 1, 1].
           - beta: tensor min of the cost. Shape [1, 1].
           - norm: bool. Default false. If true, the cost will be normalized.
        outputs:
        --------
            - Shifted (and normalized) costs. Shape [k, 1, 1].
        '''
        shift = tf.math.subtract(cost, beta)
        if norm:
            max = tf.reduce_max(shift, 0)
            return tf.divide(shift, max)
        return shift

    def exp_arg(self, scope, arg):
        '''
        Compute the argument of the exponential function $\frac{-1}{\lambda}*arg$ (argument of eq 25)
        
        inputs:
        -------
            - scope: string, the tensorflow scope.
            - arg: tensor, to be mutiplied by -1/lambda. Shape [k, 1, 1]
        
        outputs:
        --------
            - argument for the exp. Shape [k, 1, 1]
        '''
        return tf.math.multiply(-1./self._lam, arg)

    def exp(self, scope, arg):
        '''
        Compute the exponential of eq 25.

        inputs:
        -------
            - scope: string, the tensorflow scope.
            - arg: tensor, the exponential argument. Shape [k, 1, 1]

        outputs:
        --------
            - the exponential. Shape [k, 1, 1].
        '''
        return tf.math.exp(arg)

    def nu(self, scope, arg):
        '''
        Computes the normalizing factor for every samples. (eq 25)
        
        inputs:
        -------
            - scope: string, the tensorflow scope.
            - arg: tensor contraining every sample costs. Shape [k, 1, 1]
        
        outputs:
        --------
            - $nu$ The normalizing term. Shape [1, 1]
        '''
        return tf.math.reduce_sum(arg, 0)

    def weights(self, scope, arg, nu):
        '''
        Compute the weights according to eq 24.

        inputs:
        -------
            - scope: string, the tensorflow scope.
            - arg: tensor, The exponenetial results in eq 24. Shape [k, 1, 1]
            - nu: tensor, the normalization term computed in eq 25. Shape [1, 1]
        
        outputs:
        --------
            - weights: tensor, the weights $\omega(v)$ (eq 23) used for the importance
                sampling. Shape [k, 1, 1]
        '''
        return tf.realdiv(arg, nu)

    def weighted_noise(self, scope, weights, noises):
        '''
        Computes the update term used to update the action sequence (eq 26).

        inputs:
        -------
            - scope: string, the tensorflow scope.
            - weights: tensor, the weights corresponding to the samples costs.
                shape [k, 1, 1].
            - noises: tensor, the noise matrix used for the rollouts.
                shape [k, tau, aDim, 1].

        outputs:
        --------
            - weighted_noise: tensor. The noise weighted according to their 
                contribution. Shape [k, 1, 1].
        '''
        return tf.math.reduce_sum(
                tf.math.multiply(
                    weights[..., None],
                    noises),
                0)

    def filter_seq(seq):
        '''
        Filters the sequence with a savgol_filter as in Graddy's thesis.

        inputs:
        ------
            - seq: np.array of floats. The action sequence. Shape: [Tau, aDim, 1]
        
        outputs:
        --------
            - the filtered sequence. Shape [Tau, aDim, 1]
        '''

        filtered = scipy.signal.savgol_filter(
            seq[..., 0], 
            window_length=5, polyorder=3,
            deriv=0, delta=1.0, axis=0
        )[..., None]
        return filtered


class MPPIClassic(MPPIBase):
    def __init__(
        self, model, cost, data,
        k=1, tau=1, lam=1, upsilon=1,
        sigma=None, initSeq=None,
        norm=False, filter=False,
        graph=False,
        log=False, logPath=None,
        debug=False):
        super(MPPIClassic, self).__init__(
            model, cost, data,
            k, tau, lam, upsilon,
            sigma, initSeq,
            norm, filter,
            graph,
            log, logPath,
            debug)

    def rollout(self, scope, k, x, noises, seq):
        with tf.name_scope("setup") as s:
            cost = tf.zeros(shape=(k, 1, 1), dtype=dtype)
            trajs = []
        with tf.name_scope("rollout"):
            for i in range(self._tau):
                with tf.name_scope("prep_data"):
                    action = seq[i]
                    noise = noises[:, i]
                    toApply = tf.add(action, noise)
                    x.prep(toApply)
                with tf.name_scope(f"step_{i}") as s:
                    nextState = self._model(x)
                    trajs = tf.concat([trajs, nextState[:, None, ...]], axis=1)
                with tf.name_scope(f"cost_{i}") as c:
                    tmp = self._cost.step(c, nextState, action, noise)
                    cost = tf.add(cost, tmp)
                with tf.name_scope(f"state_update_{i}") as s:
                    x.add_state(nextState)
                    x.add_action(toApply)
        with tf.name_scope("Terminal_cost") as tc:
            fCost = self._cost.final(c, nextState)
        with tf.name_scope("Rollout_cost") as rc:
            samplesCost = tf.add(fCost, cost)
        
        self._observer.write_control("Sample_cost", samplesCost)
        return samplesCost, trajs


class DMDMPPI(MPPIBase):
    def __ini__(self):
        pass

    def rollout(self, scope, k, x, noises, seq):
        pass