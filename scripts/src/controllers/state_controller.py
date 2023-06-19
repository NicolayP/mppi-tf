import tensorflow as tf

from ..misc.utile import assert_shape, push_to_tensor, dtype
from .controller_base import ControllerBase
from ..observer.observer_base import ObserverBase
import numpy as np


class StateModelController(ControllerBase):
    def __init__(
        self, model, cost, k=1, tau=1, sDim=1, aDim=1, 
        lam=1., upsilon=1., sigma=np.array([]), initSeq=np.array([]),
        normalizeCost=False, filterSeq=False,
        configDict=None, taskDict=None, modelDict=None,
        log=False, logPath=None, graphMode=False, debug=False
    ):
        super(StateModelController, self).__init__(
            model=model, cost=cost, k=k, tau=tau, sDim=sDim, aDim=aDim,
            lam=lam, upsilon=upsilon, sigma=sigma, initSeq=initSeq,
            normalizeCost=normalizeCost,  filterSeq=filterSeq,
            configDict=configDict, taskDict=taskDict, modelDict=modelDict,
            log=log, logPath=logPath, graphMode=graphMode, debug=debug
        )
        self.set_observer(ObserverBase(
            logPath=logPath, log=log, debug=debug,
            k=k, tau=tau, lam=lam,
            configDict=configDict,
            taskDict=taskDict,
            modelDict=modelDict,
            aDim=aDim, sDim=sDim, modelName=self._model.get_name()))
        if self._log:
            self._tracing = True
            self._observer.save_graph(self._next_fct, self._graphMode)
            self._tracing = False

    def predict(self, model_input, actionSeq, xNext):
        x, u = model_input
        # TODO: get first predicted next state, predict action sequence,
        # compute diff between first next state and next state
        # log error on prediction, predicted cost, action input, state input.
        nextState = self._model.predict(tf.expand_dims(x, axis=0),
                                        tf.expand_dims(u, axis=0))
        error = self.state_error(xNext, nextState)
        dist = self._cost.dist(x)
        costPred = self._cost.state_cost("predict", nextState)

        state = nextState

        for i in range(actionSeq.shape[0]):
            with tf.name_scope("Prepare_data_" + str(i)) as pd:
                action = tf.expand_dims(self.prepare_action(pd, actionSeq, i), axis=0)
            with tf.name_scope("Step_" + str(i)) as s:
                nextState = self._model.build_step_graph(s, state, action)
            with tf.name_scope("Cost_" + str(i)) as c:
                tmp = self._cost.state_cost(c, nextState)
                costPred = self._cost.add_cost(c, tmp, costPred)
            state = nextState

        with tf.name_scope("terminal_cost") as s:
            fCost = self._cost.build_final_step_cost_graph(s, nextState)
        with tf.name_scope("Rollout_cost"):
            sampleCosts = self._cost.add_cost(c, fCost, costPred)

        self._observer.write_predict("predicted/next_state", xNext)
        self._observer.write_predict("predicted/state", x)
        self._observer.write_predict("predicted/error", error)
        self._observer.write_predict("predicted/dist", dist)
        self._observer.write_predict("predicted/sample_cost", sampleCosts)

        return nextState

    def build_model(self, scope, k, model_input, noises, actionSeq):
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
                - sample costs. Tensor containing the cost of
                every samples. tf.Tensor, shape [k, 1, 1]
        '''
        state = model_input
        with tf.name_scope("setup") as setup:
            state = tf.expand_dims(state, axis=0)
            state = tf.broadcast_to(state,
                                    [k, self._sDim, 1],
                                    name="Broadcast_inital_state")
            nextState = tf.zeros(shape=tf.shape(state), dtype=dtype)
            cost = tf.zeros(shape=(k, 1, 1), dtype=dtype)
            trajs = state[:, None]

        # PAY ATTENTION TO THE FOR LOOPS WITH @tf.function.
        for i in range(self._tau):
            with tf.name_scope("Rollout_" + str(i)):
                with tf.name_scope("Prepare_data_" + str(i)) as pd:
                    action = self.prepare_action(pd, actionSeq, i)
                    noise = self.prepare_noise(pd, noises, i)
                    toApply = tf.add(action, noise, name="toApply")                    
                with tf.name_scope("Step_" + str(i)) as s:
                    nextState = self._model.build_step_graph(s,
                                                             state,
                                                             toApply)
                    trajs = tf.concat([trajs, nextState[:, None]], axis=1)
                with tf.name_scope("Cost_" + str(i)) as c:
                    tmp = self._cost.build_step_cost_graph(c, nextState,
                                                           action, noise)
                    cost = self._cost.add_cost(c, cost, tmp)
                state = nextState
            state = nextState

        with tf.name_scope("terminal_cost") as s:
            fCost = self._cost.build_final_step_cost_graph(s, nextState)
        with tf.name_scope("Rollout_cost"):
            sampleCosts = self._cost.add_cost(c, fCost, cost)
            
        self._observer.write_control("sample_costs", sampleCosts)
        return sampleCosts, trajs
