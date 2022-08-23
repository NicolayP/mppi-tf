
import tensorflow as tf
import numpy as np

from ..misc.utile import assert_shape, push_to_tensor, dtype
from .controller_base import ControllerBase
from ..observer.observer_base import ObserverLagged


class LaggedModelController(ControllerBase):
    def __init__(
        self, model, cost, k=1, tau=1, sDim=1, aDim=1, h=1,
        lam=1., upsilon=1., sigma=np.array([]), initSeq=np.array([]),
        normalizeCost=False, filterSeq=False,
        configDict=None, taskDict=None, modelDict=None,
        log=False, logPath=None, graphMode=False, debug=False
    ):
        super(LaggedModelController, self).__init__(
            model=model, cost=cost, k=k, tau=tau, sDim=sDim, aDim=aDim,
            lam=lam, upsilon=upsilon, sigma=sigma, initSeq=initSeq,
            normalizeCost=normalizeCost,  filterSeq=filterSeq,
            configDict=configDict, taskDict=taskDict, modelDict=modelDict,
            log=log, logPath=logPath, graphMode=graphMode, debug=debug
        )
        self._h = h
        self._observer = ObserverLagged(
            logPath=logPath, log=log, debug=debug,
            k=k, tau=tau, lam=lam,
            configDict=configDict,
            taskDict=taskDict,
            modelDict=modelDict,
            h=h, aDim=aDim, sDim=sDim, modelName=self._model.get_name())
        if self._log:
            self._observer.save_graph(self._next_fct, self._graphMode)

    def predict(self, model_input, actionSeq, xNext):
        laggedX, laggedU = model_input
        laggedX = tf.expand_dims(laggedX, axis=0) # shape [1, history, sDim, 1]
        laggedU = tf.expand_dims(laggedU, axis=0) # shape [1, history, aDim, 1]
        nextState = self._model.predict(laggedX, laggedU)
        error = self.state_error(xNext, nextState)
        dist = self._cost.dist(laggedX[0, -1])

        self._observer.write_predict("predicted/next_state", nextState)
        self._observer.write_predict("predicted/predicted_error", error)
        self._observer.write_predict("predicted/dist_to_goal", dist)
        
        return nextState

    def build_model(self, scope, k, model_input, noises, actionSeq):
        '''
            model_input: (laggedState, laggedAction) pair
                laggedState, shape: [history, sDim, 1]
                laggedAction, shape: [history-1, aDim, 1]
        '''
        print(model_input)
        laggedState, laggedAction = model_input
        with tf.name_scope("setup") as setup:
            laggedState = tf.broadcast_to(
                laggedState[None, ...], #shape [1, history, sDim, 1]
                [k, self._h, self._sDim, 1],
                name="Broacast_inital_state"
            ) # shape [k, history, sDim, 1]
            laggedAction = tf.broadcast_to(
                laggedAction[None, ...], # shape [1, history-1, aDim, 1]
                [k, self._h-1, self._aDim, 1],
                name="Broadcast_inital_aciton"
            ) # shape [k, history-1, aDim, 1]

            cost = tf.zeros(shape=(k, 1, 1), dtype=dtype)
        with tf.name_scope("Rollout") as r:
            for i in range(self._tau):
                with tf.name_scope(f"Rollout_{i}"):
                    with tf.name_scope(f"Prepare_data_{i}") as pd:
                        action = self.prepare_action(pd, actionSeq, i) # shape [k, aDim, 1]
                        noise = self.prepare_noise(pd, noises, i) # shape [k, aDim, 1]
                        toApply = self.prepare_to_apply(pd, action, noise, laggedAction) # shape [k, history, aDim, 1]
                    with tf.name_scope(f"Step_{i}") as s:
                        nextState = self._model.build_step_graph(
                            s,
                            laggedState,
                            toApply
                        ) # shape [k, sDim, 1]
                    with tf.name_scope(f"Cost_{i}") as c:
                        tmp = self._cost.build_step_cost_graph(c, nextState, action, noise)
                        cost = self._cost.add_cost(c, cost, tmp)
                with tf.name_scope("State_update"):
                    laggedState = push_to_tensor(laggedState, nextState)
                    laggedAction = push_to_tensor(laggedAction, toApply[:, -1])
        
        with tf.name_scope("Terminal_cost") as tc:
            fCost = self._cost.build_final_step_cost_graph(tc, nextState)
        with tf.name_scope("Rollout_cost") as rc:
            samplesCost = self._cost.add_cost(rc, fCost, cost)
        
        self._observer.write_control("samples_cost", samplesCost)
        return samplesCost

    def prepare_to_apply(self, scope, action, noise, laggedAction):
        '''
            input:
            ------
                action: shape [aDim, 1]
                noise: shape [k, aDim, 1]
                laggedAction: shape [k, history-1, aDim, 1]
            output:
            -------
                laggedInput: shape [k, history, aDim, 1]
        '''
        tmp = tf.expand_dims(tf.add(action, noise, name=""), axis=1)
        act = tf.concat([laggedAction, tmp], axis=1) # shape [k, history, adim, 1]
        print("-"*10)
        print(laggedAction)
        print("-"*10)
        print(tmp)
        print("-"*10)
        print(act)
        return act