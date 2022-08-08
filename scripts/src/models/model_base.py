import tensorflow as tf


class ModelBase(tf.Module):
    '''
        Model base class for the MPPI controller.
        Every model should inherit this class.
    '''
    def __init__(self, modelDict,
                 stateDim=2,
                 actionDim=1,
                 k=1,
                 dt=0.1,
                 name="model",
                 inertialFrameId="world"):
        '''
            Model constructor.

            - input:
            --------
                - state_dim: int. the state space dimension.
                - action_dim: int. the action space dimension.
                - name: string. the model name. Used for logging.
        '''
        self._modelDict = modelDict
        self._k = self._k = tf.Variable(k,
                              trainable=False,
                              dtype=tf.int32,
                              name="samples")
        self._inertialFrameId = inertialFrameId
        self._stateDim = stateDim
        self._actionDim = actionDim
        self._modelVars = {}
        self._dt = dt
        self._optimizer = tf.optimizers.Adam(learning_rate=0.5)
        self._currentLoss = None
        self._name = name
        self._observer = None

    def build_step_graph(self, scope, state, action):
        '''
            Abstract method, need to be overwritten in child class.
            Step graph for the model. This computes the prediction
            for $hat{f}(x, u)$

            - input:
            --------
                - scope: String, the tensorflow scope name.
                - state: State tensor. Shape [k, sDim, 1]
                - action: Action tensor. Shape [k, aDim, 1]

            - output:
            ---------
                - the next state.
        '''
        raise NotImplementedError

    def predict(self, state, action):
        '''
            Performs one step prediction for error visualisation.

            - input:
            --------
                - state: the state tensor. Shape [1, sDim, 1]
                - action: the action tensor. Shape [1, aDim, 1]

            - output:
            ---------
                - the predicted next state. Shape [1, sDim, 1]
        '''
        self.set_k(1)
        return self.build_step_graph("predict", state, action)

    def get_name(self):
        '''
            Get the name of the model.

            - output:
            ---------
                - String, the name of the model
        '''
        return self._name

    def get_state_dim(self):
        return self._stateDim

    def get_action_dim(self):
        return self._actionDim

    def set_k(self, k):
        self._k.assign(k)

    def set_observer(self, observer):
        self._observer = observer

    def save_params(self, path, step):
        raise NotImplementedError

    def load_params(self, path):
        raise NotImplementedError