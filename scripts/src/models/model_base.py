import tensorflow as tf


class ModelBase(tf.Module):
    '''
        Model base class for the MPPI controller.
        Every model should inherit this class.
    '''
    def __init__(self,
                 stateDim=2,
                 actionDim=1,
                 k=tf.Variable(1),
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
        self._k = k
        self._inertialFrameId = inertialFrameId
        self._stateDim = stateDim
        self._actionDim = actionDim
        self._modelVars = {}
        self._dt = dt
        self._optimizer = tf.optimizers.Adam(learning_rate=0.5)
        self._currentLoss = None
        self._name = name
        self._observer = None

    def add_model_vars(self, name, var):
        '''
            Add model variables to the dictionnary of variables.
            Used for logging and learning (if enabled)

            - input:
            --------
                - name: string. Unique variable name for identification.
                - var: the variable object.
                    Tensorflow variable with trainable enabled.

            - output:
            ---------
                None.
        '''
        self._modelVars[name] = var

    def build_loss_graph(self, gt, x, a):
        '''
            Computes the loss function for a given batch of samples.
            Can be overwritten by child classes.
            Standard is l2 loss function.

            - input:
            --------
                - gt: the ground truth tensor. Shape [batch_size, sDim, 1]
                - x: the previous state. Shape [batch_size, sDim, 1]
                - a: the action to apply. Shape [batch_size, aDim, 1]

            - output:
            ---------
                - the loss function between one step prediction
                "model(x, a)" and gt.
        '''

        pred = self.build_step_graph("train", x, a)
        return tf.reduce_mean(tf.math.squared_difference(pred, gt),
                              name="Loss")

    def is_trained(self):
        '''
            Tells whether the model is trained or not.

            - output:
            ---------
                - bool, true if training finished.
        '''
        if (self._currentLoss is not None) and self._currentLoss < 5e-5:
            return True
        return False

    # DEPRECATED: TODO: remove this function as the learner does all this.
    def train_step(self, gt, x, a, step=None, writer=None, log=False):
        '''
            Performs one step of training.

            - input:
            --------
                - gt. the ground truth tensor. Shape [batch_size, sDim, 1]
                - x. the input state tensor. Shape [batch_size, sDim, 1]
                - a. the input action tensor. Shape [batch_size, aDim, 1]
                - step. Int, The current learning step.
                - writer. Tensorflow summary writer.
                - log. bool. If true, logs learning info in tensorboard.

            - output:
            ---------
                None
        '''

        gt = tf.convert_to_tensor(gt, dtype=tf.float64)
        x = tf.convert_to_tensor(x, dtype=tf.float64)
        a = tf.convert_to_tensor(a, dtype=tf.float64)
        with tf.GradientTape() as tape:
            for key in self._modelVars:
                tape.watch(self._modelVars[key])
            self._currentLoss = self.build_loss_graph(gt, x, a)

        grads = tape.gradient(self._currentLoss,
                              list(self._modelVars.values()))

        self._optimizer.apply_gradients(list(
                                         zip(
                                          grads,
                                          list(self._modelVars.values()))))

        if log:
            with writer.as_default():
                for key in self._modelVars:
                    if tf.size(self._modelVars[key]).numpy() == 1:
                        tf.summary.scalar("training/{}".format(key),
                                          self._modelVars[key].numpy()[0, 0],
                                          step=step)
                    else:
                        tf.summary.histogram("training/{}".format(key),
                                             self._modelVars[key],
                                             step=step)

                tf.summary.scalar("training/loss",
                                  self._currentLoss.numpy(),
                                  step=step)

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

    def get_stats(self):
        raise NotImplementedError

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