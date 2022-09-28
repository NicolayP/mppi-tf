import tensorflow as tf
from utile import dtype, append_to_tensor

class ModelDtype(tf.Module):
    def __init__(self, x, u, h=1):
        super(ModelDtype, self).__init__(name="ModelDtype")
        self.reset(x, u)
        self._h = h

    def __call__(self):
        return (self._x[:, -self._h:], self._u[:, -self._h:])

    def reset(self, x, u):
        self._x = tf.convert_to_tensor(x, dtype=dtype) #shape = [k, h, sDim, 1]
        self._u = tf.convert_to_tensor(u, dtype=dtype) #shape = [k, h, aDim, 1]

    def add(self, x=None, u=None):
        if x is not None:
            self._x = append_to_tensor(self._x, x)
        if u is not None:
            self._u = append_to_tensor(self._u, u)
    
    def traj(self):
        return self._x[:, self._h:]


class ModelBase(tf.Module):
    '''
        Model base class for the MPPI controller.
        Every model should inherit this class.
    '''
    def __init__(self,
                 k=1, dt=0.1,
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
        super(ModelBase, self).__init__(name="Predictive_model")
        self._k = tf.Variable(k, trainable=False, dtype=tf.int32, name="samples")
        self._inertialFrameId = inertialFrameId
        self._name = name

    def __call__(self, scope, x, u):
        '''
            Abstract method, need to be overwritten in child class.
            Step graph for the model. This computes the prediction
            for $hat{f}(x, u)$

            - input:
            --------
                - scope: String, the tensorflow scope name.
                - x: State tensor. Shape [k, h, sDim, 1]
                - u: Action tensor. Shape [k, h, aDim, 1]

            - output:
            ---------
                - the next state. Shape [k, sDim, 1]
        '''
        raise NotImplementedError

    def predict(self, x, u):
        '''
            Performs one step prediction for error visualisation.

            - input:
            --------
                - x: the state tensor. Shape [1, h, sDim, 1]
                - u: the action tensor. Shape [1, h, aDim, 1]

            - output:
            ---------
                - the predicted next state. Shape [1, sDim, 1]
        '''
        prev_k = self._k.numpy()
        self.set_k(1)
        pred = self("predict", x, u)
        self.set_k(prev_k)
        return pred

    def get_name(self):
        '''
            Get the name of the model.

            - output:
            ---------
                - String, the name of the model
        '''
        return self._name

    def set_k(self, k):
        self._k.assign(k)

class FooModel(ModelBase):
    def __init__(self, k=1, dt=0.1, name="foo"):
        super(FooModel, self).__init__(k=k, dt=dt, name=name)
    
    def __call__(self, scope, x, u):
        return x[:, -1]