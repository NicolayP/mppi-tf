from types import DynamicClassAttribute
import tensorflow as tf
from .model_base import ModelBase
import numpy as np
import os

class NNModel(ModelBase):
    '''
        Neural network based model class.
    '''
    # PUBLIC
    def __init__(self,
                 stateDim=2,
                 actionDim=1,
                 k=tf.Variable(1),
                 name="nn_model",
                 inertialFrameId="world",
                 weightFile=None):
        '''
            Neural network model constructor.

            - input:
            --------
                - stateDim: int the state space dimension.
                - actionDim: int the action space dimension.
                - name: string the model name.
        '''

        ModelBase.__init__(self,
                           stateDim=stateDim,
                           actionDim=actionDim,
                           k=k,
                           name=name,
                           inertialFrameId=inertialFrameId)
        tf.keras.backend.set_floatx('float64')
        self.nn = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='linear',
                                  input_shape=(stateDim+actionDim-3,)),
            tf.keras.layers.Dense(32, activation='linear'),
            tf.keras.layers.Dense(32, activation='linear'),
            tf.keras.layers.Dense(stateDim, activation='linear')
        ])

        if weightFile is not None:
            self.load_params(weightFile)
        
        self.Xmean = np.zeros(shape=(stateDim+actionDim-3))
        self.Xstd = np.ones(shape=(stateDim+actionDim-3))

        self.Ymean = np.zeros(shape=(stateDim))
        self.Ystd = np.ones(shape=(stateDim))

    def set_Xmean_Xstd(self, mean, std):
        '''
            Sets the mean and std of the data. This will be used as normalization
            parameters for the data. 

            input:
            ------
                - mean: The mean of the data for each input feature/axis 
                    shape: [15/16, 1]
                - std: the std of the data for each input feature/axis
                    shape: [15/16, 1] 
        '''
        self.Xmean = tf.constant(mean, dtype=tf.float64)
        self.Xstd = tf.constant(std, dtype=tf.float64)

    def set_Ymean_Ystd(self, mean, std):
        '''
            Sets the mean and std of the output. This will be used as normalization
            parameters for the data. 

            input:
            ------
                - mean: The mean of the data for each output feature/axis 
                    shape: [15/16, 1]
                - std: the std of the data for each output feature/axis
                    shape: [15/16, 1] 
        '''
        self.Ymean = tf.constant(mean, dtype=tf.float64)
        self.Ystd = tf.constant(std, dtype=tf.float64)

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

    def get_weights(self):
        ret = []
        for entry in self.nn.trainable_variables:
            ret.append(tf.identity(entry))
        return ret

    def weights(self):
        return self.nn.trainable_variables

    def update_weights(self, var, msg=True):
        if msg:
            newVariables = self.extract_variables(var)
        else:
            newVariables = var
        
        for oldVar, newVar in zip(self.nn.trainable_variables, newVariables):
            oldVar.assign(newVar)

    def save_params(self, path, step):
        path = os.path.join(path, "weights_step{}".format(step))
        self.nn.save(path)
    
    def load_params(self, path):
        self.nn = tf.keras.models.load_model(path)

    # PRIVATE
    def _extract_variables(self, var):
        vars = []
        for layer in var:
            weights = self._get_2d_tensor(layer.weights,layer.weights_name)
            bias = self._get_1d_tensor(layer.bias, layer.bias_name)
            vars.append(weights)
            vars.append(bias)
        
        return vars

    def _get_2d_tensor(self, tensor2d, name):
        tensor = []
        for row in tensor2d.tensor:
            tensor.append(row.tensor)
        tensor_np = np.array(tensor)
        tensor = tf.Variable(tensor_np,
                             True,
                             name=name,
                             dtype=tf.float64)
        return tensor

    def _get_1d_tensor(self, tensor1d, name):
        tensor = np.array(tensor1d.tensor)
        tensor = tf.Variable(tensor,
                             True,
                             name=name,
                             dtype=tf.float64)
        return tensor

    def _predict_nn(self, scope, input):
        return self.nn(input)

# TODO: CHECK difference between quaterion and euler.
# TODO: Is adding quaternions a smart idea in next_state?
class NNAUVModel(NNModel):
    '''
        Neural network representation for AUV model. Assumes that
        the model uses quaternion representation. The network predicts
        the next state expressed in the previous state body frame based on:
        the orientation of the body frame (for restoring forces), the velocity
        (in body frame) and the input forces.
    '''

    def __init__(self,
                 inertialFrameId="world",
                 k=tf.Variable(1),
                 stateDim=13,
                 actionDim=6,
                 mask=np.array([[[1], [1], [1],
                                 [0], [0], [0], [0],
                                 [0], [0], [0],
                                 [0], [0], [0]]]),
                 name="auv_nn_model",
                 weightFile=None):

        NNModel.__init__(self,
                         inertialFrameId=inertialFrameId,
                         stateDim=stateDim,
                         actionDim=actionDim,
                         name=name,
                         k=k,
                         weightFile=weightFile)
        
        self.mask = tf.constant(mask,
                                name="mask",
                                dtype=tf.float64)

    def build_step_graph(self, scope, state, action):
        '''
            Predicts the next step using the neural network model.

            Input:
            ------
                - scope: String, the tensorflow scope of the network
                - state: The current state of the system using quaternion
                representation. Shape [k, 12/13, 1].
                - action: The 6D force/torque tensor. Shape [k, 6, 1]


            Output:
            -------
                - nextState: The next state of the system.
                [X_pose_{t+1}_I.T, X_vel_{t+1}_B].T. Shape [k, 13, 1]
        '''
        state = tf.convert_to_tensor(state, dtype=tf.float64)
        with tf.name_scope(scope) as scope:
            x = self.prepare_data(state, action)
            normDelta = self._predict_nn("nn", x)
            delta = self.denormalize(normDelta)
            delta = tf.expand_dims(delta, axis=-1)
            nextState = self.next_state(state, delta)
        return nextState

    def prepare_training_data(self, stateT, stateT1, action, norm=True):
        '''
            Create new frame such that:
                - stateT's position is on the origin.
                - stateT1 is expressed in this new frame.
            It then creates the training data by computing deltaT from
            the two new representations of stateT and stateT1.

            Input:
            ------
                - stateT: state of the plant at time t.
                    Tensor with shape [k, 12/13, 1].
                - stateT1: state of the plant at time t+1.
                    Tensor with shape [k, 12/13, 1].
                - action: applied action on the plant at time t.
                    Tensor with shape [k, 6, 1]

            Output:
            -------
                - (X, y) pair containing:
                    X: The concatenation of the state and the action, 
                    without the position as it's set to 0.
                    Tensor with shape [k, 16]

                    y: The delta between stateT and stateT1.
                        shape: [k, 12/13]
        '''
        if not tf.is_tensor(stateT):
            stateT = tf.convert_to_tensor(stateT, dtype=tf.float64)
        
        if not tf.is_tensor(stateT1):
            stateT1 = tf.convert_to_tensor(stateT1, dtype=tf.float64)
        
        if not tf.is_tensor(action):
            action = tf.convert_to_tensor(action, dtype=tf.float64)

        tFrom = self.mask*stateT
        poseBIt = stateT - tFrom
        poseBIt1 = stateT1 - tFrom
        X = tf.squeeze(tf.concat([stateT[:, 3:], action], axis=1), axis=-1)
        Y = tf.squeeze(poseBIt1 - poseBIt, axis=-1)

        if norm:
            X = (X-self.Xmean)/self.Xstd
            Y = (Y-self.Ymean)/self.Ystd

        return (X, Y)
    
    def prepare_data(self, state, action):
        data = tf.concat([state[:, 3:], action], axis=1)
        data = tf.squeeze(data, axis=-1)
        data = (data-self.Xmean)/self.Xstd
        return data

    def denormalize(self, normDelta):
        norm = normDelta*self.Ystd + self.Ymean
        return norm

    def next_state(self, state, delta):
        return tf.add(state, delta)
