from types import DynamicClassAttribute
import tensorflow as tf
from .model_base import ModelBase
import numpy as np
import os

class NNModel(ModelBase):
    '''
        Neural network based model class.
    '''
    def __init__(self,
                 stateDim=2,
                 actionDim=1,
                 k=tf.Variable(1),
                 name="nn_model",
                 inertialFrameId="world"):
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
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(stateDim+actionDim-3,)),
            tf.keras.layers.Dense(10, activation=tf.nn.relu),
            tf.keras.layers.Dense(stateDim)
        ])

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

    def predict_nn(self, scope, input):
        return tf.expand_dims(self.model(input), axis=-1)

    # Define a new train function for NN using keras.
    def train_step(self, gt, x, a, step=None, writer=None, log=False):
        with tf.GradientTape() as tape:
            loss = self.build_loss_graph(gt, x, a)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self._optimizer.apply_gradients(zip(grads,
                                            self.model.trainable_variables))
        return loss

    def get_var(self):
        return self.model.trainable_variables

    def set_var(self, var):
        newVariables = self.extract_variables(var)
        for oldVar, newVar in zip(self.model.trainable_variables, newVariables):
            oldVar = newVar

    def extract_variables(self, var):
        vars = []
        for layer in var:
            weights = self.get_2d_tensor(layer.weights,layer.weights_name)
            bias = self.get_1d_tensor(layer.bias, layer.bias_name)
            vars.append(weights)
            vars.append(bias)
        
        return vars

    def get_2d_tensor(self, tensor2d, name):
        tensor = []
        for row in tensor2d.tensor:
            tensor.append(row.tensor)
        tensor_np = np.array(tensor)
        tensor = tf.Variable(tensor_np,
                             True,
                             name=name,
                             dtype=tf.float64)
        return tensor

    def get_1d_tensor(self, tensor1d, name):
        tensor = np.array(tensor1d.tensor)
        tensor = tf.Variable(tensor,
                             True,
                             name=name,
                             dtype=tf.float64)
        return tensor

    def save_params(self, path, step):
        file = os.path.join(path, "weights_step{}.keras".format(step))
        self.model.save(file)
    
    def load_params(self, path):
        file = os.path.join(path, "weights.keras")
        self.model = tf.keras.models.load_model(file)


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
                 name="auv_nn_model"):

        NNModel.__init__(self,
                         inertialFrameId=inertialFrameId,
                         stateDim=stateDim,
                         actionDim=actionDim,
                         name=name,
                         k=k)
        
        self.mask = tf.constant([[[1],
                                  [1],
                                  [1],
                                  [0],
                                  [0],
                                  [0],
                                  [0],
                                  [0],
                                  [0],
                                  [0],
                                  [0],
                                  [0],
                                  [0]]],
                                name="mask",
                                dtype=tf.float64)

    def build_step_graph(self, scope, state, action):
        '''
            Predicts the next step using the neural network model.

            Input:
            ------
                - scope: String, the tensorflow scope of the network
                - state: The current state of the system using quaternion
                representation. Shape [k, 13, 1].
                - action: The 6D force/torque tensor. Shape [k, 6, 1]


            Output:
            -------
                - nextState: The next state of the system.
                [X_pose_{t+1}_I.T, X_vel_{t+1}_B].T. Shape [k, 13, 1]
        '''
        state = tf.convert_to_tensor(state, dtype=tf.float64)
        with tf.name_scope(scope) as scope:
            x = self.prepare_data(state, action)
            delta = self.predict_nn("nn", x)
            nextState = self.next_state(state, delta)
        return nextState

    def prepare_training_data(self, stateT, stateT1, action):
        tFrom = self.mask*stateT
        poseBIt = stateT - tFrom
        poseBIt1 = stateT1 - tFrom
        X = tf.concat([stateT[:, 3:-1], action], axis=1)
        Y = poseBIt1 - poseBIt
        return (X, Y)
    
    def prepare_data(self, state, action):
        data = tf.concat([state[:, 3:13], action], axis=1)
        return tf.squeeze(data, axis=-1)

    def next_state(self, state, delta):
        return tf.add(state, delta)
