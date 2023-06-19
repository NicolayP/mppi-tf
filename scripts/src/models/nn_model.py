import tensorflow_graphics.geometry.transformation as tfgt
import tensorflow as tf

from .model_utils import ToSE3Mat, SE3int, FlattenSE3
from .model_base import ModelBase
import numpy as np
import os
from ..misc.utile import dtype

# Gets rid of the error triggered when running
# tfg in graph mode.
import sys
module = sys.modules['tensorflow_graphics.util.shape']
def _get_dim(tensor, axis):
    """Returns dimensionality of a tensor for a given axis."""
    return tf.compat.v1.dimension_value(tensor.shape[axis])

module._get_dim = _get_dim
sys.modules['tensorflow_graphics.util.shape'] = module


class NNModel(ModelBase):
    '''
        Neural network based model class.
    '''
    # PUBLIC
    def __init__(self,
                 modelDict,
                 stateDim=2,
                 actionDim=1,
                 limMax=tf.ones(shape=(1,), dtype=tf.float64),
                 limMin=-tf.ones(shape=(1,), dtype=tf.float64),
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

        ModelBase.__init__(self, modelDict,
                           stateDim=stateDim,
                           actionDim=actionDim,
                           limMax=limMax,
                           limMin=limMin,
                           k=k,
                           name=name,
                           inertialFrameId=inertialFrameId)
        self.nn = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu',
                                  input_shape=(stateDim+actionDim-3,), dtype=dtype),
            tf.keras.layers.Dense(32, activation='relu', dtype=dtype),
            tf.keras.layers.Dense(32, activation='relu', dtype=dtype),
            tf.keras.layers.Dense(stateDim, activation='linear', dtype=dtype)
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
        self.Xmean = tf.constant(mean, dtype=dtype)
        self.Xstd = tf.constant(std, dtype=dtype)

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
        self.Ymean = tf.constant(mean, dtype=dtype)
        self.Ystd = tf.constant(std, dtype=dtype)

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
            newVariables = self._extract_variables(var)
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
                             dtype=dtype)
        return tensor

    def _get_1d_tensor(self, tensor1d, name):
        tensor = np.array(tensor1d.tensor)
        tensor = tf.Variable(tensor,
                             True,
                             name=name,
                             dtype=dtype)
        return tensor

    def _predict_nn(self, scope, X):
        return self.nn(X)

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
                 modelDict,
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

        NNModel.__init__(self, modelDict,
                         inertialFrameId=inertialFrameId,
                         stateDim=stateDim,
                         actionDim=actionDim,
                         limMax=limMax,
                         limMin=limMin,
                         name=name,
                         k=k,
                         weightFile=weightFile)
        
        self.mask = tf.constant(mask,
                                name="mask",
                                dtype=dtype)

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
        state = tf.convert_to_tensor(state, dtype=dtype)
        with tf.name_scope(scope) as scope:
            x = self.prepare_data(state, action)
            normDelta = self._predict_nn("nn", x)
            delta = self.denormalizeY(normDelta)
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
            stateT = tf.convert_to_tensor(stateT, dtype=dtype)
        
        if not tf.is_tensor(stateT1):
            stateT1 = tf.convert_to_tensor(stateT1, dtype=dtype)
        
        if not tf.is_tensor(action):
            action = tf.convert_to_tensor(action, dtype=dtype)

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

    def denormalizeY(self, normY):
        norm = normY*self.Ystd + self.Ymean
        return norm
    
    def denormalizeX(self, normX):
        norm = normX*self.Xstd + self.Xmean
        return norm

    def next_state(self, state, delta):
        return tf.add(state, delta)


class NNAUVModelSpeed(NNAUVModel):
    '''
        Neural network representation for AUV model. Assumes that
        the model uses quaternion representation. The network predicts
        the next state expressed in the previous state body frame based on:
        the orientation of the body frame (for restoring forces), the velocity
        (in body frame) and the input forces.
    '''

    def __init__(self,
                 modelDict,
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

        NNAUVModel.__init__(self, modelDict,
                            inertialFrameId=inertialFrameId,
                            stateDim=stateDim,
                            actionDim=actionDim,
                            mask=mask,
                            name=name,
                            k=k,
                            weightFile=weightFile)

        self.nn = tf.keras.Sequential([
            tf.keras.layers.Dense(16, activation='relu', kernel_regularizer='l2',
                                  input_shape=(stateDim+actionDim-3 -1,), name="dense1"),
            tf.keras.layers.Dense(16, activation='relu', kernel_regularizer='l2', name="dense2"),
            tf.keras.layers.Dense(16, activation='relu', kernel_regularizer='l2', name="dense3"),
            tf.keras.layers.Dense(6, activation='linear', name="dense4")
        ])

        if weightFile is not None:
            self.load_params(weightFile)
        
        # Input is (state + aciton) - position 
        # and rotation expressed in euler representation 
        # so stateDim + actionDim - 3 (postion) - 1 (quat2Euler)
        self.Xmean = np.zeros(shape=(stateDim + actionDim - 3 - 1))
        self.Xstd = np.ones(shape=(stateDim + actionDim - 3 - 1))

        self.Ymean = np.zeros(shape=(6))
        self.Ystd = np.ones(shape=(6))

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
        state = tf.convert_to_tensor(state, dtype=dtype)
        with tf.name_scope(scope) as scope:
            x = self.prepare_data(state, action)
            normVelDelta = self._predict_nn("nn", x)
            velDelta = self.denormalizeY(normVelDelta)
            velDelta = tf.expand_dims(velDelta, axis=-1)
            nextState = self.next_state(state, velDelta)

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
                    Tensor with shape [k, 13, 1].
                - stateT1: state of the plant at time t+1.
                    Tensor with shape [k, 13, 1].
                - action: applied action on the plant at time t.
                    Tensor with shape [k, 6, 1]

            Output:
            -------
                - (X, y) pair containing:
                    X: The concatenation of the state and the action, 
                    without the position as it's set to 0. and the 
                    rotation expressed in euler representation.
                    Tensor with shape [k, 15]

                    y: The delta between stateT and stateT1.
                        shape: [k, 6]
        '''
        if not tf.is_tensor(stateT):
            stateT = tf.convert_to_tensor(stateT, dtype=dtype)
        
        if not tf.is_tensor(stateT1):
            stateT1 = tf.convert_to_tensor(stateT1, dtype=dtype)
        
        if not tf.is_tensor(action):
            action = tf.convert_to_tensor(action, dtype=dtype)

        tFrom = self.mask*stateT
        poseBIt = stateT - tFrom
        poseBIt1 = stateT1 - tFrom

        stateEuler = self.to_euler(stateT)

        X = tf.squeeze(tf.concat([stateEuler[:, 3:], action],
                                 axis=1),
                       axis=-1)
        Y = tf.squeeze((poseBIt1 - poseBIt)[:, 7:], axis=-1)

        if norm:
            X = (X-self.Xmean)/self.Xstd
            Y = (Y-self.Ymean)/self.Ystd

        return (X, Y)
    
    def prepare_data(self, state, action):
        '''
            Prepares the data before feeding it to the network. 
            Changes the state to the euler representation and 
            appends the action vector a the end of it.

            - Input:
            --------
                - state: The state of the system in quaternion
                    representation. Shape [k, 13, 1].

                - action: The action applied on the system.
                    Shape [k, 6, 1].
            
            - Output:
            ---------
                - the input vector for the network.
                    shape [k, 15, 1]
        '''
        stateEuler = self.to_euler(state)
        X = tf.concat([stateEuler[:, 3:], action], axis=1)
        X = tf.squeeze(X, axis=-1)
        X = (X-self.Xmean)/self.Xstd
        return X

    def next_state(self, state, delta):
        pose = state[:, 0:7, :]
        speed = state[:, 7:13, :]
        pDot = tf.matmul(self.get_jacobian(state), speed)
        nextPose = pose + pDot*self._dt
        nextPose = self.normalize_quat(nextPose)
        nextVel = speed + delta
        return tf.concat([nextPose, nextVel], axis=1)

    def normalize_quat(self, pose):
        '''
            Normalizes the quaternions.

            input:
            ------
                - pose. Float64 Tensor. Shape [k, 13, 1]

            ouput:
            ------
                - the pose with normalized quaternion. Float64 Tensor.
                    Shape [k, 13, 1]
        '''

        pos = pose[:, 0:3]
        quat = tf.squeeze(pose[:, 3:7], axis=-1)
        vel = pose[:, 7:13]

        quat = tf.math.l2_normalize(quat, axis=-1)
        quat = tf.expand_dims(quat, axis=-1)
        #quat = tf.divide(quat, tf.linalg.norm(quat, axis=1, keepdims=True))
        pose = tf.concat([pos, quat, vel], axis=1)
        return pose

    def get_jacobian(self, state):
        '''
        Returns J(nu) in $mathbb{R}^{7 cross 7}$
                     ---------------------------------------
            J(nu) = | q_{n}^{b}(Theta) 0^{3 cross 3}    |
                     | 0^{3 cross 3} T_{theta}(theta)   |
                     ---------------------------------------
        '''
        k = state.shape[0]
        OPad3x3 = tf.zeros(shape=(k, 3, 3), dtype=dtype)
        OPad4x3 = tf.zeros(shape=(k, 4, 3), dtype=dtype)
        rotBtoI, TBtoIquat = self.body2inertial_transform(state)
        jacR1 = tf.concat([rotBtoI, OPad3x3], axis=-1)

        jacR2 = tf.concat([OPad4x3, TBtoIquat], axis=-1)
        jac = tf.concat([jacR1, jacR2], axis=1)

        return jac

    def body2inertial_transform(self, pose):
        '''
            Computes the rotational transform from
            body to inertial Rot_{n}^{b}(q)
            and the attitude transformation T_{q}(q).

            input:
            ------
                - pose the robot pose expressed in inertial frame.
                    Shape [k, 7, 1]

        '''
        quat = pose[:, 3:7, :]
        w = quat[:, 3]
        x = quat[:, 0]
        y = quat[:, 1]
        z = quat[:, 2]

        r1 = tf.expand_dims(tf.concat([1 - 2 * (tf.pow(y, 2) + tf.pow(z, 2)),
                                       2 * (x * y - z * w),
                                       2 * (x * z + y * w)], axis=-1),
                            axis=1)

        r2 = tf.expand_dims(tf.concat([2 * (x * y + z * w),
                                       1 - 2 * (tf.pow(x, 2) + tf.pow(z, 2)),
                                       2 * (y * z - x * w)], axis=-1),
                            axis=1)

        r3 = tf.expand_dims(tf.concat([2 * (x * z - y * w),
                                       2 * (y * z + x * w),
                                       1 - 2 * (tf.pow(x, 2) + tf.pow(y, 2))],
                                      axis=-1),
                            axis=1)

        rotBtoI = tf.concat([r1, r2, r3], axis=1)

        r1t = tf.expand_dims(tf.concat([-x, -y, -z], axis=-1), axis=1)

        r2t = tf.expand_dims(tf.concat([w, -z, y], axis=-1), axis=1)

        r3t = tf.expand_dims(tf.concat([z, w, -x], axis=-1), axis=1)

        r4t = tf.expand_dims(tf.concat([-y, x, w], axis=-1), axis=1)

        TBtoIquat = 0.5 * tf.concat([r1t, r2t, r3t, r4t], axis=1)

        return rotBtoI, TBtoIquat

    def to_euler(self, stateQ):
        '''
            Converts a state from quaternion representation to
            a euler representation.

            - input:
            --------
                - stateQ, the quaternion representation of the 
                state. shape [k, 13, 1]
            
            - output:
            ---------
                - stateEuler, the euler representaiton of the state.
                shape [k, 12, 1]
        '''
        samples = stateQ.shape[0]
        pos = stateQ[:, 0:3, :]
        quats = tf.squeeze(stateQ[:, 3:7, :], axis=-1)
        euler = tfgt.euler.from_quaternion(quats)
        
        euler = tf.expand_dims(euler, axis=-1)

        vel = stateQ[:, 7:, :]
        state_euler = tf.concat([pos, euler, vel], axis=1)
        return state_euler


class VelPred(tf.Module):
    def __init__(self, in_size=21, topology=[64]):
        self.n = "nn-veloctiy"
        bias = False
        layers = [tf.keras.layers.InputLayer(shape=(in_size,))]
        for i, s in enumerate(topology):
            if i == 0:
                self.n += f"_{in_size}"
                layer = tf.keras.layers.Dense(s, use_bias=bias)
            else:
                self.n += f"x{topology[i-1]}"
                layer = tf.keras.layers.Dense(s, use_bias=bias)
            layers.append(layer)
            layers.append(tf.keras.layers.LeakyReLU(alpha=0.1))
        
        self.n += f"x{topology[-1]}x6"
        layer = tf.keras.layers.Dense(6, activation="linear", use_bias=bias)
        layers.append(layer)
        self.nn = tf.keras.Sequential(layers)

    def forward(self, x, u):
        return self.nn(tf.concat((x, u), axis=1))


class Predictor(tf.Module):
    def __init__(self, internal, dt, h=1, onnx=False):
        self.internal = internal
        self.int = SE3int()
        self.toMat = ToSE3Mat()
        self.flat = FlattenSE3()
        self.dt = dt
        if onnx:
            self.n = "Predictor_onnx_model"
        else:
            self.n = f"Predictor_{self.internal.n}"
        self.h = h
        self.onnx = onnx

    def forward(self, x, u):
        k = tf.shape(x)[0]
        x_red = x[:, :, 3:]
        x_flat = tf.reshape(x_red, (k, self.h*15))
        u_flat = tf.reshape(u, (k, self.h*6))

        if self.onnx:
            pred = self.internal(x=x_flat, u=u_flat)['vel']
        else:
            pred = self.internal.forward(x_flat, u_flat)

        v_next = tf.cast(pred, dtype=dtype, name="casting_output")
        x_last = x[:, -1]
        v = x[:, -1, 12:]
        tau = v*self.dt
        M = self.toMat.forward(x_last)
        M = self.int.forward(M, tau)
        nextState = self.flat.forward(M, v_next)
        return nextState


class LaggedNNAUVSpeed(ModelBase):
    def __init__(self, k, h, dt, sDim=18, aDim=6, topology=[128, 128, 128], velPred=None):
        super(LaggedNNAUVSpeed, self).__init__(
            {}, stateDim=sDim, actionDim=aDim, k=k, dt=dt
        )
        onnx = False
        if velPred is not None:
            self.velPred = velPred
            onnx = True
        else:
            self.velPred = VelPred(in_size=h*(sDim-3+aDim), topology=topology)

        self.pred = Predictor(self.velPred, dt=dt, h=h, onnx=onnx)

    def build_step_graph(self, scope, state, action):
        state = tf.squeeze(state, axis=-1)
        action = tf.squeeze(action, axis=-1)
        foo = self.pred.forward(state, action)
        return foo[..., None]
