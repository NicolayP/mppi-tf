import tensorflow as tf
from model_base import ModelBase


class NNModel(ModelBase):
    '''
        Neural network based model class.
    '''
    def __init__(self,
                 stateDim=2,
                 actionDim=1,
                 k=1,
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
                           stateDim,
                           actionDim,
                           k,
                           name,
                           inertialFrameId)

        self._leakyReluApha = 0.2
        self.initializer = tf.initializers.glorot_uniform()
        self.addModelVars("first",
                          self.get_weights((stateDim+actionDim, 5),
                                           "first"))

        self.addModelVars("second",
                          self.get_weights((5, 5),
                                           "second"))

        self.addModelVars("final",
                          self.get_weights((5, stateDim),
                                           "final"))

    def build_step_graph(self, scope, state, action):
        '''
            Abstract method, need to be overwritten in child class.
            Step graph for the model. This computes the prediction
            for $hat{f}(x, u)$

            - input:
            --------
                - scope: String, the tensorflow scope name.
                - state: State tensor. Shape [k, s_dim, 1]
                - action: Action tensor. Shape [k, a_dim, 1]

            - output:
            ---------
                - the next state.
        '''
        # expand and broadcast state vector to match dim of action
        sshape = state.shape
        ashape = action.shape

        if len(sshape) < 3 and len(ashape) == 3:
            state = tf.broadcast_to(state,
                                    [ashape[0],
                                     sshape[0],
                                     sshape[1]])

        input = tf.squeeze(tf.concat([state, action], axis=1), -1)
        return self.predict(scope, input)

    def predict(self, scope, input):

        init = self.dense(input, self._modelVars["first"])
        second = self.dense(init, self._modelVars["second"])
        return tf.expand_dims(self.final(second,
                                         self._modelVars["final"]), -1)

    def final(self, inputs, weights):
        '''
            Computes the output layer of the neural network.

            - input:
            --------
                - inputs: the input tensor. Shape []
                - weights: the weights tensor. Shape []

            - output:
            ---------
                - the output tensor w^T*x. Shape []
        '''
        return tf.matmul(inputs, weights)

    def dense(self, inputs, weights):
        '''
            Computes the middle layers of the nn. Leaky relu activated.

            - input:
            --------
                - inputs: the input tensor. Shape []
                - weights: the weights tensor. Shape []

            - output:
            ---------
                - the output tensor leaky_relu(w^T*x). Shape []
        '''

        return tf.nn.leaky_relu(tf.matmul(inputs, weights),
                                alpha=self._leakyReluApha)

    def get_weights(self, shape, name):
        '''
            initalize the weights of a given shape

            - input:
            --------
                - shape: list, the shape of the weights.
                - name: string, the name of the shapes.
        '''
        return tf.Variable(self.initializer(shape, dtype=tf.float64),
                           name=name,
                           trainable=True,
                           dtype=tf.float64)


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
                 k=1,
                 stateDim=13,
                 actionDim=6,
                 name="nn_model"):

        NNModel.__init__(self, stateDim, actionDim, name, k)
        assert inertialFrameId in ["world", "world_ned"]

        # TODO: Not used.
        self._inertialFrameId = inertialFrameId
        if self._inertialFrameId == 'world':
            self._bodyFrameId = 'base_link'
        else:
            self._bodyFrameId = 'base_link_ned'

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
        with tf.name_scope(scope) as scope:
            x = self.prepare_data(state, action)
            delta = self.predict("nn", x)
            nextState = self.pred_2_inertial(state, delta)
        return nextState

    def pred_2_inertial(self, state, delta):
        '''
            Transforms delta, expressed in body_t frame, into Inertial frame.

            Input:
            ------
                - State: the state a time t. Shape [k, 13, 1]. Allows to
                compute the transform.
                    [x_pose_t_I.T, x_vel_t_B.T].T
                - Delta: the prediciton
                    [x_pose_{t+1}_Bt.T, x_vel_{t+1}_B{t+1}.T].T

            Output:
            -------
                - Delta expressed in the Inertial frame for the pose and in
                the body frame for the velocity.
                    [x_pose_{t+1}_I.T, x_vel_{t+1}_B{t+1}.T].T
        '''
        poseT0Bt0 = state[:, 0:7]
        poseT1Bt0 = delta[:, 0:7]
        poseT1Bt1 = self.transform(poseT0Bt0, poseT1Bt0)
        return poseT1Bt1

    def prepare_data(self, state, action):
        '''
            Prepares the data for inference.

            Input:
            ------
                - State [k, 13, 1]. [x_pose_I.T, x_vel_B.T].T. The current
                state of the system. The first part, x_pose_I, is the pose of
                the robot using quaternion representation [k, 7, 1] and
                expressed in the inertial frame.
                - Action [k, 6, 1]. The aciton expressed in the body frame.

            Output:
            -------
                - X [k, 16, 1]. [x_rot_I_t.T, x_vel_B_t.T, u_B_t.T].T
                The input to the neural network.
        '''
        return tf.concat([state[:, 3:13], action], axis=1)

    def prepare_training_data(self, stateT, stateT1, action):
        '''
            Prepares the training data. First convert the next state from
            the inertial frame to the body frame of the previous step.

            Input:
            ------
                - state_t: [x_pose_t_I.T, x_vel_t_Bt.T].T The state at time t.
                Shape [k, 13, 1].
                - state_t1: [x_pose_t+1_I.T, x_vel_t_B{t+1}.T].T
                The state at time t+1. Shape [k, 13, 1].
                - action (u): The action tensor. Shape [k, 6, 1].

            Output:
            -------
                - (X, y) training pair.
                    X is [x_rot_t_It.T, x_vel_t_Bt.T, u_Bt.T].T
                    Shape [k, 16, 1].
                    y is [x_pose_{t+1}_Bt.T, x_vel_{t+1}_B{t+1}.T].T
                    Shape [k, 13, 1].
        '''

        poseT = stateT[:, 0:7]
        poseT1 = stateT1[:, 0:7]
        inv = self.invTransform(poseT)
        poseT1Bt = self.transform(inv, poseT1)
        X = tf.concat([stateT[:, 3:13], action], axis=1)
        y = tf.concat([poseT1Bt, stateT1[:, 7:13]], axis=1)
        return (X, y)

    def rot_I_to_B(self, pose):
        return tf.transpose(self.rot_B_to_I(pose), perm=[0, 2, 1])

    def rot_B_to_I(self, pose):
        quat = pose[:, 3:7, :]

        w = quat[:, 0]
        x = quat[:, 1]
        y = quat[:, 2]
        z = quat[:, 3]

        r1 = tf.expand_dims(tf.concat([1 - 2 * (tf.pow(y, 2) + tf.pow(z, 2)),
                                       2 * (x * y - z * w),
                                       2 * (x * z + y * w)], axis=-1), axis=1)

        r2 = tf.expand_dims(tf.concat([2 * (x * y + z * w),
                                       1 - 2 * (tf.pow(x, 2) + tf.pow(z, 2)),
                                       2 * (y * z - x * w)], axis=-1), axis=1)

        r3 = tf.expand_dims(tf.concat([2 * (x * z - y * w),
                                       2 * (y * z + x * w),
                                       1 - 2 * (tf.pow(x, 2) + tf.pow(y, 2))],
                                      axis=-1), axis=1)

        return tf.concat([r1, r2, r3], axis=1)

    def transform(self, poseFrom, poseTo):
        tFrom = poseFrom[:, 0:3]
        tTo = poseTo[:, 0:3]
        qFrom = poseFrom[:, 3:7]
        qTo = poseTo[:, 3:7]
        tRes = tFrom + self.rot_vec(qFrom, tTo)
        qRes = self.qMul(qFrom, qTo)
        return tf.concat([tRes, qRes], axis=1)

    def inv_transform(self, pose):
        t = pose[:, 0:3]
        q = pose[:, 3:7]
        tRes = -self.rot_vec(self.inv_q(q), t)
        qRes = self.inv_q(q)
        return tf.concat([tRes, qRes], axis=1)

    def inv_q(self, q):
        inv = tf.constant([[1.], [-1.], [-1.], [-1.]], dtype=tf.float64)
        return inv*q

    def pure_q(self, vec):
        pad = tf.zeros(shape=(vec.shape[0], 1, 1), dtype=tf.float64)
        return tf.concat([pad, vec], axis=1)

    def rot_vec(self, q, vec):
        tmp = self.q_mul(self.pure_q(vec), self.inv_q(q))
        return self.q_mul(q, tmp)[:, 1:4]

    def q_mul(self, q1, q2):
        w = q1[:, 0]
        x = q1[:, 1]
        y = q1[:, 2]
        z = q1[:, 3]

        r1t = tf.expand_dims(tf.concat([w, -x, -y, -z], axis=-1), axis=1)

        r2t = tf.expand_dims(tf.concat([x, w, -z, y], axis=-1), axis=1)

        r3t = tf.expand_dims(tf.concat([y, z, w, -x], axis=-1), axis=1)

        r4t = tf.expand_dims(tf.concat([z, -y, x, w], axis=-1), axis=1)

        t = tf.concat([r1t, r2t, r3t, r4t], axis=1)

        return tf.matmul(t, q2)
