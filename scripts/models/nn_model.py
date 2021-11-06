import tensorflow as tf
import numpy as np
from model_base import ModelBase

from utile import Arrow3D
import matplotlib.pyplot as plt

class NNModel(ModelBase):
    '''
        Neural network based model class.
    '''
    def __init__(self, state_dim=2, action_dim=1, k=1, name="nn_model", inertial_frame_id="world"):
        '''
            Neural network model constructor.

            - input:
            --------
                - state_dim: int the state space dimension.
                - action_dim: int the action space dimension.
                - name: string the model name.
        '''

        ModelBase.__init__(self, state_dim, action_dim, k, name, inertial_frame_id)
        self.leaky_relu_alpha = 0.2
        self.initializer = tf.initializers.glorot_uniform()
        self.addModelVars("first", self.getWeights((state_dim+action_dim, 5), "first"))
        self.addModelVars("second", self.getWeights((5, 5), "second"))
        self.addModelVars("final", self.getWeights((5, state_dim), "final"))

    def build_step_graph(self, scope, state, action):
        '''
            Abstract method, need to be overwritten in child class.
            Step graph for the model. This computes the prediction for $\hat{f}(x, u)$

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
            state = tf.broadcast_to(state, [ashape[0], sshape[0], sshape[1]])
        
        input = tf.squeeze(tf.concat([state, action], axis=1), -1)
        return self.predict(scope, input)

    def predict(self, scope, input):

        init = self.dense(input, self.model_vars["first"])
        second = self.dense(init, self.model_vars["second"])
        return tf.expand_dims(self.final(second, self.model_vars["final"]), -1)

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

        return tf.nn.leaky_relu( tf.matmul(inputs, weights), alpha=self.leaky_relu_alpha)

    def getWeights(self, shape, name):
        '''
            initalize the weights of a given shape

            - input:
            --------
                - shape: list, the shape of the weights.
                - name: string, the name of the shapes.
        '''
        return tf.Variable(self.initializer(shape, dtype=tf.float64), name=name, trainable=True, dtype=tf.float64)


class NNAUVModel(NNModel):
    '''
        Neural network representation for AUV model. Assumes that
        the model uses quaternion representation. The network predicts
        the next state expressed in the previous state body frame based on:
        the orientation of the body frame (for restoring forces), the velocity
        (in body frame) and the input forces.
    '''

    def __init__(self,
                 inertial_frame_id="world",
                 k=1,
                 state_dim=13,
                 action_dim=6,
                 name="nn_model"):
        '''

        '''
        NNModel.__init__(self, state_dim, action_dim, name, k)
        assert inertial_frame_id in ["world", "world_ned"]
        self.inertial_frame_id = inertial_frame_id
        if self.inertial_frame_id == 'world':
            self.body_frame_id = 'base_link'
        else:
            self.body_frame_id = 'base_link_ned'

        pass

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
                - next_state: The next state of the system.
                [X_pose_{t+1}_I.T, X_vel_{t+1}_B].T. Shape [k, 13, 1]
        '''
        with tf.name_scope(scope) as scope:
            x = self.prepareData(state, action)
            delta = self.predict("nn", x)
            next_state = self.pred2Inertial(state, delta)
        return next_state

    def pred2Inertial(self, state, delta):
        '''
            Transforms delta, expressed in body_t frame, into Inertial frame.

            Input:
            ------
                - State: the state a time t. Shape [k, 13, 1]. Allows to
                compute the transform. [x_pose_t_I.T, x_vel_t_B.T].T
                - Delta: the prediciton [x_pose_{t+1}_Bt.T, x_vel_{t+1}_B{t+1}.T].T

            Output:
            -------
                - Delta expressed in the Inertial frame for the pose and in
                the body frame for the velocity. [x_pose_{t+1}_I.T, x_vel_{t+1}_B{t+1}.T].T
        '''
        pose_t0_Bt0 = state[:, 0:7]
        pose_t1_Bt0 = delta[:, 0:7]
        pose_t1_Bt1 = self.transform(pose_t0_Bt0, pose_t1_Bt0)
        return pose_t1_Bt1

    def prepareData(self, state, action):
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
                - X [k, 16, 1]. [x_rot_I_t.T, x_vel_B_t.T, u_B_t.T].T The input to the
                the neural network.
        '''
        return tf.concat([state[:, 3:13], action], axis=1)

    def prepareTrainingData(self, state_t, state_t1, action):
        '''
            Prepares the training data. First convert the next state from
            the inertial frame to the body frame of the previous step.

            Input:
            ------
                - state_t: [x_pose_t_I.T, x_vel_t_Bt.T].T The state at time t.
                Shape [k, 13, 1].
                - state_t1: [x_pose_t+1_I.T, x_vel_t_B{t+1}.T].T The state at time t+1.
                Shape [k, 13, 1].
                - action (u): The action tensor. Shape [k, 6, 1].

            Output:
            -------
                - (X, y) training pair. X is [x_rot_t_It.T, x_vel_t_Bt.T, u_Bt.T].T
                Shape [k, 16, 1]. y is [x_pose_{t+1}_Bt.T, x_vel_{t+1}_B{t+1}.T].T
                Shape [k, 13, 1].
        '''

        pose_t = state_t[:, 0:7]
        pose_t1 = state_t1[:, 0:7]
        inv = self.invTransform(pose_t)
        pose_t1_Bt = self.transform(inv, pose_t1)
        X = tf.concat([state_t[:, 3:13], action], axis=1)
        y = tf.concat([pose_t1_Bt, state_t1[:, 7:13]], axis=1)
        return (X, y)

    def rotItoB(self, pose):
        return tf.transpose(self.rotBtoI(pose), perm=[0, 2, 1])

    def rotBtoI(self, pose):
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
                                        1 - 2 * (tf.pow(x, 2) + tf.pow(y, 2))], axis=-1), axis=1)

        return tf.concat([r1, r2, r3], axis=1)

    def transform(self, pose_from, pose_to):
        t_from = pose_from[:, 0:3]
        t_to = pose_to[:, 0:3]
        q_from = pose_from[:, 3:7]
        q_to = pose_to[:, 3:7]
        t_res = t_from + self.rotVec(q_from, t_to)
        q_res = self.qMul(q_from, q_to)
        return tf.concat([t_res, q_res], axis=1)

    def invTransform(self, pose):
        t = pose[:, 0:3]
        q = pose[:, 3:7]
        t_res = -self.rotVec(self.invQ(q), t)
        q_res = self.invQ(q)
        return tf.concat([t_res, q_res], axis=1)

    def invQ(self, q):
        inv = tf.constant([[1.], [-1.], [-1.], [-1.]], dtype=tf.float64)
        return inv*q

    def pureQ(self, vec):
        pad = tf.zeros(shape=(vec.shape[0], 1, 1), dtype=tf.float64)
        return tf.concat([pad, vec], axis=1)

    def rotVec(self, q, vec):
        tmp = self.qMul(self.pureQ(vec), self.invQ(q))
        return self.qMul(q, tmp)[:, 1:4]

    def qMul(self, q1, q2):
        w = q1[:, 0]
        x = q1[:, 1]
        y = q1[:, 2]
        z = q1[:, 3]

        r1_t = tf.expand_dims(tf.concat([w, -x, -y, -z], axis=-1), axis=1)

        r2_t = tf.expand_dims(tf.concat([x, w, -z, y], axis=-1), axis=1)

        r3_t = tf.expand_dims(tf.concat([y, z, w, -x], axis=-1), axis=1)

        r4_t = tf.expand_dims(tf.concat([z, -y, x, w], axis=-1), axis=1)

        t = tf.concat([r1_t, r2_t, r3_t, r4_t], axis=1)

        return tf.matmul(t, q2)

def plotFrame(t, x, y, z):

    plt.figure()
    ax = plt.axes(projection="3d")
    x_ar = Arrow3D([0, 1],
                   [0, 0],
                   [0, 0],
                   mutation_scale=20,
                   lw=1, arrowstyle="->", color="r")

    y_ar = Arrow3D([0, 0],
                   [0, 1],
                   [0, 0],
                   mutation_scale=20,
                   lw=1, arrowstyle="->", color="g")

    z_ar = Arrow3D([0, 0],
                   [0, 0],
                   [0, 1],
                   mutation_scale=20,
                   lw=1, arrowstyle="->", color="b")

    ax.add_artist(x_ar)
    ax.add_artist(y_ar)
    ax.add_artist(z_ar)


    for i in range(t.shape[0]):
        x_ar = Arrow3D([t[i, 0, 0], x[i, 0, 0]],
                       [t[i, 1, 0], x[i, 1, 0]],
                       [t[i, 2, 0], x[i, 2, 0]],
                       mutation_scale=20,
                       lw=1, arrowstyle="->", color="r")
        y_ar = Arrow3D([t[i, 0, 0], y[i, 0, 0]],
                       [t[i, 1, 0], y[i, 1, 0]],
                       [t[i, 2, 0], y[i, 2, 0]],
                       mutation_scale=20,
                       lw=1, arrowstyle="->", color="g")
        z_ar = Arrow3D([t[i, 0, 0], z[i, 0, 0]],
                       [t[i, 1, 0], z[i, 1, 0]],
                       [t[i, 2, 0], z[i, 2, 0]],
                       mutation_scale=20,
                       lw=1, arrowstyle="->", color="b")
        ax.add_artist(x_ar)
        ax.add_artist(y_ar)
        ax.add_artist(z_ar)

def visuTrainData(nn_auv):
    pose_t0_Bt0 = np.array([
                            [
                             [1.0], [1.0], [0.5], #Position
                             [1.0], [0.0], [0.0], [0.0], #Quaterinon
                            ],
                            [
                             [1.0], [-1.75],[0.5],
                             [0.6532815], [0.6532815], [0.2705981], [0.2705981],
                            ]
                           ])

    pose_t1_Bt1 = np.array([
                            [
                             [1.5], [2.3], [0.9], #Position
                             [-0.1913417], [0.8001031], [0.4619398], [0.3314136] #Quaterinon
                            ],
                            [
                             [0.0], [-1.],[2.5],
                             [0.4615897], [0.8446119], [0.0560099], [0.2653839]
                            ]
                           ])

    inv = nn_auv.invTransform(pose_t0_Bt0)
    pose_t1_Bt0 = nn_auv.transform(inv, pose_t1_Bt1)

    t_t0_Bt0 = pose_t0_Bt0[:, 0:3]
    q_t0_Bt0 = pose_t0_Bt0[:, 3:7]

    t_t1_Bt1 = pose_t1_Bt1[:, 0:3]
    q_t1_Bt1 = pose_t1_Bt1[:, 3:7]

    t_t1_Bt0 = pose_t1_Bt0[:, 0:3]
    q_t1_Bt0 = pose_t1_Bt0[:, 3:7]

    x = np.array([[[1.0], [0.0], [0.0]]])
    y = np.array([[[0.0], [1.0], [0.0]]])
    z = np.array([[[0.0], [0.0], [1.0]]])

    x_t1_Bt0 = (t_t1_Bt0 + nn_auv.qMul(nn_auv.qMul(q_t1_Bt0, nn_auv.pureQ(x)), nn_auv.invQ(q_t1_Bt0))[:, 1:4]).numpy()
    y_t1_Bt0 = (t_t1_Bt0 + nn_auv.qMul(nn_auv.qMul(q_t1_Bt0, nn_auv.pureQ(y)), nn_auv.invQ(q_t1_Bt0))[:, 1:4]).numpy()
    z_t1_Bt0 = (t_t1_Bt0 + nn_auv.qMul(nn_auv.qMul(q_t1_Bt0, nn_auv.pureQ(z)), nn_auv.invQ(q_t1_Bt0))[:, 1:4]).numpy()

    x_t1_Bt1 = (t_t1_Bt1 + nn_auv.qMul(nn_auv.qMul(q_t1_Bt1, nn_auv.pureQ(x)), nn_auv.invQ(q_t1_Bt1))[:, 1:4]).numpy()
    y_t1_Bt1 = (t_t1_Bt1 + nn_auv.qMul(nn_auv.qMul(q_t1_Bt1, nn_auv.pureQ(y)), nn_auv.invQ(q_t1_Bt1))[:, 1:4]).numpy()
    z_t1_Bt1 = (t_t1_Bt1 + nn_auv.qMul(nn_auv.qMul(q_t1_Bt1, nn_auv.pureQ(z)), nn_auv.invQ(q_t1_Bt1))[:, 1:4]).numpy()

    x_t0_Bt0 = (t_t0_Bt0 + nn_auv.qMul(nn_auv.qMul(q_t0_Bt0, nn_auv.pureQ(x)), nn_auv.invQ(q_t0_Bt0))[:, 1:4]).numpy()
    y_t0_Bt0 = (t_t0_Bt0 + nn_auv.qMul(nn_auv.qMul(q_t0_Bt0, nn_auv.pureQ(y)), nn_auv.invQ(q_t0_Bt0))[:, 1:4]).numpy()
    z_t0_Bt0 = (t_t0_Bt0 + nn_auv.qMul(nn_auv.qMul(q_t0_Bt0, nn_auv.pureQ(z)), nn_auv.invQ(q_t0_Bt0))[:, 1:4]).numpy()

    plotFrame(np.concatenate([t_t0_Bt0, t_t1_Bt1], axis=0),
              np.concatenate([x_t0_Bt0, x_t1_Bt1], axis=0),
              np.concatenate([y_t0_Bt0, y_t1_Bt1], axis=0),
              np.concatenate([z_t0_Bt0, z_t1_Bt1], axis=0))

    plotFrame(t_t1_Bt0,
              x_t1_Bt0,
              y_t1_Bt0,
              z_t1_Bt0)

    #plt.show()

def visuPred2Inertial(nn_auv):
    pose_t0_Bt0 = np.array([
                        [
                            [1.0], [1.0], [0.5], #Position
                            [1.0], [0.0], [0.0], [0.0], #Quaterinon
                        ],
                        [
                            [1.0], [-1.75],[0.5],
                            [0.6532815], [0.6532815], [0.2705981], [0.2705981],
                        ]
                        ])

    pose_t1_Bt0 = np.array([
                            [
                             [1.5], [2.3], [0.9], #Position
                             [-0.1913417], [0.8001031], [0.4619398], [0.3314136] #Quaterinon
                            ],
                            [
                             [0.0], [-1.],[2.5],
                             [0.4615897], [0.8446119], [0.0560099], [0.2653839]
                            ]
                           ])

    pose_t1_Bt1 = nn_auv.transform(pose_t0_Bt0, pose_t1_Bt0)

    t_t0_Bt0 = pose_t0_Bt0[:, 0:3]
    q_t0_Bt0 = pose_t0_Bt0[:, 3:7]

    t_t1_Bt1 = pose_t1_Bt1[:, 0:3]
    q_t1_Bt1 = pose_t1_Bt1[:, 3:7]

    t_t1_Bt0 = pose_t1_Bt0[:, 0:3]
    q_t1_Bt0 = pose_t1_Bt0[:, 3:7]

    x = np.array([[[1.0], [0.0], [0.0]]])
    y = np.array([[[0.0], [1.0], [0.0]]])
    z = np.array([[[0.0], [0.0], [1.0]]])

    x_t1_Bt0 = (t_t1_Bt0 + nn_auv.qMul(nn_auv.qMul(q_t1_Bt0, nn_auv.pureQ(x)), nn_auv.invQ(q_t1_Bt0))[:, 1:4]).numpy()
    y_t1_Bt0 = (t_t1_Bt0 + nn_auv.qMul(nn_auv.qMul(q_t1_Bt0, nn_auv.pureQ(y)), nn_auv.invQ(q_t1_Bt0))[:, 1:4]).numpy()
    z_t1_Bt0 = (t_t1_Bt0 + nn_auv.qMul(nn_auv.qMul(q_t1_Bt0, nn_auv.pureQ(z)), nn_auv.invQ(q_t1_Bt0))[:, 1:4]).numpy()

    x_t1_Bt1 = (t_t1_Bt1 + nn_auv.qMul(nn_auv.qMul(q_t1_Bt1, nn_auv.pureQ(x)), nn_auv.invQ(q_t1_Bt1))[:, 1:4]).numpy()
    y_t1_Bt1 = (t_t1_Bt1 + nn_auv.qMul(nn_auv.qMul(q_t1_Bt1, nn_auv.pureQ(y)), nn_auv.invQ(q_t1_Bt1))[:, 1:4]).numpy()
    z_t1_Bt1 = (t_t1_Bt1 + nn_auv.qMul(nn_auv.qMul(q_t1_Bt1, nn_auv.pureQ(z)), nn_auv.invQ(q_t1_Bt1))[:, 1:4]).numpy()

    x_t0_Bt0 = (t_t0_Bt0 + nn_auv.qMul(nn_auv.qMul(q_t0_Bt0, nn_auv.pureQ(x)), nn_auv.invQ(q_t0_Bt0))[:, 1:4]).numpy()
    y_t0_Bt0 = (t_t0_Bt0 + nn_auv.qMul(nn_auv.qMul(q_t0_Bt0, nn_auv.pureQ(y)), nn_auv.invQ(q_t0_Bt0))[:, 1:4]).numpy()
    z_t0_Bt0 = (t_t0_Bt0 + nn_auv.qMul(nn_auv.qMul(q_t0_Bt0, nn_auv.pureQ(z)), nn_auv.invQ(q_t0_Bt0))[:, 1:4]).numpy()

    plotFrame(np.concatenate([t_t0_Bt0, t_t1_Bt1], axis=0),
              np.concatenate([x_t0_Bt0, x_t1_Bt1], axis=0),
              np.concatenate([y_t0_Bt0, y_t1_Bt1], axis=0),
              np.concatenate([z_t0_Bt0, z_t1_Bt1], axis=0))

    plotFrame(t_t1_Bt0,
              x_t1_Bt0,
              y_t1_Bt0,
              z_t1_Bt0)
    plt.show()

def main():
    '''
        Debugging main function, not the main use of this program:

        1. Test prepareTrainingData for: 1 element, 2 elements and N batchesize.
        2. Test prepareData for: 1 element, 2 elements and N batchesize.
        3. Test pred2Inertial for: 1 element, 2 elements and N batchsize.

    '''
    # Test 1: Visualize the training data in the world frame as given by gazebo/robot.
    # And in the body_t frame. Mostly visual verification. After that still need to verify the
    # Unit tests.
    nn_auv = NNAUVModel()
    #visuTrainData(nn_auv)
    visuPred2Inertial(nn_auv)



    pass

if __name__ == "__main__":
    main()