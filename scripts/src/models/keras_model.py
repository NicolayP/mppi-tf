import tensorflow as tf
import tensorflow_graphics.geometry.transformation as tfgt

# Gets rid of the error triggered when running
# tfg in graph mode.
import sys
module = sys.modules['tensorflow_graphics.util.shape']
def _get_dim(tensor, axis):
    """Returns dimensionality of a tensor for a given axis."""
    return tf.compat.v1.dimension_value(tensor.shape[axis])

module._get_dim = _get_dim
sys.modules['tensorflow_graphics.util.shape'] = module

from mppi_ros.scripts.mppi_tf.scripts.src.models.model_base import ModelBase


class PrepData(tf.keras.layers.Layer):
    def __init__(self, angIdx, velIdx, actIdx, maxAct,
                 angRep="Euler", velMean=0., velVar=1.):
        '''
            Data Preparation layer. Preprocess the data so that
            it's easier to use by the network.
            
            inputs:
            -------
                - angIdx: slice object indicating the angles index
                    in the state vector.
                - velIdx: slice object indicating the velocities index
                    in the state vector.
                - actIdx: slice index indicating the actions index
                    in the state vector.
                - maxAct: the max action value used for normalization of
                    the data.
                - angRep: the angle representation used in the state vector.
                    can be either "Euler" or "Quaterion".
        '''
        super(PrepData, self).__init__()
        self.angIdx = angIdx
        self.velIdx = velIdx
        self.actIdx = actIdx
        self.angRep = angRep
        self.maxAct = maxAct
        self.normVel = tf.keras.layers.experimental.preprocessing.Normalization(mean=velMean, variance=velVar)
        if angRep not in ["Euler", "Quaternion"]:
            raise ValueError(f'Unknow angle representaiton, must be Euler or Quaternion, got {angRep}')

    def sin_cos(self, angles):
        '''
            Converts angle representation to cos(angles) & sin(angles). The
            cos & sin representation provides the network with a continuous
            angle representation i.e no jump from 0 to 359 degrees for example.
            If "Quaterion" change representation to euler.

            inputs: 
            -------
                - angles: tensor representing either quaterions or euler angles.
                    shape: [k, 4/3]

            outputs:
            --------
                - sin_cos: tensor with sin and cos value of the euler angles.
                    shape: [k, 6] with values [sin(roll), sin(pitch), sin(yaw),
                                               cos(roll), cos(pitch), cos(yaw)]
        '''
        if self.angRep in ["Quaternion"]:
            angles = tfgt.euler.from_quaternion(angles)
        cos = tf.math.cos(angles)
        sin = tf.math.sin(angles)

        return tf.concat([sin, cos], axis=-1)

    def vel_norm(self, vel):
        '''
            Normalizes the velocity vector.

            inputs:
            -------
                - vel: tensor representing the velocity vector.
                    shape: [k, velDim]
            
            outputs:
            --------
                - normedVel: normed velocity vector.
                    shape: [k, velDim]
        '''
        return self.normVel(vel)

    def act_norm(self, act):
        '''
            Normalizes the action vector according to the max output
            an actoin can generate

            inputs:
            -------
                - act: tensor representing the action applied on the system.
                    shape: [k, aDim]

            outputs:
            --------
                - normedAct: tensor with the normed action.
                    shape: [k, adim]
        '''
        return act/self.maxAct

    def call(self, inputs):
        '''
            Prepare Data call function. Extracts the angles, the velocity
            and the action vectors from the state vector. Changes the angle
            representation and normalized the vel and action.

            inputs:
            -------
                - inputs: tensor representing the state of the system.
                    Should be [position, angles, velocities, actions] with
                    the first dimension representing the batch size.

            outputs:
            --------
                - input that will be fed to the network. The data is
                    normalized depending on what the represent and the position
                    is cropped from the state resulting in:
                    [sin_cos(angles), normedVel, normedAct] with the first
                    dimension is the batch size.
        '''
        angles = inputs[:, self.angIdx]
        vel = inputs[:, self.velIdx]
        act = inputs[:, self.actIdx]
        angRep = self.sin_cos(angles)
        velNorm = self.vel_norm(vel)
        actNorm = self.act_norm(act)
        return tf.concat([angRep, velNorm, actNorm], axis=-1)


class NextState(tf.keras.layers.Layer):
    def __init__(self, poseIdx, posIdx, angIdx, velIdx, actIdx,
                 maxAct, angRep="Euler", dt=0.1):
        '''
            Next Step Layer. This Layer computes the kinematics of
            a generic plant. From the velocity vector and the pose vector,
            The Layer predicts th enext state.

            inputs:
            -------
                - poseIdx: slice object inidcating the pose (position, orientation)
                    indicies in the state vector.
                - posIdx: slice object indicating the position indicies in the 
                    input vector.
                - angIdx: slice object indicating the angles indicies in the 
                    input vector.
                - velIdx: slice object indicating the velocities indicies in the 
                    input vector.
                - actIdx: slice object indicating the action indicies in the
                    input vector.
                - maxAct: the max input value. Can be either a scalar or a array of 
                    shape [aDim]
                - angRep: the angle representation used by the state vector. 
                    either "Euler" or "Quaternion".
                - dt: the time step interval.
        '''
        super(NextState, self).__init__()
        self.poseIdx = poseIdx
        self.posIdx = posIdx
        self.angIdx = angIdx
        self.velIdx = velIdx
        self.actIdx = actIdx
        self.maxAct = maxAct
        self.angRep = angRep
        self.dt = dt
        if angRep not in ["Euler", "Quaternion"]:
            raise ValueError(f'Unknow angle representaiton, must be Euler or Quaternion, got {angRep}')

    def denorm(self, delta):
        '''
            If the training data used for the model has been normalized
            (input and output), this denormalizes the output delta. 

            - inputs:
            ---------
                - deltaNormed: tensor representing the normalized delta predicted by the internal model.
                    shape: [k, 6]

            - outputs:
            ----------
                - delta: tensor representing the predicted delta. Shape: [k, 6]
        '''
        return delta

    def normalize_quat(self, pose):
        '''
            Normalizes batch of quaternions.

            input:
            ------
                - pose. Float64 Tensor. Shape [k, 13]

            ouput:
            ------
                - the pose with normalized quaternion. Float64 Tensor.
                    Shape [k, 13]
        '''

        pos = pose[:, self.posIdx]
        quat = pose[:, self.angIdx]

        quat = tf.math.l2_normalize(quat, axis=-1)
        pose = tf.concat([pos, quat], axis=1)
        return pose

    def get_jacobian(self, angles):
        '''
        Computes the jacobian that transforms a state vector in body frame
        to the inerttial frame.
        Returns J(nu) in $mathbb{R}^{7 cross 7}$
                     +------------------+------------------+
            J(nu) =  | q_{n}^{b}(Quat) | 0^{3 cross 3}    |
                     +------------------+------------------+
                     | 0^{3 cross 4}    | T_{Quat}(Quat) |
                     +------------------+------------------+
        
            - inputs:
            ---------
                - angles: tensor of angles with shape (k, 4). Only supports
                    quaternion atm.
            
            - outputs:
            ----------
                - jacobian: the jacobian that transforms a body frame pose vector into
                    a inertial frame pose vector.
        '''
        k = angles.shape[0]
        OPad3x3 = tf.zeros(shape=(k, 3, 3), dtype=tf.float32)
        OPad4x3 = tf.zeros(shape=(k, 4, 3), dtype=tf.float32)
        rotBtoI, TBtoIquat = self.body2inertial_transform(angles)
        jacR1 = tf.concat([rotBtoI, OPad3x3], axis=-1)

        jacR2 = tf.concat([OPad4x3, TBtoIquat], axis=-1)
        jac = tf.concat([jacR1, jacR2], axis=1)

        return jac

    def body2inertial_transform(self, angles):
        '''
            Computes the rotational transform from
            body to inertial Rot_{n}^{b}(q)
            and the attitude transformation T_{q}(q).

            - inputs:
            --------
                - angles: orentation of the robot pose expressed in
                    inertial frame. Shape [k, 4]
            
            - outputs:
            ----------
                - rotBtoI: tensor, rotational transform from 
                    body to inertial Rot_{n}^{b}(q), shape: [k, 3, 3]
                - TBtoIquat: tenso, attitude transformation T_{q}(q),
                    shape: [k, 3, 4]

        '''
        quat = tf.expand_dims(angles, axis=-1)
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

    def call(self, state, deltaNormed):
        '''
            Compute the next state for a given state and an delta.
            The state being expressed in world frame and the delta 
            expressed in body frame.
        '''
        delta = self.denorm(deltaNormed)
        pose = state[:, self.poseIdx]
        angles = state[:, self.angIdx]
        vel = state[:, self.velIdx]
        jac = self.get_jacobian(angles)

        pDot = tf.squeeze(tf.matmul(jac, tf.expand_dims(vel, axis=-1)), axis=-1)
        nextPose = pose + pDot*self.dt
        nextPose = self.normalize_quat(nextPose)
        nextVel = vel + delta
        return tf.concat([nextPose, nextVel], axis=1)


class StepLayer(tf.keras.layers.Layer):
    def __init__(self, poseIdx, posIdx, angIdx,
                 velIdx, actIdx, stateIdx,
                 maxAct, angRep="Euler", dt=0.1):
        super(StepLayer, self).__init__()
        self.poseIdx = poseIdx
        self.posIdx = posIdx
        self.angIdx = angIdx
        self.velIdx = velIdx
        self.actIdx = actIdx
        self.stateIdx = stateIdx
        self.angRep = angRep
        self.dt = dt
        self.prep_data = PrepData(angIdx=angIdx,
                                  velIdx=velIdx,
                                  actIdx=actIdx,
                                  maxAct=maxAct,
                                  angRep=angRep)

        self.next_state = NextState(poseIdx=poseIdx,
                                    posIdx=posIdx,
                                    angIdx=angIdx,
                                    velIdx=velIdx,
                                    actIdx=actIdx,
                                    maxAct=maxAct,
                                    angRep=angRep,
                                    dt=dt)

        self.nn_predict = tf.keras.Sequential([
                            tf.keras.layers.Dense(
                                16,
                                activation='relu',
                                kernel_regularizer='l2',
                                input_shape=(18,),
                                name="dense1"),
                            tf.keras.layers.Dense(
                                16,
                                activation='relu',
                                kernel_regularizer='l2',
                                name="dense2"),
                            tf.keras.layers.Dense(
                                16,
                                activation='relu',
                                kernel_regularizer='l2',
                                name="dense3"),
                            tf.keras.layers.Dense(
                                6,
                                activation='linear',
                                name="dense4")
                         ])

    def call(self, inputs):
        dataNormed = self.prep_data(inputs)
        deltaNormed = self.nn_predict(dataNormed)
        nextState = self.next_state(inputs, deltaNormed)
        return nextState


class MultiStep(tf.keras.Model):
    def __init__(self, poseIdx, posIdx, angIdx, velIdx,
                 actIdx, stateIdx, maxAct, angRep="Quaterion", dt=0.1):
        super(MultiStep, self).__init__()
        self.step = StepLayer(poseIdx, posIdx, angIdx,
                              velIdx, actIdx, stateIdx,
                              maxAct, angRep=angRep, dt=dt)

    def call(self, inputs):
        state, actionSeq = inputs
        tau = actionSeq.shape[1]
        k = state.shape[0]
        dim = state.shape[1]
        i = tf.constant(0)
        ta = tf.TensorArray(tf.float32, size=tau+1, clear_after_read=False)
        loop_vars = [state, ta, i]
        cond = lambda state, ta, i: i < tau
        body = lambda state, ta, i: [self.step(tf.concat([state, actionSeq[:, i, :]], axis=-1)),
                                     ta.write(i, state),
                                     i+1
                                    ]
        loop = tf.while_loop(cond, body, loop_vars, parallel_iterations=1)
        loop[1].write(loop[2], loop[0])
        return tf.reshape(loop[1].stack(), shape=(k, tau+1, dim))

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred,
                                      regularization_losses=self.losses)
            # Compute gradients
            trainable_vars = self.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)
            # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

