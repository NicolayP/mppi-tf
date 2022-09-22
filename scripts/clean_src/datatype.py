import tensorflow as tf
import numpy as np
from utile import npdtype, dtype, push_to_numpy, push_to_tensor
from scipy.spatial.transform import Rotation as R

class DataType(object):
    def __init__(self, stateNames, actionNames=None, h=1):
        self._init = False
        self._stateNames = stateNames
        self._actionNames = actionNames
        self._states = np.zeros((h, self.sDim, 1), dtype=npdtype)
        self._actions = np.zeros((h-1, self.aDim, 1), dtype=npdtype)
        self._h = h
        self._state_ptr = 0
        self._action_ptr = 0
        self._angIdx = None

    @property
    def init(self):
        return (self._state_ptr >= self._h) and (self._action_ptr >= (self._h-1))

    @property
    def sDim(self):
        return len(self._stateNames)

    @property
    def aDim(self):
        return len(self._actionNames)

    def add_state(self, state):
        if self._state_ptr < self._h:
            self._states[self._state_ptr] = state
            self._state_ptr += 1
        else:
            self._states = push_to_numpy(self._states, state)
    
    def add_action(self, action):
        if self._h <= 1:
            return
        if self._action_ptr < (self._h-1):
            self._actions[self._action_ptr] = action
            self._action_ptr += 1
        else:
            self._actions = push_to_numpy(self._actions, action)

    def get_input(self):
        if self._h <= 1:
            return (self._states, None)
        return (self._states, self._actions)

    def to_r(self, state):
        raise NotImplementedError

    def flat_angle(self):
        raise NotImplementedError

    def split(self, state):
        pos = state[..., :self._angIdx[0], :]
        angles = state[..., self._angIdx, :]
        vel = state[..., self._angIdx[-1]+1:, :]
        return pos, angles, vel
    
    def fake_input(self, k=1):
        state = np.zeros(shape=(k, self._h, self.sDim, 1), dtype=npdtype)
        state[:, :, self._angIdx] = self.flat_angle()
        if self._h <= 1:
            return (state, None)
        action = np.zeros(shape=(k, self._h-1, self.aDim, 1), dtype=npdtype)
        return (state, action)

    def to_euler(self, state): 
        state_prefix, angles, state_suffix = self.split(state)
        s = list(angles.shape)
        if len(angles.shape) > 3:
            angles = angles.reshape((-1, s[-2], 1))
        r = self.to_r(angles)
        
        s[-2] = 3
        euler = r.as_euler("XYZ").reshape(s)

        return np.concatenate([state_prefix, euler, state_suffix], axis=-2)


class DataTypeQuat(DataType):
    '''
        Datatype for MPPI using quaternion. The representation assumes that 
        the quaternion uses (qx, qy, qz, qw) format where qw is the real
        part of the quaternion.
    '''
    def __init__(self, stateNames, actionNames=None, h=1, quatIdx=None):
        super(DataTypeQuat, self).__init__(stateNames=stateNames, actionNames=actionNames, h=h)
        if quatIdx is None:
            quatIdx = np.arange(3, 7)
        self._angIdx = quatIdx

    def flat_angle(self):
        return np.array([[0.], [0.], [0.], [1.]], dtype=npdtype)

    def to_r(self, quat):
        quat = np.squeeze(quat, axis=-1)
        return R.from_quat(quat)


class DataTypeRot(DataType):
    '''
        Datatype for MPPI using rotation matrix.
    '''
    def __init__(self, stateNames, actionNames=None, h=1, rotIdx=None):
        super(DataTypeRot, self).__init__(stateNames=stateNames, actionNames=actionNames, h=h)
        if rotIdx is None:
            rotIdx = np.arange(3, 3+9)
        self._angIdx = rotIdx

    def flat_angle(self):
        return np.eye(3, dtype=npdtype).reshape((-1, 1))

    def to_r(self, rot):
        s = list(rot.shape)
        s[-1] = s[-2] = 3
        mat = np.reshape(rot, s)
        return R.from_matrix(mat)
