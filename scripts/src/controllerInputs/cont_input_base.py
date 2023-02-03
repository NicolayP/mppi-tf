import numpy as np
from scipy.spatial.transform import Rotation as R

import tensorflow as tf

from ..misc.utile import assert_shape, push_to_tensor, dtype, npdtype

class ControllerInputBase(object):
    def __init__(self, stateNames, actionNames=None):
        self._init = False
        self._stateNames = stateNames
        self._actionNames = actionNames
        self._states = None
        self._actions = None

    def is_init(self):
        return self._init

    def get_fake_input(self):
        raise NotImplementedError

    def human_readable(self, input):
        raise NotImplementedError
    
    def get_input(self):
        raise NotImplementedError

    def add_state(self, state):
        raise NotImplementedError
    
    def add_action(self, acton):
        raise NotImplementedError

class ContInput(object):
    def __init__(self, history, rot="quat"):
        self._states = np.zeros((history, 13, 1), dtype=npdtype)
        self._actions = np.zeros((history-1, 6, 1), dtype=npdtype)

        self._h_state = history
        self._h_action = history - 1
        self._rot = rot
        self._current_action = 0
        self._current_state = 0
        self._filled_state = False
        self._filled_action = False
        self._filled = False

    def add_state(self, state):
        if not self._filled_state:
            self._states[self._current_state] = state
            self._current_state += 1
            if self._current_state >= self._h_state:
                self._filled_state = True
        else:
            tmpState = self._states[1:]
            self._states = np.concatenate([tmpState, state[None, ...]], axis=0)

    def add_action(self, action):
        if self._h_action < 1:
            self._filled_action = True
            return

        if not self._filled_action:
            self._actions[self._current_action] = action
            self._current_action += 1
            if self._current_action >= self._h_action:
                self._filled_action = True
        else:
            tmpAct = self._actions[:-1]
            self._actions = np.concatenate([tmpAct, action[None, ...]], axis=0)

    def is_filled(self):
        return self._filled_action and self._filled_state

    def get_input(self):
        return self.pred_controller_input(self._states, self._actions)

    def pred_controller_input(self, lagged_state, lagged_action):
        if self._rot == "rot":
            lagged_state_rot = self.quat_to_rot(lagged_state)
        else:
            lagged_state_rot = lagged_state
        if self._h_action < 1:
            return (lagged_state_rot.astype(npdtype), None)
        return (lagged_state_rot.astype(npdtype), lagged_action.astype(npdtype))

    def quat_to_rot(self, lagged_state):
        pos = lagged_state[:, :3]
        quat = lagged_state[:, 3:7]
        vel = lagged_state[:, -6:]
        r = R.from_quat(quat[:, :, 0])
        rot = r.as_matrix().reshape((-1, 9, 1))

        return np.concatenate([pos, rot, vel], axis=1)

class AUVContInput(ControllerInputBase):
    def __init__(self, stateName, actionName, history, rot="quat", inp_rot="rot"):
        super(AUVContInput, self).__init__(stateName, actionName)
        self._h = history
        self._rep = rot
        self._inp_rep = inp_rot

        if rot == "euler":
            self._rot_idx = [3, 4, 5]
        elif rot == "quat":
            self._rot_idx = [3, 4, 5, 6]
        elif rot == "rot":
            self._rot_idx = [
                3, 4, 5,
                6, 7, 8,
                9, 10, 11

            ]
        else:
            raise NotImplementedError
    
    def add_state(self, state):
        if self._h_s < self._h:
            self._states[self._h_s] = state
            self._h_s += 1
        else:
            tmpStates = self._states[1:]
            self._states = np.concatenate([tmpStates, state[None, ...]], axis=0)

    def add_action(self, action):
        if self._h <= 1:
            return

        if self._h_a < (self._h-1):
            self._actions[self._h_a] = action
            self._h_a += 1
        else:
            tmpAction = self._actions[1:]
            self._actions = np.concatenate([tmpAction, action[None, ...]], axis=0)

    def is_init(self):
        return ((self._h_a == self._h) and (self._h_s == (self._h - 1)))

    def get_input(self):
        if self._rep != self._int_rep:
            lagged_state_rot = self.rot_state(self._states)
        else:
            lagged_state_rot = self._states
        if self._h <= 1:
            lagged_action = None
        else:
            lagged_action = self._actions

        return (lagged_state_rot, lagged_action)

    def rot_state(self, state):
        pos = state[:, :3]
        rot = state[:, self._rot_idx]
        vel = state[:, -6:]
        if self._rep == "rot":
            r = R.from_matrix(rot)
        elif self._rep == "quat":
            r = R.from_quat(rot)
        elif self._rep == "euler":
            r = R.from_euler(rot)
        else:
            raise NotImplementedError

        if self._inp_rep == "rot":
            rot = r.as_matrix()
        elif self._inp_rep == "quat":
            rot = r.as_quat()
        elif self._inp_rep == "euler":
            rot = r.as_euler("XYZ")
        else:
            raise NotImplementedError

        return np.concatenate([pos, rot, vel], axis=1)