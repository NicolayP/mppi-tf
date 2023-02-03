import numpy as np
import tensorflow as tf

from ..src.controllerInputs.cont_input_base import AUVContInput


class TestControllerInput(tf.test.TestCase):
    def setUp(self):
        state_name = [
            "x", "y", "z",
            "roll", "pitch", "yaw",
            "u", "v", "w",
            "p", "q", "r",
        ]
        action_names = ["Fx", "Fy" "Fz", "Tx", "Ty", "Tz"]
        h = 1
        inputs = AUVContInput(state_name, action_names, h, "quat", "rot")
        pass

    def 