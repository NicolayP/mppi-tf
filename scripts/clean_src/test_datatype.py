import numpy as np
import tensorflow as tf

from datatype import DataType, DataTypeQuat, DataTypeRot
from model import ModelDtype
from utile import push_to_numpy, push_to_tensor, append_to_tensor

class TestDataType(tf.test.TestCase):
    def setUp(self):
        self.state_euler_name = [
            "x", "y", "z",
            "roll", "pitch", "yaw",
            "u", "v", "w",
            "p", "q", "r",
        ]

        self.state_quat_name = [
            "x", "y", "z",
            "qx", "qy", "qz", "qw",
            "u", "v", "w",
            "p", "q", "r",
        ]

        self.state_rot_name = [
            "x", "y", "z",
            "r00", "r01", "r02",
            "r10", "r11", "r12",
            "r20", "r21", "r22",
            "u", "v", "w",
            "p", "q", "r",
        ]

        self.action_names = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]

    def testBaseNotImpl(self):
        dt = DataType(self.state_euler_name, self.action_names, h=1)
        foo = np.zeros(shape=(1, dt.sDim, 1))
        with self.assertRaises(NotImplementedError):
            dt.to_r(None)

        with self.assertRaises(NotImplementedError):
            dt.flat_angle()

        with self.assertRaises(NotImplementedError):
            dt.fake_input()

        with self.assertRaises(TypeError):
            dt.split(foo)


class TestDataTypeH1Quat(tf.test.TestCase):
    def setUp(self):
        self.state_quat_name = [
            "x", "y", "z",
            "qx", "qy", "qz", "qw",
            "u", "v", "w",
            "p", "q", "r",
        ]
        self.action_names = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]
        self.dt_quat = DataTypeQuat(self.state_quat_name, self.action_names, h=1)
        pass

    def testFakeInputQuatH1(self):
        foo_quat = self.dt_quat.fake_input()
        gt_quat = (np.array([[[
                [0.], [0.], [0.], #x, y, z
                [0.], [0.], [0.], [1.], # qx, qy, qz, qw
                [0.], [0.], [0.], # u, v, w
                [0.], [0.], [0.], # p, q, r
              ]]]), None)
        self.assertAllClose(gt_quat[0], foo_quat[0])

    def testAddStateQuatH1(self):
        states = np.array([
                [
                    [0.], [0.], [0.], #x, y, z
                    [0.1], [1.0], [1.0], [1.0], #qx, qy, qz, qw
                    [0.], [0.], [0.], # u, v, w
                    [0.], [0.], [0.], # p, q, r
                ],
                [
                    [1.0], [2.0], [3.0], #x, y, z
                    [0.1], [1.0], [1.0], [1.0], #qx, qy, qz, qw
                    [10.], [20.], [30.], # u, v, w
                    [40.], [50.], [60.], # p, q, r
                ],
                [
                    [4.], [5.], [6.], #x, y, z
                    [0.1], [1.0], [1.0], [1.0], #qx, qy, qz, qw
                    [70.], [80.], [90.], # u, v, w
                    [30.], [20.], [10.], # p, q, r
                ],
            ])
        actions = np.array([
            [[0.], [0.], [0.], [0.], [0.], [0.]], # Fx, Fy, Fz, Tx, Ty, Tz
            [[1.], [1.], [1.], [1.], [1.], [1.]], # Fx, Fy, Fz, Tx, Ty, Tz
            [[2.], [2.], [2.], [2.], [2.], [2.]], # Fx, Fy, Fz, Tx, Ty, Tz
        ])
        
        self.assertEqual(self.dt_quat.init, False)

        self.dt_quat.add_state(states[0])
        self.dt_quat.add_action(actions[0])
        self.assertEqual(self.dt_quat.init, True)
        inputs = self.dt_quat.get_input()
        self.assertAllClose(inputs[0], states[0][None])
        self.assertEqual(inputs[1], None)

        self.dt_quat.add_state(states[1])
        self.dt_quat.add_action(actions[1])
        self.assertEqual(self.dt_quat.init, True)
        inputs = self.dt_quat.get_input()
        self.assertAllClose(inputs[0], states[1][None])
        self.assertEqual(inputs[1], None)

        self.dt_quat.add_state(states[2])
        self.dt_quat.add_action(actions[2])
        self.assertEqual(self.dt_quat.init, True)
        inputs = self.dt_quat.get_input()
        self.assertAllClose(inputs[0], states[2][None])
        self.assertEqual(inputs[1], None)

    def testSplitQuatH1(self):
        foo = np.array([
            [0.], [0.], [0.],
            [0.], [0.], [0.], [1.],
            [0.], [0.], [0.],
            [0.], [0.], [0.],
        ])
        pos, angles, vel = self.dt_quat.split(foo)
        self.assertAllClose(np.array([[0.], [0.], [0.]]), pos)
        self.assertAllClose(np.array([[0.], [0.], [0.], [1.],]), angles)
        self.assertAllClose(np.array([
                [0.], [0.], [0.],
                [0.], [0.], [0.]]),
        vel)


class TestDataTypeH3Quat(tf.test.TestCase):
    def setUp(self):
        self.h = 3
        self.state_quat_name = [
            "x", "y", "z",
            "qx", "qy", "qz", "qw",
            "u", "v", "w",
            "p", "q", "r",
        ]
        self.action_names = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]
        self.dt_quat = DataTypeQuat(self.state_quat_name, self.action_names, h=self.h)

    def testFakeInputQuatH3(self):    
        foo_quat = self.dt_quat.fake_input()
        gt_quat = (
            np.array([[
                [
                    [0.], [0.], [0.], #x, y, z
                    [0.], [0.], [0.], [1.], # qx, qy, qz, qw
                    [0.], [0.], [0.], # u, v, w
                    [0.], [0.], [0.], # p, q, r
                ],
                [
                    [0.], [0.], [0.], #x, y, z
                    [0.], [0.], [0.], [1.], # qx, qy, qz, qw
                    [0.], [0.], [0.], # u, v, w
                    [0.], [0.], [0.], # p, q, r
                ],
                [
                    [0.], [0.], [0.], #x, y, z
                    [0.], [0.], [0.], [1.], # qx, qy, qz, qw
                    [0.], [0.], [0.], # u, v, w
                    [0.], [0.], [0.], # p, q, r
                ],
            ]]),
            np.array([[
                [[0.], [0.], [0.], [0.], [0.], [0.]], # Fx, Fy, Fz, Tx, Ty, Tz
                [[0.], [0.], [0.], [0.], [0.], [0.]], # Fx, Fy, Fz, Tx, Ty, Tz
            ]]) 
            
        )

        self.assertAllClose(gt_quat[0], foo_quat[0])
        self.assertAllClose(gt_quat[1], foo_quat[1])

    def testAddStateQuatH3(self):
        states = np.array([
                [
                    [0.], [0.], [0.], #x, y, z
                    [0.1], [1.0], [1.0], [1.0], #qx, qy, qz, qw
                    [0.], [0.], [0.], # u, v, w
                    [0.], [0.], [0.], # p, q, r
                ],
                [
                    [1.0], [1.0], [1.0], #x, y, z
                    [0.1], [1.0], [1.0], [1.0], #qx, qy, qz, qw
                    [1.], [1.], [1.], # u, v, w
                    [1.], [1.], [1.], # p, q, r
                ],
                [
                    [2.], [2.], [2.], #x, y, z
                    [0.1], [1.0], [1.0], [1.0], #qx, qy, qz, qw
                    [2.], [2.], [2.], # u, v, w
                    [2.], [2.], [2.], # p, q, r
                ],
                [
                    [3.], [3.], [3.], #x, y, z
                    [0.1], [1.0], [1.0], [1.0], #qx, qy, qz, qw
                    [3.], [3.], [3.], # u, v, w
                    [3.], [3.], [3.], # p, q, r
                ],
                [
                    [4.], [4.], [4.], #x, y, z
                    [0.1], [1.0], [1.0], [1.0], #qx, qy, qz, qw
                    [4.], [4.], [4.], # u, v, w
                    [4.], [4.], [4.], # p, q, r
                ],
                [
                    [5.], [5.], [5.], #x, y, z
                    [0.1], [1.0], [1.0], [1.0], #qx, qy, qz, qw
                    [5.], [5.], [5.], # u, v, w
                    [5.], [5.], [5.], # p, q, r
                ],
            ])
        actions = np.array([
            [[0.], [0.], [0.], [0.], [0.], [0.]], # Fx, Fy, Fz, Tx, Ty, Tz
            [[1.], [1.], [1.], [1.], [1.], [1.]], # Fx, Fy, Fz, Tx, Ty, Tz
            [[2.], [2.], [2.], [2.], [2.], [2.]], # Fx, Fy, Fz, Tx, Ty, Tz
            [[3.], [3.], [3.], [3.], [3.], [3.]], # Fx, Fy, Fz, Tx, Ty, Tz
            [[4.], [4.], [4.], [4.], [4.], [4.]], # Fx, Fy, Fz, Tx, Ty, Tz
            [[5.], [5.], [5.], [5.], [5.], [5.]], # Fx, Fy, Fz, Tx, Ty, Tz
        ])
        
        self.assertEqual(self.dt_quat.init, False)
        self.dt_quat.add_state(states[0])
        self.dt_quat.add_action(actions[0])
        self.assertEqual(self.dt_quat.init, False)
        self.dt_quat.add_state(states[1])
        self.dt_quat.add_action(actions[1])
        self.assertEqual(self.dt_quat.init, False)
        self.dt_quat.add_state(states[2])
        self.dt_quat.add_action(actions[2])
        self.assertEqual(self.dt_quat.init, True)

        steps = states.shape[0]

        for i in range(self.h, steps):
            inputs = self.dt_quat.get_input()
            self.assertAllClose(inputs[0], states[i-self.h:i])
            self.assertAllClose(inputs[1], actions[i-(self.h-1):i])

            self.dt_quat.add_state(states[i])
            self.dt_quat.add_action(actions[i])
            self.assertEqual(self.dt_quat.init, True)

    def testSplitQuatH3(self):
        foo = np.array([
            [0.], [0.], [0.],
            [0.], [0.], [0.], [1.],
            [0.], [0.], [0.],
            [0.], [0.], [0.],
        ])
        pos, angles, vel = self.dt_quat.split(foo)
        self.assertAllClose(np.array([[0.], [0.], [0.]]), pos)
        self.assertAllClose(np.array([[0.], [0.], [0.], [1.]]), angles)
        self.assertAllClose(np.array([
                [0.], [0.], [0.],
                [0.], [0.], [0.]]),
        vel)

    def testToEulerQuatState(self):
        #Trajectory in rotation format, shape = [sDim, 1]
        foo = np.array([
                [0.], [0.], [0.],
                [0.], [0.], [0.], [1.],
                [0.], [0.], [0.],
                [0.], [0.], [0.],
        ])
        gt_euler = np.zeros(shape=(12, 1))

        foo_euler = self.dt_quat.to_euler(foo)

        self.assertAllClose(gt_euler, foo_euler)

    def testToEulerQuatTraj(self):
        #Trajectory in rotation format, shape = [Tau, sDim, 1]
        foo = np.array([
            [
                [0.], [0.], [0.],
                [0.], [0.], [0.], [1.],
                [0.], [0.], [0.],
                [0.], [0.], [0.],
            ],
            [
                [0.], [0.], [0.],
                [0.], [0.], [0.], [1.],
                [0.], [0.], [0.],
                [0.], [0.], [0.],
            ],
            [
                [0.], [0.], [0.],
                [0.], [0.], [0.], [1.],
                [0.], [0.], [0.],
                [0.], [0.], [0.],
            ],
            [
                [0.], [0.], [0.],
                [0.], [0.], [0.], [1.],
                [0.], [0.], [0.],
                [0.], [0.], [0.],
            ],
            [
                [0.], [0.], [0.],
                [0.], [0.], [0.], [1.],
                [0.], [0.], [0.],
                [0.], [0.], [0.],
            ],
        ])
        gt_euler = np.zeros(shape=(5, 12, 1))

        foo_euler = self.dt_quat.to_euler(foo)

        self.assertAllClose(gt_euler, foo_euler)

    def testToEulerQuatSamples(self):
        #Trajectory in rotation format, shape = [Tau, sDim, 1]
        foo = np.array([
            [
                [0.], [0.], [0.],
                [0.], [0.], [0.], [1.],
                [0.], [0.], [0.],
                [0.], [0.], [0.],
            ],
            [
                [0.], [0.], [0.],
                [0.], [0.], [0.], [1.],
                [0.], [0.], [0.],
                [0.], [0.], [0.],
            ],
            [
                [0.], [0.], [0.],
                [0.], [0.], [0.], [1.],
                [0.], [0.], [0.],
                [0.], [0.], [0.],
            ],
            [
                [0.], [0.], [0.],
                [0.], [0.], [0.], [1.],
                [0.], [0.], [0.],
                [0.], [0.], [0.],
            ],
            [
                [0.], [0.], [0.],
                [0.], [0.], [0.], [1.],
                [0.], [0.], [0.],
                [0.], [0.], [0.],
            ],
        ])
        foo = np.broadcast_to(foo, (3, 5, 13, 1))
        gt_euler = np.zeros(shape=(3, 5, 12, 1))

        foo_euler = self.dt_quat.to_euler(foo)

        self.assertAllClose(gt_euler, foo_euler)


class TestDataTypeH1Rot(tf.test.TestCase):
    def setUp(self):
        self.state_rot_name = [
            "x", "y", "z",
            "r00", "r01", "r02",
            "r10", "r11", "r12",
            "r20", "r21", "r22",
            "u", "v", "w",
            "p", "q", "r",
        ]

        self.action_names = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]
        self.h = 1
        self.dt_rot = DataTypeRot(self.state_rot_name, self.action_names, h=self.h)

    def testFakeInputRotH1(self):
        foo_rot = self.dt_rot.fake_input()
        gt_rot = (np.array([[[
                [0.], [0.], [0.], #x, y, z
                [1.], [0.], [0.], # r00, r01, r02
                [0.], [1.], [0.], # r10, r11, r12
                [0.], [0.], [1.], # r20, r21, r22
                [0.], [0.], [0.], # u, v, w
                [0.], [0.], [0.], # p, q, r
              ]]]), None)
        self.assertAllClose(gt_rot[0], foo_rot[0])

    def testBaseAddStateRotH1(self):
        states = np.array([
                [
                    [0.], [0.], [0.], #x, y, z
                    [1.], [0.], [0.], # r00, r01, r02
                    [0.], [1.], [0.], # r10, r11, r12
                    [0.], [0.], [1.], # r20, r21, r22
                    [0.], [0.], [0.], # u, v, w
                    [0.], [0.], [0.], # p, q, r
                ],
                [
                    [1.0], [2.0], [3.0], #x, y, z
                    [1.0], [0.5], [0.5], # r00, r01, r02
                    [0.5], [1.0], [0.5], # r10, r11, r12
                    [0.5], [0.5], [1.0], # r20, r21, r22
                    [10.], [20.], [30.], # u, v, w
                    [40.], [50.], [60.], # p, q, r
                ],
                [
                    [4.], [5.], [6.], #x, y, z
                    [0.1], [1.0], [1.0], # r00, r01, r02
                    [1.0], [0.1], [1.0], # r10, r11, r12
                    [1.0], [1.0], [0.1], # r20, r21, r22
                    [70.], [80.], [90.], # u, v, w
                    [30.], [20.], [10.], # p, q, r
                ],
            ])
        actions = np.array([
            [[0.], [0.], [0.], [0.], [0.], [0.]], # Fx, Fy, Fz, Tx, Ty, Tz
            [[1.], [1.], [1.], [1.], [1.], [1.]], # Fx, Fy, Fz, Tx, Ty, Tz
            [[2.], [2.], [2.], [2.], [2.], [2.]], # Fx, Fy, Fz, Tx, Ty, Tz
        ])
        
        self.assertEqual(self.dt_rot.init, False)

        self.dt_rot.add_state(states[0])
        self.dt_rot.add_action(actions[0])
        self.assertEqual(self.dt_rot.init, True)
        inputs = self.dt_rot.get_input()
        self.assertAllClose(inputs[0], states[0][None])
        self.assertEqual(inputs[1], None)

        self.dt_rot.add_state(states[1])
        self.dt_rot.add_action(actions[1])
        self.assertEqual(self.dt_rot.init, True)
        inputs = self.dt_rot.get_input()
        self.assertAllClose(inputs[0], states[1][None])
        self.assertEqual(inputs[1], None)

        self.dt_rot.add_state(states[2])
        self.dt_rot.add_action(actions[2])
        self.assertEqual(self.dt_rot.init, True)
        inputs = self.dt_rot.get_input()
        self.assertAllClose(inputs[0], states[2][None])
        self.assertEqual(inputs[1], None)

    def testSplitRotH1(self):
        foo = np.array([
            [0.], [0.], [0.],
            [1.], [0.], [0.],
            [0.], [1.], [0.],
            [0.], [0.], [1.],
            [0.], [0.], [0.],
            [0.], [0.], [0.],
        ])
        pos, angles, vel = self.dt_rot.split(foo)
        self.assertAllClose(np.array([[0.], [0.], [0.]]), pos)
        self.assertAllClose(np.array([
                [1.], [0.], [0.],
                [0.], [1.], [0.],
                [0.], [0.], [1.]]),
        angles)
        self.assertAllClose(np.array([
                [0.], [0.], [0.],
                [0.], [0.], [0.]]),
        vel)


class TestDataTypeH3Rot(tf.test.TestCase):
    def setUp(self):
        self.state_rot_name = [
            "x", "y", "z",
            "r00", "r01", "r02",
            "r10", "r11", "r12",
            "r20", "r21", "r22",
            "u", "v", "w",
            "p", "q", "r",
        ]

        self.action_names = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]
        self.h = 3
        self.dt_rot = DataTypeRot(self.state_rot_name, self.action_names, h=self.h)

    def testFakeInputRotH3(self):
        foo_rot = self.dt_rot.fake_input()
        gt_rot = (
            np.array([[
                [
                    [0.], [0.], [0.], #x, y, z
                    [1.], [0.], [0.], # r00, r01, r02
                    [0.], [1.], [0.], # r10, r11, r12
                    [0.], [0.], [1.], # r20, r21, r22
                    [0.], [0.], [0.], # u, v, w
                    [0.], [0.], [0.], # p, q, r
                ],
                [
                    [0.], [0.], [0.], #x, y, z
                    [1.], [0.], [0.], # r00, r01, r02
                    [0.], [1.], [0.], # r10, r11, r12
                    [0.], [0.], [1.], # r20, r21, r22
                    [0.], [0.], [0.], # u, v, w
                    [0.], [0.], [0.], # p, q, r
                ],
                [
                    [0.], [0.], [0.], #x, y, z
                    [1.], [0.], [0.], # r00, r01, r02
                    [0.], [1.], [0.], # r10, r11, r12
                    [0.], [0.], [1.], # r20, r21, r22
                    [0.], [0.], [0.], # u, v, w
                    [0.], [0.], [0.], # p, q, r
                ],
            ]]),
            np.array([[
                [[0.], [0.], [0.], [0.], [0.], [0.]], # Fx, Fy, Fz, Tx, Ty, Tz
                [[0.], [0.], [0.], [0.], [0.], [0.]], # Fx, Fy, Fz, Tx, Ty, Tz
            ]]) 
            
        )

        self.assertAllClose(gt_rot[0], foo_rot[0])
        self.assertAllClose(gt_rot[1], foo_rot[1])

    def testBaseAddStateRotH3(self):
        states = np.array([
                [
                    [0.], [0.], [0.], #x, y, z
                    [1.], [0.], [0.], # r00, r01, r02
                    [0.], [1.], [0.], # r10, r11, r12
                    [0.], [0.], [1.], # r20, r21, r22
                    [0.], [0.], [0.], # u, v, w
                    [0.], [0.], [0.], # p, q, r
                ],
                [
                    [1.0], [1.0], [1.0], #x, y, z
                    [1.0], [0.5], [0.5], # r00, r01, r02
                    [0.5], [1.0], [0.5], # r10, r11, r12
                    [0.5], [0.5], [1.0], # r20, r21, r22
                    [1.], [1.], [1.], # u, v, w
                    [1.], [1.], [1.], # p, q, r
                ],
                [
                    [2.], [2.], [2.], #x, y, z
                    [0.1], [1.0], [1.0], # r00, r01, r02
                    [1.0], [0.1], [1.0], # r10, r11, r12
                    [1.0], [1.0], [0.1], # r20, r21, r22
                    [2.], [2.], [2.], # u, v, w
                    [2.], [2.], [2.], # p, q, r
                ],
                [
                    [3.], [3.], [3.], #x, y, z
                    [0.1], [1.0], [1.0], # r00, r01, r02
                    [1.0], [0.1], [1.0], # r10, r11, r12
                    [1.0], [1.0], [0.1], # r20, r21, r22
                    [3.], [3.], [3.], # u, v, w
                    [3.], [3.], [3.], # p, q, r
                ],
                [
                    [4.], [4.], [4.], #x, y, z
                    [0.1], [1.0], [1.0], # r00, r01, r02
                    [1.0], [0.1], [1.0], # r10, r11, r12
                    [1.0], [1.0], [0.1], # r20, r21, r22
                    [4.], [4.], [4.], # u, v, w
                    [4.], [4.], [4.], # p, q, r
                ],
                [
                    [5.], [5.], [5.], #x, y, z
                    [0.1], [1.0], [1.0], # r00, r01, r02
                    [1.0], [0.1], [1.0], # r10, r11, r12
                    [1.0], [1.0], [0.1], # r20, r21, r22
                    [5.], [5.], [5.], # u, v, w
                    [5.], [5.], [5.], # p, q, r
                ],
            ])
        actions = np.array([
            [[0.], [0.], [0.], [0.], [0.], [0.]], # Fx, Fy, Fz, Tx, Ty, Tz
            [[1.], [1.], [1.], [1.], [1.], [1.]], # Fx, Fy, Fz, Tx, Ty, Tz
            [[2.], [2.], [2.], [2.], [2.], [2.]], # Fx, Fy, Fz, Tx, Ty, Tz
            [[3.], [3.], [3.], [3.], [3.], [3.]], # Fx, Fy, Fz, Tx, Ty, Tz
            [[4.], [4.], [4.], [4.], [4.], [4.]], # Fx, Fy, Fz, Tx, Ty, Tz
            [[5.], [5.], [5.], [5.], [5.], [5.]], # Fx, Fy, Fz, Tx, Ty, Tz
        ])
        
        self.assertEqual(self.dt_rot.init, False)
        self.dt_rot.add_state(states[0])
        self.dt_rot.add_action(actions[0])
        self.assertEqual(self.dt_rot.init, False)
        self.dt_rot.add_state(states[1])
        self.dt_rot.add_action(actions[1])
        self.assertEqual(self.dt_rot.init, False)
        self.dt_rot.add_state(states[2])
        self.dt_rot.add_action(actions[2])
        self.assertEqual(self.dt_rot.init, True)

        steps = states.shape[0]

        for i in range(self.h, steps):
            inputs = self.dt_rot.get_input()
            self.assertAllClose(inputs[0], states[i-self.h:i])
            self.assertAllClose(inputs[1], actions[i-(self.h-1):i])

            self.dt_rot.add_state(states[i])
            self.dt_rot.add_action(actions[i])
            self.assertEqual(self.dt_rot.init, True)

    def testSplitRotH3(self):
        foo = np.array([
            [0.], [0.], [0.],
            [1.], [0.], [0.],
            [0.], [1.], [0.],
            [0.], [0.], [1.],
            [0.], [0.], [0.],
            [0.], [0.], [0.],
        ])
        pos, angles, vel = self.dt_rot.split(foo)
        self.assertAllClose(np.array([[0.], [0.], [0.]]), pos)
        self.assertAllClose(np.array([
                [1.], [0.], [0.],
                [0.], [1.], [0.],
                [0.], [0.], [1.]]),
        angles)
        self.assertAllClose(np.array([
                [0.], [0.], [0.],
                [0.], [0.], [0.]]),
        vel)

    def testToEulerRotState(self):
        #Trajectory in rotation format, shape = [sDim, 1]
        foo = np.array([
                [0.], [0.], [0.],
                [1.], [0.], [0.],
                [0.], [1.], [0.],
                [0.], [0.], [1.],
                [0.], [0.], [0.],
                [0.], [0.], [0.],
        ])
        gt_euler = np.zeros(shape=(12, 1))

        foo_euler = self.dt_rot.to_euler(foo)

        self.assertAllClose(gt_euler, foo_euler)

    def testToEulerRotTraj(self):
        #Trajectory in rotation format, shape = [Tau, sDim, 1]
        foo = np.array([
            [
                [0.], [0.], [0.],
                [1.], [0.], [0.],
                [0.], [1.], [0.],
                [0.], [0.], [1.],
                [0.], [0.], [0.],
                [0.], [0.], [0.],
            ],
            [
                [0.], [0.], [0.],
                [1.], [0.], [0.],
                [0.], [1.], [0.],
                [0.], [0.], [1.],
                [0.], [0.], [0.],
                [0.], [0.], [0.],
            ],
            [
                [0.], [0.], [0.],
                [1.], [0.], [0.],
                [0.], [1.], [0.],
                [0.], [0.], [1.],
                [0.], [0.], [0.],
                [0.], [0.], [0.],
            ],
            [
                [0.], [0.], [0.],
                [1.], [0.], [0.],
                [0.], [1.], [0.],
                [0.], [0.], [1.],
                [0.], [0.], [0.],
                [0.], [0.], [0.],
            ],
            [
                [0.], [0.], [0.],
                [1.], [0.], [0.],
                [0.], [1.], [0.],
                [0.], [0.], [1.],
                [0.], [0.], [0.],
                [0.], [0.], [0.],
            ],
        ])
        gt_euler = np.zeros(shape=(5, 12, 1))

        foo_euler = self.dt_rot.to_euler(foo)

        self.assertAllClose(gt_euler, foo_euler)

    def testToEulerRotSamples(self):
        #Trajectory in rotation format, shape = [Tau, sDim, 1]
        foo = np.array([
            [
                [0.], [0.], [0.],
                [1.], [0.], [0.],
                [0.], [1.], [0.],
                [0.], [0.], [1.],
                [0.], [0.], [0.],
                [0.], [0.], [0.],
            ],
            [
                [0.], [0.], [0.],
                [1.], [0.], [0.],
                [0.], [1.], [0.],
                [0.], [0.], [1.],
                [0.], [0.], [0.],
                [0.], [0.], [0.],
            ],
            [
                [0.], [0.], [0.],
                [1.], [0.], [0.],
                [0.], [1.], [0.],
                [0.], [0.], [1.],
                [0.], [0.], [0.],
                [0.], [0.], [0.],
            ],
            [
                [0.], [0.], [0.],
                [1.], [0.], [0.],
                [0.], [1.], [0.],
                [0.], [0.], [1.],
                [0.], [0.], [0.],
                [0.], [0.], [0.],
            ],
            [
                [0.], [0.], [0.],
                [1.], [0.], [0.],
                [0.], [1.], [0.],
                [0.], [0.], [1.],
                [0.], [0.], [0.],
                [0.], [0.], [0.],
            ],
        ])
        foo = np.broadcast_to(foo, (3, 5, 18, 1))
        gt_euler = np.zeros(shape=(3, 5, 12, 1))

        foo_euler = self.dt_rot.to_euler(foo)

        self.assertAllClose(gt_euler, foo_euler)


class TestModelDataTypeH1(tf.test.TestCase):
    def setUp(self):
        self.h = 1
        self.x = np.array([
            [[
                [0.], [0.], [0.],
                [0.], [0.], [0.], [1.],
                [0.], [0.], [0.],
                [0.], [0.], [0.]
            ]],
            [[
                [1.], [1.], [1.],
                [0.], [0.], [0.], [1.],
                [1.], [1.], [1.],
                [1.], [1.], [1.]
            ]]
        ]) # shape [2, 1, 13, 1]
        self.u = np.array([
            [[
                [0.], [0.], [0.], [0.], [0.], [0.]
            ]],
            [[
                [1.], [1.], [1.], [1.], [1.], [1.]
            ]]
        ]) # shape [2, 1, 6, 1]

        self.x_new = np.array([
            [[
                [-1.], [-1.], [-1.],
                [0.], [0.], [0.], [1.],
                [-1.], [-1.], [-1.],
                [-1.], [-1.], [-1.]
            ]],
            [[
                [2.], [2.], [2.],
                [0.], [0.], [0.], [1.],
                [2.], [2.], [2.],
                [2.], [2.], [2.]
            ]]
        ])
        self.u_new = np.array([
            [[
                [-1.], [-1.], [-1.], [-1.], [-1.], [-1.]
            ]],
            [[
                [2.], [2.], [2.], [2.], [2.], [2.]
            ]]
        ]) # shape [2, 1, 6, 1]

        self.h=1
        self.dtype = ModelDtype(self.x, self.u, h=self.h)
    
    def testInit(self):
        x = self.dtype._x
        u = self.dtype._u
        self.assertAllClose(x, self.x)
        self.assertAllClose(u, self.u)

    def testAdd(self):
        self.dtype.add(self.x_new, self.u_new)
        gt_x = np.concatenate([self.x, self.x_new], axis=1)
        gt_u = np.concatenate([self.u, self.u_new], axis=1)
        x = self.dtype._x
        u = self.dtype._u
        self.assertAllClose(x, gt_x)
        self.assertAllClose(u, gt_u)

    def testCall(self):
        x, u = self.dtype()
        self.assertAllClose(x, self.x)
        self.assertAllClose(u, self.u)
        
        self.dtype.add(self.x_new, self.u_new)
        x, u = self.dtype()
        self.assertAllClose(x, self.x_new)
        self.assertAllClose(u, self.u_new)


class TestModelDataTypeH3(tf.test.TestCase):
    def setUp(self):
        self.h = 3
        self.x = np.array([
            [
                [
                    [0.], [0.], [0.],
                    [0.], [0.], [0.], [1.],
                    [0.], [0.], [0.],
                    [0.], [0.], [0.]
                ],
                [
                    [-1.], [-1.], [-1.],
                    [0.], [0.], [0.], [1.],
                    [-1.], [-1.], [-1.],
                    [-1.], [-1.], [-1.]
                ],
                [
                    [-2.], [-2.], [-2.],
                    [0.], [0.], [0.], [1.],
                    [-2.], [-2.], [-2.],
                    [-2.], [-2.], [-2.]
                ]
            ],
            [
                [
                    [1.], [1.], [1.],
                    [0.], [0.], [0.], [1.],
                    [1.], [1.], [1.],
                    [1.], [1.], [1.]
                ],
                [
                    [2.], [2.], [2.],
                    [0.], [0.], [0.], [1.],
                    [2.], [2.], [2.],
                    [2.], [2.], [2.]
                ],
                [
                    [3.], [3.], [3.],
                    [0.], [0.], [0.], [1.],
                    [3.], [3.], [3.],
                    [3.], [3.], [3.]
                ]
            ]
        ]) # shape [2, 3, 13, 1]
        self.u = np.array([
            [
                [
                    [0.], [0.], [0.], [0.], [0.], [0.]
                ],
                [
                    [1.], [1.], [1.], [1.], [1.], [1.]
                ],
                [
                    [2.], [2.], [2.], [2.], [2.], [2.]
                ]
            ],
            [
                [
                    [-1.], [-1.], [-1.], [-1.], [-1.], [-1.]
                ],
                [
                    [-2.], [-2.], [-2.], [-2.], [-2.], [-2.]
                ],
                [
                    [-3.], [-3.], [-3.], [-3.], [-3.], [-3.]
                ]
            ]
        ]) # shape [2, 3, 6, 1]

        self.x_new = np.array([
            [[
                [-1.], [-1.], [-1.],
                [0.], [0.], [0.], [1.],
                [-1.], [-1.], [-1.],
                [-1.], [-1.], [-1.]
            ]],
            [[
                [2.], [2.], [2.],
                [0.], [0.], [0.], [1.],
                [2.], [2.], [2.],
                [2.], [2.], [2.]
            ]]
        ])
        self.u_new = np.array([
            [[
                [-1.], [-1.], [-1.], [-1.], [-1.], [-1.]
            ]],
            [[
                [2.], [2.], [2.], [2.], [2.], [2.]
            ]]
        ]) # shape [2, 1, 6, 1]

        self.dtype = ModelDtype(self.x, self.u, h=self.h)
    
    def testInit(self):
        x = self.dtype._x
        u = self.dtype._u
        self.assertAllClose(x, self.x)
        self.assertAllClose(u, self.u)

    def testAdd(self):
        self.dtype.add(self.x_new, self.u_new)
        gt_x = np.concatenate([self.x, self.x_new], axis=1)
        gt_u = np.concatenate([self.u, self.u_new], axis=1)
        x = self.dtype._x
        u = self.dtype._u
        self.assertAllClose(x, gt_x)
        self.assertAllClose(u, gt_u)

    def testCall(self):
        x, u = self.dtype()
        self.assertAllClose(x, self.x)
        self.assertAllClose(u, self.u)
        
        self.dtype.add(self.x_new, self.u_new)
        x, u = self.dtype()
        gt_x = np.concatenate([self.x[:, 1:], self.x_new], axis=1)
        gt_u = np.concatenate([self.u[:, 1:], self.u_new], axis=1)

        self.assertAllClose(x, gt_x)
        self.assertAllClose(u, gt_u)

    def testTraj(self):
        t = self.dtype.traj()
        shape_np = np.zeros(shape=(2, 0, 13, 1))
        self.assertShapeEqual(t, shape_np)
        self.dtype.add(self.x_new, self.u_new)
        self.dtype.add(self.x_new, self.u_new)
        self.dtype.add(self.x_new, self.u_new)

        t = self.dtype.traj()
        gt_t = np.concatenate([self.x_new, self.x_new, self.x_new], axis=1)
        self.assertAllClose(t, gt_t)


if __name__ == '__main__':
    tf.test.main()
