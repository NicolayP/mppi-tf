import numpy as np
import tensorflow as tf

from datatype import DataType, DataTypeQuat, DataTypeRot


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

    # Rot matrix format
    def testFakeInputRotH1(self):
        dt_rot = DataTypeRot(self.state_rot_name, self.action_names, h=1)
        foo_rot = dt_rot.fake_input()
        gt_rot = (np.array([[[
                [0.], [0.], [0.], #x, y, z
                [1.], [0.], [0.], # r00, r01, r02
                [0.], [1.], [0.], # r10, r11, r12
                [0.], [0.], [1.], # r20, r21, r22
                [0.], [0.], [0.], # u, v, w
                [0.], [0.], [0.], # p, q, r
              ]]]), None)
        self.assertAllClose(gt_rot[0], foo_rot[0])

    def testFakeInputRotH3(self):
        dt_rot = DataTypeRot(self.state_rot_name, self.action_names, h=3)
        foo_rot = dt_rot.fake_input()
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

    def testBaseAddStateRotH1(self):
        dt_rot = DataTypeRot(self.state_rot_name, self.action_names, h=1)
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
        
        self.assertEqual(dt_rot.init, False)

        dt_rot.add_state(states[0])
        dt_rot.add_action(actions[0])
        self.assertEqual(dt_rot.init, True)
        inputs = dt_rot.get_input()
        self.assertAllClose(inputs[0], states[0][None])
        self.assertEqual(inputs[1], None)

        dt_rot.add_state(states[1])
        dt_rot.add_action(actions[1])
        self.assertEqual(dt_rot.init, True)
        inputs = dt_rot.get_input()
        self.assertAllClose(inputs[0], states[1][None])
        self.assertEqual(inputs[1], None)

        dt_rot.add_state(states[2])
        dt_rot.add_action(actions[2])
        self.assertEqual(dt_rot.init, True)
        inputs = dt_rot.get_input()
        self.assertAllClose(inputs[0], states[2][None])
        self.assertEqual(inputs[1], None)

    def testBaseAddStateRotH3(self):
        h = 3
        dt_rot = DataTypeRot(self.state_rot_name, self.action_names, h=h)
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
        
        self.assertEqual(dt_rot.init, False)
        dt_rot.add_state(states[0])
        dt_rot.add_action(actions[0])
        self.assertEqual(dt_rot.init, False)
        dt_rot.add_state(states[1])
        dt_rot.add_action(actions[1])
        self.assertEqual(dt_rot.init, False)
        dt_rot.add_state(states[2])
        dt_rot.add_action(actions[2])
        self.assertEqual(dt_rot.init, True)

        steps = states.shape[0]

        for i in range(h, steps):
            inputs = dt_rot.get_input()
            self.assertAllClose(inputs[0], states[i-h:i])
            self.assertAllClose(inputs[1], actions[i-(h-1):i])

            dt_rot.add_state(states[i])
            dt_rot.add_action(actions[i])
            self.assertEqual(dt_rot.init, True)

    def testSplitRotH1(self):
        h = 1
        dt_rot = DataTypeRot(self.state_rot_name, self.action_names, h=h)
        foo = np.array([
            [0.], [0.], [0.],
            [1.], [0.], [0.],
            [0.], [1.], [0.],
            [0.], [0.], [1.],
            [0.], [0.], [0.],
            [0.], [0.], [0.],
        ])
        pos, angles, vel = dt_rot.split(foo)
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

    def testSplitRotH3(self):
        h = 3
        dt_rot = DataTypeRot(self.state_rot_name, self.action_names, h=h)
        foo = np.array([
            [0.], [0.], [0.],
            [1.], [0.], [0.],
            [0.], [1.], [0.],
            [0.], [0.], [1.],
            [0.], [0.], [0.],
            [0.], [0.], [0.],
        ])
        pos, angles, vel = dt_rot.split(foo)
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
        h = 3
        dt_rot = DataTypeRot(self.state_rot_name, self.action_names, h=h)
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

        foo_euler = dt_rot.to_euler(foo)

        self.assertAllClose(gt_euler, foo_euler)

    def testToEulerRotTraj(self):
        h = 3
        dt_rot = DataTypeRot(self.state_rot_name, self.action_names, h=h)
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

        foo_euler = dt_rot.to_euler(foo)

        self.assertAllClose(gt_euler, foo_euler)

    def testToEulerRotSamples(self):
        h = 3
        dt_rot = DataTypeRot(self.state_rot_name, self.action_names, h=h)
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

        foo_euler = dt_rot.to_euler(foo)

        self.assertAllClose(gt_euler, foo_euler)

    # Quat matrix format
    def testFakeInputQuatH1(self):
        dt_quat = DataTypeQuat(self.state_quat_name, self.action_names, h=1)
        foo_quat = dt_quat.fake_input()
        gt_quat = (np.array([[[
                [0.], [0.], [0.], #x, y, z
                [0.], [0.], [0.], [1.], # qx, qy, qz, qw
                [0.], [0.], [0.], # u, v, w
                [0.], [0.], [0.], # p, q, r
              ]]]), None)
        self.assertAllClose(gt_quat[0], foo_quat[0])

    def testFakeInputQuatH3(self):
        dt_quat = DataTypeQuat(self.state_quat_name, self.action_names, h=3)
        foo_quat = dt_quat.fake_input()
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

    def testAddStateQuatH1(self):
        dt_quat = DataTypeRot(self.state_quat_name, self.action_names, h=1)
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
        
        self.assertEqual(dt_quat.init, False)

        dt_quat.add_state(states[0])
        dt_quat.add_action(actions[0])
        self.assertEqual(dt_quat.init, True)
        inputs = dt_quat.get_input()
        self.assertAllClose(inputs[0], states[0][None])
        self.assertEqual(inputs[1], None)

        dt_quat.add_state(states[1])
        dt_quat.add_action(actions[1])
        self.assertEqual(dt_quat.init, True)
        inputs = dt_quat.get_input()
        self.assertAllClose(inputs[0], states[1][None])
        self.assertEqual(inputs[1], None)

        dt_quat.add_state(states[2])
        dt_quat.add_action(actions[2])
        self.assertEqual(dt_quat.init, True)
        inputs = dt_quat.get_input()
        self.assertAllClose(inputs[0], states[2][None])
        self.assertEqual(inputs[1], None)

    def testAddStateQuatH3(self):
        h = 3
        dt_quat = DataTypeQuat(self.state_quat_name, self.action_names, h=h)
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
        
        self.assertEqual(dt_quat.init, False)
        dt_quat.add_state(states[0])
        dt_quat.add_action(actions[0])
        self.assertEqual(dt_quat.init, False)
        dt_quat.add_state(states[1])
        dt_quat.add_action(actions[1])
        self.assertEqual(dt_quat.init, False)
        dt_quat.add_state(states[2])
        dt_quat.add_action(actions[2])
        self.assertEqual(dt_quat.init, True)

        steps = states.shape[0]

        for i in range(h, steps):
            inputs = dt_quat.get_input()
            self.assertAllClose(inputs[0], states[i-h:i])
            self.assertAllClose(inputs[1], actions[i-(h-1):i])

            dt_quat.add_state(states[i])
            dt_quat.add_action(actions[i])
            self.assertEqual(dt_quat.init, True)

    def testSplitQuatH1(self):
        h = 1
        dt_quat = DataTypeQuat(self.state_quat_name, self.action_names, h=h)
        foo = np.array([
            [0.], [0.], [0.],
            [0.], [0.], [0.], [1.],
            [0.], [0.], [0.],
            [0.], [0.], [0.],
        ])
        pos, angles, vel = dt_quat.split(foo)
        self.assertAllClose(np.array([[0.], [0.], [0.]]), pos)
        self.assertAllClose(np.array([[0.], [0.], [0.], [1.],]), angles)
        self.assertAllClose(np.array([
                [0.], [0.], [0.],
                [0.], [0.], [0.]]),
        vel)

    def testSplitQuatH3(self):
        h = 3
        dt_quat = DataTypeQuat(self.state_quat_name, self.action_names, h=h)
        foo = np.array([
            [0.], [0.], [0.],
            [0.], [0.], [0.], [1.],
            [0.], [0.], [0.],
            [0.], [0.], [0.],
        ])
        pos, angles, vel = dt_quat.split(foo)
        self.assertAllClose(np.array([[0.], [0.], [0.]]), pos)
        self.assertAllClose(np.array([[0.], [0.], [0.], [1.]]), angles)
        self.assertAllClose(np.array([
                [0.], [0.], [0.],
                [0.], [0.], [0.]]),
        vel)

    def testToEulerQuatState(self):
        h = 3
        dt_quat = DataTypeQuat(self.state_quat_name, self.action_names, h=h)
        #Trajectory in rotation format, shape = [sDim, 1]
        foo = np.array([
                [0.], [0.], [0.],
                [0.], [0.], [0.], [1.],
                [0.], [0.], [0.],
                [0.], [0.], [0.],
        ])
        gt_euler = np.zeros(shape=(12, 1))

        foo_euler = dt_quat.to_euler(foo)

        self.assertAllClose(gt_euler, foo_euler)

    def testToEulerQuatTraj(self):
        h = 3
        dt_quat = DataTypeQuat(self.state_quat_name, self.action_names, h=h)
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

        foo_euler = dt_quat.to_euler(foo)

        self.assertAllClose(gt_euler, foo_euler)

    def testToEulerQuatSamples(self):
        h = 3
        dt_quat = DataTypeQuat(self.state_quat_name, self.action_names, h=h)
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

        foo_euler = dt_quat.to_euler(foo)

        self.assertAllClose(gt_euler, foo_euler)

if __name__ == '__main__':
    tf.test.main()