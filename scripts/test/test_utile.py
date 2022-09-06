import numpy as np
import tensorflow as tf
gpu = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(device=gpu[0], enable=True)

from ..src.models.model_utils import ToSE3Mat, SE3int, SO3int, Skew, FlattenSE3
from ..src.misc.utile import dtype, npdtype

class TestSkewOP(tf.test.TestCase):
    def setUp(self):
        self.skewOP = Skew()

    def test_k1_valid_vec(self):
        vec = np.array([[0.1, 0.2, 0.3]], dtype=npdtype)
        gt_skew = np.array([
            [
                [0., -0.3, 0.2],
                [0.3, 0., -0.1],
                [-0.2, 0.1, 0.]
            ]
        ], dtype=npdtype)
        skew = self.skewOP.forward(vec)
        self.assertAllClose(gt_skew, skew)

    def test_k5_valid_vec(self):
        vec = np.array([
            [0.1, 0.2, 0.3],
            [1.0, 2.0, 3.0],
            [0.4, 0.5, 0.6],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
            ], dtype=npdtype)
        gt_skew = np.array([
            [
                [0., -0.3, 0.2],
                [0.3, 0., -0.1],
                [-0.2, 0.1, 0.]
            ],
            [
                [0., -3., 2.],
                [3., 0., -1.],
                [-2., 1., 0.]
            ],
            [
                [0., -0.6, 0.5],
                [0.6, 0., -0.4],
                [-0.5, 0.4, 0.]
            ],
            [
                [0., -6.0, 5.0],
                [6.0, 0., -4.0],
                [-5.0, 4.0, 0.]
            ],
            [
                [0., -9.0, 8.0],
                [9.0, 0., -7.0],
                [-8.0, 7.0, 0.]
            ]
        ], dtype=npdtype)
        skew = self.skewOP.forward(vec)
        self.assertAllClose(gt_skew, skew)

    def test_k1_invalid_length_vec(self):
        vec = np.array([[0.2, 0.3]], dtype=npdtype)
        with self.assertRaises(IndexError):
            skew = self.skewOP.forward(vec)

    def test_k1_invalid_shape_vec(self):
        vec = np.array([0.1, 0.2, 0.3], dtype=npdtype)
        with self.assertRaises(IndexError):
            skew = self.skewOP.forward(vec)

    def test_k2_invalid_length_vec(self):
        vec = np.array([
            [0.2, 0.3],
            [2.0, 3.0]
        ], dtype=npdtype)
        with self.assertRaises(IndexError):
            skew = self.skewOP.forward(vec)

    def test_k2_invalid_shape_vec(self):
        vec = np.array([[
            [0.1, 0.2, 0.3],
            [1.0, 2.0, 3.0]
        ]], dtype=npdtype)
        with self.assertRaises(IndexError):
            skew = self.skewOP.forward(vec)


class TestFlattenSE3(tf.test.TestCase):
    def setUp(self):
        self.flattenSE3op = FlattenSE3()

    def test_k1_valid_input(self):
        M = np.array([
            [
                [1., 0., 0., 0.],
                [0., 1., 0., 0.],
                [0., 0., 1., 0.],
                [0., 0., 0., 1.]
            ]
        ], dtype=npdtype)
        vel = np.array([[0., 0., 0., 0., 0., 0.]], dtype=npdtype)
        gt_flat = np.array([
            [0., 0., 0.,
             1., 0., 0.,
             0., 1., 0.,
             0., 0., 1.,
             0., 0., 0., 0., 0., 0.],
        ], dtype=npdtype)
        flat = self.flattenSE3op.forward(M, vel)
        
        self.assertAllClose(gt_flat, flat)

    def test_k3_valid_input(self):
        M = np.array([
            [
                [1., 0., 0., 0.],
                [0., 1., 0., 0.],
                [0., 0., 1., 0.],
                [0., 0., 0., 1.]
            ],
            [
                [0.1, 0.5, 0.0, 10.],
                [0.0, 0.1, 0.0, 20.],
                [0.0, 0.4, 0.2, 30.],
                [0.0, 0.0, 0.0, 1.0]
            ],
            [
                [1., 2., 3., 40.],
                [4., 5., 6., 50.],
                [7., 8., 9., 60.],
                [0., 0., 0., 1.]
            ]
        ], dtype=npdtype)

        vel = np.array([
            [0., 0., 0., 0., 0., 0.],
            [1., 2., 3., 4., 5., 6.],
            [7., 8., 9., 10., 11., 12.]
        ], dtype=npdtype)

        gt_flat = np.array([
            [0., 0., 0.,
             1., 0., 0.,
             0., 1., 0.,
             0., 0., 1.,
             0., 0., 0., 0., 0., 0.],
            [10., 20., 30.,
             0.1, 0.5, 0.0,
             0.0, 0.1, 0.0,
             0.0, 0.4, 0.2,
             1., 2., 3., 4., 5., 6.],
            [40., 50., 60.,
             1., 2., 3.,
             4., 5., 6.,
             7., 8., 9.,
             7., 8., 9., 10., 11., 12.],
        ], dtype=npdtype)
        flat = self.flattenSE3op.forward(M, vel)
        
        self.assertAllClose(gt_flat, flat)

    def test_k1_invalid_input(self):
        M = np.array([
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.]
        ], dtype=npdtype)
        vel = np.array([0., 0., 0., 0., 0., 0.], dtype=npdtype)
        with self.assertRaises(IndexError):
            flat = self.flattenSE3op.forward(M, vel)

    def test_k3_invalid_input(self):
        M = np.array([
            [
                [1., 0., 0.],
                [0., 1., 0.],
                [0., 0., 1.]
            ],
            [
                [0.1, 0.5, 0.0],
                [0.0, 0.1, 0.0],
                [0.0, 0.4, 0.2]
            ],
            [
                [1., 2., 3.],
                [4., 5., 6.],
                [7., 8., 9.]
            ]
        ], dtype=npdtype)

        vel = np.array([
            [0., 0., 0., 0., 0., 0.],
            [1., 2., 3., 4., 5., 6.],
            [7., 8., 9., 10., 11., 12.]
        ], dtype=npdtype)
        with self.assertRaises(IndexError):
            flat = self.flattenSE3op.forward(M, vel)


class TestToSE3MatOP(tf.test.TestCase):
    def setUp(self):
        self.toSE3matOP = ToSE3Mat()

    def test_k1_valid_input(self):
        vec = np.array([
            [
                1., 2., 3.,
                1., 0., 0.,
                0., 1., 0.,
                0., 0., 1.,
                4., 5., 6., 7., 8., 9.
            ]
        ], dtype=npdtype)
        M_gt = np.array([
            [
                [1., 0., 0., 1.],
                [0., 1., 0., 2.],
                [0., 0., 1., 3.],
                [0., 0., 0., 1.],
            ]
        ], dtype=npdtype)
        M = self.toSE3matOP.forward(vec)
        self.assertAllClose(M_gt, M)

    def test_k3_valid_input(self):
        vec = np.array([
            [
                1., 2., 3.,
                1., 0., 0.,
                0., 1., 0.,
                0., 0., 1.,
                4., 5., 6., 7., 8., 9.
            ],
            [
                4., 5., 6.,
                0.1, 0.2, 0.3,
                0.4, 0.5, 0.6,
                0.7, 0.8, 0.9,
                7., 8., 9., 10., 11., 12.
            ],
            [
                7., 8., 9.,
                0.9, 0.8, 0.7,
                0.6, 0.5, 0.4,
                0.3, 0.2, 0.1,
                10., 11., 12., 13., 14., 15.
            ]
        ], dtype=npdtype)
        M_gt = np.array([
            [
                [1., 0., 0., 1.],
                [0., 1., 0., 2.],
                [0., 0., 1., 3.],
                [0., 0., 0., 1.],
            ],
            [
                [0.1, 0.2, 0.3, 4],
                [0.4, 0.5, 0.6, 5],
                [0.7, 0.8, 0.9, 6],
                [0., 0., 0., 1.]
            ],
            [
                [0.9, 0.8, 0.7, 7],
                [0.6, 0.5, 0.4, 8],
                [0.3, 0.2, 0.1, 9],
                [0., 0., 0., 1.]
            ]
        ], dtype=npdtype)
        M = self.toSE3matOP.forward(vec)
        self.assertAllClose(M_gt, M)


class TestSO3intOP(tf.test.TestCase):
    def setUp(self):
        self.so3intOP = SO3int()
    
    def test_k1_so3_int_valid(self):
        r = np.array([
            [
                [1., 0., 0.],
                [0., 1., 0.],
                [0., 0., 1.],
            ]
        ], dtype=npdtype)
        theta = np.pi/4.
        u = np.array([[0.36, 0.8, 0.48]], dtype=npdtype)
        tau = theta*u

        r_next_gt = np.array([
            [
                [0.74506574, -0.25505801, 0.61629737],
                [0.4237645, 0.89455844, -0.14208745],
                [-0.51507348, 0.36702944, 0.77458938]
            ]
        ], dtype=npdtype)

        r_next = self.so3intOP.forward(r, tau)
        det = tf.linalg.det(r_next)
        i = r_next @ tf.linalg.matrix_transpose(r_next)
        self.assertAllClose(r_next_gt, r_next)
        self.assertAllClose(det, [1])        
        self.assertAllClose(np.eye(3)[None, ...], i)

    def test_k3_so3_int_valid(self):
        r = np.array([
            [
                [1., 0., 0.],
                [0., 1., 0.],
                [0., 0., 1.],
            ],
            [
                [1., 0., 0.],
                [0., 1., 0.],
                [0., 0., 1.]
            ],
            [
                [0.98438195, 0.17604595, 0.],
                [-0.17604595, 0.98438195, 0.],
                [0., 0., 1.]
            ]
        ], dtype=npdtype)
        theta = np.array([np.pi/4., 0., np.pi/6.], dtype=npdtype)
        u = np.array([
            [0.36, 0.80, 0.48],
            [0.60, 0.64, 0.48],
            [0.00, 0.00, 1.00]
        ], dtype=npdtype)

        r_next_gt = np.array([
            [
                [0.74506574, -0.25505801, 0.61629737],
                [0.4237645, 0.89455844, -0.14208745],
                [-0.51507348, 0.36702944, 0.77458938]
            ],
            [
                [1., 0., 0.,],
                [0., 1., 0.],
                [0., 0., 1.]
            ],
            [
                [0.94052275, -0.33973071, 0.],
                [0.33973071, 0.94052275, 0.],
                [0., 0., 1.]
            ]
        ], dtype=npdtype)
        det_gt = np.ones(3)
        tau = theta[:, None]*u

        r_next = self.so3intOP.forward(r, tau)
        det = tf.linalg.det(r_next)
        i = r_next @ tf.linalg.matrix_transpose(r_next)
        theta = tf.math.acos(tf.clip_by_value((tf.linalg.trace(i)-1.)/2., -1, 1))
        self.assertAllClose(r_next_gt, r_next)
        self.assertAllClose(det_gt, det)        
        self.assertAllClose(np.vstack([np.identity(3)[None, ...] for i in range(3)]), i)

class TestSE3intOP(tf.test.TestCase):
    def setUp(self):
        self.se3intOP = SE3int()
    
    def test_k1_se3_int_valid(self):
        m = np.array([
            [
                [1., 0., 0., 0.],
                [0., 1., 0., 0.],
                [0., 0., 1., 0.],
                [0., 0., 0., 1.],
            ]
        ], dtype=npdtype)
        tau = np.array([
            [0., 0., 0., 0., 0., 0.]
        ], dtype=npdtype)
        m_next_gt = np.array([
            [
                [1., 0., 0., 0.],
                [0., 1., 0., 0.],
                [0., 0., 1., 0.],
                [0., 0., 0., 1.],
            ]
        ], dtype=npdtype)

        m_next = self.se3intOP.forward(m, tau)
        self.assertAllClose(m_next_gt, m_next)

    def test_k3_se3_int_valid(self):
        m = np.array([
            [
                [1., 0., 0., 0.],
                [0., 1., 0., 0.],
                [0., 0., 1., 0.],
                [0., 0., 0., 1.],
            ],
            [
                [1., 0., 0., 0.],
                [0., 1., 0., 0.],
                [0., 0., 1., 0.],
                [0., 0., 0., 1.],
            ],
            [
                [1., 0., 0., 0.],
                [0., 1., 0., 0.],
                [0., 0., 1., 0.],
                [0., 0., 0., 1.],
            ]
        ], dtype=npdtype)
        tau = np.array([
            [0., 0., 0., 0., 0., 0.],
            [0.1, 0.2, 0.3, np.pi/4.*0.36, np.pi/4.*0.80, np.pi/4.*0.48],
            [1., 2., 3., np.pi/6.*0.60, np.pi/6.*0.64, np.pi/6.*0.48],
        ], dtype=npdtype)
        v_gt = np.array([
            [
                [1., 0., 0.,],
                [0., 1., 0.,],
                [0., 0., 1.,],
            ],
            [
                [0.91323532, -0.15029425, 0.31556392],
                [0.20771205, 0.96411387, -0.09597383],
                [-0.28111324, 0.1725309,  0.92328344],
            ],
            [
                [0.97115498, -0.10551185, 0.17673874],
                [0.14012587, 0.97339047, -0.13967797],
                [-0.15077823, 0.16736919, 0.96531387],
            ],
        ])
        m_next_gt = np.array([
            [
                [1., 0., 0., 0.],
                [0., 1., 0., 0.],
                [0., 0., 1., 0.],
                [0., 0., 0., 1.],
            ],
            [
                [0.74506574, -0.25505801, 0.61629737, 0.15593386],
                [0.4237645, 0.89455844, -0.14208745, 0.18480183],
                [-0.51507348, 0.36702944, 0.77458938, 0.28337989],
                [0., 0., 0., 1.],
            ],
            [
                [0.91425626, -0.18855376, 0.35858468, 1.2903475],
                [0.29144624, 0.9209014, -0.258843, 1.66787291],
                [-0.28141532, 0.341157, 0.89689315, 3.07990175],
                [0., 0., 0., 1.],
            ],
        ], dtype=npdtype)

        v = self.se3intOP.v(tau[:, -3:])
        m_next = self.se3intOP.forward(m, tau)

        self.assertAllClose(v_gt, v)
        self.assertAllClose(m_next_gt, m_next)
