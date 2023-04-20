import numpy as np
import tensorflow as tf
import tensorflow_graphics.geometry.transformation as tfgt

from ..src.costs.cost_base import CostBase
from ..src.costs.cost_base import PrimitiveObstacles
from ..src.costs.cost_base import CylinderObstacle
from ..src.costs.static_cost import StaticCost
from ..src.costs.elipse_cost import ElipseCost
from ..src.costs.elipse_cost import ElipseCost3D
from ..src.misc.utile import dtype, npdtype

class TestCost(tf.test.TestCase):
    def setUp(self):
        self.dt=0.1

    def testStepCost_s2_a2_l1(self):

        state=np.array([[[0.], [1.]]], dtype=npdtype)
        action=np.array([[1.], [1.]], dtype=npdtype)
        noise=np.array([[[1.], [1.]]], dtype=npdtype)
        Sigma=np.array([[1., 0.], [0., 1.]], dtype=npdtype)
        lam=np.array([1.], dtype=npdtype)
        gamma = 1.
        upsilon = 1.

        cost = CostBase(lam, gamma, upsilon, Sigma)

        exp_a_c = np.array([[[ 0.5*(gamma*(2. + 4.) + lam[0]*(1-1./upsilon)*(0.) ) ]]], dtype=npdtype)


        with self.assertRaises(NotImplementedError):
            _ = cost.state_cost("", state)

        a_c = cost.action_cost("", action, noise)
        self.assertAllClose(exp_a_c, a_c)

    def testStepCost_s4_a2_l1(self):

        state=np.array([[[0.], [0.5], [2.], [0.]]], dtype=npdtype)
        action=np.array([[0.5], [2.]], dtype=npdtype)
        noise=np.array([[[0.5], [1.]]], dtype=npdtype)
        Sigma=np.array([[1., 0.], [0., 1.]], dtype=npdtype)

        lam=np.array([1.], dtype=npdtype)
        lam=np.array([1.], dtype=npdtype)
        gamma = 1.
        upsilon = 1.

        cost = CostBase(lam, gamma, upsilon, Sigma)

        exp_a_c = np.array([[[0.5*(gamma*(4.25 + 2.*2.25) + lam[0]*(1-1./upsilon)*(1.25) )]]], dtype=npdtype)

        with self.assertRaises(NotImplementedError):
            _ = cost.state_cost("", state)

        a_c = cost.action_cost("", action, noise)
        self.assertAllClose(exp_a_c, a_c)

    def testStepCost_s4_a3_l1(self):

        state=np.array([[[0.], [0.5], [2.], [0.]],
                        [[0.], [2.], [0.], [0.]],
                        [[10.], [2.], [2.], [3.]],
                        [[1.], [1.], [1.], [2.]],
                        [[3.], [4.], [5.], [6.]]], dtype=npdtype)
        action=np.array([[0.5], [2.], [0.25]], dtype=npdtype)
        noise=np.array([[[0.5], [1.], [2.]],
                        [[0.5], [2.], [0.25]],
                        [[-2], [-0.2], [-1]],
                        [[0.], [0.], [0.]],
                        [[1.], [0.5], [3.]]], dtype=npdtype)
        Sigma=np.array([[1., 0., 0.],
                        [0., 1., 0.],
                        [0., 0., 1.]], dtype=npdtype)
        lam=np.array([1.], dtype=npdtype)
        gamma = 1.
        upsilon = 1.

        cost = CostBase(lam, gamma, upsilon, Sigma)


        exp_a_c = np.array([[[0.5*(gamma*(4.3125 + 2.*2.75) + lam[0]*(1-1./upsilon)*(5.25) ) ]],
                            [[0.5*(gamma*(4.3125 + 2.*4.3125) + lam[0]*(1-1./upsilon)*(4.3125) ) ]],
                            [[0.5*(gamma*(4.3125 + 2.*-1.65) + lam[0]*(1-1./upsilon)*(5.04) ) ]],
                            [[0.5*(gamma*(4.3125 + 2.*0) + lam[0]*(1-1./upsilon)*(0) ) ]],
                            [[0.5*(gamma*(4.3125 + 2.*2.25) + lam[0]*(1-1./upsilon)*(10.25) ) ]]], dtype=npdtype)


        with self.assertRaises(NotImplementedError):
            _ = cost.state_cost("", state)

        a_c = cost.action_cost("", action, noise)
        self.assertAllClose(exp_a_c, a_c)

    def testStepCost_s4_a3_l10_g2_u3(self):

        state=np.array([[[0.], [0.5], [2.], [0.]],
                        [[0.], [2.], [0.], [0.]],
                        [[10.], [2.], [2.], [3.]],
                        [[1.], [1.], [1.], [2.]],
                        [[3.], [4.], [5.], [6.]]], dtype=npdtype)
        action=np.array([[0.5], [2.], [0.25]], dtype=npdtype)
        noise=np.array([[[0.5], [1.], [2.]],
                        [[0.5], [2.], [0.25]],
                        [[-2], [-0.2], [-1]],
                        [[0.], [0.], [0.]],
                        [[1.], [0.5], [3.]]], dtype=npdtype)
        Sigma=np.array([[1., 0., 0.],
                        [0., 1., 0.],
                        [0., 0., 1.]], dtype=npdtype)
        lam=np.array([10.], dtype=npdtype)
        gamma = 2.
        upsilon = 3.

        cost = CostBase(lam, gamma, upsilon, Sigma)


        exp_a_c = np.array([[[0.5*(gamma*(4.3125 + 2.*2.75) + lam[0]*(1-1./upsilon)*(5.25) ) ]],
                            [[0.5*(gamma*(4.3125 + 2.*4.3125) + lam[0]*(1-1./upsilon)*(4.3125) ) ]],
                            [[0.5*(gamma*(4.3125 + 2.*-1.65) + lam[0]*(1-1./upsilon)*(5.04) ) ]],
                            [[0.5*(gamma*(4.3125 + 2.*0) + lam[0]*(1-1./upsilon)*(0) ) ]],
                            [[0.5*(gamma*(4.3125 + 2.*2.25) + lam[0]*(1-1./upsilon)*(10.25) ) ]]], dtype=npdtype)


        with self.assertRaises(NotImplementedError):
            _ = cost.state_cost("", state)

        a_c = cost.action_cost("", action, noise)
        self.assertAllClose(exp_a_c, a_c)

    def testStepCost_s4_a3_l15_g20_u30(self):

        state=np.array([[[0.], [0.5], [2.], [0.]],
                        [[0.], [2.], [0.], [0.]],
                        [[10.], [2.], [2.], [3.]],
                        [[1.], [1.], [1.], [2.]],
                        [[3.], [4.], [5.], [6.]]], dtype=npdtype)
        action=np.array([[0.5], [2.], [0.25]], dtype=npdtype)
        noise=np.array([[[0.5], [1.], [2.]],
                        [[0.5], [2.], [0.25]],
                        [[-2], [-0.2], [-1]],
                        [[0.], [0.], [0.]],
                        [[1.], [0.5], [3.]]], dtype=npdtype)
        Sigma=np.array([[1., 0., 0.],
                        [0., 1., 0.],
                        [0., 0., 1.]], dtype=npdtype)
        lam=np.array([15.], dtype=npdtype)
        gamma = 20.
        upsilon = 30.

        cost = CostBase(lam, gamma, upsilon, Sigma)


        exp_a_c = np.array([[[0.5*(gamma*(4.3125 + 2.*2.75) + lam[0]*(1-1./upsilon)*(5.25) ) ]],
                            [[0.5*(gamma*(4.3125 + 2.*4.3125) + lam[0]*(1-1./upsilon)*(4.3125) ) ]],
                            [[0.5*(gamma*(4.3125 + 2.*-1.65) + lam[0]*(1-1./upsilon)*(5.04) ) ]],
                            [[0.5*(gamma*(4.3125 + 2.*0) + lam[0]*(1-1./upsilon)*(0) ) ]],
                            [[0.5*(gamma*(4.3125 + 2.*2.25) + lam[0]*(1-1./upsilon)*(10.25) ) ]]], dtype=npdtype)


        with self.assertRaises(NotImplementedError):
            _ = cost.state_cost("", state)

        a_c = cost.action_cost("", action, noise)
        self.assertAllClose(exp_a_c, a_c)


class TestPrimitiveCollision(tf.test.TestCase):
    def setUp(self):
        self.obstacle = PrimitiveObstacles("none-type")

    def testPrimitiveCollision(self):
        k, sDim = 5, 13
        state = tf.zeros(shape=(k, sDim))
        with self.assertRaises(NotImplementedError):
            self.obstacle.collide(state)


class TestCylinderCollision(tf.test.TestCase):
    def setUp(self):
        p1 = np.array([[0.], [0.], [0.]], dtype=npdtype)
        p2 = np.array([[0.], [0.], [1.]], dtype=npdtype)
        r = 1.
        self.obst = CylinderObstacle(p1, p2, r)
        pass

    def testCollision_k1(self):
        state = np.array([[
            [0.], [0.], [-1.]
        ]], dtype=npdtype)
        gt_mask = np.array([[[False]]], dtype=np.bool)
        mask = self.obst.collide(state)
        self.assertAllEqual(gt_mask, mask)
        pass

    def testCollision_k5(self):
        state = np.array([
            [[0.], [0.], [-1.]],
            [[-.5], [0.], [0.5]],
            [[1.], [1.], [0.5]],
            [[0.], [0.], [0.]],
            [[0.], [0.], [2.]]
        ], dtype=npdtype)
        gt_mask = np.array([
            [[False]],
            [[True]],
            [[False]],
            [[True]],
            [[False]]
        ], dtype=np.bool)
        mask = self.obst.collide(state)
        self.assertAllEqual(gt_mask, mask)
        pass


class TestStaticCost(tf.test.TestCase):
    def setUp(self):
        self.dt=0.1

    def testStepStaticCost_s2_a2_l1(self):

        state=np.array([[[0.], [1.]]], dtype=npdtype)
        goal=np.array([[1.], [1.]], dtype=npdtype)
        action=np.array([[1.], [1.]], dtype=npdtype)
        noise=np.array([[[1.], [1.]]], dtype=npdtype)
        Sigma=np.array([[1., 0.], [0., 1.]], dtype=npdtype)
        Q=np.array([[1., 0.], [0., 1.]], dtype=npdtype)
        lam=np.array([1.], dtype=npdtype)
        gamma = 1.
        upsilon = 1.

        cost = StaticCost(lam, gamma, upsilon, Sigma, goal, Q)

        exp_a_c = np.array([[[ 0.5*(gamma*(2. + 4.) + lam[0]*(1-upsilon)*(0.) ) ]]], dtype=npdtype)
        exp_s_c = np.array([[[1.]]], dtype=npdtype)


        exp_c = exp_a_c + exp_s_c

        a_c = cost.action_cost("", action, noise)
        c = cost.build_step_cost_graph("", state, action, noise)

        self.assertAllClose(exp_a_c, a_c)
        self.assertAllClose(exp_c, c)

    def testStepStaticCost_s4_a2_l1(self):

        state=np.array([[[0.], [0.5], [2.], [0.]]], dtype=npdtype)
        goal=np.array([[1.], [1.], [1.], [2.]], dtype=npdtype)
        action=np.array([[0.5], [2.]], dtype=npdtype)
        noise=np.array([[[0.5], [1.]]], dtype=npdtype)
        Sigma=np.array([[1., 0.], [0., 1.]], dtype=npdtype)
        Q=np.array([[1., 0., 0., 0.],
                   [0., 1., 0., 0.],
                   [0., 0., 10., 0.],
                   [0., 0., 0., 10.]], dtype=npdtype)

        lam=np.array([1.], dtype=npdtype)
        gamma = 1.
        upsilon = 1.

        cost = StaticCost(lam, gamma, upsilon, Sigma, goal, Q)

        exp_a_c = np.array([[[0.5*(gamma*(4.25 + 2.*2.25) + lam[0]*(1-upsilon)*(1.25) )]]], dtype=npdtype)

        exp_s_c = np.array([[[51.25]]], dtype=npdtype)

        exp_c = exp_a_c + exp_s_c


        a_c = cost.action_cost("", action, noise)
        c = cost.build_step_cost_graph("", state, action, noise)

        self.assertAllClose(exp_a_c, a_c)
        self.assertAllClose(exp_c, c)

    def testStepStaticCost_s4_a3_l1(self):
        state=np.array([[[0.], [0.5], [2.], [0.]],
                        [[0.], [2.], [0.], [0.]],
                        [[10.], [2.], [2.], [3.]],
                        [[1.], [1.], [1.], [2.]],
                        [[3.], [4.], [5.], [6.]]], dtype=npdtype)
        goal=np.array([[1.], [1.], [1.], [2.]], dtype=npdtype)
        action=np.array([[0.5], [2.], [0.25]], dtype=npdtype)
        noise=np.array([[[0.5], [1.], [2.]],
                        [[0.5], [2.], [0.25]],
                        [[-2], [-0.2], [-1]],
                        [[0.], [0.], [0.]],
                        [[1.], [0.5], [3.]]], dtype=npdtype)
        Sigma=np.array([[1., 0., 0.],
                        [0., 1., 0.],
                        [0., 0., 1.]], dtype=npdtype)
        Q=np.array([[1., 0., 0., 0.],
                   [0., 1., 0., 0.],
                   [0., 0., 10., 0.],
                   [0., 0., 0., 10.]], dtype=npdtype)
        lam=np.array([1.], dtype=npdtype)
        gamma = 1.
        upsilon = 1.

        cost = StaticCost(lam, gamma, upsilon, Sigma, goal, Q)

        exp_a_c = np.array([[[0.5*(gamma*(4.3125 + 2.*2.75) + lam[0]*(1-upsilon)*(5.25) ) ]],
                            [[0.5*(gamma*(4.3125 + 2.*4.3125) + lam[0]*(1-upsilon)*(4.3125) ) ]],
                            [[0.5*(gamma*(4.3125 + 2.*-1.65) + lam[0]*(1-upsilon)*(5.04) ) ]],
                            [[0.5*(gamma*(4.3125 + 2.*0) + lam[0]*(1-upsilon)*(0) ) ]],
                            [[0.5*(gamma*(4.3125 + 2.*2.25) + lam[0]*(1-upsilon)*(10.25) ) ]]], dtype=npdtype)

        exp_s_c = np.array([[[51.25]], [[52]], [[102]], [[0.]], [[333]]], dtype=npdtype)

        exp_c = exp_a_c + exp_s_c

        a_c = cost.action_cost("", action, noise)
        c = cost.build_step_cost_graph("", state, action, noise)

        self.assertAllClose(exp_a_c, a_c)
        self.assertAllClose(exp_c, c)

    def testStepStaticCost_s13_a6_l1(self):
        state=np.array([
                        [[0.], [0.5], [2.],
                         [0.], [0.], [0.], [1.],
                         [1.], [2.], [3.],
                         [4.], [5.], [6.]
                        ],
                        [
                         [0.], [2.], [0.],
                         [0.], [0.5], [0.5], [0.],
                         [4.], [5.], [6.],
                         [1.], [2.], [3.]
                        ]
                       ], dtype=npdtype)

        goal=np.array([
                       [1.], [1.], [2.],
                       [0.], [0.], [0.], [1.],
                       [0.], [0.], [0.],
                       [0.], [0.], [0.]
                      ], dtype=npdtype)

        action=np.array([
                         [0.5], [2.], [0.25], [4.], [1.], [1.5]
                        ], dtype=npdtype)

        noise=np.array([
                        [[0.5], [1.], [2.], [3.], [4.], [5.]],
                        [[0.5], [2.], [0.25], [1.25], [2.5], [0.75]]
                       ], dtype=npdtype)

        Sigma=np.array([
                        [1., 0., 0., 0., 0., 0.],
                        [0., 1., 0., 0., 0., 0.],
                        [0., 0., 1., 0., 0., 0.],
                        [0., 0., 0., 1., 0., 0.],
                        [0., 0., 0., 0., 1., 0.],
                        [0., 0., 0., 0., 0., 1.]
                       ], dtype=npdtype)

        Q=np.array([
                    [ # X
                     1., 0., 0.,
                     0., 0., 0., 0.,
                     0., 0., 0.,
                     0., 0., 0.
                    ],
                    [ # Y
                     0., 1., 0.,
                     0., 0., 0., 0.,
                     0., 0., 0.,
                     0., 0., 0.
                    ],
                    [ # Z
                     0., 0., 1.,
                     0., 0., 0., 0.,
                     0., 0., 0.,
                     0., 0., 0.
                    ],
                    [ # Q.X
                     0., 0., 0.,
                     1., 0., 0., 0.,
                     0., 0., 0.,
                     0., 0., 0.
                    ],
                    [ # Q.Y
                     0., 0., 0.,
                     0., 1., 0., 0.,
                     0., 0., 0.,
                     0., 0., 0.
                    ],
                    [ # Q.Z
                     0., 0., 0.,
                     0., 0., 1., 0.,
                     0., 0., 0.,
                     0., 0., 0.
                    ],
                    [ # Q.W
                     0., 0., 0.,
                     0., 0., 0., 1.,
                     0., 0., 0.,
                     0., 0., 0.
                    ],
                    [ # X DOT
                     0., 0., 0.,
                     0., 0., 0., 0.,
                     10., 0., 0.,
                     0., 0., 0.
                    ],
                    [ # Y DOT
                     0., 0., 0.,
                     0., 0., 0., 0.,
                     0., 10., 0.,
                     0., 0., 0.
                    ],
                    [ # Z DOT
                     0., 0., 0.,
                     0., 0., 0., 0.,
                     0., 0., 10.,
                     0., 0., 0.
                    ],
                    [ # ROLL DOT
                     0., 0., 0.,
                     0., 0., 0., 0.,
                     0., 0., 0.,
                     10., 0., 0.
                    ],
                    [ # PITCH DOT
                     0., 0., 0.,
                     0., 0., 0., 0.,
                     0., 0., 0.,
                     0., 10., 0.
                    ],
                    [ # YAW DOT
                     0., 0., 0.,
                     0., 0., 0., 0.,
                     0., 0., 0.,
                     0., 0., 10.
                    ],
                   ], dtype=npdtype)

        lam=np.array([1.], dtype=npdtype)
        gamma = 1.
        upsilon = 1.

        cost = StaticCost(lam, gamma, upsilon, Sigma, goal, Q)

        exp_a_c = np.array([
                            [
                             [0.5*(gamma*(23.5625 + 2.*26.25) + lam[0]*(1-1./upsilon)*(55.25) ) ]
                            ],
                            [
                             [0.5*(gamma*(23.5625 + 2.*(12.9375)) + lam[0]*(1-1./upsilon)*(12.6875) ) ]
                            ]
                           ], dtype=npdtype)

        exp_s_c = np.array([
                            [
                             [911.25]
                            ],
                            [
                             [917.5]
                            ]
                           ], dtype=npdtype)

        exp_c = exp_a_c + exp_s_c

        a_c = cost.action_cost("", action, noise)
        c = cost.build_step_cost_graph("", state, action, noise)

        self.assertAllClose(exp_a_c, a_c)
        self.assertAllClose(exp_c, c)


class TestStaticWCollision(tf.test.TestCase):
    def setUp(self):

        goal=np.array([
            [1.], [1.], [1.],
            [0.], [0.], [0.], [1.],
            [0.], [0.], [0.],
            [0.], [0.], [0.]
        ], dtype=npdtype)

        Sigma=np.array([
            [1., 0., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0., 0.],
            [0., 0., 1., 0., 0., 0.],
            [0., 0., 0., 1., 0., 0.],
            [0., 0., 0., 0., 1., 0.],
            [0., 0., 0., 0., 0., 1.]
        ], dtype=npdtype)

        Q=np.array([
            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
        ], dtype=npdtype)
        self.lam=np.array([1.], dtype=npdtype)
        self.gamma = 1.
        self.upsilon = 1.

        self.cost = StaticCost(self.lam, self.gamma, self.upsilon, Sigma, goal, Q)

        p11 = np.array([[0.], [0.], [0.]], dtype=npdtype)
        p12 = np.array([[0.], [0.], [1.]], dtype=npdtype)
        r = 1.
        obs1 = CylinderObstacle(p11, p12, r)
        self.cost.add_obstacle(obs1)
        pass

    def testStepStaticCostNoColl_k2(self):
        state = np.array([[
            [0.], [0.5], [2.],
            [0.], [0.], [0.], [1.],
            [1.], [2.], [3.],
            [4.], [5.], [6.]
        ],[
            [0.], [2.], [0.],
            [0.], [0.5], [0.5], [0.],
            [4.], [5.], [6.],
            [1.], [2.], [3.]
        ]], dtype=npdtype)

        action=np.array(
            [[0.5], [2.], [0.25], [4.], [1.], [1.5]],
        dtype=npdtype)

        noise=np.array([
            [[0.5], [1.], [2.], [3.], [4.], [5.]],
            [[0.5], [2.], [0.25], [1.25], [2.5], [0.75]]
        ], dtype=npdtype)
        
        exp_a_c = np.array([
            [[0.5*(self.gamma*(23.5625 + 2.*26.25) + self.lam[0]*(1-1./self.upsilon)*(55.25) )]],
            [[0.5*(self.gamma*(23.5625 + 2.*(12.9375)) + self.lam[0]*(1-1./self.upsilon)*(12.6875) )]]
        ], dtype=npdtype)

        exp_s_c = np.array([
            [[93.25]],
            [[95.5 ]]
        ], dtype=npdtype)

        exp_c_c = np.array([
            [[0.]],
            [[0.]]
        ])

        exp_c = exp_a_c + exp_s_c + exp_c_c

        a_c = self.cost.action_cost("", action, noise)
        c_c = self.cost.collision_cost("", state)
        c = self.cost.build_step_cost_graph("", state, action, noise)

        self.assertAllClose(exp_a_c, a_c)
        self.assertAllClose(exp_c_c, c_c)
        self.assertAllClose(exp_c, c)

    def testStepStaticCostColl_k5(self):
        state = np.array([[
            [0.], [0.5], [2.],
            [0.], [0.], [0.], [1.],
            [1.], [2.], [3.],
            [4.], [5.], [6.]
        ],[
            [0.], [2.], [0.],
            [0.], [0.5], [0.5], [0.],
            [4.], [5.], [6.],
            [1.], [2.], [3.]
        ],[
            [0.], [0.5], [0.5],
            [0.], [0.], [0.], [1.],
            [1.], [2.], [3.],
            [4.], [5.], [6.]
        ],[
            [0.], [-0.5], [0.5],
            [0.], [0.], [0.], [1.],
            [1.], [2.], [3.],
            [4.], [5.], [6.]
        ],[
            [0.], [0.5], [0.3],
            [0.], [0.], [0.], [1.],
            [1.], [2.], [3.],
            [4.], [5.], [6.]
        ]], dtype=npdtype)

        action=np.array(
            [[0.5], [2.], [0.25], [4.], [1.], [1.5]],
        dtype=npdtype)

        noise=np.array([
            [[0.5], [1.], [2.], [3.], [4.], [5.]],
            [[0.5], [2.], [0.25], [1.25], [2.5], [0.75]],
            [[0.5], [1.], [2.], [3.], [4.], [5.]],
            [[0.5], [2.], [0.25], [1.25], [2.5], [0.75]],
            [[0.5], [1.], [2.], [3.], [4.], [5.]],
        ], dtype=npdtype)
        
        exp_a_c = np.array([
            [[0.5*(self.gamma*(23.5625 + 2.*26.25) + self.lam[0]*(1-1./self.upsilon)*(55.25) )]],
            [[0.5*(self.gamma*(23.5625 + 2.*(12.9375)) + self.lam[0]*(1-1./self.upsilon)*(12.6875) )]],
            [[0.5*(self.gamma*(23.5625 + 2.*26.25) + self.lam[0]*(1-1./self.upsilon)*(55.25) )]],
            [[0.5*(self.gamma*(23.5625 + 2.*(12.9375)) + self.lam[0]*(1-1./self.upsilon)*(12.6875) )]],
            [[0.5*(self.gamma*(23.5625 + 2.*26.25) + self.lam[0]*(1-1./self.upsilon)*(55.25) )]],
        ], dtype=npdtype)

        exp_s_c = np.array([
            [[93.25]],
            [[95.5 ]],
            [[92.5 ]],
            [[94.5 ]],
            [[92.74]]
        ], dtype=npdtype)

        exp_c_c = np.array([
            [[0.]],
            [[0.]],
            [[np.inf]],
            [[np.inf]],
            [[np.inf]],
        ])

        exp_c = exp_a_c + exp_s_c + exp_c_c

        a_c = self.cost.action_cost("", action, noise)
        c_c = self.cost.collision_cost("", state)
        c = self.cost.build_step_cost_graph("", state, action, noise)

        self.assertAllClose(exp_a_c, a_c)
        self.assertAllClose(exp_c_c, c_c)
        self.assertAllClose(exp_c, c)


class TestElipseCost(tf.test.TestCase):
    def setUp(self):
        self.dt=0.1
        self.tau=1
        self.a=1
        self.b=1
        self.cx=0
        self.cy=0
        self.r=1
        self.s=1
        self.m_s=1
        self.m_v=1

    def testStepElipseCost_s4_l1_k1(self):

        state=np.array([[[0.], [0.5], [1.], [0.]]], dtype=npdtype)
        sigma=np.array([[1., 0.], [0., 1.]], dtype=npdtype)
        lam=np.array([1.], dtype=npdtype)
        gamma = 1.
        upsilon = 1.

        cost = ElipseCost(lam,
                          gamma,
                          upsilon,
                          sigma,
                          self.a, self.b, self.cx, self.cy,
                          self.s, self.m_s, self.m_v)

        exp_s_c = np.array([[[0.25]]], dtype=npdtype)

        s_c = cost.state_cost("", state)


        self.assertAllClose(exp_s_c, s_c)

    def testStepElipseCost_s4_l1_k5(self):

        state=np.array([[[0.], [0.5], [1.], [0.]],
                        [[0.], [2.], [0.], [0.]],
                        [[10.], [2.], [2.], [3.]],
                        [[1.], [1.], [1.], [2.]],
                        [[3.], [4.], [5.], [6.]]], dtype=npdtype)
        sigma=np.array([[1., 0., 0.],
                        [0., 1., 0.],
                        [0., 0., 1.]], dtype=npdtype)
        lam=np.array([1.], dtype=npdtype)
        gamma = 1.
        upsilon = 1.

        cost = ElipseCost(lam,
                          gamma,
                          upsilon,
                          sigma,
                          self.a, self.b, self.cx, self.cy,
                          self.s, self.m_s, self.m_v)

        exp_s_c = np.array([[[0.25]],
                            [[2]],
                            [[103+6.788897449072021]],
                            [[1+1.5278640450004208]],
                            [[33+38.57779489814404]]], dtype=npdtype)

        s_c = cost.state_cost("", state)
        self.assertAllClose(exp_s_c, s_c)


class TestElipse3DCost(tf.test.TestCase):
    def setUp(self):
        self.lam = 1.
        self.gamma = 1.
        self.upsilon = 1.
        self.sigma = np.array([
                               [1., 0., 0., 0., 0., 0.],
                               [0., 1., 0., 0., 0., 0.],
                               [0., 0., 1., 0., 0., 0.],
                               [0., 0., 0., 1., 0., 0.],
                               [0., 0., 0., 0., 1., 0.],
                               [0., 0., 0., 0. ,0., 1.]
                              ], dtype=npdtype)
        self.speed = 1.
        self.v_speed = 1.
        self.m_state = 1.
        self.m_vel = 1.
        self.axis = np.array([[2.], [1.5]], dtype=npdtype)

    def test_prep_const(self):
        normal = np.array([[0.], [0.], [1.]], dtype=npdtype)
        aVec = np.array([[1.], [0.], [0.]], dtype=npdtype)
        center = np.array([[0.], [0.], [0.]], dtype=npdtype)

        cost = ElipseCost3D(self.lam, self.gamma, self.upsilon,
                            self.sigma, normal, aVec, self.axis, center,
                            self.speed, self.m_state,
                            self.m_vel)
        exp_b = np.array([[0.], [1.], [0.]], dtype=npdtype)
        exp_r = np.array([
                          [1., 0., 0.],
                          [0., 1., 0.],
                          [0., 0., 1.]
                         ], dtype=npdtype)
        exp_t = center
        self.assertAllClose(cost.bVec, exp_b)
        self.assertAllClose(cost.R, exp_r)
        self.assertAllClose(cost.t, exp_t)

        normal = np.array([[0.], [1.], [1.]], dtype=npdtype)
        aVec = np.array([[1.], [0.], [0.]], dtype=npdtype)
        center = np.array([[0.], [1.], [-2.]], dtype=npdtype)

        cost = ElipseCost3D(self.lam, self.gamma, self.upsilon,
                            self.sigma, normal, aVec, self.axis, center,
                            self.speed, self.m_state,
                            self.m_vel)
        exp_b = np.array([[0.], [1.], [-1.]], dtype=npdtype)
        exp_r = np.array([
                          [1., 0., 0.],
                          [0., 0.5, -0.5],
                          [0., 0.5, 0.5]
                         ], dtype=npdtype).T
        exp_t = center
        self.assertAllClose(cost.bVec, exp_b)
        self.assertAllClose(cost.R, exp_r)
        self.assertAllClose(cost.t, exp_t)

    def test_position_error(self):
        # Already expressed in elipse frame
        position = np.array([
                             [[0.1], [0.4], [0.2]],
                             [[1.], [1.], [-2]],
                             [[2.], [0.], [0.]]
                            ], dtype=npdtype)
        normal = np.array([[0.], [1.], [1.]], dtype=npdtype)
        aVec = np.array([[1.], [0.], [0.]], dtype=npdtype)
        center = np.array([[0.], [1.], [-2.]], dtype=npdtype)

        cost = ElipseCost3D(self.lam, self.gamma, self.upsilon,
                            self.sigma, normal, aVec, self.axis, center,
                            self.speed, self.m_state,
                            self.m_vel)
        error = cost.position_error(position)
        
        exp_er = np.array([[[0.8863888888888889]],
                           [[3.6944444444444446]],
                           [[0.]]], dtype=npdtype)
        self.assertAllClose(error, exp_er)

    def test_orientation_error(self):
        pose = np.array([
                                [
                                 [0.1], [0.4], [0.2],
                                 [0.0], [0.0], [0.0], [1.] #0 roll, 0 pitch, 0 yaw.
                                ],
                                [
                                 [1.], [1.], [-2],
                                 [0.48038446], [0.32025631], [0.16012815], [0.80064077]
                                ],
                                [
                                 [2.], [1.], [-2],
                                 [0.20628425], [-0.30942637], [-0.92827912], [0.]
                                ]
                               ], dtype=npdtype)

        normal = np.array([[0.], [1.], [1.]], dtype=npdtype)
        aVec = np.array([[1.], [0.], [0.]], dtype=npdtype)
        center = np.array([[0.], [1.], [-2.]], dtype=npdtype)

        cost = ElipseCost3D(self.lam, self.gamma, self.upsilon,
                            self.sigma, normal, aVec, self.axis, center,
                            self.speed, self.m_state,
                            self.m_vel)

        tg_vec = np.array([[(-(self.axis[0, 0]/self.axis[1, 0])**2) * pose[0, 1, 0], pose[0, 0, 0], 0.],
                           [(-(self.axis[0, 0]/self.axis[1, 0])**2) * pose[1, 1, 0], pose[1, 0, 0], 0.],
                           [(-(self.axis[0, 0]/self.axis[1, 0])**2) * pose[2, 1, 0], pose[2, 0, 0], 0.]])[..., None]
        tg_vec /= np.linalg.norm(tg_vec, axis=1)[..., None]

        q_tg = np.array([[0., tg_vec[0, 2, 0], tg_vec[0, 1, 0], 1 + tg_vec[0, 0, 0]],
                         [0., tg_vec[1, 2, 0], tg_vec[1, 1, 0], 1 + tg_vec[1, 0, 0]],
                         [0., tg_vec[2, 2, 0], tg_vec[2, 1, 0], 1 + tg_vec[2, 0, 0]]])[..., None]
        q_tg /= np.linalg.norm(q_tg, axis=1)[..., None]

        exp_er_tg = 2* np.arccos(np.sum(q_tg[..., 0] * pose[:, 3:, 0], axis=-1))
        # Implementation differences between tensorflow and numpy makes a pi difference.
        exp_er_tg = np.array([
                           [[3.0018840093006306]],
                           [[2.4098026419889416]],
                           [[1.1216650246733544]]
                          ], dtype=npdtype)
        error_tg = cost.orientation_error_tg(pose)
        self.assertAllClose(error_tg, exp_er_tg)

        perp_vec = np.array([
            [pose[0, 0, 0], ((self.axis[0, 0]/self.axis[1, 0])**2) * pose[0, 1, 0], 0.],
            [pose[1, 0, 0], ((self.axis[0, 0]/self.axis[1, 0])**2) * pose[1, 1, 0], 0.],
            [pose[2, 0, 0], ((self.axis[0, 0]/self.axis[1, 0])**2) * pose[2, 1, 0], 0.],
        ])[..., None]
        perp_vec /= np.linalg.norm(perp_vec, axis=1)[..., None]

        q_perp = np.array([[0., perp_vec[0, 2, 0], -perp_vec[0, 1, 0], 1 + perp_vec[0, 0, 0]],
                           [0., perp_vec[1, 2, 0], -perp_vec[1, 1, 0], 1 + perp_vec[1, 0, 0]],
                           [0., perp_vec[2, 2, 0], -perp_vec[2, 1, 0], 1 + perp_vec[2, 0, 0]]])[..., None]
        q_perp /= np.linalg.norm(q_perp, axis=1)[..., None]

        exp_er_perp = 2* np.arccos(np.sum(q_perp[..., 0] * pose[:, 3:, 0], axis=-1))
        exp_er_perp = np.array([
                           [[1.43108746]],
                           [[1.3777549]],
                           [[2.46921356]],
                          ], dtype=npdtype)

        error_perp = cost.orientation_error_perp(pose)
        self.assertAllClose(error_perp, exp_er_perp)

    def test_velocity_error(self):
        velocitiy = np.array([
                                [
                                 [0.1], [0.4], [0.2],
                                 [0.0], [0.0], [0.0],
                                ],
                                [
                                 [1.], [1.], [-2],
                                 [0.3], [0.2], [0.1],
                                ],
                                [
                                 [2.], [1.], [-2],
                                 [0.2], [-0.3], [-0.9],
                                ]
                               ], dtype=npdtype)
        normal = np.array([[0.], [1.], [1.]], dtype=npdtype)
        aVec = np.array([[1.], [0.], [0.]], dtype=npdtype)
        center = np.array([[0.], [1.], [-2.]], dtype=npdtype)

        cost = ElipseCost3D(self.lam, self.gamma, self.upsilon,
                            self.sigma, normal, aVec, self.axis, center,
                            self.speed, self.m_state,
                            self.m_vel)
        error = cost.velocity_error(velocitiy)
        exp_er = np.abs(np.array([[[0.21-1]], [[6-1]], [[9-1]]]), dtype=npdtype)
        self.assertAllClose(error, exp_er)

    def test_state_cost(self):
        state = np.array([
                          [
                           [0.1], [0.4], [0.2],
                           [0.0], [0.0], [0.0], [1.],
                           [0.3], [0.7], [2.],
                           [1.], [2.4], [5.0]
                          ],
                          [
                           [1.], [1.], [-2],
                           [0.3], [0.2], [0.1], [0.5],
                           [0.4], [2.7], [2.],
                           [0.], [0.0], [0.0]
                          ],
                          [
                           [2.], [1.], [-2],
                           [0.2], [-0.3], [-0.9], [0.0],
                           [2.3], [1.7], [0.],
                           [0.1], [0.4], [0.01]
                          ] 
                         ], dtype=npdtype)
        normal = np.array([[0.], [1.], [1.]], dtype=npdtype)
        aVec = np.array([[1.], [0.], [0.]], dtype=npdtype)
        center = np.array([[0.], [1.], [-2.]], dtype=npdtype)


        cost = ElipseCost3D(self.lam, self.gamma, self.upsilon,
                            self.sigma, normal, aVec, self.axis, center,
                            self.speed, self.m_state,
                            self.m_vel)
        c = cost.state_cost("foo", state)
        exp_shape = np.zeros(shape=(3, 1, 1), dtype=npdtype)
        self.assertShapeEqual(exp_shape, c)

    def test_tf_rot(self):
        # plane is the zy plane
        self.q = np.array([0., 0.7071068, 0., 0.7071068], dtype=npdtype)

        # Pose x, y, z and roll 90, pitch 0, yaw 0.
        position = np.array([1., 2., 3.], dtype=npdtype)
        quat = np.array([0.7071068, 0., 0., 0.7071068], dtype=npdtype)
        posPf = tfgt.quaternion.rotate(position, self.q)
        posPf = tf.expand_dims(posPf, axis=-1)

        quatPf = tfgt.quaternion.multiply(self.q, quat)
        quatPf = tf.expand_dims(quatPf, axis=-1)

        exp_pos = np.array([[3.], [2.], [-1.]], dtype=npdtype)
        # roll = 90, pitch = 0, yaw = -90
        exp_quat = np.array([[0.5], [0.5], [-0.5], [0.5]], dtype=npdtype)
        self.assertAllClose(posPf, exp_pos)
        self.assertAllClose(quatPf, exp_quat)

