import numpy as np
import tensorflow as tf

from utile import npdtype, dtype

from cost import CostBase, FooCost

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

if __name__ == "__main__":
    tf.test.main()
