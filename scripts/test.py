import tensorflow as tf
from model_base import ModelBase
from point_mass_model import PointMassModel
from cost_base import CostBase
from static_cost import StaticCost
from elipse_cost import ElipseCost
from controller_base import ControllerBase
import numpy as np

class TestModel(tf.test.TestCase):
    def setUp(self):
        self.dt=0.1
        pass

    def test_step1_k1_s2_a1_m1(self):
        s = 2
        a = 1
        m = 1.
        model = PointMassModel(m, self.dt, s, a)

        state_in = np.array([[[0.], [0.]]])
        action_in = np.array([[[1.]]])

        acc = self.dt*self.dt/(2.*m)
        vel = self.dt/m
        exp_u = np.array([[[acc], [vel]]])
        exp_x = np.array([[[0.], [0.]]])
        exp = exp_u + exp_x

        x_pred = model.buildFreeStepGraph("", state_in)
        u_pred = model.buildActionStepGraph("", action_in)
        pred = model.buildStepGraph("", state_in, action_in)

        self.assertAllClose(exp_u, u_pred)
        self.assertAllClose(exp_x, x_pred)
        self.assertAllClose(exp, pred)

    def test_step1_k1_s4_a2_m1(self):
        s = 4
        a = 2
        m = 1.
        model = PointMassModel(m, self.dt, s, a)

        state_in = np.array([[[0.], [0.], [0.], [0.]]])
        action_in = np.array([[[1.], [1.]]])

        acc = self.dt*self.dt/(2.*m)
        vel = self.dt/m
        exp_u = np.array([[[acc], [vel], [acc], [vel]]])
        exp_x = np.array([[[0.], [0.], [0.], [0.]]])
        exp = exp_u + exp_x

        x_pred = model.buildFreeStepGraph("", state_in)
        u_pred = model.buildActionStepGraph("", action_in)
        pred = model.buildStepGraph("", state_in, action_in)

        self.assertAllClose(exp_u, u_pred)
        self.assertAllClose(exp_x, x_pred)
        self.assertAllClose(exp, pred)

    def test_step1_k5_s6_a3_m1d5(self):
        s = 6
        a = 3
        m = 1.5
        model = PointMassModel(m, self.dt, s, a)

        state_in = np.array([[[0.], [0.], [0.], [0.], [0.], [0.]],
                             [[2.], [1.], [5.], [0.], [-1.], [-2.]],
                             [[0.5], [0.5], [0.5], [0.5], [0.5], [0.5]],
                             [[1.], [0.], [1.], [0.], [1.], [0.]],
                             [[-1.], [0.5], [-3.], [2.], [0.], [0.]]])
        action_in = np.array([[[1.], [1.], [1.]],
                              [[2.], [0.], [-1.]],
                              [[0.], [0.], [0.]],
                              [[0.5], [-0.5], [0.5]],
                              [[3.], [3.], [3.]]])

        acc = self.dt*self.dt/(2.*m)
        vel = self.dt/m
        exp_u = np.array([[[acc], [vel], [acc], [vel], [acc], [vel]],
                          [[2.*acc], [2.*vel], [0.*acc], [0.*vel], [-1.*acc], [-1.*vel]],
                          [[0.*acc], [0.*vel], [0.*acc], [0.*vel], [0.*acc], [0.*vel]],
                          [[0.5*acc], [0.5*vel], [-0.5*acc], [-0.5*vel], [0.5*acc], [0.5*vel]],
                          [[3.*acc], [3.*vel], [3.*acc], [3.*vel], [3.*acc], [3.*vel]]])

        exp_x = np.array([[[0.], [0.], [0.], [0.], [0.], [0.]],
                          [[2.+self.dt], [1.], [5.], [0.], [-1.-2.*self.dt], [-2.]],
                          [[0.5+0.5*self.dt], [0.5], [0.5+0.5*self.dt], [0.5], [0.5+0.5*self.dt], [0.5]],
                          [[1.], [0.], [1.], [0.], [1.], [0.]],
                          [[-1.+0.5*self.dt], [0.5], [-3.+2.*self.dt], [2.], [0.], [0.]]])
        exp = exp_u + exp_x

        x_pred = model.buildFreeStepGraph("", state_in)
        u_pred = model.buildActionStepGraph("", action_in)
        pred = model.buildStepGraph("", state_in, action_in)

        self.assertAllClose(exp_u, u_pred)
        self.assertAllClose(exp_x, x_pred)
        self.assertAllClose(exp, pred)

    def test_init_k5_s6_a3_m1d5(self):
        s = 6
        a = 3
        m = 1.5
        model = PointMassModel(m, self.dt, s, a)

        state_in = np.array([[[-1.], [0.5], [-3.], [2.], [0.], [0.]]])
        action_in = np.array([[[1.], [1.], [1.]],
                              [[2.], [0.], [-1.]],
                              [[0.], [0.], [0.]],
                              [[0.5], [-0.5], [0.5]],
                              [[3.], [3.], [3.]]])

        acc = self.dt*self.dt/(2.*m)
        vel = self.dt/m
        exp_u = np.array([[[acc], [vel], [acc], [vel], [acc], [vel]],
                          [[2.*acc], [2.*vel], [0.*acc], [0.*vel], [-1.*acc], [-1.*vel]],
                          [[0.*acc], [0.*vel], [0.*acc], [0.*vel], [0.*acc], [0.*vel]],
                          [[0.5*acc], [0.5*vel], [-0.5*acc], [-0.5*vel], [0.5*acc], [0.5*vel]],
                          [[3.*acc], [3.*vel], [3.*acc], [3.*vel], [3.*acc], [3.*vel]]])

        exp_x = np.array([[[-1+0.5*self.dt], [0.5], [-3.+2.*self.dt], [2.], [0.], [0.]],
                          [[-1+0.5*self.dt], [0.5], [-3.+2.*self.dt], [2.], [0.], [0.]],
                          [[-1+0.5*self.dt], [0.5], [-3.+2.*self.dt], [2.], [0.], [0.]],
                          [[-1+0.5*self.dt], [0.5], [-3.+2.*self.dt], [2.], [0.], [0.]],
                          [[-1+0.5*self.dt], [0.5], [-3.+2.*self.dt], [2.], [0.], [0.]]])

        exp = exp_u + exp_x

        exp_x = np.expand_dims(exp_x[0, :, :], 0)

        x_pred = model.buildFreeStepGraph("", state_in)
        u_pred = model.buildActionStepGraph("", action_in)
        pred = model.buildStepGraph("", state_in, action_in)

        self.assertAllClose(exp_u, u_pred)
        self.assertAllClose(exp_x, x_pred)
        self.assertAllClose(exp, pred)

    def test_step3_k5_s6_a3_m1d5(self):
        s = 6
        a = 3
        m = 1.5
        model = PointMassModel(m, self.dt, s, a)

        state_in = np.array([[[0.], [0.], [0.], [0.], [0.], [0.]],
                             [[2.], [1.], [5.], [0.], [-1.], [-2.]],
                             [[0.5], [0.5], [0.5], [0.5], [0.5], [0.5]],
                             [[1.], [0.], [1.], [0.], [1.], [0.]],
                             [[-1.], [0.5], [-3.], [2.], [0.], [0.]]])
        action_in = np.array([[[1.], [1.], [1.]],
                              [[2.], [0.], [-1.]],
                              [[0.], [0.], [0.]],
                              [[0.5], [-0.5], [0.5]],
                              [[3.], [3.], [3.]]])

        acc = self.dt*self.dt/(2.*m)
        vel = self.dt/m

        # exp = A**3 * state_in + B*action_in * ( A**2 + A + I)

        # A**3 = | 1 self.dt**3 |
        #        | 0        1   |
        # A**2 = | 1 self.dt**2 |
        #        | 0        1   |

        B_u = np.array([[[1.0*3*(acc+vel*self.dt)], [1.0*vel*3], [1.0*3*(acc+vel*self.dt)], [1.0*vel*3], [1.0*3*(acc+vel*self.dt)], [1.0*vel*3]],
                        [[2.0*3*(acc+vel*self.dt)], [2.0*vel*3], [0.0*3*(acc+vel*self.dt)], [0.0*vel*3], [-1.0*3*(acc+vel*self.dt)], [-1.0*vel*3]],
                        [[0.0*3*(acc+vel*self.dt)], [0.0*vel*3], [0.0*3*(acc+vel*self.dt)], [0.0*vel*3], [0.0*3*(acc+vel*self.dt)], [0.0*vel*3]],
                        [[0.5*3*(acc+vel*self.dt)], [0.5*vel*3], [-0.5*3*(acc+vel*self.dt)], [-0.5*vel*3], [0.5*3*(acc+vel*self.dt)], [0.5*vel*3]],
                        [[3.0*3*(acc+vel*self.dt)], [3.0*vel*3], [3.0*3*(acc+vel*self.dt)], [3.0*vel*3], [3.0*3*(acc+vel*self.dt)], [3.0*vel*3]]])

        exp_x = np.array([[[0.], [0.], [0.], [0.], [0.], [0.]],
                          [[2.+self.dt*3], [1.], [5.], [0.], [-1.-2.*self.dt*3], [-2.]],
                          [[0.5+0.5*self.dt*3], [0.5], [0.5+0.5*self.dt*3], [0.5], [0.5+0.5*self.dt*3], [0.5]],
                          [[1.], [0.], [1.], [0.], [1.], [0.]],
                          [[-1.+0.5*self.dt*3], [0.5], [-3.+2.*self.dt*3], [2.], [0.], [0.]]])

        exp = B_u + exp_x

        pred = model.buildStepGraph("", state_in, action_in)
        pred = model.buildStepGraph("", pred, action_in)
        pred = model.buildStepGraph("", pred, action_in)

        self.assertAllClose(exp, pred)

    def test_training(self):
        s = 4
        a = 2
        m = 1.5
        model = PointMassModel(m, self.dt, s, a)

        x = np.array([[0.5], [0.1], [0.2], [-0.1]])
        u = np.array([[0.2], [-0.15]])
        gt = np.array([[0.514], [0.14], [0.187], [-0.13]])

        model.train_step(gt, x, u)

        mt = model.getMass()

        self.assertAllLessEqual(mt, m)

class TestCost(tf.test.TestCase):
    def setUp(self):
        self.dt=0.1

    def testStepCost_s2_a2_l1(self):

        state=np.array([[[0.], [1.]]])
        action=np.array([[1.], [1.]])
        noise=np.array([[[1.], [1.]]])
        Sigma=np.array([[1., 0.], [0., 1.]])
        lam=np.array([1.])
        gamma = 1.
        upsilon = 1.

        cost = CostBase(lam, gamma, upsilon, Sigma)

        exp_a_c = np.array([[[ 0.5*(gamma*(2. + 4.) + lam[0]*(1-1./upsilon)*(0.) ) ]]])


        with self.assertRaises(NotImplementedError):
            _ = cost.state_cost("", state)

        a_c_dict = cost.action_cost("", action, noise)
        self.assertAllClose(exp_a_c, a_c_dict["action_cost"])

    def testStepCost_s4_a2_l1(self):

        state=np.array([[[0.], [0.5], [2.], [0.]]])
        action=np.array([[0.5], [2.]])
        noise=np.array([[[0.5], [1.]]])
        Sigma=np.array([[1., 0.], [0., 1.]])

        lam=np.array([1.])
        lam=np.array([1.])
        gamma = 1.
        upsilon = 1.

        cost = CostBase(lam, gamma, upsilon, Sigma)

        exp_a_c = np.array([[[0.5*(gamma*(4.25 + 2.*2.25) + lam[0]*(1-1./upsilon)*(1.25) )]]])

        with self.assertRaises(NotImplementedError):
            _ = cost.state_cost("", state)

        a_c_dict = cost.action_cost("", action, noise)
        self.assertAllClose(exp_a_c, a_c_dict["action_cost"])

    def testStepCost_s4_a3_l1(self):

        state=np.array([[[0.], [0.5], [2.], [0.]],
                        [[0.], [2.], [0.], [0.]],
                        [[10.], [2.], [2.], [3.]],
                        [[1.], [1.], [1.], [2.]],
                        [[3.], [4.], [5.], [6.]]])
        action=np.array([[0.5], [2.], [0.25]])
        noise=np.array([[[0.5], [1.], [2.]],
                        [[0.5], [2.], [0.25]],
                        [[-2], [-0.2], [-1]],
                        [[0.], [0.], [0.]],
                        [[1.], [0.5], [3.]]])
        Sigma=np.array([[1., 0., 0.],
                        [0., 1., 0.],
                        [0., 0., 1.]])
        lam=np.array([1.])
        gamma = 1.
        upsilon = 1.

        cost = CostBase(lam, gamma, upsilon, Sigma)


        exp_a_c = np.array([[[0.5*(gamma*(4.3125 + 2.*2.75) + lam[0]*(1-1./upsilon)*(5.25) ) ]],
                            [[0.5*(gamma*(4.3125 + 2.*4.3125) + lam[0]*(1-1./upsilon)*(4.3125) ) ]],
                            [[0.5*(gamma*(4.3125 + 2.*-1.65) + lam[0]*(1-1./upsilon)*(5.04) ) ]],
                            [[0.5*(gamma*(4.3125 + 2.*0) + lam[0]*(1-1./upsilon)*(0) ) ]],
                            [[0.5*(gamma*(4.3125 + 2.*2.25) + lam[0]*(1-1./upsilon)*(10.25) ) ]]])


        with self.assertRaises(NotImplementedError):
            _ = cost.state_cost("", state)

        a_c_dict = cost.action_cost("", action, noise)
        self.assertAllClose(exp_a_c, a_c_dict["action_cost"])

    def testStepCost_s4_a3_l10_g2_u3(self):

        state=np.array([[[0.], [0.5], [2.], [0.]],
                        [[0.], [2.], [0.], [0.]],
                        [[10.], [2.], [2.], [3.]],
                        [[1.], [1.], [1.], [2.]],
                        [[3.], [4.], [5.], [6.]]])
        action=np.array([[0.5], [2.], [0.25]])
        noise=np.array([[[0.5], [1.], [2.]],
                        [[0.5], [2.], [0.25]],
                        [[-2], [-0.2], [-1]],
                        [[0.], [0.], [0.]],
                        [[1.], [0.5], [3.]]])
        Sigma=np.array([[1., 0., 0.],
                        [0., 1., 0.],
                        [0., 0., 1.]])
        lam=np.array([10.])
        gamma = 2.
        upsilon = 3.

        cost = CostBase(lam, gamma, upsilon, Sigma)


        exp_a_c = np.array([[[0.5*(gamma*(4.3125 + 2.*2.75) + lam[0]*(1-1./upsilon)*(5.25) ) ]],
                            [[0.5*(gamma*(4.3125 + 2.*4.3125) + lam[0]*(1-1./upsilon)*(4.3125) ) ]],
                            [[0.5*(gamma*(4.3125 + 2.*-1.65) + lam[0]*(1-1./upsilon)*(5.04) ) ]],
                            [[0.5*(gamma*(4.3125 + 2.*0) + lam[0]*(1-1./upsilon)*(0) ) ]],
                            [[0.5*(gamma*(4.3125 + 2.*2.25) + lam[0]*(1-1./upsilon)*(10.25) ) ]]])


        with self.assertRaises(NotImplementedError):
            _ = cost.state_cost("", state)

        a_c_dict = cost.action_cost("", action, noise)
        self.assertAllClose(exp_a_c, a_c_dict["action_cost"])

    def testStepCost_s4_a3_l15_g20_u30(self):

        state=np.array([[[0.], [0.5], [2.], [0.]],
                        [[0.], [2.], [0.], [0.]],
                        [[10.], [2.], [2.], [3.]],
                        [[1.], [1.], [1.], [2.]],
                        [[3.], [4.], [5.], [6.]]])
        action=np.array([[0.5], [2.], [0.25]])
        noise=np.array([[[0.5], [1.], [2.]],
                        [[0.5], [2.], [0.25]],
                        [[-2], [-0.2], [-1]],
                        [[0.], [0.], [0.]],
                        [[1.], [0.5], [3.]]])
        Sigma=np.array([[1., 0., 0.],
                        [0., 1., 0.],
                        [0., 0., 1.]])
        lam=np.array([15.])
        gamma = 20.
        upsilon = 30.

        cost = CostBase(lam, gamma, upsilon, Sigma)


        exp_a_c = np.array([[[0.5*(gamma*(4.3125 + 2.*2.75) + lam[0]*(1-1./upsilon)*(5.25) ) ]],
                            [[0.5*(gamma*(4.3125 + 2.*4.3125) + lam[0]*(1-1./upsilon)*(4.3125) ) ]],
                            [[0.5*(gamma*(4.3125 + 2.*-1.65) + lam[0]*(1-1./upsilon)*(5.04) ) ]],
                            [[0.5*(gamma*(4.3125 + 2.*0) + lam[0]*(1-1./upsilon)*(0) ) ]],
                            [[0.5*(gamma*(4.3125 + 2.*2.25) + lam[0]*(1-1./upsilon)*(10.25) ) ]]])


        with self.assertRaises(NotImplementedError):
            _ = cost.state_cost("", state)

        a_c_dict = cost.action_cost("", action, noise)
        self.assertAllClose(exp_a_c, a_c_dict["action_cost"])


class TestStaticCost(tf.test.TestCase):
    def setUp(self):
        self.dt=0.1

    def testStepStaticCost_s2_a2_l1(self):

        state=np.array([[[0.], [1.]]])
        goal=np.array([[1.], [1.]])
        action=np.array([[1.], [1.]])
        noise=np.array([[[1.], [1.]]])
        Sigma=np.array([[1., 0.], [0., 1.]])
        Q=np.array([[1., 0.], [0., 1.]])
        lam=np.array([1.])
        gamma = 1.
        upsilon = 1.

        cost = StaticCost(lam, gamma, upsilon, Sigma, goal, Q)

        exp_a_c = np.array([[[ 0.5*(gamma*(2. + 4.) + lam[0]*(1-upsilon)*(0.) ) ]]])
        exp_s_c = np.array([[[1.]]])


        exp_c = exp_a_c + exp_s_c

        a_c_dict = cost.action_cost("", action, noise)
        c_dict = cost.build_step_cost_graph("", state, action, noise)

        self.assertAllClose(exp_a_c, a_c_dict["action_cost"])
        self.assertAllClose(exp_s_c, c_dict["state_cost"])
        self.assertAllClose(exp_c, c_dict["cost"])

    def testStepStaticCost_s4_a2_l1(self):

        state=np.array([[[0.], [0.5], [2.], [0.]]])
        goal=np.array([[1.], [1.], [1.], [2.]])
        action=np.array([[0.5], [2.]])
        noise=np.array([[[0.5], [1.]]])
        Sigma=np.array([[1., 0.], [0., 1.]])
        Q=np.array([[1., 0., 0., 0.],
                   [0., 1., 0., 0.],
                   [0., 0., 10., 0.],
                   [0., 0., 0., 10.]])

        lam=np.array([1.])
        gamma = 1.
        upsilon = 1.

        cost = StaticCost(lam, gamma, upsilon, Sigma, goal, Q)

        exp_a_c = np.array([[[0.5*(gamma*(4.25 + 2.*2.25) + lam[0]*(1-upsilon)*(1.25) )]]])

        exp_s_c = np.array([[[51.25]]])

        exp_c = exp_a_c + exp_s_c


        a_c_dict = cost.action_cost("", action, noise)
        c_dict = cost.build_step_cost_graph("", state, action, noise)

        self.assertAllClose(exp_a_c, a_c_dict["action_cost"])
        self.assertAllClose(exp_s_c, c_dict["state_cost"])
        self.assertAllClose(exp_c, c_dict["cost"])

    def testStepStaticCost_s4_a3_l1(self):

        state=np.array([[[0.], [0.5], [2.], [0.]],
                        [[0.], [2.], [0.], [0.]],
                        [[10.], [2.], [2.], [3.]],
                        [[1.], [1.], [1.], [2.]],
                        [[3.], [4.], [5.], [6.]]])
        goal=np.array([[1.], [1.], [1.], [2.]])
        action=np.array([[0.5], [2.], [0.25]])
        noise=np.array([[[0.5], [1.], [2.]],
                        [[0.5], [2.], [0.25]],
                        [[-2], [-0.2], [-1]],
                        [[0.], [0.], [0.]],
                        [[1.], [0.5], [3.]]])
        Sigma=np.array([[1., 0., 0.],
                        [0., 1., 0.],
                        [0., 0., 1.]])
        Q=np.array([[1., 0., 0., 0.],
                   [0., 1., 0., 0.],
                   [0., 0., 10., 0.],
                   [0., 0., 0., 10.]])
        lam=np.array([1.])
        gamma = 1.
        upsilon = 1.

        cost = StaticCost(lam, gamma, upsilon, Sigma, goal, Q)

        exp_a_c = np.array([[[0.5*(gamma*(4.3125 + 2.*2.75) + lam[0]*(1-upsilon)*(5.25) ) ]],
                            [[0.5*(gamma*(4.3125 + 2.*4.3125) + lam[0]*(1-upsilon)*(4.3125) ) ]],
                            [[0.5*(gamma*(4.3125 + 2.*-1.65) + lam[0]*(1-upsilon)*(5.04) ) ]],
                            [[0.5*(gamma*(4.3125 + 2.*0) + lam[0]*(1-upsilon)*(0) ) ]],
                            [[0.5*(gamma*(4.3125 + 2.*2.25) + lam[0]*(1-upsilon)*(10.25) ) ]]])

        exp_s_c = np.array([[[51.25]], [[52]], [[102]], [[0.]], [[333]]])

        exp_c = exp_a_c + exp_s_c

        a_c_dict = cost.action_cost("", action, noise)
        c_dict = cost.build_step_cost_graph("", state, action, noise)

        self.assertAllClose(exp_a_c, a_c_dict["action_cost"])
        self.assertAllClose(exp_s_c, c_dict["state_cost"])
        self.assertAllClose(exp_c, c_dict["cost"])


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

        state=np.array([[[0.], [0.5], [1.], [0.]]])
        sigma=np.array([[1., 0.], [0., 1.]])
        lam=np.array([1.])
        gamma = 1.
        upsilon = 1.

        cost = ElipseCost(lam,
                          gamma,
                          upsilon,
                          sigma,
                          self.a, self.b, self.cx, self.cy,
                          self.s, self.m_s, self.m_v)

        exp_s_c = np.array([[[0.25]]])

        s_c_dict = cost.state_cost("", state)


        self.assertAllClose(exp_s_c, s_c_dict["state_cost"])

    def testStepElipseCost_s4_l1_k5(self):

        state=np.array([[[0.], [0.5], [1.], [0.]],
                        [[0.], [2.], [0.], [0.]],
                        [[10.], [2.], [2.], [3.]],
                        [[1.], [1.], [1.], [2.]],
                        [[3.], [4.], [5.], [6.]]])
        sigma=np.array([[1., 0., 0.],
                        [0., 1., 0.],
                        [0., 0., 1.]])
        lam=np.array([1.])
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
                            [[33+38.57779489814404]]])

        s_c_dict = cost.state_cost("", state)
        self.assertAllClose(exp_s_c, s_c_dict["state_cost"])


class TestController(tf.test.TestCase):
    def setUp(self):
        self.k = 5
        self.tau = 3
        self.a_dim = 2
        self.s_dim = 4
        self.dt = 0.01
        self.mass = 1.
        self.lam = 1.
        self.gamma = 1.
        self.upsilon = 1.

        self.goal = np.array([[0.], [1.]])
        self.sigma = np.array([[1., 0.], [0., 1.]])
        self.Q = np.array([[1., 0., 0., 0.],
                           [0., 1., 0., 0.],
                           [0., 0., 1., 0.],
                           [0., 0., 0., 1.]])

        self.c = np.array([[[3.]], [[10.]], [[0.]], [[1.]], [[5.]]])
        self.n = np.array([[[[1.], [-0.5]], [[1.], [-0.5]], [[2.], [1.]]],
                           [[[0.3], [0.]], [[2.], [0.2]], [[1.2], [3.]]],
                           [[[0.5], [0.5]], [[0.5], [0.5]], [[0.5], [0.5]]],
                           [[[0.6], [0.7]], [[0.2], [-0.3]], [[0.1], [-0.4]]],
                           [[[-2.], [-3.]], [[-4.], [-1.]], [[0.], [0.]]]])

        self.a = np.array([[[1.], [0.5]], [[2.3], [4.5]], [[2.1], [-0.4]]])
        model = PointMassModel(self.mass, self.dt, self.s_dim, self.a_dim)
        cost = StaticCost(self.lam, self.gamma, self.upsilon, self.sigma, self.goal, self.Q)
        self.cont = ControllerBase(model,
                                   cost,
                                   self.k,
                                   self.tau,
                                   self.dt,
                                   self.s_dim,
                                   self.a_dim,
                                   self.lam,
                                   self.sigma)

    def testDataPrep(self):
        exp_a0 = np.array([[1.], [0.5]])
        exp_a1 = np.array([[2.3], [4.5]])
        exp_a2 = np.array([[2.1], [-0.4]])
        exp_n0 = np.array([[[1.], [-0.5]], [[0.3], [0.]], [[0.5], [0.5]], [[0.6], [0.7]], [[-2.], [-3.]]])
        exp_n1 = np.array([[[1.], [-0.5]], [[2.], [0.2]], [[0.5], [0.5]], [[0.2], [-0.3]], [[-4], [-1]]])
        exp_n2 = np.array([[[2.], [1.]], [[1.2], [3.]], [[0.5], [0.5]], [[0.1], [-0.4]], [[0.], [0.]]])


        a0 = self.cont.prepareAction("", self.a, 0)
        n0 = self.cont.prepareNoise("", self.n, 0)

        a1 = self.cont.prepareAction("", self.a, 1)
        n1 = self.cont.prepareNoise("", self.n, 1)

        a2 = self.cont.prepareAction("", self.a, 2)
        n2 = self.cont.prepareNoise("", self.n, 2)

        self.assertAllClose(a0, exp_a0)
        self.assertAllClose(n0, exp_n0)

        self.assertAllClose(a1, exp_a1)
        self.assertAllClose(n1, exp_n1)

        self.assertAllClose(a2, exp_a2)
        self.assertAllClose(n2, exp_n2)

    def testUpdate(self):
        beta = np.array([[0]])
        exp_arg = np.array([[[-3.]], [[-10.]], [[0.]], [[-1.]], [[-5.]]])
        exp = np.array([[[0.049787068367863944]],
                        [[4.5399929762484854e-05]],
                        [[1]],
                        [[0.36787944117144233]],
                        [[0.006737946999085467]]])

        nabla = np.array([[1.424449856468154]])
        weights = np.array([[[0.034951787275480706]],
                           [[3.1871904480408675e-05]],
                           [[0.7020254138530686]],
                           [[0.2582607169364174]],
                           [[0.004730210030553017]]])

        expected = np.array([[[0.034951787275480706*1. + 3.1871904480408675e-05*0.3 + 0.7020254138530686*0.5 + 0.2582607169364174*0.6 + 0.004730210030553017*(-2)],
                              [0.034951787275480706*(-0.5) + 3.1871904480408675e-05*0 + 0.7020254138530686*0.5 + 0.2582607169364174*0.7 + 0.004730210030553017*(-3)]],
                             [[0.034951787275480706*1 + 3.1871904480408675e-05*2 + 0.7020254138530686*0.5 + 0.2582607169364174*0.2 + 0.004730210030553017*(-4)],
                              [0.034951787275480706*(-0.5) + 3.1871904480408675e-05*0.2 + 0.7020254138530686*0.5 + 0.2582607169364174*(-0.3) + 0.004730210030553017*(-1)]],
                             [[0.034951787275480706*2 + 3.1871904480408675e-05*1.2 + 0.7020254138530686*0.5 + 0.2582607169364174*0.1 + 0.004730210030553017*0],
                              [0.034951787275480706*1 + 3.1871904480408675e-05*3 + 0.7020254138530686*0.5 + 0.2582607169364174*(-0.4) + 0.004730210030553017*0]]])

        b = self.cont.beta("", self.c)
        arg = self.cont.normArg("", self.c, b, False)
        e_arg = self.cont.expArg("", arg)
        e = self.cont.exp("", e_arg)
        nab = self.cont.nabla("", e)
        w = self.cont.weights("", e, nab)
        w_n = self.cont.weightedNoise("", w, self.n)
        sum = tf.reduce_sum(w)


        self.assertAllClose(b, beta)
        self.assertAllClose(e_arg, exp_arg)

        self.assertAllClose(e, exp)
        self.assertAllClose(nab, nabla)

        self.assertAllClose(w, weights)
        self.assertAllClose(w_n, expected)
        self.assertAllClose(sum, 1.)

    def testNew(self):
        next1 = np.array([[[1.], [0.5]]])
        next2 = np.array([[[1.], [0.5]], [[2.3], [4.5]]])
        next3 = np.array([[[1.], [0.5]], [[2.3], [4.5]], [[2.1], [-0.4]]])

        n1 = self.cont.getNext("", self.a, 1)
        n2 = self.cont.getNext("", self.a, 2)
        n3 = self.cont.getNext("", self.a, 3)

        self.assertAllClose(n1, next1)
        self.assertAllClose(n2, next2)
        self.assertAllClose(n3, next3)

    def testShiftAndInit(self):
        init1 = np.array([[[1.], [0.5]]])
        init2 = np.array([[[1.], [0.5]], [[2.3], [4.5]]])

        exp1 = np.array([[[2.3], [4.5]], [[2.1], [-0.4]], [[1.], [0.5]]])
        exp2 = np.array([[[2.1], [-0.4]], [[1.], [0.5]], [[2.3], [4.5]]])

        n1 = self.cont.shift("", self.a, init1, 1)
        n2 = self.cont.shift("", self.a, init2, 2)

        self.assertAllClose(n1, exp1)
        self.assertAllClose(n2, exp2)

    def testAll(self):
        pass


tf.test.main()
