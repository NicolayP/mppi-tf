import tensorflow as tf
from model_base import ModelBase
from cost_base import CostBase
from controller_base import ControllerBase
import numpy as np

class TestModel(tf.test.TestCase):
    def setUp(self):
        self.dt=0.1
        pass

    def test_step1_k1_s2_a1_m1(self):
        k = 1
        s = 2
        a = 1
        m = 1.
        model = ModelBase(m, self.dt, s, a)

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
        k = 1
        s = 4
        a = 2
        m = 1.
        model = ModelBase(m, self.dt, s, a)

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
        k = 5
        s = 6
        a = 3
        m = 1.5
        model = ModelBase(m, self.dt, s, a)

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
        k = 5
        s = 6
        a = 3
        m = 1.5
        model = ModelBase(m, self.dt, s, a)

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
        k = 5
        s = 6
        a = 3
        m = 1.5
        model = ModelBase(m, self.dt, s, a)

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


class TestCost(tf.test.TestCase):
    def setUp(self):
        self.dt=0.1

    def testStepCost_s2_a2_l1(self):

        state=np.array([[[0.], [1.]]])
        goal=np.array([[1.], [1.]])
        action=np.array([[1.], [1.]])
        noise=np.array([[[1.], [1.]]])
        Sigma=np.array([[1., 0.], [0., 1.]])
        Q=np.array([[1., 0.], [0., 1.]])
        lam=np.array([1.])

        cost = CostBase(lam, Sigma, goal, Q)

        exp_a_c = np.array([[[2.]]])
        exp_s_c = np.array([[[1.]]])


        exp_c = exp_a_c + exp_s_c

        a_c = cost.action_cost("", action, noise)
        s_c = cost.state_cost("", state)
        c = cost.build_step_cost_graph("", state, action, noise)

        self.assertAllClose(exp_a_c, a_c)
        self.assertAllClose(exp_s_c, s_c)
        self.assertAllClose(exp_c, c)

    def testStepCost_s4_a2_l1(self):

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

        cost = CostBase(lam, Sigma, goal, Q)

        exp_a_c = np.array([[[2.25]]])
        exp_s_c = np.array([[[51.25]]])

        exp_c = exp_a_c + exp_s_c


        a_c = cost.action_cost("", action, noise)
        s_c = cost.state_cost("", state)
        c = cost.build_step_cost_graph("", state, action, noise)


        self.assertAllClose(exp_a_c, a_c)
        self.assertAllClose(exp_s_c, s_c)
        self.assertAllClose(exp_c, c)


    def testStepCost_s4_a3_l1(self):

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

        cost = CostBase(lam, Sigma, goal, Q)

        exp_a_c = np.array([[[2.75]], [[4.3125]], [[-1.65]], [[0]], [[2.25]]])
        exp_s_c = np.array([[[51.25]], [[52]], [[102]], [[0.]], [[333]]])

        exp_c = exp_a_c + exp_s_c

        a_c = cost.action_cost("", action, noise)
        s_c = cost.state_cost("", state)
        c = cost.build_step_cost_graph("", state, action, noise)

        self.assertAllClose(exp_a_c, a_c)
        self.assertAllClose(exp_s_c, s_c)
        self.assertAllClose(exp_c, c)


class TestController(tf.test.TestCase):
    def setUp(self):
        pass

    def testDataPrep(self):
        pass

    def testUpdate(self):
        pass

    def testNew(self):
        pass

    def testShiftAndInit(self):
        pass

    def testAll(self):
        pass


tf.test.main()
