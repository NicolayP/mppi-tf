import numpy as np

import tensorflow as tf

from ..src.controllers.controller_base import ControllerBase
from ..src.controllers.state_controller import StateModelController
from ..src.controllers.lagged_controller import LaggedModelController
from ..src.models.point_mass_model import PointMassModel
from ..src.models.auv_model import AUVModel
from ..src.models.nn_model import LaggedNNAUVSpeed
from ..src.costs.static_cost import StaticCost
from ..src.misc.utile import dtype, npdtype


class TestControllerBase(tf.test.TestCase):
    def setUp(self):
        self.k = 5
        self.tau = 3
        self.aDim = 2
        self.sDim = 4
        self.dt = 0.01
        self.mass = 1.
        self.lam = 1.
        self.gamma = 1.
        self.upsilon = 1.

        self.goal = np.array([[0.], [1.], [0.], [1.]], dtype=npdtype)
        self.sigma = np.array([[1., 0.], [0., 1.]], dtype=npdtype)
        self.Q = np.array([[1., 0., 0., 0.],
                           [0., 1., 0., 0.],
                           [0., 0., 1., 0.],
                           [0., 0., 0., 1.]], dtype=npdtype)

        self.c = np.array([[[3.]], [[10.]], [[0.]], [[1.]], [[5.]]], dtype=npdtype)
        self.n = np.array([[[[1.], [-0.5]], [[1.], [-0.5]], [[2.], [1.]]],
                           [[[0.3], [0.]], [[2.], [0.2]], [[1.2], [3.]]],
                           [[[0.5], [0.5]], [[0.5], [0.5]], [[0.5], [0.5]]],
                           [[[0.6], [0.7]], [[0.2], [-0.3]], [[0.1], [-0.4]]],
                           [[[-2.], [-3.]], [[-4.], [-1.]], [[0.], [0.]]]], dtype=npdtype)

        self.a = np.array([[[1.], [0.5]], [[2.3], [4.5]], [[2.1], [-0.4]]], dtype=npdtype)
        model = PointMassModel(self.mass, self.dt, self.sDim, self.aDim)
        cost = StaticCost(self.lam, self.gamma, self.upsilon, self.sigma, self.goal, self.Q)
        self.cont = ControllerBase(model=model,
                                   cost=cost,
                                   k=self.k,
                                   tau=self.tau,
                                   sDim=self.sDim,
                                   aDim=self.aDim,
                                   lam=self.lam,
                                   sigma=self.sigma)

    def testDataPrep(self):
        exp_a0 = np.array([[1.], [0.5]], dtype=npdtype)
        exp_a1 = np.array([[2.3], [4.5]], dtype=npdtype)
        exp_a2 = np.array([[2.1], [-0.4]], dtype=npdtype)
        exp_n0 = np.array([[[1.], [-0.5]], [[0.3], [0.]], [[0.5], [0.5]], [[0.6], [0.7]], [[-2.], [-3.]]], dtype=npdtype)
        exp_n1 = np.array([[[1.], [-0.5]], [[2.], [0.2]], [[0.5], [0.5]], [[0.2], [-0.3]], [[-4], [-1]]], dtype=npdtype)
        exp_n2 = np.array([[[2.], [1.]], [[1.2], [3.]], [[0.5], [0.5]], [[0.1], [-0.4]], [[0.], [0.]]], dtype=npdtype)


        a0 = self.cont.prepare_action("", self.a, 0)
        n0 = self.cont.prepare_noise("", self.n, 0)

        a1 = self.cont.prepare_action("", self.a, 1)
        n1 = self.cont.prepare_noise("", self.n, 1)

        a2 = self.cont.prepare_action("", self.a, 2)
        n2 = self.cont.prepare_noise("", self.n, 2)

        self.assertAllClose(a0, exp_a0)
        self.assertAllClose(n0, exp_n0)

        self.assertAllClose(a1, exp_a1)
        self.assertAllClose(n1, exp_n1)

        self.assertAllClose(a2, exp_a2)
        self.assertAllClose(n2, exp_n2)

    def testUpdate(self):
        beta = np.array([[0]], dtype=npdtype)
        exp_arg = np.array([[[-3.]], [[-10.]], [[0.]], [[-1.]], [[-5.]]], dtype=npdtype)
        exp = np.array([[[0.049787068367863944]],
                        [[4.5399929762484854e-05]],
                        [[1]],
                        [[0.36787944117144233]],
                        [[0.006737946999085467]]], dtype=npdtype)

        nabla = np.array([[1.424449856468154]], dtype=npdtype)
        weights = np.array([[[0.034951787275480706]],
                           [[3.1871904480408675e-05]],
                           [[0.7020254138530686]],
                           [[0.2582607169364174]],
                           [[0.004730210030553017]]], dtype=npdtype)

        expected = np.array([[[0.034951787275480706*1. + 3.1871904480408675e-05*0.3 + 0.7020254138530686*0.5 + 0.2582607169364174*0.6 + 0.004730210030553017*(-2)],
                              [0.034951787275480706*(-0.5) + 3.1871904480408675e-05*0 + 0.7020254138530686*0.5 + 0.2582607169364174*0.7 + 0.004730210030553017*(-3)]],
                             [[0.034951787275480706*1 + 3.1871904480408675e-05*2 + 0.7020254138530686*0.5 + 0.2582607169364174*0.2 + 0.004730210030553017*(-4)],
                              [0.034951787275480706*(-0.5) + 3.1871904480408675e-05*0.2 + 0.7020254138530686*0.5 + 0.2582607169364174*(-0.3) + 0.004730210030553017*(-1)]],
                             [[0.034951787275480706*2 + 3.1871904480408675e-05*1.2 + 0.7020254138530686*0.5 + 0.2582607169364174*0.1 + 0.004730210030553017*0],
                              [0.034951787275480706*1 + 3.1871904480408675e-05*3 + 0.7020254138530686*0.5 + 0.2582607169364174*(-0.4) + 0.004730210030553017*0]]], dtype=npdtype)

        b = self.cont.beta("", self.c)
        arg = self.cont.norm_arg("", self.c, b, False)
        e_arg = self.cont.exp_arg("", arg)
        e = self.cont.exp("", e_arg)
        nab = self.cont.nabla("", e)
        w = self.cont.weights("", e, nab)
        w_n = self.cont.weighted_noise("", w, self.n)
        sum = tf.reduce_sum(w)


        self.assertAllClose(b, beta)
        self.assertAllClose(e_arg, exp_arg)

        self.assertAllClose(e, exp)
        self.assertAllClose(nab, nabla)

        self.assertAllClose(w, weights)
        self.assertAllClose(w_n, expected)
        self.assertAllClose(sum, 1.)

    def testNew(self):
        next1 = np.array([[[1.], [0.5]]], dtype=npdtype)
        next2 = np.array([[[1.], [0.5]], [[2.3], [4.5]]], dtype=npdtype)
        next3 = np.array([[[1.], [0.5]], [[2.3], [4.5]], [[2.1], [-0.4]]], dtype=npdtype)

        n1 = self.cont.get_next("", self.a, 1)
        n2 = self.cont.get_next("", self.a, 2)
        n3 = self.cont.get_next("", self.a, 3)

        self.assertAllClose(n1, next1)
        self.assertAllClose(n2, next2)
        self.assertAllClose(n3, next3)

    def testShiftAndInit(self):
        init1 = np.array([[[1.], [0.5]]], dtype=npdtype)
        init2 = np.array([[[1.], [0.5]], [[2.3], [4.5]]], dtype=npdtype)

        exp1 = np.array([[[2.3], [4.5]], [[2.1], [-0.4]], [[1.], [0.5]]], dtype=npdtype)
        exp2 = np.array([[[2.1], [-0.4]], [[1.], [0.5]], [[2.3], [4.5]]], dtype=npdtype)

        n1 = self.cont.shift("", self.a, init1, 1)
        n2 = self.cont.shift("", self.a, init2, 2)

        self.assertAllClose(n1, exp1)
        self.assertAllClose(n2, exp2)


class TestStateController(tf.test.TestCase):
    def setUp(self):
        self.k, self.tau, self.sDim, self.aDim, self.dt = 5, 3, 13, 6, 0.1
        self.lam, self.gamma, self.upsilon = 1., 1., 1.

        self.goal = np.zeros(shape=(self.sDim, 1), dtype=npdtype)
        self.sigma = np.eye(self.aDim, dtype=npdtype)
        self.Q = np.eye(self.sDim, dtype=npdtype)

        # self.model = NNAUVModelSpeed({}, stateDim=self.sDim, actionDim=self.aDim, k=self.k)

        self.params = dict()
        self.params["mass"] = 1000
        self.params["volume"] = 1.5
        self.params["density"] = 1000
        self.params["height"] = 1.6
        self.params["length"] = 2.5
        self.params["width"] = 1.5
        self.params["cog"] = [0, 0, 0]
        self.params["cob"] = [0, 0, 0.5]
        self.params["Ma"] = [[500., 0., 0., 0., 0., 0.],
                             [0., 500., 0., 0., 0., 0.],
                             [0., 0., 500., 0., 0., 0.],
                             [0., 0., 0., 500., 0., 0.],
                             [0., 0., 0., 0., 500., 0.],
                             [0., 0., 0., 0., 0., 500.]]
        self.params["linear_damping"] = [-70., -70., -700., -300., -300., -100.]
        self.params["quad_damping"] = [-740., -990., -1800., -670., -770., -520.]
        self.params["linear_damping_forward_speed"] = [1., 2., 3., 4., 5., 6.]
        self.inertial = dict()
        self.inertial["ixx"] = 650.0
        self.inertial["iyy"] = 750.0
        self.inertial["izz"] = 550.0
        self.inertial["ixy"] = 1.0
        self.inertial["ixz"] = 2.0
        self.inertial["iyz"] = 3.0
        self.params["inertial"] = self.inertial
        self.params["rk"] = 2
        self.model = AUVModel({}, actionDim=6, dt=0.1, parameters=self.params)


        self.cost = StaticCost(self.lam, self.gamma, self.upsilon, self.sigma, self.goal, self.Q)
        self.controller = StateModelController(
            model=self.model,
            cost=self.cost,
            k=self.k,
            tau=self.tau,
            sDim=self.sDim,
            aDim=self.aDim,
            lam=self.lam,
            sigma=self.sigma
        )

    def testBuildModel(self):
        dummy_input = tf.random.normal(shape=(self.sDim, 1), dtype=dtype)
        dummy_noise = tf.random.normal(shape=(self.k, self.tau, self.aDim, 1), dtype=dtype)
        dummy_seq = tf.random.normal(shape=(self.tau, self.aDim, 1), dtype=dtype)
        self.controller._model.set_k(self.k)
        costs, trajs = self.controller.build_model("test", self.k, dummy_input, dummy_noise, dummy_seq)
        costs_gt = np.zeros(shape=(self.k, 1, 1), dtype=npdtype)
        trajs_gt = np.zeros(shape=(self.k, self.tau+1, self.sDim, 1), dtype=npdtype)
        self.assertShapeEqual(costs_gt, costs)
        self.assertShapeEqual(trajs_gt, trajs)

    def testStateController(self):
        dummy_input = tf.random.normal(shape=(self.sDim, 1), dtype=dtype)
        next_act, trajs, weights = self.controller.next(dummy_input)

        next_act = tf.constant(next_act, dtype=dtype)
        trajs = tf.constant(trajs, dtype=dtype)
        weights = tf.constant(weights, dtype=dtype)

        act_gt_shape = np.zeros(shape=(self.aDim, ), dtype=npdtype)
        trajs_gt_shape = np.zeros(shape=(self.k, self.tau+1, self.sDim), dtype=npdtype)
        weights_gt_shape = np.zeros(shape=(self.k, ), dtype=npdtype)

        self.assertShapeEqual(act_gt_shape, next_act)
        self.assertShapeEqual(trajs_gt_shape, trajs)
        self.assertShapeEqual(weights_gt_shape, weights)


class TestLaggedStateController(tf.test.TestCase):
    def setUp(self):
        self.h, self.k, self.tau, self.sDim, self.aDim, self.dt = 3, 5, 3, 18, 6, 0.1
        self.lam, self.gamma, self.upsilon = 1., 1., 1.

        self.goal = np.zeros(shape=(self.sDim, 1), dtype=npdtype)
        self.sigma = np.eye(self.aDim, dtype=npdtype)
        self.Q = np.eye(self.sDim, dtype=npdtype)

        self.model = LaggedNNAUVSpeed(k=self.k, h=self.h, dt=self.dt)
        self.cost = StaticCost(self.lam, self.gamma, self.upsilon, self.sigma, self.goal, self.Q)
        self.controller = LaggedModelController(
            model=self.model,
            cost=self.cost,
            k=self.k,
            h=self.h,
            tau=self.tau,
            sDim=self.sDim,
            aDim=self.aDim,
            lam=self.lam,
            sigma=self.sigma
        )

    def testPrepareToApply(self):
        dummy_action = tf.random.normal(shape=(self.aDim, 1), dtype=dtype)
        dummy_noise = tf.random.normal(shape=(self.k, self.aDim, 1), dtype=dtype)
        dummy_lagged_action = tf.random.normal(shape=(self.k, self.h-1, self.aDim, 1), dtype=dtype)
        preped = self.controller.prepare_to_apply("test", dummy_action, dummy_noise, dummy_lagged_action)
        gt_shape = np.zeros(shape=(self.k, self.h, self.aDim, 1), dtype=npdtype)
        self.assertShapeEqual(gt_shape, preped)

    def testBuildModel(self):
        dummy_input = (
            tf.random.normal(shape=(self.h, self.sDim, 1), dtype=dtype),
            tf.random.normal(shape=(self.h-1, self.aDim, 1), dtype=dtype)
        )
        dummy_noise = tf.random.normal(shape=(self.k, self.tau, self.aDim, 1), dtype=dtype)
        dummy_seq = tf.random.normal(shape=(self.tau, self.aDim, 1), dtype=dtype)
        costs = self.controller.build_model("test", self.k, dummy_input, dummy_noise, dummy_seq)
        gt_shape = np.zeros(shape=(self.k, 1, 1), dtype=npdtype)

        self.assertShapeEqual(gt_shape, costs)

    def testLaggedController(self):
        dummy_input = (
            tf.random.normal(shape=(self.h, self.sDim, 1), dtype=dtype),
            tf.random.normal(shape=(self.h-1, self.aDim, 1), dtype=dtype)
        )
        next_act = tf.constant(self.controller.next(dummy_input), dtype=dtype)

        gt_shape = np.zeros(shape=(self.aDim, ), dtype=npdtype)
        self.assertShapeEqual(gt_shape, next_act)
