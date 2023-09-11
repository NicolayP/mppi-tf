import unittest
import torch

from scripts.controllers.mppi_base import ControllerBase, MPPIBase, MPPIPypose, Update
from scripts.models.rnn_auv import AUVRNNDeltaV, AUVLSTMDeltaV, AUVNNDeltaV, AUVStep
from scripts.models.fossen import AUVFossen
from scripts.observers.observer_base import ObserverBase
from scripts.costs.static import Static, StaticPypose
from scripts.inputs.ControllerInput import ControllerInput, ControllerInputPypose
from scripts.inputs.ModelInput import ModelInput, ModelInputPypose



class TestUpdate(unittest.TestCase):
    def setUp(self):
        self.lam = 0.5
        self.update = Update(self.lam)

    def test_beta(self):
        cost = torch.tensor([3., 5., 20., 4., 2.])
        beta = self.update.beta(cost)
        self.assertEqual(beta, 2.)

    def test_arg_unnormed(self):
        cost = torch.tensor([3., 5., 20., 4., 2.])
        beta = self.update.beta(cost)
        arg = self.update.arg(cost, beta)
        expected_arg = torch.tensor([1., 3., 18., 2., 0.])
        self.assertEqual(arg.shape, (5, ))
        self.assertTrue(torch.allclose(arg, expected_arg, rtol=1e-6))

    def test_arg_normed(self):
        cost = torch.tensor([3., 5., 20., 4., 2.])
        beta = self.update.beta(cost)
        arg = self.update.arg(cost, beta, True)
        expected_arg = torch.tensor([1./18., 3./18., 18./18., 2./18., 0.])
        self.assertEqual(arg.shape, (5, ))
        self.assertTrue(torch.allclose(arg, expected_arg, rtol=1e-6))

    def test_exp_arg(self):
        arg = torch.tensor([1., 3., 18., 2., 0.])
        exp_arg = self.update.exp_arg(arg)
        expected_exp_arg = torch.tensor([-2., -6., -36., -4., 0.])
        self.assertEqual(exp_arg.shape, (5, ))
        self.assertTrue(torch.allclose(exp_arg, expected_exp_arg, rtol=1e-6))

    def test_exp(self):
        exp_arg = torch.tensor([-2., -6., -36., -4., 0.])
        exp = self.update.exp(exp_arg)
        expected_exp = torch.tensor([1.35335283e-01, 2.47875218e-03, 2.31952283e-16, 1.83156389e-02, 1.])
        self.assertEqual(exp.shape, (5, ))
        self.assertTrue(torch.allclose(exp, expected_exp, rtol=1e-6))

    def test_eta(self):
        exp = torch.tensor([1.35335283e-01, 2.47875218e-03, 2.31952283e-16, 1.83156389e-02, 1.])
        eta = self.update.eta(exp)
        expected_eta = torch.tensor(1.1561296743020135)
        self.assertEqual(eta.shape, ( ))
        self.assertTrue(torch.allclose(eta, expected_eta, rtol=1e-6))

    def test_weights(self):
        exp = torch.tensor([1.35335283e-01, 2.47875218e-03, 2.31952283e-16, 1.83156389e-02, 1.])
        eta = torch.tensor([1.1561296743020135])
        weights = self.update.weights(exp, eta)
        expected_weight = torch.tensor([1.17058913e-01, 2.14400878e-03, 2.00628258e-16, 1.58422012e-02, 8.64954877e-01])
        self.assertEqual(weights.shape, (5, ))
        self.assertTrue(torch.allclose(weights, expected_weight, rtol=1e-6))

    def test_weighted_noise(self):
        k, tau, aDim = 4, 3, 2
        w = torch.tensor([2., 1., 4., 3.])
        n = torch.tensor([[[0.0, 0.0], [0.0, 0.0], [0.1, 0.1]],
                          [[1.0, 1.0], [1.0, 1.0], [1.1, 1.1]],
                          [[0.5, 0.5], [1.5, 1.5], [2.5, 2.5]],
                          [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]])
        wn = self.update.weighted_noise(w, n)
        expected_wn = torch.tensor([[6., 6.], [13., 13.], [20.3, 20.3]])
        self.assertEqual(wn.shape, (tau, aDim))
        self.assertTrue(torch.allclose(wn, expected_wn, rtol=1e-6))

    def test_weighted_noise_rand(self):
        k, tau, aDim = 200, 50, 6
        w = torch.randn(k)
        n = torch.randn(k, tau, aDim)
        wn = self.update.weighted_noise(w, n)
        self.assertEqual(wn.shape, (tau, aDim))

    def test_fowrard(self):
        k, tau, aDim = 200, 50, 6
        cost  = torch.randn(k)
        noise = torch.randn(k, tau, aDim)
        wn, eta = self.update(cost, noise)
        self.assertEqual(eta.shape, ( ))
        self.assertEqual(wn.shape, (tau, aDim))


class TestControllerBase(unittest.TestCase):
    def setUp(self):
        self.k = 10
        self.steps = 1
        self.tau = 50
        self.lam = 0.5
        self.gamma = 0.2
        self.upsilon = 2.0
        self.sigma = [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]

        self.goal_pose = [1.0, 2.0, -1.0, 0.0, 0.0, 0.0, 1.0]
        self.goal_vel = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Example goal vector
        self.Q = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                  1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # Example weight vector

        self.cost = StaticPypose(self.lam, self.gamma, self.upsilon, self.sigma,
                                 self.goal_pose, self.goal_vel, self.Q)
        self.model = AUVStep()

        self.observer = ObserverBase(False)  # Replace with your observer class
        self.controller = ControllerBase(self.model, self.cost, self.observer, self.k, self.tau, self.lam,
                                         self.upsilon, self.sigma)

    def test_noise(self):
        n = self.controller.noise()
        self.assertEqual(n.shape, (self.k, self.tau, 6))

    def test_rollout_cost(self):
        state = ControllerInputPypose(self.steps)
        model_input = ModelInputPypose(self.k, self.steps)
        model_input.init(state)
        n = self.controller.noise()
        A = self.controller.act_sequence.clone()
        cost = self.controller.rollout_cost(model_input, n, A)
        self.assertEqual(cost.shape, (self.k, ))

    def test_control(self):
        state = ControllerInputPypose(self.steps)
        model_input = ModelInputPypose(self.k, self.steps)
        model_input.init(state)
        A = self.controller.act_sequence.clone()
        next_a, next_A = self.controller.control(model_input, A)
        self.assertEqual(next_a.shape, (6, ))
        self.assertEqual(next_A.shape, (self.tau, 6))

    def test_forward(self):        
        state = ControllerInputPypose(self.steps)  # Replace with your controller input
        with self.assertRaises(NotImplementedError):
            self.controller(state)

# Use AUF Fossen Model.
class TestMPPIBase(unittest.TestCase):
    def setUp(self) -> None:
        self.k = 10
        self.steps = 1
        self.tau = 50
        self.lam = 0.5
        self.gamma = 0.2
        self.upsilon = 2.0
        self.sigma = [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]

        self.goal_pose = [1.0, 2.0, -1.0, 0.0, 0.0, 0.0, 1.0]
        self.goal_vel = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Example goal vector
        self.Q = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                  1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # Example weight vector

        self.cost = Static(lam=self.lam, gamma=self.gamma, upsilon=self.upsilon, sigma=self.sigma,
                           goal_pose=self.goal_pose, goal_vel=self.goal_vel, Q=self.Q)
        self.model = AUVFossen()

        self.observer = ObserverBase(False)  # Replace with your observer class
        self.controller = MPPIBase(self.model, self.cost, self.observer, self.k, self.tau, self.lam,
                                   self.upsilon, self.sigma)

    def test_noise(self):
        n = self.controller.noise()
        self.assertEqual(n.shape, (self.k, self.tau, 6))

    def test_rollout_cost(self):
        state = ControllerInput(self.steps)
        model_input = ModelInput(self.k, self.steps)
        model_input.init(state)
        n = self.controller.noise()
        A = self.controller.act_sequence.clone()
        cost = self.controller.rollout_cost(model_input, n, A)
        self.assertEqual(cost.shape, (self.k, ))

    def test_control(self):
        state = ControllerInput(self.steps)
        model_input = ModelInput(self.k, self.steps)
        model_input.init(state)
        A = self.controller.act_sequence.clone()
        next_a, next_A = self.controller.control(model_input, A)
        self.assertEqual(next_a.shape, (6, ))
        self.assertEqual(next_A.shape, (self.tau, 6))

    def test_forward(self):
        state = ControllerInput(self.steps)  # Replace with your controller input
        action = self.controller(state)
        self.assertEqual(action.shape, (6, ))


class TestMPPIPyposeRNN(unittest.TestCase):
    def setUp(self) -> None:
        self.k = 10
        self.steps = 1
        self.tau = 2
        self.lam = 0.5
        self.gamma = 0.2
        self.upsilon = 2.0
        self.sigma = [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]

        self.goal_pose = [1.0, 2.0, -1.0, 0.0, 0.0, 0.0, 1.0]
        self.goal_vel = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Example goal vector
        self.Q = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                  1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # Example weight vector

        self.cost = StaticPypose(self.lam, self.gamma, self.upsilon, self.sigma,
                                 self.goal_pose, self.goal_vel, self.Q)
        self.model = AUVStep()

        self.observer = ObserverBase(False)  # Replace with your observer class
        self.controller = MPPIPypose(self.model, self.cost, self.observer, self.k, self.tau, self.lam,
                                   self.upsilon, self.sigma)

    def test_noise(self):
        n = self.controller.noise()
        self.assertEqual(n.shape, (self.k, self.tau, 6))

    def test_rollout_cost(self):
        state = ControllerInputPypose(self.steps)
        model_input = ModelInputPypose(self.k, self.steps)
        model_input.init(state)
        n = self.controller.noise()
        A = self.controller.act_sequence.clone()
        cost = self.controller.rollout_cost(model_input, n, A)
        self.assertEqual(cost.shape, (self.k, ))

    def test_control(self):
        state = ControllerInputPypose(self.steps)
        model_input = ModelInputPypose(self.k, self.steps)
        model_input.init(state)
        A = self.controller.act_sequence.clone()
        next_a, next_A = self.controller.control(model_input, A)
        self.assertEqual(next_a.shape, (6, ))
        self.assertEqual(next_A.shape, (self.tau, 6))

    def test_forward(self):
        state = ControllerInputPypose(self.steps)  # Replace with your controller input
        action = self.controller(state)
        self.assertEqual(action.shape, (6, ))


class TestMPPIPyposeNN(unittest.TestCase):
    def setUp(self) -> None:
        self.k = 10
        self.steps = 4
        self.tau = 50
        self.lam = 0.5
        self.gamma = 0.2
        self.upsilon = 2.0
        self.sigma = [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]

        self.goal_pose = [1.0, 2.0, -1.0, 0.0, 0.0, 0.0, 1.0]
        self.goal_vel = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Example goal vector
        self.Q = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                  1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # Example weight vector

        self.cost = StaticPypose(self.lam, self.gamma, self.upsilon, self.sigma,
                                 self.goal_pose, self.goal_vel, self.Q)
        self.model = AUVStep()
        self.model.update_model(AUVNNDeltaV())

        self.observer = ObserverBase(False)  # Replace with your observer class
        self.controller = MPPIPypose(self.model, self.cost, self.observer, self.k, self.tau, self.lam,
                                   self.upsilon, self.sigma)

    def test_noise(self):
        n = self.controller.noise()
        self.assertEqual(n.shape, (self.k, self.tau, 6))

    def test_rollout_cost(self):
        state = ControllerInputPypose(self.steps)
        model_input = ModelInputPypose(self.k, self.steps)
        model_input.init(state)
        n = self.controller.noise()
        A = self.controller.act_sequence.clone()
        cost = self.controller.rollout_cost(model_input, n, A)
        self.assertEqual(cost.shape, (self.k, ))

    def test_control(self):
        state = ControllerInputPypose(self.steps)
        model_input = ModelInputPypose(self.k, self.steps)
        model_input.init(state)
        A = self.controller.act_sequence.clone()
        next_a, next_A = self.controller.control(model_input, A)
        self.assertEqual(next_a.shape, (6, ))
        self.assertEqual(next_A.shape, (self.tau, 6))

    def test_forward(self):
        state = ControllerInputPypose(self.steps)  # Replace with your controller input
        action = self.controller(state)
        self.assertEqual(action.shape, (6, ))


class TestMPPIPyposeLSTM(unittest.TestCase):
    def setUp(self) -> None:
        self.k = 10
        self.steps = 1
        self.tau = 50
        self.lam = 0.5
        self.gamma = 0.2
        self.upsilon = 2.0
        self.sigma = [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]

        self.goal_pose = [1.0, 2.0, -1.0, 0.0, 0.0, 0.0, 1.0]
        self.goal_vel = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Example goal vector
        self.Q = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                  1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # Example weight vector

        self.cost = StaticPypose(self.lam, self.gamma, self.upsilon, self.sigma,
                                 self.goal_pose, self.goal_vel, self.Q)
        self.model = AUVStep()
        self.model.update_model(AUVLSTMDeltaV())

        self.observer = ObserverBase(False)  # Replace with your observer class
        self.controller = MPPIPypose(self.model, self.cost, self.observer, self.k, self.tau, self.lam,
                                   self.upsilon, self.sigma)

    def test_noise(self):
        n = self.controller.noise()
        self.assertEqual(n.shape, (self.k, self.tau, 6))

    def test_rollout_cost(self):
        state = ControllerInputPypose(self.steps)
        model_input = ModelInputPypose(self.k, self.steps)
        model_input.init(state)
        n = self.controller.noise()
        A = self.controller.act_sequence.clone()
        cost = self.controller.rollout_cost(model_input, n, A)
        self.assertEqual(cost.shape, (self.k, ))

    def test_control(self):
        state = ControllerInputPypose(self.steps)
        model_input = ModelInputPypose(self.k, self.steps)
        model_input.init(state)
        A = self.controller.act_sequence.clone()
        next_a, next_A = self.controller.control(model_input, A)
        self.assertEqual(next_a.shape, (6, ))
        self.assertEqual(next_A.shape, (self.tau, 6))

    def test_forward(self):
        state = ControllerInputPypose(self.steps)  # Replace with your controller input
        action = self.controller(state)
        self.assertEqual(action.shape, (6, ))



if __name__ == '__main__':
    unittest.main()
