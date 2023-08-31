import torch
import unittest
from scripts.controllers.mppi_base import ControllerBase
from scripts.models.rnn_auv import AUVRNNDeltaV, AUVLSTMDeltaV, AUVNNDeltaV, AUVStep
from scripts.models.fossen import AUVFossen
from scripts.observers.observer_base import ObserverBase
from scripts.costs.static import Static
from scripts.inputs.ControllerInput import ControllerInput, ControllerInputPypose
from scripts.inputs.ModelInput import ModelInput, ModelInputPypose

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

        self.goal = [[1.0], [2.0], [-1.0], [0.0], [0.0], [0.0], [1.0],
                     [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]  # Example goal vector
        self.Q = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                  1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # Example weight vector

        self.cost = Static(self.lam, self.gamma, self.upsilon, self.sigma, self.goal, self.Q)
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
        pass

    def test_forward(self):
        # state = ControllerInput(self.steps)  # Replace with your controller input
        # action = self.controller(state)

        # self.assertEqual(action.shape, (self.controller.aDim, 1))
        pass

    def test_control(self):
        pass


# class TestAUVFossen(unittest.TestCase):
#     def test_fossen(self):
#         model = AUVFossen()
#         x = torch.randn(1, 6)
#         u = torch.randn(1, 6)
#         v_next = model.forward(x, u)
        
#         self.assertEqual(v_next.shape, (1, 6))

# class TestAUVStep(unittest.TestCase):
#     def test_AUVRNNDeltaV(self):
#         dv_model = AUVRNNDeltaV()
#         auv_step = AUVStep({"model": {"rnn": {"rnn_layer": 2, "rnn_hidden_size": 16}}})
#         auv_step.change_model(dv_model)
        
#         x = YourModelInput()  # Replace with your ModelInput
#         v = torch.randn(1, 6)
#         u = torch.randn(1, 6)
#         h0 = torch.zeros(2, 1, 16)  # Example hidden state
#         x_next, v_next, dv, h_next = auv_step.forward(x, v, u, h0)
        
#         self.assertEqual(x_next.shape, (1, 6))
#         self.assertEqual(v_next.shape, (1, 6))
#         self.assertEqual(dv.shape, (1, 6, 1))
#         self.assertEqual(h_next.shape, (2, 1, 16))
    
#     def test_AUVLSTMDeltaV(self):
#         dv_model = AUVLSTMDeltaV()
#         auv_step = AUVStep({"model": {"lstm": {"lstm_layers": 2, "lstm_hidden_size": 16}}})
#         auv_step.change_model(dv_model)
        
#         x = YourModelInput()  # Replace with your ModelInput
#         v = torch.randn(1, 6)
#         u = torch.randn(1, 6)
#         h0 = (torch.zeros(2, 1, 16), torch.zeros(2, 1, 16))  # Example hidden states
#         x_next, v_next, dv, h_next = auv_step.forward(x, v, u, h0)
        
#         self.assertEqual(x_next.shape, (1, 6))
#         self.assertEqual(v_next.shape, (1, 6))
#         self.assertEqual(dv.shape, (1, 6, 1))
#         self.assertEqual(h_next[0].shape, (2, 1, 16))
#         self.assertEqual(h_next[1].shape, (2, 1, 16))
    
#     def test_AUVNNDeltaV(self):
#         dv_model = AUVNNDeltaV()
#         auv_step = AUVStep({"model": {"fc": {"topology": [32, 32]}}})
#         auv_step.change_model(dv_model)
        
#         x = YourModelInput()  # Replace with your ModelInput
#         v = torch.randn(1, 6)
#         u = torch.randn(1, 6)
#         x_next, v_next, dv, h_next = auv_step.forward(x, v, u)
        
#         self.assertEqual(x_next.shape, (1, 6))
#         self.assertEqual(v_next.shape, (1, 6))
#         self.assertEqual(dv.shape, (1, 6, 1))
#         self.assertIsNone(h_next)  # No hidden state for fully connected model

if __name__ == '__main__':
    unittest.main()
