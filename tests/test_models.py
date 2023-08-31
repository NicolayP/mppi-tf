import unittest
import torch
from torch.testing import assert_allclose
from scripts.utils.utils import tdtype  # Import the required dtype
from scripts.models.fossen import AUVFossen
from scripts.models.model_base import ModelBase
from scripts.models.rnn_auv import AUVRNNDeltaV, AUVLSTMDeltaV, AUVNNDeltaV, AUVStep
import pypose as pp

class TestModelBase(unittest.TestCase):
    def setUp(self):
        self.model = ModelBase()

    def test_forward(self):
        x = torch.randn(2, 13, 1)
        u = torch.randn(2, 6, 1)
        with self.assertRaises(NotImplementedError):
            self.model.forward(x, u)

    def test_init_param(self):
        with self.assertRaises(NotImplementedError):
            self.model.init_param()


class TestAUVFossen(unittest.TestCase):
    def setUp(self):
        self.dt = 0.1
        self.auv = AUVFossen(dt=self.dt)

    def test_forward_random_state(self):
        state = torch.randn(2, 13, 1, dtype=tdtype)
        action = torch.randn(2, 6, 1, dtype=tdtype)
        new_state_rk1 = self.auv.forward(state, action, rk=1)
        new_state_rk2 = self.auv.forward(state, action, rk=2)
        self.assertEqual(new_state_rk1.shape, torch.Size([2, 13, 1]))  # Check shape
        self.assertEqual(new_state_rk2.shape, torch.Size([2, 13, 1]))  # Check shape

    def test_forward_non_random_state(self):
        state = torch.tensor([[[1.0], [2.0], [3.0], [0.5], [0.2], [0.7], [0.1], [0.3], [0.4], [0.2], [0.5], [0.7], [0.9]]], dtype=tdtype)
        action = torch.tensor([[[0.1], [0.2], [0.3], [0.4], [0.5], [0.6]]], dtype=tdtype)
        new_state_rk1 = self.auv.forward(state, action, rk=1)
        new_state_rk2 = self.auv.forward(state, action, rk=2)
        self.assertEqual(new_state_rk1.shape, torch.Size([1, 13, 1]))  # Check shape
        self.assertEqual(new_state_rk2.shape, torch.Size([1, 13, 1]))  # Check shape

    def test_x_dot_random_state(self):
        state = torch.randn(2, 13, 1, dtype=tdtype)
        action = torch.randn(2, 6, 1, dtype=tdtype)
        x_dot = self.auv.x_dot(state, action)
        self.assertEqual(x_dot.shape, torch.Size([2, 13, 1]))  # Check shape

    def test_x_dot_non_random_state(self):
        state = torch.tensor([[[1.0], [2.0], [3.0], [0.5], [0.2], [0.7], [0.1], [0.3], [0.4], [0.2], [0.5], [0.7], [0.9]]], dtype=tdtype)
        action = torch.tensor([[[0.1], [0.2], [0.3], [0.4], [0.5], [0.6]]], dtype=tdtype)
        x_dot = self.auv.x_dot(state, action)
        self.assertEqual(x_dot.shape, torch.Size([1, 13, 1]))  # Check shape

    def test_norm_quat(self):
        quat_state = torch.tensor([[[0.5], [0.5], [0.5], [0.5], [0.2], [0.7], [0.1], [0.3], [0.4], [0.2], [0.5], [0.7], [0.9]]], dtype=tdtype)
        normalized_quat_state = self.auv.norm_quat(quat_state.clone())
        self.assertTrue(torch.allclose(torch.linalg.vector_norm(normalized_quat_state[:, 3:7]), 
                                       torch.tensor([1], dtype=tdtype), atol=1e-6)) # check normalization

    def test_damping(self):
        v = torch.tensor([[[0.1], [0.2], [0.3], [0.4], [0.5], [0.6]]], dtype=tdtype)
        damping_result = self.auv.damping(v)
        expected_result = torch.diag_embed(
            torch.tensor([[[144, 268, 1240, 568, 685, 412]]], dtype=tdtype)
        )
        self.assertTrue(torch.allclose(damping_result, expected_result, atol=1e-6))

    def test_restoring(self):
        rotBtoI, _ = self.auv.body2inertial(torch.tensor([[[1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]], dtype=tdtype))
        restoring_result = self.auv.restoring(rotBtoI)
        expected_result = torch.tensor([[0.0, 0.0, -(1028.0 - 100) * 9.81, 0.0, 0.0, 0.0]], dtype=tdtype)
        self.assertTrue(torch.allclose(restoring_result, expected_result, atol=1e-6))

    def test_coriolis(self):
        v = torch.tensor([[[0.1], [0.2], [0.3], [0.4], [0.5], [0.6]]], dtype=tdtype)
        coriolis_result = self.auv.coriolis(v)
        expected_result = torch.tensor([[[0.0, 0.0, 0.0, 0.0, 30, -20],
                                         [0.0, 0.0, 0.0, -30, 0.0, 10],
                                         [0.0, 0.0, 0.0, 20, -10, 0.0],
                                         [0.0, 30, -20, 0.0, 263, -272],
                                         [-30, 0.0, 10, -263, 0.0, 207],
                                         [20, -10, 0.0, 272, -207, 0.0]]], dtype=tdtype)
        self.assertTrue(torch.allclose(coriolis_result, expected_result, atol=1e-6))


class TestAUVRNNDeltaV(unittest.TestCase):
    def setUp(self):
        self.auv_rnn = AUVRNNDeltaV()

    def test_forward(self):
        k = 2
        x = pp.randn_SE3(k, 1, dtype=tdtype)
        v = torch.randn(k, 1, 6, dtype=tdtype)
        u = torch.randn(k, 1, 6, dtype=tdtype)
        dv = self.auv_rnn(x, v, u)
        self.assertEqual(dv.shape, (k, 1, 6))

    def test_init_hidden(self):
        k = 2
        device = torch.device("cpu")
        h0 = self.auv_rnn.init_hidden(k, device)
        self.assertEqual(h0.shape, (self.auv_rnn.rnn_layers, k, self.auv_rnn.rnn_hidden_size))


class TestAUVLSTMDeltaV(unittest.TestCase):
    def setUp(self):
        self.auv_lstm = AUVLSTMDeltaV()

    def test_forward(self):
        k = 2
        x = pp.randn_SE3(k, 1, dtype=tdtype)  # Example SE3 state (batch size, sequence length)
        v = torch.randn(k, 1, 6, dtype=tdtype)  # Example velocity
        u = torch.randn(k, 1, 6, dtype=tdtype)  # Example action
        dv = self.auv_lstm(x, v, u)
        self.assertEqual(dv.shape, (k, 1, 6))  # Output shape


class TestAUVNNDeltaV(unittest.TestCase):
    def setUp(self):
        self.auv_nn = AUVNNDeltaV()

    def test_forward(self):
        k = 2
        n = 4
        x = pp.randn_SE3(k, n, dtype=tdtype)  # Example SE3 state (batch size, sequence length)
        v = torch.randn(k, n, 6, dtype=tdtype)  # Example velocity
        u = torch.randn(k, n, 6, dtype=tdtype)  # Example action
        dv = self.auv_nn(x, v, u)
        self.assertEqual(dv.shape, (k, 1, 6))  # Output shape


class TestAUVStep(unittest.TestCase):
    def setUp(self):
        self.k = 5
        self.auv_step = AUVStep()

    def test_forward_with_AUVRNNDeltaV(self):
        model = AUVRNNDeltaV()
        self.auv_step.update_model(model)

        x = pp.randn_SE3(self.k, 1, dtype=tdtype)
        v = torch.randn(self.k, 1, 6, dtype=tdtype)
        u = torch.randn(self.k, 1, 6, dtype=tdtype)
        x_next, v_next = self.auv_step(x, v, u)

        self.assertEqual(x_next.shape, (self.k, 1, 7))
        self.assertEqual(v_next.shape, (self.k, 1, 6))

    def test_forward_with_AUVLSTMDeltaV(self):
        model = AUVLSTMDeltaV()
        self.auv_step.update_model(model)

        x = pp.randn_SE3(self.k, 1,dtype=tdtype)
        v = torch.randn(self.k, 1, 6, dtype=tdtype)
        u = torch.randn(self.k, 1, 6, dtype=tdtype)
        x_next, v_next = self.auv_step(x, v, u)

        self.assertEqual(x_next.shape, (self.k, 1, 7))
        self.assertEqual(v_next.shape, (self.k, 1, 6))

    def test_forward_with_AUVNNDeltaV(self):
        model = AUVNNDeltaV()
        n = 4
        self.auv_step.update_model(model)

        x = pp.randn_SE3(self.k, n, dtype=tdtype)
        v = torch.randn(self.k, n, 6, dtype=tdtype)
        u = torch.randn(self.k, n, 6, dtype=tdtype)
        x_next, v_next = self.auv_step(x, v, u)

        self.assertEqual(x_next.shape, (self.k, 1, 7))
        self.assertEqual(v_next.shape, (self.k, 1, 6))


if __name__ == '__main__':
    unittest.main()
