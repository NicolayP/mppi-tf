import unittest
import torch
import pypose as pp

from scripts.utils.utils import tdtype  # Import the required dtype
from scripts.models.fossen import AUVFossen
from scripts.models.model_base import ModelBase
from scripts.models.rnn_auv import AUVRNNDeltaV, AUVLSTMDeltaV, AUVNNDeltaV, AUVStep



class TestModelBase(unittest.TestCase):
    def setUp(self):
        self.model = ModelBase()

    def test_forward(self):
        x = torch.randn(2, 1, 7)
        v = torch.randn(2, 1, 6)
        u = torch.randn(2, 1, 6)
        with self.assertRaises(NotImplementedError):
            self.model.forward(x, v, u)

    def test_init_param(self):
        with self.assertRaises(NotImplementedError):
            self.model.init_param()

    def test_reset(self):
        with self.assertRaises(NotImplementedError):
            self.model.reset()


class TestAUVFossen(unittest.TestCase):
    def setUp(self):
        self.dt = 0.1
        self.auv = AUVFossen(dt=self.dt)

    def test_forward_random_state(self):
        x = torch.randn(3, 1, 7, dtype=tdtype)
        v = torch.randn(3, 1, 6, dtype=tdtype)
        u = torch.randn(3, 1, 6, dtype=tdtype)
        new_state_rk1 = self.auv.forward(x, v, u, rk=1)
        new_state_rk2 = self.auv.forward(x, v, u, rk=2)
        self.assertEqual(new_state_rk1[0].shape, torch.Size([3, 1, 7]))  # Check shape
        self.assertEqual(new_state_rk1[1].shape, torch.Size([3, 1, 6]))  # Check shape
        self.assertEqual(new_state_rk2[0].shape, torch.Size([3, 1, 7]))  # Check shape
        self.assertEqual(new_state_rk2[1].shape, torch.Size([3, 1, 6]))  # Check shape

    def test_forward_non_random_state(self):
        x = torch.tensor([[[1.0, 2.0, 3.0, 0.5, 0.2, 0.7, 0.1]]], dtype=tdtype)
        v = torch.tensor([[[0.3, 0.4, 0.2, 0.5, 0.7, 0.9]]], dtype=tdtype)
        u = torch.tensor([[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]], dtype=tdtype)
        new_state_rk1 = self.auv.forward(x, v, u, rk=1)
        new_state_rk2 = self.auv.forward(x, v, u, rk=2)
        self.assertEqual(new_state_rk1[0].shape, torch.Size([1, 1, 7]))  # Check shape
        self.assertEqual(new_state_rk1[1].shape, torch.Size([1, 1, 6]))  # Check shape
        self.assertEqual(new_state_rk2[0].shape, torch.Size([1, 1, 7]))  # Check shape
        self.assertEqual(new_state_rk2[1].shape, torch.Size([1, 1, 6]))  # Check shape

    def test_x_dot_random_state(self):
        x = torch.randn(2, 7, dtype=tdtype)
        v = torch.randn(2, 6, dtype=tdtype)
        u = torch.randn(2, 6, dtype=tdtype)
        x_dot, v_dot = self.auv.x_dot(x, v, u)
        self.assertEqual(x_dot.shape, torch.Size([2, 7]))  # Check shape
        self.assertEqual(v_dot.shape, torch.Size([2, 6]))  # Check shape

    def test_x_dot_non_random_state(self):
        x = torch.tensor([[1.0, 2.0, 3.0, 0.5, 0.2, 0.7, 0.1]], dtype=tdtype)
        v = torch.tensor([[0.3, 0.4, 0.2, 0.5, 0.7, 0.9]], dtype=tdtype)
        u = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]], dtype=tdtype)
        x_dot, v_dot = self.auv.x_dot(x, v, u)
        self.assertEqual(x_dot.shape, torch.Size([1, 7]))  # Check shape
        self.assertEqual(v_dot.shape, torch.Size([1, 6]))  # Check shape

    def test_norm_quat(self):
        quat_x = torch.tensor([[0.5, 0.5, 0.5, 0.5, 0.2, 0.7, 0.1],
                               [0.5, 0.5, 0.5, 0.5, 0.2, 0.7, 0.1]], dtype=tdtype)

        normalized_quat_state = self.auv.norm_quat(quat_x.clone())
        self.assertTrue(torch.allclose(torch.linalg.vector_norm(normalized_quat_state[:, 3:7], dim=-1), 
                                       torch.tensor([1., 1.], dtype=tdtype), atol=1e-6)) # check normalization

    def test_damping(self):
        v = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]], dtype=tdtype)
        damping_result = self.auv.damping(v)
        expected_result = torch.diag_embed(
            torch.tensor([[144, 268, 1240, 568, 685, 412]], dtype=tdtype)
        )
        self.assertTrue(torch.allclose(damping_result, expected_result, atol=1e-6))

    def test_restoring(self):
        state = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=tdtype)
        rotBtoI, _ = self.auv.body2inertial(state)
        restoring_result = self.auv.restoring(rotBtoI)
        expected_result = torch.tensor([[0.0, 0.0, -(1028.0 - 100) * 9.81, 0.0, 0.0, 0.0]], dtype=tdtype)
        self.assertTrue(torch.allclose(restoring_result, expected_result, atol=1e-6))

    def test_coriolis(self):
        v = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]], dtype=tdtype)
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

    def test_reset(self):
        k = 2
        device = torch.device("cpu")
        h0 = self.auv_rnn.init_hidden(k, device)
        self.assertNotEqual(None, h0)
        self.auv_rnn.reset()
        h0 = self.auv_rnn.hidden
        self.assertEqual(None, h0)


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

    def test_init_hidden(self):
        k = 2
        device = torch.device("cpu")
        h0 = self.auv_lstm.init_hidden(k, device)
        self.assertEqual(h0[0].shape, (self.auv_lstm.lstm_layers, k, self.auv_lstm.lstm_hidden_size))
        self.assertEqual(h0[1].shape, (self.auv_lstm.lstm_layers, k, self.auv_lstm.lstm_hidden_size))

    def test_reset(self):
        k = 2
        device = torch.device("cpu")
        h0 = self.auv_lstm.init_hidden(k, device)
        self.assertNotEqual(None, h0)
        self.auv_lstm.reset()
        h0 = self.auv_lstm.hidden
        self.assertEqual(None, h0)


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
