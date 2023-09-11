import unittest
import torch
import pypose as pp
import numpy as np
from scripts.utils.utils import tdtype  # Import the required dtype

from scripts.costs.cost_base import CostBase
from scripts.costs.static import Static, StaticPypose


class TestCostBase(unittest.TestCase):
    def setUp(self):
        self.lam = 0.5
        self.gamma = 0.2
        self.upsilon = 2.0
        self.sigma = [[1.0, 0.0], [0.0, 1.0]]  # Example sigma matrix

        self.cost = CostBase(self.lam, self.gamma, self.upsilon, self.sigma)

    def test_action_cost(self):
        action = torch.randn(4, 2, dtype=tdtype)
        noise = torch.randn(4, 2, dtype=tdtype)
        cost = self.cost.action_cost(action, noise)
        self.assertEqual(cost.shape, torch.Size([4]))

    def test_state_cost(self):
        pose = torch.randn(4, 5, dtype=tdtype)
        velocity = torch.randn(4, 5, dtype=tdtype)
        with self.assertRaises(NotImplementedError):
            self.cost.state_cost(pose, velocity)

    def test_final_cost(self):
        pose = torch.randn(4, 5, dtype=tdtype)
        velocity = torch.randn(4, 5, dtype=tdtype)
        with self.assertRaises(NotImplementedError):
            self.cost.final_cost(pose, velocity)

    def test_forward(self):
        pose = torch.randn(4, 5, dtype=tdtype)
        velocity = torch.randn(4, 5, dtype=tdtype)
        action = torch.randn(4, 2, dtype=tdtype)
        noise = torch.randn(4, 2, dtype=tdtype)
        with self.assertRaises(NotImplementedError):
            self.cost(pose, velocity, action, noise)

    def test_final_cost_forward(self):
        pose = torch.randn(2, 3, dtype=tdtype)
        velocity = torch.randn(2, 3, dtype=tdtype)
        with self.assertRaises(NotImplementedError):
            self.cost(pose, velocity, final=True)


class TestStaticCost(unittest.TestCase):
    def setUp(self):
        self.lam = 0.5
        self.gamma = 0.2
        self.upsilon = 2.0
        self.sigma = [[1.0, 0.0], [0.0, 1.0]]  # Example sigma matrix
        self.goal_p = [1.0]
        self.goal_v = [2.0]  # Example goal vector
        self.Q = [2.0, 3.0]  # Example weight vector

        self.static_cost = Static(self.lam, self.gamma, self.upsilon, self.sigma, self.goal_p, self.goal_v, self.Q)

    def test_state_cost_random_state(self):
        pose = torch.randn(4, 1, 1, dtype=tdtype)
        velocity = torch.randn(4, 1, 1, dtype=tdtype)
        cost = self.static_cost.state_cost(pose, velocity)
        self.assertEqual(cost.shape, torch.Size([4]))  # Scalar output

    def test_state_cost_non_random_state(self):
        pose = torch.tensor([[[0.5]]], dtype=tdtype)
        velocity = torch.tensor([[[1.0]]], dtype=tdtype)
        cost = self.static_cost.state_cost(pose, velocity)
        expected_cost = torch.tensor([3.5], dtype=tdtype)  # Calculate expected cost manually
        self.assertTrue(torch.allclose(cost, expected_cost, rtol=1e-6))

    def test_final_cost_random_state(self):
        pose = torch.randn(4, 1, 1, dtype=tdtype)
        velocity = torch.randn(4, 1, 1, dtype=tdtype)
        cost = self.static_cost.final_cost(pose, velocity)
        self.assertEqual(cost.shape, torch.Size([4]))  # Scalar output

    def test_final_cost_non_random_state(self):
        pose = torch.tensor([[[0.5]]], dtype=tdtype)
        velocity = torch.tensor([[[1.0]]], dtype=tdtype)

        cost = self.static_cost.final_cost(pose, velocity)
        expected_cost = torch.tensor([3.5], dtype=tdtype)  # Calculate expected cost manually
        self.assertTrue(torch.allclose(cost, expected_cost, rtol=1e-6))  # Test up to 6 decimal places


class TestStaticCostPypose(unittest.TestCase):
    def setUp(self) -> None:
        self.lam = 0.5
        self.gamma = 0.2
        self.upsilon = 2.0
        self.sigma = [[1.0, 0.0], [0.0, 1.0]]  # Example sigma matrix
        self.goal_pose = [1.0, 2.0, 0., 0., 0., 0., 1.]  # Example goal vector
        self.goal_vel = [0., 0., 0., 0., 0., 0.,]
        self.Q = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                  1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # Example weight vector

        self.static_cost = StaticPypose(self.lam, self.gamma, self.upsilon, self.sigma,
                                        self.goal_pose, self.goal_vel, self.Q, False)

    def test_state_cost_random_state(self):
        pose = pp.randn_SE3(2, 1, dtype=tdtype)
        velocity = torch.randn(2, 1, 6, dtype=tdtype)
        cost = self.static_cost.state_cost(pose, velocity)
        self.assertEqual(cost.shape, torch.Size([2]))  # Scalar output

    def test_state_cost_random_state(self):
        pose = pp.SE3([[[1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
                       [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
                       [[0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0+1e-10]],
                       [[1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0]]])
        velocity = torch.tensor([[[1.0,  1.0,  1.0,  1.0,  1.0,  1.0]],
                                 [[0.0,  0.0,  0.0,  0.0,  0.0,  0.0]],
                                 [[1.0, -1.0,  1.0, -1.0, -1.0, -1.0]],
                                 [[0.0,  0.0,  0.0,  0.0,  0.0,  0.0]]], dtype=tdtype)
        cost = self.static_cost.state_cost(pose, velocity)
        self.assertEqual(cost.shape, torch.Size([4]))
        expected_cost = torch.tensor([np.sqrt(6.0), np.sqrt(5.), np.sqrt(9 + np.pi**2), 0.0], dtype=tdtype)  # Calculate expected cost manually
        self.assertTrue(torch.allclose(cost, expected_cost, rtol=1e-6))  # Test up to 6 decimal places



if __name__ == '__main__':
    unittest.main()
