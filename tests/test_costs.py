import unittest
import torch
from scripts.utils.utils import tdtype  # Import the required dtype

from scripts.costs.cost_base import CostBase
from scripts.costs.static import Static

class TestCostBase(unittest.TestCase):
    def setUp(self):
        self.lam = 0.5
        self.gamma = 0.2
        self.upsilon = 2.0
        self.sigma = [[1.0, 0.0], [0.0, 1.0]]  # Example sigma matrix

        self.cost = CostBase(self.lam, self.gamma, self.upsilon, self.sigma)

    def test_action_cost(self):
        action = torch.randn(4, 2, 1, dtype=tdtype)
        noise = torch.randn(4, 2, 1, dtype=tdtype)
        cost = self.cost.action_cost(action, noise)
        self.assertEqual(cost.shape, torch.Size([4, 1, 1]))

    def test_state_cost(self):
        state = torch.randn(4, 5, 1, dtype=tdtype)
        with self.assertRaises(NotImplementedError):
            self.cost.state_cost(state)

    def test_final_cost(self):
        state = torch.randn(4, 5, 1, dtype=tdtype)
        with self.assertRaises(NotImplementedError):
            self.cost.final_cost(state)

    def test_forward(self):
        state = torch.randn(4, 5, 1, dtype=tdtype)
        action = torch.randn(4, 2, 1, dtype=tdtype)
        noise = torch.randn(4, 2, 1, dtype=tdtype)
        with self.assertRaises(NotImplementedError):
            self.cost(state, action, noise)

    def test_final_cost_forward(self):
        state = torch.randn(2, 3, 1, dtype=tdtype)
        with self.assertRaises(NotImplementedError):
            self.cost(state, final=True)


class TestStaticCost(unittest.TestCase):
    def setUp(self):
        self.lam = 0.5
        self.gamma = 0.2
        self.upsilon = 2.0
        self.sigma = [[1.0, 0.0], [0.0, 1.0]]  # Example sigma matrix
        self.goal = [[1.0], [2.0]]  # Example goal vector
        self.Q = [2.0, 3.0]  # Example weight vector

        self.static_cost = Static(self.lam, self.gamma, self.upsilon, self.sigma, self.goal, self.Q)

    def test_state_cost_random_state(self):
        state = torch.randn(4, 2, 1, dtype=tdtype)
        cost = self.static_cost.state_cost(state)
        self.assertEqual(cost.shape, torch.Size([4, 1, 1]))  # Scalar output

    def test_state_cost_non_random_state(self):
        state = torch.tensor([[0.5], [1.0]], dtype=tdtype)
        cost = self.static_cost.state_cost(state)
        expected_cost = torch.tensor([[[3.5]]], dtype=tdtype)  # Calculate expected cost manually
        self.assertTrue(torch.allclose(cost, expected_cost, rtol=1e-6))

    def test_final_cost_random_state(self):
        state = torch.randn(4, 2, 1, dtype=tdtype)
        cost = self.static_cost.final_cost(state)
        self.assertEqual(cost.shape, torch.Size([4, 1, 1]))  # Scalar output

    def test_final_cost_non_random_state(self):
        state = torch.tensor([[0.5], [1.0]], dtype=tdtype)
        cost = self.static_cost.final_cost(state)
        expected_cost = torch.tensor([[[3.5]]], dtype=tdtype)  # Calculate expected cost manually
        self.assertTrue(torch.allclose(cost, expected_cost, rtol=1e-6))  # Test up to 6 decimal places


if __name__ == '__main__':
    unittest.main()
