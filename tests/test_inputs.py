import unittest
import torch
from scripts.utils.utils import tdtype
from scripts.inputs.ControllerInput import ControllerInput
from scripts.inputs.ModelInput import ModelInput



class TestControllerInput(unittest.TestCase):
    def setUp(self):
        self.steps = 5
        self.sDim = 2
        self.aDim = 1
        self.input_module = ControllerInput(self.steps, self.sDim, self.aDim)

    def test_add_state(self):
        state = torch.randn(self.sDim, 1, dtype=tdtype)
        self.input_module.add_state(state)
        self.assertTrue(torch.all(torch.eq(self.input_module.states[-1], state)))

    def test_add_act(self):
        action = torch.randn(self.aDim, 1, dtype=tdtype)
        self.input_module.add_act(action)
        self.assertTrue(torch.all(torch.eq(self.input_module.actions[-1], action)))

    def test_add(self):
        state = torch.randn(self.sDim, 1, dtype=tdtype)
        action = torch.randn(self.aDim, 1, dtype=tdtype)
        self.input_module.add(state, action)
        self.assertTrue(torch.all(torch.eq(self.input_module.states[-1], state)))
        self.assertTrue(torch.all(torch.eq(self.input_module.actions[-1], action)))

    def test_get(self):
        k = 3
        states, actions = self.input_module.get(k)
        self.assertEqual(states.shape, (k, self.steps, self.sDim, 1))
        self.assertEqual(actions.shape, (k, self.steps - 1, self.aDim, 1))

    def test_get_steps(self):
        self.assertEqual(self.input_module.get_steps(), self.steps)

    def test_is_init(self):
        self.assertFalse(self.input_module.is_init)
        for _ in range(self.steps - 1):
            self.input_module.add(torch.randn(self.sDim, 1, dtype=tdtype),
                                  torch.randn(self.aDim, 1, dtype=tdtype))
        self.assertFalse(self.input_module.is_init)
        self.input_module.add_state(torch.randn(self.sDim, 1, dtype=tdtype))
        self.assertTrue(self.input_module.is_init)


class TestControllerInputSteps1(unittest.TestCase):
    def setUp(self):
        self.steps = 1
        self.sDim = 13
        self.aDim = 6
        self.input_module = ControllerInput(self.steps, self.sDim, self.aDim)

    def test_add_state(self):
        state = torch.randn(self.sDim, 1, dtype=tdtype)
        self.input_module.add_state(state)
        self.assertTrue(torch.all(torch.eq(self.input_module.states[-1], state)))

    def test_add_act(self):
        action = torch.randn(self.aDim, 1, dtype=tdtype)
        self.input_module.add_act(action)
        with self.assertRaises(IndexError):
            element = self.input_module.actions[-1]

    def test_add(self):
        state = torch.randn(self.sDim, 1, dtype=tdtype)
        action = torch.randn(self.aDim, 1, dtype=tdtype)
        self.input_module.add(state, action)
        self.assertTrue(torch.all(torch.eq(self.input_module.states[-1], state)))
        with self.assertRaises(IndexError):
            element = self.input_module.actions[-1]

    def test_get(self):
        k = 3
        states, actions = self.input_module.get(k)
        self.assertEqual(states.shape, (k, self.steps, self.sDim, 1))
        self.assertEqual(actions.shape, (k, self.steps - 1, self.aDim, 1))

    def test_get_steps(self):
        self.assertEqual(self.input_module.get_steps(), self.steps)

    def test_is_init(self):
        self.assertFalse(self.input_module.is_init)
        for _ in range(self.steps - 1):
            self.input_module.add(torch.randn(self.sDim, 1, dtype=tdtype),
                                  torch.randn(self.aDim, 1, dtype=tdtype))
        self.assertFalse(self.input_module.is_init)
        self.input_module.add_state(torch.randn(self.sDim, 1, dtype=tdtype))
        self.assertTrue(self.input_module.is_init)


class TestModelInput(unittest.TestCase):
    def setUp(self):
        self.k = 3
        self.steps = 5
        self.sDim = 13
        self.aDim = 6

        self.controller_input = ControllerInput(self.steps, self.sDim, self.aDim)
        for _ in range(self.steps - 1):
            self.controller_input.add(torch.randn(self.sDim, 1), 
                                      torch.randn(self.aDim, 1))
        self.controller_input.add_state(torch.randn(self.sDim, 1))
        
        self.model_input = ModelInput(self.k, self.steps, self.sDim, self.aDim)
        self.model_input.init(self.controller_input)

    def test_init(self):
        self.assertEqual(self.model_input.states.shape, (self.k, self.steps, self.sDim, 1))
        self.assertEqual(self.model_input.actions.shape, (self.k, self.steps, self.aDim, 1))

    def test_forward(self):
        action = torch.randn(self.k, self.aDim, 1, dtype=tdtype)
        states, actions = self.model_input(action)
        self.assertEqual(states.shape, (self.k, self.steps, self.sDim, 1))
        self.assertEqual(actions.shape, (self.k, self.steps, self.aDim, 1))
        self.assertTrue(torch.all(torch.eq(self.model_input.actions[:, -1], action)))

    def test_update(self):
        new_state = torch.randn(self.k, self.sDim, 1, dtype=tdtype)
        new_action = torch.randn(self.k, self.aDim, 1, dtype=tdtype)
        self.model_input.update(new_state, new_action)
        self.assertTrue(torch.all(torch.eq(self.model_input.states[:, -1], new_state)))
        self.assertTrue(torch.all(torch.eq(self.model_input.actions[:, -1], new_action)))


if __name__ == '__main__':
    unittest.main()