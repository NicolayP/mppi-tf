import unittest
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from scripts.utils.utils import get_device
from scripts.training.loss_fct import TrajLoss
from scripts.models.nn_auv import AUVTraj
from scripts.training.learning_utils import get_dataloader_model_input, train_step, val_step, traj_loss, train


class TestTrainStepNN_CPU(unittest.TestCase):
    def setUp(self):
        data_dir = "data/tests"
        nb_files, steps, history = 1, 3, 2
        train_params = {"batch_size": 10, "shuffle": True, "num_workers": 8}
        self.dl = get_dataloader_model_input(data_dir, nb_files, steps, history, train_params)
        model_param = {"model": {"type": "nn", "se3": True, "history": history}}
        self.device = get_device(cpu=True)
        self.model = AUVTraj(model_param).to(self.device)
        self.loss = TrajLoss().to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.epochs = 1
    
    def test_train_step(self):
        train_step(self.dl, self.model, self.loss, self.optim, None, self.epochs, self.device)

    def test_val_step(self):
        val_step(self.dl, self.model, self.loss, None, self.epochs, self.device)

    def test_train(self):
        pass


class TestTrainStepNN_GPU(unittest.TestCase):
    def setUp(self):
        data_dir = "data/tests"
        nb_files, steps, history = 1, 3, 2
        train_params = {"batch_size": 10, "shuffle": True, "num_workers": 8}
        self.dl = get_dataloader_model_input(data_dir, nb_files, steps, history, train_params)
        model_param = {"model": {"type": "nn", "se3": True, "history": history}}
        self.device = get_device(0)
        self.model = AUVTraj(model_param).to(self.device)
        self.loss = TrajLoss().to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.epochs = 1
    
    def test_train_step(self):
        train_step(self.dl, self.model, self.loss, self.optim, None, self.epochs, self.device)

    def test_val_step(self):
        val_step(self.dl, self.model, self.loss, None, self.epochs, self.device)

    def test_train(self):
        pass


class TestTrainStepRNN_CPU(unittest.TestCase):
    def setUp(self):
        data_dir = "data/tests"
        nb_files, steps, history = 1, 3, 1
        train_params = {"batch_size": 10, "shuffle": True, "num_workers": 8}
        self.dl = get_dataloader_model_input(data_dir, nb_files, steps, history, train_params)
        model_param = {"model": {"type": "rnn", "se3": True}}
        self.device = get_device(cpu=True)
        self.model = AUVTraj(model_param).to(self.device)
        self.loss = TrajLoss().to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.epochs = 1
    
    def test_train_step(self):
        train_step(self.dl, self.model, self.loss, self.optim, None, self.epochs, self.device)

    def test_val_step(self):
        val_step(self.dl, self.model, self.loss, None, self.epochs, self.device)

    def test_train(self):
        pass


class TestTrainStepRNN_GPU(unittest.TestCase):
    def setUp(self):
        data_dir = "data/tests"
        nb_files, steps, history = 1, 3, 2
        train_params = {"batch_size": 10, "shuffle": True, "num_workers": 8}
        self.dl = get_dataloader_model_input(data_dir, nb_files, steps, history, train_params)
        model_param = {"model": {"type": "rnn", "se3": True}}
        self.device = get_device(0)
        self.model = AUVTraj(model_param).to(self.device)
        self.loss = TrajLoss().to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.epochs = 1
    
    def test_train_step(self):
        train_step(self.dl, self.model, self.loss, self.optim, None, self.epochs, self.device)

    def test_val_step(self):
        val_step(self.dl, self.model, self.loss, None, self.epochs, self.device)

    def test_train(self):
        pass


class TestTrainStepLSTM_GPU(unittest.TestCase):
    def setUp(self):
        data_dir = "data/tests"
        nb_files, steps, history = 1, 3, 2
        train_params = {"batch_size": 10, "shuffle": True, "num_workers": 8}
        self.dl = get_dataloader_model_input(data_dir, nb_files, steps, history, train_params)
        model_param = {"model": {"type": "lstm", "se3": True}}
        self.device = get_device(0)
        self.model = AUVTraj(model_param).to(self.device)
        self.loss = TrajLoss().to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.epochs = 1
    
    def test_train_step(self):
        train_step(self.dl, self.model, self.loss, self.optim, None, self.epochs, self.device)

    def test_val_step(self):
        val_step(self.dl, self.model, self.loss, None, self.epochs, self.device)

    def test_train(self):
        pass


class TestTrainStepLSTM_GPU(unittest.TestCase):
    def setUp(self):
        data_dir = "data/tests"
        nb_files, steps, history = 1, 3, 2
        train_params = {"batch_size": 10, "shuffle": True, "num_workers": 8}
        self.dl = get_dataloader_model_input(data_dir, nb_files, steps, history, train_params)
        model_param = {"model": {"type": "lstm", "se3": True}}
        self.device = get_device(0)
        self.model = AUVTraj(model_param).to(self.device)
        self.loss = TrajLoss().to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.epochs = 1
    
    def test_train_step(self):
        train_step(self.dl, self.model, self.loss, self.optim, None, self.epochs, self.device)

    def test_val_step(self):
        val_step(self.dl, self.model, self.loss, None, self.epochs, self.device)

    def test_train(self):
        pass


if __name__ == '__main__':
    unittest.main()