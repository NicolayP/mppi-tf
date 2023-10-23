# import unittest
# import os
# import shutil
# import torch
# from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter

# from scripts.utils.utils import get_device
# from scripts.training.loss_fct import TrajLoss
# from scripts.models.nn_auv import AUVTraj
# from scripts.training.learning_utils import get_datasets, train_step, val_step, train

# import wandb

# wandb.init(mode="disabled")

# class TestTrainStepNN_CPU(unittest.TestCase):
#     def setUp(self):
#         data_dir = "data/csv/tests"
#         history = 2
#         parameter = {"dir": data_dir,
#                      "samples": 2,
#                      "split": 0.5,
#                      "frame": "body",
#                      "steps": 20,
#                      "history": history,
#                      "batch_size": 10,
#                      "shuffle": True,
#                      "num_workers": 8}
#         self.dls = get_datasets(parameter)
#         self.dl_train = self.dls[0]
#         self.dl_val = self.dls[1]
#         model_param = {"type": "auv_nn", "se3": True, "history": history}
#         self.device = get_device(cpu=True)
#         self.model = AUVTraj(model_param).to(self.device)
#         self.loss = TrajLoss().to(self.device)
#         self.optim = torch.optim.Adam(self.model.parameters(), lr=1e-4)
#         self.epoch = 1
#         self.epochs = 2

#     def test_train_step_writer(self):
#         train_step(self.dl_train, self.model, self.loss, self.epoch, self.device)

#     def test_val_step_writer(self):
#         val_step(self.dl_val, self.model, self.loss, self.epoch, self.device)

#     def test_train_writer(self):
#         train(self.dls, self.model, self.loss, self.optim, self.epochs, self.device)

#     def tearDown(self) -> None:
#         # need to delete every loged information
#         pass


# class TestTrainStepNN_GPU(unittest.TestCase):
#     def setUp(self):
#         data_dir = "data/csv/tests"
#         history = 2
#         parameter = {"dir": data_dir,
#                      "samples": 2,
#                      "split": 0.5,
#                      "frame": "body",
#                      "steps": 20,
#                      "history": history,
#                      "batch_size": 10,
#                      "shuffle": True,
#                      "num_workers": 8}
#         self.dls = get_datasets(parameter)
#         self.dl_train = self.dls[0]
#         self.dl_val = self.dls[1]
#         model_param = {"type": "auv_nn", "se3": True, "history": history}
#         self.device = get_device(0)
#         self.model = AUVTraj(model_param).to(self.device)
#         self.loss = TrajLoss().to(self.device)
#         self.optim = torch.optim.Adam(self.model.parameters(), lr=1e-4)
#         self.epoch = 1
#         self.epochs = 2

#     def test_train_step_writer(self):
#         train_step(self.dl_train, self.model, self.loss, self.optim, self.epoch, self.device)

#     def test_val_step_writer(self):
#         val_step(self.dl_val, self.model, self.loss, self.epoch, self.device)

#     def test_train_writer(self):
#         train(self.dls, self.model, self.loss, self.optim, self.epochs, self.device)

#     def tearDown(self) -> None:
#         # need to delete every loged information
#         pass


# class TestTrainStepRNN_CPU(unittest.TestCase):
#     def setUp(self):
#         data_dir = "data/csv/tests"
#         history = 2
#         parameter = {"dir": data_dir,
#                      "samples": 2,
#                      "split": 0.5,
#                      "frame": "body",
#                      "steps": 20,
#                      "history": history,
#                      "batch_size": 10,
#                      "shuffle": True,
#                      "num_workers": 8}
#         self.dls = get_datasets(parameter)
#         self.dl_train = self.dls[0]
#         self.dl_val = self.dls[1]
#         model_param = {"type": "auv_rnn", "se3": True}
#         self.device = get_device(cpu=True)
#         self.model = AUVTraj(model_param).to(self.device)
#         self.loss = TrajLoss().to(self.device)
#         self.optim = torch.optim.Adam(self.model.parameters(), lr=1e-4)
#         self.epoch = 1
#         self.epochs = 2

#     def test_train_step_writer(self):
#         train_step(self.dl_train, self.model, self.loss, self.optim, self.epoch, self.device)

#     def test_val_step_writer(self):
#         val_step(self.dl_val, self.model, self.loss, self.epoch, self.device)

#     def test_train_writer(self):
#         train(self.dls, self.model, self.loss, self.optim, self.epochs, self.device)

#     def tearDown(self) -> None:
#         # need to delete every loged information
#         pass


# class TestTrainStepRNN_GPU(unittest.TestCase):
#     def setUp(self):
#         data_dir = "data/csv/tests"
#         history = 2
#         parameter = {"dir": data_dir,
#                      "samples": 2,
#                      "split": 0.5,
#                      "frame": "body",
#                      "steps": 20,
#                      "history": history,
#                      "batch_size": 10,
#                      "shuffle": True,
#                      "num_workers": 8}
#         self.dls = get_datasets(parameter)
#         self.dl_train = self.dls[0]
#         self.dl_val = self.dls[1]
#         model_param = {"type": "auv_rnn", "se3": True}
#         self.device = get_device(0)
#         self.model = AUVTraj(model_param).to(self.device)
#         self.loss = TrajLoss().to(self.device)
#         self.optim = torch.optim.Adam(self.model.parameters(), lr=1e-4)
#         self.epoch = 1
#         self.epochs = 2

#     def test_train_step_writer(self):
#         train_step(self.dl_train, self.model, self.loss, self.optim, self.epoch, self.device)

#     def test_val_step_writer(self):
#         val_step(self.dl_val, self.model, self.loss, self.epoch, self.device)

#     def test_train_writer(self):
#         train(self.dls, self.model, self.loss, self.optim, self.epochs, self.device)

#     def tearDown(self) -> None:
#         # need to delete every loged information
#         pass


# class TestTrainStepLSTM_GPU(unittest.TestCase):
#     def setUp(self):
#         data_dir = "data/csv/tests"
#         history = 2
#         parameter = {"dir": data_dir,
#                      "samples": 2,
#                      "split": 0.5,
#                      "frame": "body",
#                      "steps": 20,
#                      "history": history,
#                      "batch_size": 10,
#                      "shuffle": True,
#                      "num_workers": 8}
#         self.dls = get_datasets(parameter)
#         self.dl_train = self.dls[0]
#         self.dl_val = self.dls[1]
#         model_param = {"type": "auv_lstm", "se3": True}
#         self.device = get_device(0)
#         self.model = AUVTraj(model_param).to(self.device)
#         self.loss = TrajLoss().to(self.device)
#         self.optim = torch.optim.Adam(self.model.parameters(), lr=1e-4)
#         self.epoch = 1
#         self.epochs = 2

#     def test_train_step_writer(self):
#         train_step(self.dl_train, self.model, self.loss, self.optim, self.epoch, self.device)

#     def test_val_step_writer(self):
#         val_step(self.dl_val, self.model, self.loss, self.epoch, self.device)

#     def test_train_writer(self):
#         train(self.dls, self.model, self.loss, self.optim, self.epochs, self.device)

#     def tearDown(self) -> None:
#         # need to delete every loged information
#         pass


# class TestTrainStepLSTM_GPU(unittest.TestCase):
#     def setUp(self):
#         data_dir = "data/csv/tests"
#         history = 2
#         parameter = {"dir": data_dir,
#                      "samples": 2,
#                      "split": 0.5,
#                      "frame": "body",
#                      "steps": 20,
#                      "history": history,
#                      "batch_size": 10,
#                      "shuffle": True,
#                      "num_workers": 8}
#         self.dls = get_datasets(parameter)
#         self.dl_train = self.dls[0]
#         self.dl_val = self.dls[1]
#         model_param = {"type": "auv_lstm", "se3": True}
#         self.device = get_device(0)
#         self.model = AUVTraj(model_param).to(self.device)
#         self.loss = TrajLoss().to(self.device)
#         self.optim = torch.optim.Adam(self.model.parameters(), lr=1e-4)
#         self.epoch = 1
#         self.epochs = 2

#     def test_train_step_writer(self):
#         train_step(self.dl_train, self.model, self.loss, self.optim, self.epoch, self.device)

#     def test_val_step_writer(self):
#         val_step(self.dl_val, self.model, self.loss, self.epoch, self.device)

#     def test_train_writer(self):
#         train(self.dls, self.model, self.loss, self.optim, self.epochs, self.device)

#     def tearDown(self) -> None:
#         # need to delete every loged information
#         pass


if __name__ == '__main__':
    unittest.main()