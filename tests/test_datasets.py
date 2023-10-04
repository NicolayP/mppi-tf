import unittest
import torch
from torch.utils.data import DataLoader
import pypose as pp
import os

import random
from scripts.training.datasets import DatasetList3D, DatasetListModelInput
from scripts.utils.utils import tdtype, read_files


class TestDatasetList3D(unittest.TestCase):
    def setUp(self) -> None:
        # data in this directory should be of used for testing, 3 trajectories with 500 steps.
        data_dir = "data/tests"
        self.nb_files = 3
        self.steps = 10
        files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
        files = random.sample(files, self.nb_files)
        dfs = read_files(data_dir, files)
        self.batch_size = 10
        train_params = {"batch_size": self.batch_size, "shuffle": True, "num_workers": 8}
        self.ds = DatasetList3D(
                data_list=dfs,
                steps=self.steps,
                v_frame="body",
                dv_frame="body",
                rot="quat",
                act_normed=False,
                se3=True,
                out_normed=False,
                stats=None)
        self.dl = DataLoader(self.ds, **train_params)

    def test_len(self):
        # The lenght should be equal to nb_files * (transitions - steps) -> 3 * (500-10)
        self.assertEqual(len(self.ds), self.nb_files * (500 - self.steps))

    def test_get_item(self):
        x, u, sub_traj, sub_vel, sub_dv = self.ds[1]
        self.assertEqual(x.shape, (1, 7+6))
        self.assertEqual(u.shape, (self.steps, 6))
        self.assertEqual(sub_traj.shape, (self.steps, 7))
        self.assertEqual(sub_vel.shape, (self.steps, 6))
        self.assertEqual(sub_dv.shape, (self.steps, 6))

    def test_nb_trajs(self):
        self.assertEqual(self.ds.nb_trajs, 3)

    def test_get_traj(self):
        idx = 4 # Index error
        with self.assertRaises(IndexError):
            self.ds.get_traj(idx)

        traj = self.ds.get_traj(2)
        self.assertEqual(traj.shape, (500, 7 + 6))

    def test_get_trajs(self):
        trajs, vels, dvs, acitons = self.ds.get_trajs()
        self.assertEqual(trajs.shape, (3, 500, 7))
        self.assertEqual(vels.shape, (3, 500, 6))
        self.assertEqual(dvs.shape, (3, 500, 6))
        self.assertEqual(acitons.shape, (3, 500, 6))

    def test_get_stats(self):
        mean, std = self.ds.get_stats()
        pass

    def test_get_batch(self):
        for data in self.dl:
            x, u, sub_traj, sub_vel, sub_dv = data
            self.assertEqual(x.shape, (self.batch_size, 1, 7+6))
            self.assertEqual(u.shape, (self.batch_size, self.steps, 6))
            self.assertEqual(sub_traj.shape, (self.batch_size, self.steps, 7))
            self.assertEqual(sub_vel.shape, (self.batch_size, self.steps, 6))
            self.assertEqual(sub_dv.shape, (self.batch_size, self.steps, 6))


class TestDatasetModelInput(unittest.TestCase):
    def setUp(self) -> None:
        # data in this directory should be of used for testing, 3 trajectories with 500 steps.
        data_dir = "data/tests"
        self.nb_files = 3
        self.steps = 3
        self.history = 2
        files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
        files = random.sample(files, self.nb_files)
        dfs = read_files(data_dir, files)
        self.batch_size = 10
        train_params = {"batch_size": self.batch_size, "shuffle": True, "num_workers": 8}
        self.ds = DatasetListModelInput(
                data_list=dfs,
                steps=self.steps,
                history=self.history,
                v_frame="body",
                dv_frame="body",
                act_normed=False,
                se3=True,
                out_normed=False,
                stats=None)

        self.dl = DataLoader(self.ds, **train_params)

    def test_len(self):
        # The lenght should be equal to nb_files * (transitions - steps) -> 3 * (500-10)
        self.assertEqual(len(self.ds), self.nb_files * (500 - self.steps - (self.history - 1)))

    def test_get_item(self):
        pose_past, vel_past, u_past, u, sub_traj, sub_vel, sub_dv = self.ds[1]
        self.assertEqual(pose_past.shape, (self.history, 7))
        self.assertEqual(vel_past.shape, (self.history, 6))
        self.assertEqual(u_past.shape, (self.history-1, 6))
        self.assertEqual(u.shape, (self.steps, 6))
        self.assertEqual(sub_traj.shape, (self.steps, 7))
        self.assertEqual(sub_vel.shape, (self.steps, 6))
        self.assertEqual(sub_dv.shape, (self.steps, 6))

    def test_nb_trajs(self):
        self.assertEqual(self.ds.nb_trajs, 3)

    def test_get_traj(self):
        with self.assertRaises(IndexError):
            self.ds.get_traj(4)

        pose_init, vel_init, act_int, traj, vel, dv, action = self.ds.get_traj(2)
        self.assertEqual(pose_init.shape, (self.history, 7))
        self.assertEqual(vel_init.shape, (self.history, 6))
        self.assertEqual(act_int.shape, (self.history-1, 6))
        self.assertEqual(traj.shape, (500, 7))
        self.assertEqual(vel.shape, (500, 6))
        self.assertEqual(dv.shape, (500, 6))
        self.assertEqual(action.shape, (500, 6))

    def test_get_trajs(self):
        poses_init, vels_init, acts_init, trajs, vels, dvs, actions = self.ds.get_trajs()
        self.assertEqual(poses_init.shape, (3, self.history, 7))
        self.assertEqual(vels_init.shape, (3, self.history, 6))
        self.assertEqual(acts_init.shape, (3, self.history-1, 6))
        self.assertEqual(trajs.shape, (3, 500, 7))
        self.assertEqual(vels.shape, (3, 500, 6))
        self.assertEqual(dvs.shape, (3, 500, 6))
        self.assertEqual(actions.shape, (3, 500, 6))

    def test_get_stats(self):
        pass

    def test_get_batch(self):
        it = iter(self.dl)
        data = next(it)
        pose_past, vel_past, u_past, u, sub_traj, sub_vel, sub_dv = data
        self.assertEqual(pose_past.shape, (self.batch_size, self.history, 7))
        self.assertEqual(vel_past.shape, (self.batch_size, self.history, 6))
        self.assertEqual(u_past.shape, (self.batch_size, self.history - 1, 6))
        self.assertEqual(u.shape, (self.batch_size, self.steps, 6))
        self.assertEqual(sub_traj.shape, (self.batch_size, self.steps, 7))
        self.assertEqual(sub_vel.shape, (self.batch_size, self.steps, 6))
        self.assertEqual(sub_dv.shape, (self.batch_size, self.steps, 6))

if __name__ == '__main__':
    unittest.main()