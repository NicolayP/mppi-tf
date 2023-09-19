import unittest
import torch
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
        train_params = {"batch_size": 512, "shuffle": True, "num_workers": 8}
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

    def test_len(self):
        # The lenght should be equal to nb_files * (transitions - steps) -> 3 * (500-10)
        self.assertEqual(len(self.ds), self.nb_files * (500 - self.steps - (self.history - 1)))

    def test_get_item(self):
        x, u, sub_traj, sub_vel, sub_dv = self.ds[1]
        self.assertEqual(x.shape, (1, 7+6))
        self.assertEqual(u.shape, (self.steps, 6))
        self.assertEqual(sub_traj.shape, (self.steps, 7))
        self.assertEqual(sub_vel.shape, (self.steps, 6))
        self.assertEqual(sub_dv.shape, (self.steps, 6))




if __name__ == '__main__':
    unittest.main()