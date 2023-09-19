import unittest
import torch
import pypose as pp

from scripts.training.loss_fct import GeodesicLoss, TrajLoss
from scripts.utils.utils import tdtype



class TestGeodesicLoss(unittest.TestCase):
    def setUp(self) -> None:
        self.geodesic_loss = GeodesicLoss()

    def test_geo_loss_random_sample(self):
        X1 = pp.randn_SE3(1, dtype=tdtype)
        X2 = pp.randn_SE3(1, dtype=tdtype)
        loss = self.geodesic_loss(X1, X2)
        self.assertEqual(loss.shape, (1, 6))

    def test_geo_loss_random_sample_batch(self):
        X1 = pp.randn_SE3(10, 5, dtype=tdtype)
        X2 = pp.randn_SE3(10, 5, dtype=tdtype)
        loss = self.geodesic_loss(X1, X2)
        self.assertEqual(loss.shape, (10, 5, 6))
    

class TestTrajLoss(unittest.TestCase):
    def setUp(self) -> None:
        self.k = 100
        self.tau = 50
        self.alpha = 1.
        self.beta = 1.
        self.gamma = 1.
        self.traj_loss = TrajLoss(self.alpha, self.beta, self.gamma)

    def test_has_v(self):
        self.assertTrue(self.traj_loss.has_v())
        self.traj_loss.beta = 0.
        self.assertFalse(self.traj_loss.has_v())

    def test_has_dv(self):
        self.assertTrue(self.traj_loss.has_dv())
        self.traj_loss.gamma = 0.
        self.assertFalse(self.traj_loss.has_dv())

    def test_split(self):
        t1 = pp.randn_SE3(self.k, self.tau)
        t2 = pp.randn_SE3(self.k, self.tau)
        v1 = torch.randn(self.k, self.tau, 6)
        v2 = torch.randn(self.k, self.tau, 6)
        dv1 = torch.randn(self.k, self.tau, 6)
        dv2 = torch.randn(self.k, self.tau, 6)

        t_l, v_l, dv_l = self.traj_loss.split_loss(t1, t2, v1, v2, dv1, dv2)

        self.assertEqual(t_l.shape, (6, ))
        self.assertEqual(v_l.shape, (6, ))
        self.assertEqual(dv_l.shape, (6, ))

    def test_loss(self):
        t1 = pp.randn_SE3(self.k, self.tau)
        t2 = pp.randn_SE3(self.k, self.tau)
        v1 = torch.randn(self.k, self.tau, 6)
        v2 = torch.randn(self.k, self.tau, 6)
        dv1 = torch.randn(self.k, self.tau, 6)
        dv2 = torch.randn(self.k, self.tau, 6)

        l = self.traj_loss.loss(t1, t2, v1, v2, dv1, dv2)

        self.assertEqual(l.shape, ( )) # scalar

    def test_forward(self):
        t1 = pp.randn_SE3(self.k, self.tau)
        t2 = pp.randn_SE3(self.k, self.tau)
        v1 = torch.randn(self.k, self.tau, 6)
        v2 = torch.randn(self.k, self.tau, 6)
        dv1 = torch.randn(self.k, self.tau, 6)
        dv2 = torch.randn(self.k, self.tau, 6)

        l = self.traj_loss(t1, t2, v1, v2, dv1, dv2)

        self.assertEqual(l.shape, ( )) # scalar

    def test_forward_split(self):
        t1 = pp.randn_SE3(self.k, self.tau)
        t2 = pp.randn_SE3(self.k, self.tau)
        v1 = torch.randn(self.k, self.tau, 6)
        v2 = torch.randn(self.k, self.tau, 6)
        dv1 = torch.randn(self.k, self.tau, 6)
        dv2 = torch.randn(self.k, self.tau, 6)

        t_l, v_l, dv_l = self.traj_loss(t1, t2, v1, v2, dv1, dv2, split=True)

        self.assertEqual(t_l.shape, (6, ))
        self.assertEqual(v_l.shape, (6, ))
        self.assertEqual(dv_l.shape, (6, ))



if __name__ == '__main__':
    unittest.main()