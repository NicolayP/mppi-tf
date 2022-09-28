import numpy as np
import tensorflow as tf

from utile import dtype, npdtype
from model import ModelDtype, FooModel
from controller import MPPIBase, MPPIClassic
from cost import FooCost


class TestController(tf.test.TestCase):
    def setUp(self):
        self.h = 2
        self.tau = 3
        self.dt=0.1
        self.lam=1.
        self.gamma=1.
        self.upsilon=1.
        self.k = 2
        self.sigma=np.eye(6)
        fake_x = np.zeros(shape=(1, self.h, 13, 1), dtype=npdtype)
        fake_u = np.zeros(shape=(1, self.h, 6, 1), dtype=npdtype)
        self.data = ModelDtype(fake_x, fake_u, self.h)
        self.model = FooModel(k=3, dt=self.dt)
        self.cost = FooCost(self.lam, self.gamma, self.upsilon, self.sigma)
    
    def testMPPIBase(self):
        cont = MPPIBase(
            self.model, self.cost, self.data, self.k,
            self.tau, self.lam, self.upsilon, self.sigma,
        )


if __name__ == "__main__":
    tf.test.main()