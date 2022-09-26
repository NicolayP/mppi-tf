import numpy as np
import tensorflow as tf

from model import ModelBase, FooModel

class TestModelBase(tf.test.TestCase):
    def setUp(self):
        self.k = 3
        self.m = ModelBase(self.k)

        self.x = np.zeros(shape=(3, 1, 13, 1))
        self.u = np.zeros(shape=(3, 1, 6, 1))

    def testBase(self):
        with self.assertRaises(NotImplementedError):
            self.m("foo", self.x, self.u)
        
        with self.assertRaises(NotImplementedError):
            self.m.predict(self.x, self.u)


class TestFooModel(tf.test.TestCase):
    def setUp(self):
        self.k = 3
        self.m = FooModel(self.k)

        self.x = np.zeros(shape=(3, 1, 13, 1))
        self.u = np.zeros(shape=(3, 1, 6, 1))

    def testCall(self):
        x = self.m("foo", self.x, self.u)
        self.assertAllClose(x, np.squeeze(self.x, axis=1))

    def testPredict(self):
        self.assertAllClose(self.m._k.numpy(), 3)
        pred = self.m.predict(self.x[:1], self.u[:1])
        self.assertAllClose(self.x[:1, 0], pred)
        self.assertAllClose(self.m._k.numpy(), 3)


if __name__ == '__main__':
    tf.test.main()
