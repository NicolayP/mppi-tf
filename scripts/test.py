import tensorflow as tf
from model_base import ModelBase
from cost_base import CostBase
from controller_base import ControllerBase

class TestModel(tf.test.TestCase):
    def setUp(self):
        pass

    def testStep1(self):
        pass

    def testStep2(self):
        pass

    def testStep3(self):
        pass

    def testStep4(self):
        pass

class TestCost(tf.test.TestCase):
    def setUp(self):
        pass

    def testStateCost(self):
        pass

    def testStepCost(self):
        pass

class TestController(tf.test.TestCase):
    def setUp(self):
        pass

    def testDataPrep(self):
        pass

    def testUpdate(self):
        pass

    def testNew(self):
        pass

    def testShiftAndInit(self):
        pass

    def testAll(self):
        pass


tf.test.main()
