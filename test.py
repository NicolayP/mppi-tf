import tensorflow as tf
from scripts.test.test_controller import TestControllerBase, TestStateController, TestLaggedStateController
from scripts.test.test_models import TestAUVModel, TestVelPred, TestPredictor
from scripts.test.test_costs import TestCost, TestStaticCost, TestElipseCost, TestElipse3DCost


if __name__ == '__main__':
    tf.test.main()