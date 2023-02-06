import tensorflow as tf

from scripts.test.test_models import TestAUVModel #, TestVelPred, TestPredictor

from scripts.test.test_costs import TestCost, TestPrimitiveCollision, TestCylinderCollision, TestStaticCost, TestStaticWCollision, TestElipseCost, TestElipse3DCost
from scripts.test.test_utile import TestSkewOP #, TestFlattenSE3, TestToSE3MatOP, TestSO3intOP, TestSE3intOP
from scripts.test.test_controller import TestControllerBase, TestStateController #, TestLaggedStateController


if __name__ == '__main__':
    tf.test.main()