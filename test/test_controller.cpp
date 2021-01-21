#include "gtest/gtest.h"
#include "controller_base.hpp"

#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/summary/summary_file_writer.h"

#include "utile.hpp"

#include <string>

using namespace std;
using namespace tensorflow;
using namespace tensorflow::ops;

class ControllerBaseTest : public ::testing::Test {
protected:
    ControllerBaseTest () : root(Scope::NewRootScope()),
                            k(5), tau(3), a_dim(2),
                            cont(root, k, tau, 0.01, 1., 4, a_dim){
        c = Tensor(DT_FLOAT, TensorShape({k, 1, 1}));
        n = Tensor(DT_FLOAT, TensorShape({k, tau, a_dim, 1}));
        a = Tensor(DT_FLOAT, TensorShape({tau, a_dim, 1}));

        vector<float> cost = {3., 10., 0., 1., 5.,};
        vector<float> noise = {1., -0.5, 1., -0.5, 2., 1.,
                               0.3, 0, 2., 0.2, 1.2, 3.,
                               0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                               0.6, 0.7, 0.2, -0.3, 0.1, -0.4,
                               -2., -3., -4., -1., 0., 0.};

        vector<float> action = {1., 0.5, 2.3, 4.5, 2.1, -0.4};

        copy_n(cost.begin(), cost.size(), c.flat<float>().data());
        copy_n(noise.begin(), noise.size(), n.flat<float>().data());
        copy_n(action.begin(), action.size(), a.flat<float>().data());


    }

    Scope root;
    int k, tau, a_dim;
    Tensor c, n, a;
    ControllerBase cont;

    void test_tensor (Tensor computed,
                      vector<float>& expected,
                      vector<int>& dims, string name="") {
        float* data = computed.flat<float>().data();

        TensorShape shape(computed.shape());

        ASSERT_EQ(shape.dims(), dims.size()) << name << " Shape error at index : ";

        for (pair<TensorShapeIter<TensorShape>, vector<int>::iterator>
                 it(shape.begin(), dims.begin());
             it.first != shape.end();
             ++it.first, ++it.second) {
            ASSERT_EQ((*(it.first)).size, (*it.second)) << name
                                                        << " Dimension size error at index : "
                                                         << it.second - dims.begin();
        }

        int el = computed.NumElements();

        for (int i=0; i < el; i++) {
            EXPECT_FLOAT_EQ(data[i], expected[i]) << name << " State error at index : " << i;
        }
    }
};

TEST_F(ControllerBaseTest, testDataPrep) {
    vector<Tensor> o;

    vector<int> act_dim = {a_dim, 1};
    vector<int> noise_dim = {k, a_dim, 1};

    vector<float> exp_a0 = {1., 0.5};
    vector<float> exp_a1 = {2.3, 4.5};
    vector<float> exp_a2 = {2.1, -0.4};
    vector<float> exp_n0 = {1., -0.5, 0.3, 0, 0.5, 0.5, 0.6, 0.7, -2., -3.};
    vector<float> exp_n1 = {1., -0.5, 2., 0.2, 0.5, 0.5, 0.2, -0.3, -4, -1};
    vector<float> exp_n2 = {2., 1., 1.2, 3., 0.5, 0.5, 0.1, -0.4, 0., 0.};

    auto noises = Identity(root, n);
    auto actions = Identity(root, a);

    auto a0 = cont.mPrepareAction(root, actions, 0);
    auto n0 = cont.mPrepareNoise(root, noises, 0);

    auto a1 = cont.mPrepareAction(root, actions, 1);
    auto n1 = cont.mPrepareNoise(root, noises, 1);

    auto a2 = cont.mPrepareAction(root, actions, 2);
    auto n2 = cont.mPrepareNoise(root, noises, 2);

    ClientSession sess(root);
    TF_CHECK_OK(sess.Run({a0, n0, a1, n1, a2, n2}, &o));

    test_tensor(o[0], exp_a0, act_dim);
    test_tensor(o[1], exp_n0, noise_dim);
    test_tensor(o[2], exp_a1, act_dim);
    test_tensor(o[3], exp_n1, noise_dim);
    test_tensor(o[4], exp_a2, act_dim);
    test_tensor(o[5], exp_n2, noise_dim);


}

TEST_F(ControllerBaseTest, testUpdate) {
    vector<Tensor> o;

    vector<float> beta = {0};
    vector<int> beta_dim = {1, 1};

    vector<float> exp_arg = {-3., -10., 0, -1., -5.};
    vector<int> exp_arg_dim = {k, 1, 1};

    vector<float> exp = {0.049787068367863944,
                         4.5399929762484854e-05,
                         1,
                         0.36787944117144233,
                         0.006737946999085467};
    vector<int> exp_dim = {k, 1, 1};

    vector<float> nabla = {1.424449856468154};
    vector<int> nabla_dim = {1, 1};

    vector<float> weight = {0.034951787275480706,
                            3.1871904480408675e-05,
                            0.7020254138530686,
                            0.2582607169364174,
                            0.004730210030553017};
    vector<int> weight_dim = {k, 1, 1};

    vector<float> expected = {0.034951787275480706*1. + 3.1871904480408675e-05*0.3 + 0.7020254138530686*0.5 + 0.2582607169364174*0.6 + 0.004730210030553017*(-2),
                              0.034951787275480706*(-0.5) + 3.1871904480408675e-05*0 + 0.7020254138530686*0.5 + 0.2582607169364174*0.7 + 0.004730210030553017*(-3),
                              0.034951787275480706*1 + 3.1871904480408675e-05*2 + 0.7020254138530686*0.5 + 0.2582607169364174*0.2 + 0.004730210030553017*(-4),
                              0.034951787275480706*(-0.5) + 3.1871904480408675e-05*0.2 + 0.7020254138530686*0.5 + 0.2582607169364174*(-0.3) + 0.004730210030553017*(-1),
                              0.034951787275480706*2 + 3.1871904480408675e-05*1.2 + 0.7020254138530686*0.5 + 0.2582607169364174*0.1 + 0.004730210030553017*0,
                              0.034951787275480706*1 + 3.1871904480408675e-05*3 + 0.7020254138530686*0.5 + 0.2582607169364174*(-0.4) + 0.004730210030553017*0};
    vector<int> expected_dim = {tau, a_dim, 1};

    auto cost = Identity(root, c);
    auto noises = Identity(root, n);

    auto b = cont.mBeta(root.NewSubScope("beta"), cost);
    auto e_arg = cont.mExpArg(root.NewSubScope("e_arg"), cost, b);
    auto e = cont.mExp(root.NewSubScope("e"), e_arg);
    auto nab = cont.mNabla(root.NewSubScope("nab"), e);
    auto w = cont.mWeights(root.NewSubScope("w"), e, nab);
    auto w_n = cont.mWeightedNoise(root.NewSubScope("wn"), w, noises);
    auto sum_w = Sum(root, w, {0});

    ClientSession sess(root);
    TF_CHECK_OK(sess.Run({cost, noises, b, e_arg, e, nab, w, w_n, sum_w}, &o));

    test_tensor(o[2], beta, beta_dim, "beta");
    test_tensor(o[3], exp_arg, exp_arg_dim, "exp_arg");
    test_tensor(o[4], exp, exp_dim, "exp");
    test_tensor(o[5], nabla, nabla_dim, "nabla");
    test_tensor(o[6], weight, weight_dim, "weights");
    test_tensor(o[7], expected, expected_dim, "weighted noise");

    EXPECT_FLOAT_EQ(o[8].flat<float>().data()[0], 1);


}

TEST_F(ControllerBaseTest, testNew) {
    vector<Tensor> o;
    vector<float> next0 = {};
    vector<int> nextDim0 = {0, 2, 1};
    vector<float> next1 = {1, 0.5};
    vector<int> nextDim1 = {1, 2, 1};
    vector<float> next2 = {1, 0.5, 2.3, 4.5};
    vector<int> nextDim2 = {2, 2, 1};
    vector<float> next3 = {1., 0.5, 2.3, 4.5, 2.1, -0.4};
    vector<int> nextDim3 = {3, 2, 1};
    auto in = Identity(root, a);
    auto n0 = cont.mGetNew(root, in, 0);
    auto n1 = cont.mGetNew(root, in, 1);
    auto n2 = cont.mGetNew(root, in, 2);
    auto n3 = cont.mGetNew(root, in, 3);

    ClientSession sess(root);
    TF_CHECK_OK(sess.Run({in,
                         n0, n1, n2, n3
                     }, &o));
    test_tensor(o[1], next0, nextDim0, "0");
    test_tensor(o[2], next1, nextDim1, "1");
    test_tensor(o[3], next2, nextDim2, "2");
    test_tensor(o[4], next3, nextDim3, "3");
}

TEST_F(ControllerBaseTest, testShiftAndInit) {
    vector<Tensor> o;
    vector<float> init1 = {1, 0.5};
    vector<int> dim = {3, 2, 1};
    vector<float> init2 = {1, 0.5, 2.3, 4.5};

    Tensor i1(DT_FLOAT, TensorShape({1, 2, 1}));
    Tensor i2(DT_FLOAT, TensorShape({2, 2, 1}));

    copy_n(init1.begin(), init1.size(), i1.flat<float>().data());
    copy_n(init2.begin(), init2.size(), i2.flat<float>().data());

    vector<float> exp1 = {2.3, 4.5, 2.1, -0.4, 1., 0.5};
    vector<float> exp2 = {2.1, -0.4, 1., 0.5, 2.3, 4.5};

    auto in = Identity(root, a);
    auto n0 = cont.mShift(root, in, i1, 1);
    auto n1 = cont.mShift(root, in, i2, 2);

    Status log = utile::logGraph(root);

    ClientSession sess(root);
    TF_CHECK_OK(sess.Run({in,
                         n0, n1
                     }, &o));

    test_tensor(o[1], exp1, dim, "0");
    test_tensor(o[2], exp2, dim, "1");
}

TEST_F(ControllerBaseTest, testAll) {

}
