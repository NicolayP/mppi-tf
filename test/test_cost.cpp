#include "gtest/gtest.h"
#include "cost_base.hpp"
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/summary/summary_file_writer.h"

#include <stdlib.h>

// Test With k [1-20]
// Test With a_dim [1-10]
// Test With s_dim [1-10]

using namespace std;
using namespace tensorflow;
using namespace tensorflow::ops;

class CostBaseTest : public ::testing::Test {
 protected:
     CostBaseTest () : root1(tensorflow::Scope::NewRootScope()),
                       root2(tensorflow::Scope::NewRootScope()),
                       root3(tensorflow::Scope::NewRootScope()),
                       sess1(root1), sess2(root2), sess3(root3),
                       k1(1), s_dim1(2), a_dim1(2), lam1(1),
                       k2(1), s_dim2(4), a_dim2(2), lam2(1),
                       k3(5), s_dim3(4), a_dim3(3), lam3(1)
    {
        /* ----- 1st scenario ------ */
        // k = 1, s_dim = 2, a_dim = 2,
        //
        s1 = Tensor(DT_FLOAT, TensorShape({k1, s_dim1, 1}));
        g1 = Tensor(DT_FLOAT, TensorShape({s_dim1, 1}));
        a1 = Tensor(DT_FLOAT, TensorShape({a_dim1, 1}));
        e1 = Tensor(DT_FLOAT, TensorShape({k1, a_dim1, 1}));
        S1 = Tensor(DT_FLOAT, TensorShape({a_dim1, a_dim1}));
        Q1 = Tensor(DT_FLOAT, TensorShape({s_dim1}));
        L1 = Const(root1, lam1);

        vector<float> state1 = {0., 1.};
        vector<float> goal1 = {1., 1.};
        vector<float> action1 = {1., 1.};
        vector<float> epsilon1 = {1., 1.};
        vector<float> sig1 = {1., 0., 0., 1.};
        vector<float> q1 = {1., 1.};

        copy_n(state1.begin(), state1.size(),s1.flat<float>().data());
        copy_n(goal1.begin(), goal1.size(), g1.flat<float>().data());
        copy_n(action1.begin(), action1.size(), a1.flat<float>().data());
        copy_n(epsilon1.begin(), epsilon1.size(), e1.flat<float>().data());
        copy_n(sig1.begin(), sig1.size(), S1.flat<float>().data());
        copy_n(q1.begin(), q1.size(), Q1.flat<float>().data());

        IS1 = MatrixInverse(root1, S1);

        c1 = CostBase(1., S1, g1, Q1);
        c1.setConsts(root1);

        /* ----- 2nd scenario ------ */
        // k = 1, s_dim = 4, a_dim = 2,
        //
        s2 = Tensor(DT_FLOAT, TensorShape({k2, s_dim2, 1}));
        g2 = Tensor(DT_FLOAT, TensorShape({s_dim2, 1}));
        a2 = Tensor(DT_FLOAT, TensorShape({a_dim2, 1}));
        e2 = Tensor(DT_FLOAT, TensorShape({k2, a_dim2, 1}));
        S2 = Tensor(DT_FLOAT, TensorShape({a_dim2, a_dim2}));
        Q2 = Tensor(DT_FLOAT, TensorShape({s_dim2}));
        L2 = Const(root2, lam2);

        vector<float> state2 = {0., 0.5, 2., 0.};
        vector<float> goal2 = {1., 1., 1., 2.};
        vector<float> action2 = {0.5, 2.};
        vector<float> epsilon2 = {0.5, 1.};
        vector<float> sig2 = {1., 0., 0., 1.};
        vector<float> q2 = {1., 1., 10., 10.};

        copy_n(state2.begin(), state2.size(), s2.flat<float>().data());
        copy_n(goal2.begin(), goal2.size(), g2.flat<float>().data());
        copy_n(action2.begin(), action2.size(), a2.flat<float>().data());
        copy_n(epsilon2.begin(), epsilon2.size(), e2.flat<float>().data());
        copy_n(sig2.begin(), sig2.size(), S2.flat<float>().data());
        copy_n(q2.begin(), q2.size(), Q2.flat<float>().data());

        IS2 = MatrixInverse(root2, S2);

        c2 = CostBase(1., S2, g2, Q2);
        c2.setConsts(root2);

        /* ----- 3rd scenario ------ */
        // k = 5, a_dim = 3, s_dim = 4,
        //
        s3 = Tensor(DT_FLOAT, TensorShape({k3, s_dim3, 1}));
        g3 = Tensor(DT_FLOAT, TensorShape({s_dim3, 1}));
        a3 = Tensor(DT_FLOAT, TensorShape({a_dim3, 1}));
        e3 = Tensor(DT_FLOAT, TensorShape({k3, a_dim3, 1}));
        S3 = Tensor(DT_FLOAT, TensorShape({a_dim3, a_dim3}));
        Q3 = Tensor(DT_FLOAT, TensorShape({s_dim3}));
        L3 = Const(root3, lam3);

        vector<float> state3 = {0., 0.5, 2., 0.,
                                0., 2., 0., 0.,
                                10., 2., 2., 3,
                                1., 1., 1., 2.,
                                3., 4., 5., 6.};
        vector<float> goal3 = {1., 1., 1., 2.};
        vector<float> action3 = {0.5, 2., 0.25};
        vector<float> epsilon3 = {0.5, 1., 2.,
                                  0.5, 2., 0.25,
                                  -2, -0.2, -1,
                                  0, 0, 0,
                                  1., 0.5, 3.};
        vector<float> sig3 = {1., 0., 0.,
                              0., 1., 0.,
                              0., 0., 1.};
        vector<float> q3 = {1., 1., 10., 10.};

        copy_n(state3.begin(), state3.size(), s3.flat<float>().data());
        copy_n(goal3.begin(), goal3.size(), g3.flat<float>().data());
        copy_n(action3.begin(), action3.size(), a3.flat<float>().data());
        copy_n(epsilon3.begin(), epsilon3.size(), e3.flat<float>().data());
        copy_n(sig3.begin(), sig3.size(), S3.flat<float>().data());
        copy_n(q3.begin(), q3.size(), Q3.flat<float>().data());

        IS3 = MatrixInverse(root3, S3);

        c3 = CostBase(1., S3, g3, Q3);
        c3.setConsts(root3);
     }

     CostBase c1;
     CostBase c2;
     CostBase c3;
     Scope root1;
     Scope root2;
     Scope root3;
     ClientSession sess1;
     ClientSession sess2;
     ClientSession sess3;
     int k1, s_dim1, a_dim1;
     int k2, s_dim2, a_dim2;
     int k3, s_dim3, a_dim3;
     float lam1, lam2, lam3;
     Tensor s1, a1, e1, S1, g1, Q1;
     Tensor s2, a2, e2, S2, g2, Q2;
     Tensor s3, a3, e3, S3, g3, Q3;
     Output L1, L2, L3, IS1, IS2, IS3;

     void test_tensor (Tensor computed, vector<float>& expected, vector<int>& dims) {
         float* data = computed.flat<float>().data();

         TensorShape shape(computed.shape());

         ASSERT_EQ(shape.dims(), dims.size());

         for (pair<TensorShapeIter<TensorShape>, vector<int>::iterator>
                  it(shape.begin(), dims.begin());
              it.first != shape.end();
              ++it.first, ++it.second) {
             ASSERT_EQ((*(it.first)).size, (*it.second));
         }

         int el = computed.NumElements();

         for (int i=0; i < el; i++) {
             EXPECT_FLOAT_EQ(data[i], expected[i]) << "State error at index : " << i;
         }
     }
};

TEST_F (CostBaseTest, StateCost) {

    vector<Tensor> o1;
    vector<int> dim1 = {k1, 1, 1};
    vector<float> exp1 = {1.};

    auto in1 = Identity(root1, s1);
    auto out1 = c1.mBuildFinalStepCostGraph(root1, in1);
    TF_CHECK_OK(sess1.Run({out1}, &o1));
    test_tensor(o1[0], exp1, dim1);


    vector<Tensor> o2;
    vector<int> dim2 = {k2, 1, 1};
    vector<float> exp2 = {51.25};

    auto in2 = Identity(root2, s2);
    auto out2 = c2.mBuildFinalStepCostGraph(root2, in2);
    TF_CHECK_OK(sess2.Run({out2}, &o2));

    test_tensor(o2[0], exp2, dim2);


    vector<Tensor> o3;
    vector<int> dim3 = {k3, 1, 1};
    vector<float> exp3 = {51.25, 52, 102, 0., 333};

    auto in3 = Identity(root3, s3);
    auto out3 = c3.mBuildFinalStepCostGraph(root3, in3);
    TF_CHECK_OK(sess3.Run({out3}, &o3));

    test_tensor(o3[0], exp3, dim3);


}

TEST_F (CostBaseTest, StepCost) {

    vector<Tensor> o;
    vector<float> exp = {3.};
    vector<int> dim = {k1, 1, 1};
    auto s_in = Identity(root1.WithOpName("State1"), s1);
    auto a_in = Identity(root1.WithOpName("Action1"), a1);
    auto e_in = Identity(root1.WithOpName("noise1"), e1);
    auto out = c1.mBuildStepCostGraph(root1, s_in, a_in, e_in);
    TF_CHECK_OK(sess1.Run({out}, &o));

    test_tensor(o[0], exp, dim);


    exp = {53.5};
    dim = {k2, 1, 1};
    s_in = Identity(root2.WithOpName("State2"), s2);
    a_in = Identity(root2.WithOpName("Action2"), a2);
    e_in = Identity(root2.WithOpName("noise2"), e2);
    out = c2.mBuildStepCostGraph(root2, s_in, a_in, e_in);
    TF_CHECK_OK(sess2.Run({out, s_in, a_in, e_in}, &o));
    test_tensor(o[0], exp, dim);


    exp = {51.25 + 2.75, 52 + 4.3125, 102 - 1.65, 0. + 0, 333 + 2.25};
    dim = {k3, 1, 1};
    s_in = Identity(root3.WithOpName("State3"), s3);
    a_in = Identity(root3.WithOpName("Action3"), a3);
    e_in = Identity(root3.WithOpName("noise3"), e3);
    out = c3.mBuildStepCostGraph(root3, s_in, a_in, e_in);
    TF_CHECK_OK(sess3.Run({out}, &o));

    test_tensor(o[0], exp, dim);

}
