#include "gtest/gtest.h"
#include "model_base.hpp"
#include "utile.hpp"

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/summary/summary_file_writer.h"


using namespace std;
using namespace tensorflow;
using namespace tensorflow::ops;


class ModelBaseTest : public ::testing::Test {
protected:
    ModelBaseTest () : root1(tensorflow::Scope::NewRootScope()),
                       root2(tensorflow::Scope::NewRootScope()),
                       root3(tensorflow::Scope::NewRootScope()),
                       sess1(root1),
                       sess2(root2),
                       sess3(root3),
                       k1(1), k2(1), k3(5),
                       s_dim1(2), s_dim2(4), s_dim3(6),
                       a_dim1(1), a_dim2(2), a_dim3(3),
                       m1(1.), m2(2), m3(1.5),
                       dt1(0.01), dt2(0.01), dt3(0.01)

    {
        s1 = Tensor(DT_FLOAT, TensorShape({k1, s_dim1, 1}));
        a1 = Tensor(DT_FLOAT, TensorShape({k1, a_dim1, 1}));

        vector<float> state1 = {0., 0.};
        vector<float> action1 = {1.};

        copy_n(state1.begin(), state1.size(), s1.flat<float>().data());
        copy_n(action1.begin(), action1.size(), a1.flat<float>().data());


        s2 = Tensor(DT_FLOAT, TensorShape({k2, s_dim2, 1}));
        a2 = Tensor(DT_FLOAT, TensorShape({k2, a_dim2, 1}));

        vector<float> state2 = {0., 0., 0., 0.};
        vector<float> action2 = {1., 1.};

        copy_n(state2.begin(), state2.size(), s2.flat<float>().data());
        copy_n(action2.begin(), action2.size(), a2.flat<float>().data());

        s_init = Tensor(DT_FLOAT, TensorShape({1, s_dim3, 1}));
        s3 = Tensor(DT_FLOAT, TensorShape({k3, s_dim3, 1}));
        a3 = Tensor(DT_FLOAT, TensorShape({k3, a_dim3, 1}));

        vector<float> state3 = {0., 0., 0., 0., 0., 0.,
                                2., 1., 5., 0., -1., -2.,
                                0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                                1., 0., 1., 0., 1., 0.,
                                -1, 0.5, -3, 2., 0., 0.};

        vector<float> state_init = {-1, 0.5, -3, 2., 0., 0.};

        vector<float> action3 = {1., 1., 1.,
                                 2., 0., -1.,
                                 0., 0., 0.,
                                 0.5, -0.5, 0.5,
                                 3., 3., 3.};

        copy_n(state_init.begin(), state_init.size(), s_init.flat<float>().data());
        copy_n(state3.begin(), state3.size(), s3.flat<float>().data());
        copy_n(action3.begin(), action3.size(), a3.flat<float>().data());

        model1 = ModelBase(m1, dt1, s_dim1, a_dim1);
        model2 = ModelBase(m2, dt2, s_dim2, a_dim2);
        model3 = ModelBase(m3, dt3, s_dim3, a_dim3);
    }

    ModelBase model1, model2, model3;
    Scope root1, root2, root3;

    ClientSession sess1, sess2, sess3;

    Tensor s1, a1;
    Tensor s2, a2;
    Tensor s_init, s3, a3;

    int k1, k2, k3;
    int a_dim1, a_dim2, a_dim3;
    int s_dim1, s_dim2, s_dim3;
    float m1, m2, m3;
    float dt1, dt2, dt3;

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

TEST_F(ModelBaseTest, StepTesting1) {
    // Test 1

    float acc = (dt1*dt1)/(2.f*m1);
    float vel = (dt1)/(m1);
    vector<float> exp_u = {1.f*acc, 1.f*vel};

    vector<float> exp_s = {0.f, 0.f};
    vector<float> exp_res = {exp_u[0]+exp_s[0], exp_u[1]+exp_s[1]};

    vector<Tensor> o;
    vector<int> dims = {k1, s_dim1, 1};

    auto s_in = Identity(root1.WithOpName("State_input"), s1);
    auto a_in = Identity(root1.WithOpName("Action_input"), a1);

    auto out_step = model1.mBuildFreeStepGraph(root1, s_in);
    auto out_act = model1.mBuildActionStepGraph(root1, a_in);
    auto out = model1.mBuildModelStepGraph(root1, s_in, a_in);


    TF_CHECK_OK(sess1.Run({out_step, out_act, out}, &o));

    test_tensor(o[0], exp_s, dims, "acceleration");
    test_tensor(o[1], exp_u, dims, "velocity");
    test_tensor(o[2], exp_res, dims, "result");
}

TEST_F(ModelBaseTest, StepTesting2) {
    // Test 2

    float acc = (dt2*dt2)/(2.f*m2);
    float vel = (dt2)/(m2);
    vector<float> exp_s = {0, 0, 0, 0};
    vector<float> exp_u = {1.f*acc, 1.f*vel, 1.f*acc, 1.f*vel};
    vector<float> exp_res = {exp_u[0]+exp_s[0], exp_u[1]+exp_s[1],
                             exp_u[2]+exp_s[2], exp_u[3]+exp_s[3]};


    vector<Tensor> o;
    vector<int> dims = {k2, s_dim2, 1};

    auto s_in = Identity(root2.WithOpName("State_input"), s2);
    auto a_in = Identity(root2.WithOpName("Action_input"), a2);

    auto out_step = model2.mBuildFreeStepGraph(root2, s_in);
    auto out_act = model2.mBuildActionStepGraph(root2, a_in);
    auto out = model2.mBuildModelStepGraph(root2, s_in, a_in);


    TF_CHECK_OK(sess2.Run({out_step, out_act, out}, &o));

    test_tensor(o[0], exp_s, dims, "acceleration");
    test_tensor(o[1], exp_u, dims, "velocity");
    test_tensor(o[2], exp_res, dims, "result");

}

TEST_F(ModelBaseTest, LargeTesting) {
    //Test 3
    vector<Tensor> o;
    vector<int>  dims = {k3, s_dim3, 1};
    float acc = (dt3*dt3)/(2.f*m3);
    float vel = (dt3)/(m3);

    vector<float> exp_u = {acc, vel, acc, vel, acc, vel,
                            2.f*acc, 2.f*vel, 0.f*acc, 0*vel, -1.f*acc, -1.f*vel,
                            0.f*acc, 0.f*vel, 0.f*acc, 0*vel, 0.f*acc, 0.f*vel,
                            0.5f*acc, 0.5f*vel, -0.5f*acc, -0.5f*vel, 0.5f*acc, 0.5f*vel,
                            3.f*acc, 3.f*vel, 3.f*acc, 3.f*vel, 3.f*acc, 3.f*vel};

    vector<float> exp_s = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
                            2.f+dt3, 1.f, 5.f, 0.f, -1.f-2.f*dt3, -2.f,
                            0.5f+dt3/2.f, 0.5f, 0.5f+dt3/2.f, 0.5f, 0.5f+dt3/2.f, 0.5f,
                            1.f, 0.f, 1.f, 0.f, 1.f, 0.f,
                            -1.f+dt3/2.f, 0.5f, -3.f+2.f*dt3, 2.f, 0.f, 0.f};
    vector<float> exp_res = {};
    for (int i = 0; i < k3*s_dim3; i++) {
        exp_res.push_back(exp_u[i]+exp_s[i]);
    }

    auto s_in = Identity(root3.WithOpName("State_input"), s3);
    auto a_in = Identity(root3.WithOpName("Action_input"), a3);

    auto out_step = model3.mBuildFreeStepGraph(root3, s_in);
    auto out_act = model3.mBuildActionStepGraph(root3, a_in);
    auto out = model3.mBuildModelStepGraph(root3, s_in, a_in);


    TF_CHECK_OK(sess3.Run({out_step, out_act, out}, &o));

    test_tensor(o[0], exp_s, dims, "acceleration");
    test_tensor(o[1], exp_u, dims, "velocity");
    test_tensor(o[2], exp_res, dims, "result");
}

TEST_F(ModelBaseTest, InitTest) {
    //Test 3
    vector<Tensor> o;
    vector<int> dims = {k3, s_dim3, 1};
    vector<int> dim_state = {1, s_dim3, 1};
    float acc = (dt3*dt3)/(2.f*m3);
    float vel = (dt3)/(m3);

    vector<float> exp_u = {acc, vel, acc, vel, acc, vel,
                            2.f*acc, 2.f*vel, 0.f*acc, 0*vel, -1.f*acc, -1.f*vel,
                            0.f*acc, 0.f*vel, 0.f*acc, 0*vel, 0.f*acc, 0.f*vel,
                            0.5f*acc, 0.5f*vel, -0.5f*acc, -0.5f*vel, 0.5f*acc, 0.5f*vel,
                            3.f*acc, 3.f*vel, 3.f*acc, 3.f*vel, 3.f*acc, 3.f*vel};

    vector<float> exp_s = {-1.f+dt3/2.f, 0.5f, -3.f+2.f*dt3, 2.f, 0.f, 0.f,
                            -1.f+dt3/2.f, 0.5f, -3.f+2.f*dt3, 2.f, 0.f, 0.f,
                            -1.f+dt3/2.f, 0.5f, -3.f+2.f*dt3, 2.f, 0.f, 0.f,
                            -1.f+dt3/2.f, 0.5f, -3.f+2.f*dt3, 2.f, 0.f, 0.f,
                            -1.f+dt3/2.f, 0.5f, -3.f+2.f*dt3, 2.f, 0.f, 0.f};

    vector<float> exp_res = {};
    for (int i = 0; i < k3*s_dim3; i++) {
        exp_res.push_back(exp_u[i]+exp_s[i]);
    }


    auto s_in = Identity(root3.WithOpName("State_input"), s_init);
    auto a_in = Identity(root3.WithOpName("Action_input"), a3);

    auto out_step = model3.mBuildFreeStepGraph(root3, s_in);
    auto out_act = model3.mBuildActionStepGraph(root3, a_in);
    auto out = model3.mBuildModelStepGraph(root3, s_in, a_in);


    TF_CHECK_OK(sess3.Run({out_step, out_act, out}, &o));

    test_tensor(o[0], exp_s, dim_state, "acceleration");
    test_tensor(o[1], exp_u, dims, "velocity");
    test_tensor(o[2], exp_res, dims, "result");
}
