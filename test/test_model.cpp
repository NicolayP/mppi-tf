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

};

TEST_F(ModelBaseTest, StepTesting1) {
    // Test 1

    float acc1 = (dt1*dt1)/(2.f*m1);
    float vel1 = (dt1)/(m1);
    vector<float> exp_u1 = {1.f*acc1, 1.f*vel1};

    vector<float> exp_s1 = {0.f, 0.f};

    vector<Tensor> o1;
    vector<int> dims1 = {k1, s_dim1, 1};

    auto s_in1 = Identity(root1.WithOpName("State_input"), s1);
    auto a_in1 = Identity(root1.WithOpName("Action_input"), a1);

    auto out_step1 = model1.mBuildFreeStepGraph(root1, s_in1);
    auto out_act1 = model1.mBuildActionStepGraph(root1, a_in1);
    auto out1 = model1.mBuildModelStepGraph(root1, s_in1, a_in1);


    TF_CHECK_OK(sess1.Run({out_step1, out_act1, out1}, &o1));

    float* res_state1 = o1[0].flat<float>().data();
    float* res_act1 = o1[1].flat<float>().data();
    float* res1 = o1[2].flat<float>().data();

    TensorShape shape_state1(o1[0].shape()), shape_vel1(o1[1].shape()), shape1(o1[2].shape());

    ASSERT_EQ(o1[0].shape().dims(), 3);
    ASSERT_EQ(o1[1].shape().dims(), 3);
    ASSERT_EQ(o1[2].shape().dims(), 3);

    for (pair<TensorShapeIter<TensorShape>, vector<int>::iterator> it(shape_state1.begin(), dims1.begin());
         it.first != shape_state1.end();
         ++it.first, ++it.second) {
        EXPECT_EQ((*(it.first)).size, (*it.second));
    }

    for (pair<TensorShapeIter<TensorShape>, vector<int>::iterator> it(shape_vel1.begin(), dims1.begin());
         it.first != shape_vel1.end();
         ++it.first, ++it.second) {
        EXPECT_EQ((*(it.first)).size, (*it.second));
    }

    for (pair<TensorShapeIter<TensorShape>, vector<int>::iterator> it(shape1.begin(), dims1.begin());
         it.first != shape1.end();
         ++it.first, ++it.second) {
        EXPECT_EQ((*(it.first)).size, (*it.second));
    }

    for (int i=0; i < k1*s_dim1; i++) {
        EXPECT_FLOAT_EQ(res_state1[i], exp_s1[i]) << "State error at index : " << i;
    }

    for (int i=0; i < k1*s_dim1; i++) {
        EXPECT_FLOAT_EQ(res_act1[i], exp_u1[i]) << "Action error at index : " << i;
    }

    for (int i=0; i < k1*s_dim1; i++) {
        EXPECT_FLOAT_EQ(res1[i], exp_u1[i]+exp_s1[i]) << "Prediction error at index : " << i;
    }

}

TEST_F(ModelBaseTest, StepTesting2) {
    // Test 2

    float acc2 = (dt2*dt2)/(2.f*m2);
    float vel2 = (dt2)/(m2);
    vector<float> exp_s2 = {0, 0, 0, 0};
    vector<float> exp_u2 = {1.f*acc2, 1.f*vel2, 1.f*acc2, 1.f*vel2};


    vector<Tensor> o2;
    vector<int> dims2 = {k2, s_dim2, 1};

    auto s_in2 = Identity(root2.WithOpName("State_input"), s2);
    auto a_in2 = Identity(root2.WithOpName("Action_input"), a2);

    auto out_step2 = model2.mBuildFreeStepGraph(root2, s_in2);
    auto out_act2 = model2.mBuildActionStepGraph(root2, a_in2);
    auto out2 = model2.mBuildModelStepGraph(root2, s_in2, a_in2);


    TF_CHECK_OK(sess2.Run({out_step2, out_act2, out2}, &o2));

    float* res_state2 = o2[0].flat<float>().data();
    float* res_act2 = o2[1].flat<float>().data();
    float* res2 = o2[2].flat<float>().data();

    TensorShape shape_state2(o2[0].shape()), shape_vel2(o2[1].shape()), shape2(o2[2].shape());

    /* ------ Check the number of dimensions ------ */
    ASSERT_EQ(o2[0].shape().dims(), 3);
    ASSERT_EQ(o2[1].shape().dims(), 3);
    ASSERT_EQ(o2[2].shape().dims(), 3);

    /* ------ Check the dimension sizes ------ */
    for (pair<TensorShapeIter<TensorShape>, vector<int>::iterator> it(shape_state2.begin(), dims2.begin());
     it.first != shape_state2.end();
     ++it.first, ++it.second) {
         ASSERT_EQ((*(it.first)).size, (*it.second));
    }

    for (pair<TensorShapeIter<TensorShape>, vector<int>::iterator> it(shape_vel2.begin(), dims2.begin());
     it.first != shape_vel2.end();
     ++it.first, ++it.second) {
         ASSERT_EQ((*(it.first)).size, (*it.second));
    }

    for (pair<TensorShapeIter<TensorShape>, vector<int>::iterator> it(shape2.begin(), dims2.begin());
     it.first != shape2.end();
     ++it.first, ++it.second) {
         ASSERT_EQ((*(it.first)).size, (*it.second));
    }

    /* ------ Check the values ------ */
    for (int i=0; i < k2*s_dim2; i++) {
        EXPECT_FLOAT_EQ(res_state2[i], exp_s2[i]) << "State error at index : " << i;
    }

    for (int i=0; i < k2*s_dim2; i++) {
        EXPECT_FLOAT_EQ(res_act2[i], exp_u2[i]) << "Action error at index : " << i;
    }

    for (int i=0; i < k2*s_dim2; i++) {
        EXPECT_FLOAT_EQ(res2[i], exp_u2[i]+exp_s2[i]) << "Prediction error at index : " << i;
    }
}

TEST_F(ModelBaseTest, LargeTesting) {
    //Test 3
    vector<Tensor> o3;
    vector<int>  dims3 = {k3, s_dim3, 1};
    float acc3 = (dt3*dt3)/(2.f*m3);
    float vel3 = (dt3)/(m3);

    vector<float> exp_u3 = {acc3, vel3, acc3, vel3, acc3, vel3,
                            2.f*acc3, 2.f*vel3, 0.f*acc3, 0*vel3, -1.f*acc3, -1.f*vel3,
                            0.f*acc3, 0.f*vel3, 0.f*acc3, 0*vel3, 0.f*acc3, 0.f*vel3,
                            0.5f*acc3, 0.5f*vel3, -0.5f*acc3, -0.5f*vel3, 0.5f*acc3, 0.5f*vel3,
                            3.f*acc3, 3.f*vel3, 3.f*acc3, 3.f*vel3, 3.f*acc3, 3.f*vel3};

    vector<float> exp_s3 = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
                            2.f+dt3, 1.f, 5.f, 0.f, -1.f-2.f*dt3, -2.f,
                            0.5f+dt3/2.f, 0.5f, 0.5f+dt3/2.f, 0.5f, 0.5f+dt3/2.f, 0.5f,
                            1.f, 0.f, 1.f, 0.f, 1.f, 0.f,
                            -1.f+dt3/2.f, 0.5f, -3.f+2.f*dt3, 2.f, 0.f, 0.f};

    auto s_in3 = Identity(root3.WithOpName("State_input"), s3);
    auto a_in3 = Identity(root3.WithOpName("Action_input"), a3);

    auto out_step3 = model3.mBuildFreeStepGraph(root3, s_in3);
    auto out_act3 = model3.mBuildActionStepGraph(root3, a_in3);
    auto out3 = model3.mBuildModelStepGraph(root3, s_in3, a_in3);


    TF_CHECK_OK(sess3.Run({out_step3, out_act3, out3, s_in3, a_in3}, &o3));

    float* res_state3 = o3[0].flat<float>().data();
    float* res_act3 = o3[1].flat<float>().data();
    float* res3 = o3[2].flat<float>().data();


    TensorShape shape_state3(o3[0].shape()), shape_vel3(o3[1].shape()), shape3(o3[2].shape());

    ASSERT_EQ(o3[0].shape().dims(), 3);
    ASSERT_EQ(o3[1].shape().dims(), 3);
    ASSERT_EQ(o3[2].shape().dims(), 3);

    for (pair<TensorShapeIter<TensorShape>, vector<int>::iterator> it(shape_state3.begin(), dims3.begin());
         it.first != shape_state3.end();
         ++it.first, ++it.second) {
        EXPECT_EQ((*(it.first)).size, (*it.second));
    }

    for (pair<TensorShapeIter<TensorShape>, vector<int>::iterator> it(shape_vel3.begin(), dims3.begin());
         it.first != shape_vel3.end();
         ++it.first, ++it.second) {
        EXPECT_EQ((*(it.first)).size, (*it.second));
    }

    for (pair<TensorShapeIter<TensorShape>, vector<int>::iterator> it(shape3.begin(), dims3.begin());
         it.first != shape3.end();
         ++it.first, ++it.second) {
        EXPECT_EQ((*(it.first)).size, (*it.second));
    }

    for (int i=0; i < k3*s_dim3; i++) {
        EXPECT_FLOAT_EQ(res_state3[i], exp_s3[i]) << "State error at index : " << i;
    }

    for (int i=0; i < k3*s_dim3; i++) {
        EXPECT_FLOAT_EQ(res_act3[i], exp_u3[i]) << "Action error at index : " << i;
    }

    for (int i=0; i < k3*s_dim3; i++) {
        EXPECT_FLOAT_EQ(res3[i], exp_u3[i]+exp_s3[i]) << "Prediction error at index : " << i;
    }
}

TEST_F(ModelBaseTest, InitTest) {
    //Test 3
    vector<Tensor> o;
    vector<int> dims = {k3, s_dim3, 1};
    vector<int> dim_state = {1, s_dim3, 1};
    float acc3 = (dt3*dt3)/(2.f*m3);
    float vel3 = (dt3)/(m3);

    vector<float> exp_u = {acc3, vel3, acc3, vel3, acc3, vel3,
                            2.f*acc3, 2.f*vel3, 0.f*acc3, 0*vel3, -1.f*acc3, -1.f*vel3,
                            0.f*acc3, 0.f*vel3, 0.f*acc3, 0*vel3, 0.f*acc3, 0.f*vel3,
                            0.5f*acc3, 0.5f*vel3, -0.5f*acc3, -0.5f*vel3, 0.5f*acc3, 0.5f*vel3,
                            3.f*acc3, 3.f*vel3, 3.f*acc3, 3.f*vel3, 3.f*acc3, 3.f*vel3};

    vector<float> exp_s = {-1.f+dt3/2.f, 0.5f, -3.f+2.f*dt3, 2.f, 0.f, 0.f,
                            -1.f+dt3/2.f, 0.5f, -3.f+2.f*dt3, 2.f, 0.f, 0.f,
                            -1.f+dt3/2.f, 0.5f, -3.f+2.f*dt3, 2.f, 0.f, 0.f,
                            -1.f+dt3/2.f, 0.5f, -3.f+2.f*dt3, 2.f, 0.f, 0.f,
                            -1.f+dt3/2.f, 0.5f, -3.f+2.f*dt3, 2.f, 0.f, 0.f};

    auto s_in = Identity(root3.WithOpName("State_input"), s_init);
    auto a_in = Identity(root3.WithOpName("Action_input"), a3);

    auto out_step = model3.mBuildFreeStepGraph(root3, s_in);
    auto out_act = model3.mBuildActionStepGraph(root3, a_in);
    auto out = model3.mBuildModelStepGraph(root3, s_in, a_in);


    TF_CHECK_OK(sess3.Run({out_step, out_act, out, s_in, a_in}, &o));

    float* res_state = o[0].flat<float>().data();
    float* res_act = o[1].flat<float>().data();
    float* res = o[2].flat<float>().data();


    TensorShape shape_state(o[0].shape()), shape_vel(o[1].shape()), shape(o[2].shape());

    ASSERT_EQ(o[0].shape().dims(), 3);
    ASSERT_EQ(o[1].shape().dims(), 3);
    ASSERT_EQ(o[2].shape().dims(), 3);

    for (pair<TensorShapeIter<TensorShape>, vector<int>::iterator> it(shape_state.begin(), dim_state.begin());
         it.first != shape_state.end();
         ++it.first, ++it.second) {
        ASSERT_EQ((*(it.first)).size, (*it.second));
    }

    for (pair<TensorShapeIter<TensorShape>, vector<int>::iterator> it(shape_vel.begin(), dims.begin());
         it.first != shape_vel.end();
         ++it.first, ++it.second) {
        ASSERT_EQ((*(it.first)).size, (*it.second));
    }

    for (pair<TensorShapeIter<TensorShape>, vector<int>::iterator> it(shape.begin(), dims.begin());
         it.first != shape.end();
         ++it.first, ++it.second) {
        ASSERT_EQ((*(it.first)).size, (*it.second));
    }

    for (int i=0; i < 1*s_dim3; i++) {
        EXPECT_FLOAT_EQ(res_state[i], exp_s[i]) << "State error at index : " << i;
    }

    for (int i=0; i < k3*s_dim3; i++) {
        EXPECT_FLOAT_EQ(res_act[i], exp_u[i]) << "Action error at index : " << i;
    }

    for (int i=0; i < k3*s_dim3; i++) {
        EXPECT_FLOAT_EQ(res[i], exp_u[i]+exp_s[i]) << "Prediction error at index : " << i;
    }
}
