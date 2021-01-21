#include "gtest/gtest.h"
#include "utile.hpp"

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/summary/summary_file_writer.h"


using namespace std;
using namespace tensorflow;
using namespace tensorflow::ops;
using namespace utile;

class UtileTest : public ::testing::Test {
protected:
    UtileTest () : root(tensorflow::Scope::NewRootScope()),
                       sess(root),
                       m(1.5), dt(0.01)
    {
        A = Const(root.WithOpName("A_matrix"),
              {{1.f, dt}, {0.f, 1.f}});
        B = Const(root.WithOpName("B_matrix"),
              {{(dt*dt)/(2.f*m)}, {dt/m}});

    }

    Output A, B;
    Scope root;
    ClientSession sess;

    float m;
    float dt;

};

TEST_F(UtileTest, blockDiagTest1) {

    int n(1);
    vector<float> exp_a = {1.f, dt, 0.f, 1.f};
    vector<float> exp_b = {(dt*dt)/(2*m), dt/m};

    vector<int> dim_a = {2, 2};
    vector<int> dim_b = {2, 1};


    vector<Tensor> o;
    auto out_a = blockDiag(root, A, n);
    auto out_b = blockDiag(root, B, n);

    TF_CHECK_OK(sess.Run({out_a, out_b}, &o));

    float* a = o[0].flat<float>().data();
    float* b = o[1].flat<float>().data();

    TensorShape shape_a(o[0].shape()), shape_b(o[1].shape());

    ASSERT_EQ(shape_a.dims(), 2);
    ASSERT_EQ(shape_b.dims(), 2);

    for (pair<TensorShapeIter<TensorShape>, vector<int>::iterator> it(shape_a.begin(), dim_a.begin());
         it.first != shape_a.end();
         ++it.first, ++it.second) {
        EXPECT_EQ((*(it.first)).size, (*it.second));
    }

    for (pair<TensorShapeIter<TensorShape>, vector<int>::iterator> it(shape_b.begin(), dim_b.begin());
         it.first != shape_b.end();
         ++it.first, ++it.second) {
        EXPECT_EQ((*(it.first)).size, (*it.second));
    }


    for (int i=0; i < n*2*n*2; i++) {
        EXPECT_FLOAT_EQ(a[i], exp_a[i]) << "State error at index : " << i;
    }

    for (int i=0; i < n*2*n; i++) {
        EXPECT_FLOAT_EQ(b[i], exp_b[i]) << "Action error at index : " << i;
    }

}

TEST_F(UtileTest, blockDiagTest2) {

    int n(2);
    vector<float> exp_a = {1.f, dt, 0, 0,
                           0.f, 1.f, 0, 0,
                           0, 0, 1.f, dt,
                           0, 0, 0.f, 1.f};
    vector<float> exp_b = {(dt*dt)/(2*m), 0,
                           dt/m, 0,
                           0, (dt*dt)/(2*m),
                           0, dt/m};

    vector<int> dim_a = {2*n, 2*n};
    vector<int> dim_b = {2*n, n};


    vector<Tensor> o;
    auto out_a = blockDiag(root, A, n);
    auto out_b = blockDiag(root, B, n);

    TF_CHECK_OK(sess.Run({out_a, out_b}, &o));

    float* a = o[0].flat<float>().data();
    float* b = o[1].flat<float>().data();

    TensorShape shape_a(o[0].shape()), shape_b(o[1].shape());

    ASSERT_EQ(shape_a.dims(), 2);
    ASSERT_EQ(shape_b.dims(), 2);

    for (pair<TensorShapeIter<TensorShape>, vector<int>::iterator> it(shape_a.begin(), dim_a.begin());
         it.first != shape_a.end();
         ++it.first, ++it.second) {
        EXPECT_EQ((*(it.first)).size, (*it.second));
    }

    for (pair<TensorShapeIter<TensorShape>, vector<int>::iterator> it(shape_b.begin(), dim_b.begin());
         it.first != shape_b.end();
         ++it.first, ++it.second) {
        EXPECT_EQ((*(it.first)).size, (*it.second));
    }


    for (int i=0; i < n*2*n*2; i++) {
        EXPECT_FLOAT_EQ(a[i], exp_a[i]) << "State error at index : " << i;
    }

    for (int i=0; i < n*2*n; i++) {
        EXPECT_FLOAT_EQ(b[i], exp_b[i]) << "Action error at index : " << i;
    }

}

TEST_F(UtileTest, blockDiagTest3) {

    int n(3);
    vector<float> exp_a = {1.f, dt, 0, 0, 0, 0,
                           0.f, 1.f, 0, 0, 0, 0,
                           0, 0, 1.f, dt, 0, 0,
                           0, 0, 0.f, 1.f, 0, 0,
                           0, 0, 0, 0, 1.f, dt,
                           0, 0, 0, 0, 0.f, 1.f};
    vector<float> exp_b = {(dt*dt)/(2*m), 0, 0,
                           dt/m, 0, 0,
                           0, (dt*dt)/(2*m), 0,
                           0, dt/m, 0,
                           0, 0, (dt*dt)/(2*m),
                           0, 0, dt/m};

    vector<int> dim_a = {2*n, 2*n};
    vector<int> dim_b = {2*n, n};


    vector<Tensor> o;
    auto out_a = blockDiag(root, A, n);
    auto out_b = blockDiag(root, B, n);

    TF_CHECK_OK(sess.Run({out_a, out_b}, &o));

    float* a = o[0].flat<float>().data();
    float* b = o[1].flat<float>().data();

    TensorShape shape_a(o[0].shape()), shape_b(o[1].shape());

    ASSERT_EQ(shape_a.dims(), 2);
    ASSERT_EQ(shape_b.dims(), 2);

    for (pair<TensorShapeIter<TensorShape>, vector<int>::iterator> it(shape_a.begin(), dim_a.begin());
         it.first != shape_a.end();
         ++it.first, ++it.second) {
        EXPECT_EQ((*(it.first)).size, (*it.second));
    }

    for (pair<TensorShapeIter<TensorShape>, vector<int>::iterator> it(shape_b.begin(), dim_b.begin());
         it.first != shape_b.end();
         ++it.first, ++it.second) {
        EXPECT_EQ((*(it.first)).size, (*it.second));
    }


    for (int i=0; i < n*2*n*2; i++) {
        EXPECT_FLOAT_EQ(a[i], exp_a[i]) << "State error at index : " << i;
    }

    for (int i=0; i < n*2*n; i++) {
        EXPECT_FLOAT_EQ(b[i], exp_b[i]) << "Action error at index : " << i;
    }

}

TEST_F(UtileTest, blockDiagTest4) {

    int n(4);
    vector<float> exp_a = {1.f, dt, 0, 0, 0, 0, 0, 0,
                           0.f, 1.f, 0, 0, 0, 0, 0, 0,
                           0, 0, 1.f, dt, 0, 0, 0, 0,
                           0, 0, 0.f, 1.f, 0, 0, 0, 0,
                           0, 0, 0, 0, 1.f, dt, 0, 0,
                           0, 0, 0, 0, 0.f, 1.f, 0, 0,
                           0, 0, 0, 0, 0, 0, 1.f, dt,
                           0, 0, 0, 0, 0, 0, 0.f, 1.f};
    vector<float> exp_b = {(dt*dt)/(2*m), 0, 0, 0,
                           dt/m, 0, 0, 0,
                           0, (dt*dt)/(2*m), 0, 0,
                           0, dt/m, 0, 0,
                           0, 0, (dt*dt)/(2*m), 0,
                           0, 0, dt/m, 0,
                           0, 0, 0, (dt*dt)/(2*m),
                           0, 0, 0, dt/m};

    vector<int> dim_a = {2*n, 2*n};
    vector<int> dim_b = {2*n, n};


    vector<Tensor> o;
    auto out_a = blockDiag(root, A, n);
    auto out_b = blockDiag(root, B, n);

    TF_CHECK_OK(sess.Run({out_a, out_b}, &o));

    float* a = o[0].flat<float>().data();
    float* b = o[1].flat<float>().data();

    TensorShape shape_a(o[0].shape()), shape_b(o[1].shape());

    ASSERT_EQ(shape_a.dims(), 2);
    ASSERT_EQ(shape_b.dims(), 2);

    for (pair<TensorShapeIter<TensorShape>, vector<int>::iterator> it(shape_a.begin(), dim_a.begin());
         it.first != shape_a.end();
         ++it.first, ++it.second) {
        EXPECT_EQ((*(it.first)).size, (*it.second));
    }

    for (pair<TensorShapeIter<TensorShape>, vector<int>::iterator> it(shape_b.begin(), dim_b.begin());
         it.first != shape_b.end();
         ++it.first, ++it.second) {
        EXPECT_EQ((*(it.first)).size, (*it.second));
    }


    for (int i=0; i < n*2*n*2; i++) {
        EXPECT_FLOAT_EQ(a[i], exp_a[i]) << "State error at index : " << i;
    }

    for (int i=0; i < n*2*n; i++) {
        EXPECT_FLOAT_EQ(b[i], exp_b[i]) << "Action error at index : " << i;
    }

}
