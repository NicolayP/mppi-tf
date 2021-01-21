#include "model_base.hpp"

#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/summary/summary_file_writer.h"

#include "utile.hpp"

using namespace std;
using namespace tensorflow;
using namespace tensorflow::ops;
using namespace utile;

ModelBase::ModelBase() : m_m(1.), m_dt(0.01), m_s_dim(2), m_a_dim(1) {

}

ModelBase::ModelBase(const float mass,
                     const float dt,
                     const int s_dim,
                     const int a_dim) :
                     m_m(mass), m_dt(dt), m_s_dim(s_dim), m_a_dim(a_dim) {

}

ModelBase::~ModelBase() {}

Output ModelBase::mBuildModelStepGraph(Scope scope, Input state, Input action) {
    auto free_state = mBuildFreeStepGraph(scope, state);
    auto action_step = mBuildActionStepGraph(scope, action);
    return AddV2(scope.WithOpName("add"), free_state, action_step);
}

Output ModelBase::mBuildFreeStepGraph(Scope scope, Input state) {
    auto free_scope(scope.NewSubScope("Free_step"));
    auto A = blockDiag(free_scope,
                       Const(free_scope.WithOpName("A_matrix"),
                             {{1.f, m_dt}, {0.f, 1.f}}),
                       m_s_dim/2);
    return BatchMatMulV2(free_scope.WithOpName("A_times_X"), A, state);
}

Output ModelBase::mBuildActionStepGraph(Scope scope, Input action) {
    auto action_scope(scope.NewSubScope("Action_step"));
    auto B = blockDiag(action_scope,
                       Const(action_scope.WithOpName("B_matrix"),
                             {{(m_dt*m_dt)/(2.f*m_m)}, {m_dt/m_m}}),
                       m_a_dim);
    return BatchMatMulV2(action_scope.WithOpName("B_times_X"), B, action);
}


void ModelBase::train(){}
