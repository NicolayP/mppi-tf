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

void ModelBase::mBuildTrainGraph(Scope scope) {
    m_shapes["mass"] = TensorShape({1, 1});
    m_vars["mass"] = Variable(scope.WithOpName("mass"), m_shapes["mass"], DT_FLOAT);
    m_assigns["mass"] = Assign(scope.WithOpName("mass_assign"), m_vars["mass"],
                               Input::Initializer(0.1f, m_shapes["mass"]));
}

void ModelBase::mBuildLossGraph(Scope scope) {
    gt = Placeholder(scope.WithOpName("Expected"), DT_FLOAT);
    in_state = Placeholder(scope.WithOpName("State_input"), DT_FLOAT);
    in_act = Placeholder(scope.WithOpName("Action_input"), DT_FLOAT);
    auto pred = mBuildModelStepGraph(scope.NewSubScope("model"), in_state, in_act);

    Scope loss(scope.NewSubScope("Loss"));
    loss_var = Mean(loss.WithOpName("Loss"),
                           SquaredDifference(loss, pred, gt), {0});
    TF_CHECK_OK(loss.status());
    vector<Output> vars;
    for (pair<string, Output> i : m_vars) {
        vars.push_back(i.second);
    }

    vector<Output> gradients;
    //TF_CHECK_OK(AddSymbolicGradients(scope, {loss_var}, vars, &gradients));
}

Output ModelBase::mBuildModelStepGraph(Scope scope, Input state, Input action) {
    auto free_state = mBuildFreeStepGraph(scope, state);
    auto action_step = mBuildActionStepGraph(scope, action);
    return Add(scope.WithOpName("add"), free_state, action_step);
}

Output ModelBase::mBuildFreeStepGraph(Scope scope, Input state) {
    auto free_scope(scope.NewSubScope("Free_step"));
    auto A = blockDiag(free_scope,
                       Const(free_scope.WithOpName("A_matrix"),
                             {{1.f, m_dt}, {0.f, 1.f}}),
                       m_s_dim/2);
    return BatchMatMul(free_scope.WithOpName("A_times_X"),
                       A,
                       state);
}

Output ModelBase::mBuildActionStepGraph(Scope scope, Input action) {
    auto action_scope(scope.NewSubScope("Action_step"));
    auto dt = Const(action_scope, {{(m_dt*m_dt)/2.f}, {m_dt}});

    auto B = blockDiag(action_scope,
                       RealDiv(action_scope.WithOpName("B_matrix"),
                               dt,
                               m_vars["mass"]),
                       m_a_dim);
    return BatchMatMulV2(action_scope.WithOpName("B_times_X"),
                       B,
                       action);
}


void ModelBase::train(){}
