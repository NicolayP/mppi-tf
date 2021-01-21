#include "cost_base.hpp"

#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/summary/summary_file_writer.h"


using namespace std;
using namespace tensorflow;
using namespace tensorflow::ops;

CostBase::CostBase() {};



CostBase::CostBase(const float lambda,
                   const Tensor sigma,
                   const Tensor goal):
         m_goal(goal)
{
    in_lambda = lambda;
    in_sigma = sigma;
}

CostBase::CostBase(const float lambda,
                   const Tensor sigma,
                   const Tensor goal,
                   const Tensor Q):
         m_goal(goal)
{
    in_lambda = lambda;
    in_sigma = sigma;
    in_Q = Q;
}

CostBase::~CostBase() {}

void CostBase::setConsts(Scope scope) {
    m_lambda = Const(scope.WithOpName("lambda"), in_lambda);
    m_inv_sigma = MatrixInverse(scope.WithOpName("Inverse"), in_sigma);
    m_Q = Diag(scope, in_Q);
}

Output CostBase::mBuildStepCostGraph(Scope scope,
                                     Input state,
                                     Input action,
                                     Input noise) {
    auto state_cost = mStateCost(scope, state);
    auto action_cost = mActionCost(scope, action, noise);
    return AddV2(scope.WithOpName("Step_cost_result"), state_cost, action_cost);
}

Output CostBase::mBuildFinalStepCostGraph(Scope scope, Input state) {
    return mStateCost(scope, state);
}

Output CostBase::mStateCost(Scope scope, Input state) {
    auto diff = Subtract(scope.WithOpName("diff"), state, m_goal);
    return BatchMatMulV2(scope.WithOpName("right"),
                         diff, BatchMatMulV2(scope.WithOpName("left"), m_Q, diff),
                         BatchMatMulV2::AdjX(true));
}

Output CostBase::mActionCost(Scope scope, Input action, Input noise) {
    auto noise_cost = BatchMatMulV2(scope.WithOpName("noise"), m_inv_sigma, noise);
    auto action_cost = BatchMatMulV2(scope.WithOpName("actions"), action, noise_cost,
                                     BatchMatMulV2::AdjX(true));
    return Multiply(scope.WithOpName("lambda"), m_lambda, action_cost);
}
