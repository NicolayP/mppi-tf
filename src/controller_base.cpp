#include "controller_base.hpp"

#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/summary/summary_file_writer.h"

#include "utile.hpp"

#include <iostream>
#include <string>

using namespace std;
using namespace tensorflow;
using namespace tensorflow::ops;


ControllerBase::ControllerBase () :
                m_root(tensorflow::Scope::NewRootScope()),
                m_k(0), m_tau(0), m_s_dim(0), m_a_dim(0), m_dt(0),
                m_mass(0)
{}

ControllerBase::ControllerBase (const int k,
                                const int tau,
                                const float dt,
                                const float mass,
                                const int s_dim,
                                const int a_dim):
                m_root(tensorflow::Scope::NewRootScope()),
                m_k(k), m_tau(tau), m_s_dim(s_dim), m_a_dim(a_dim), m_dt(dt),
                m_mass(mass),
                m_sigma(DT_FLOAT, TensorShape({a_dim, a_dim})),
                m_goal(DT_FLOAT, TensorShape({s_dim, 1})) {

    vector<float> sig;
    vector<float> goal;
    vector<float> q_mat;
    float lambda(1.);
    m_lambda = lambda;

    for (int i=0; i < s_dim/2; i++) {
        goal.push_back(1.);
        goal.push_back(0.);
    }

    for (int i=0; i < a_dim; i++) {
        for (int j=0; j < a_dim; j++) {
            if (i == j) {
                sig.push_back(1.);
            } else {
                sig.push_back(0.);
            }
        }
    }

    for (int i=0; i < s_dim; i++) {
        q_mat.push_back(1.);
    }

    Tensor Q(DT_FLOAT, TensorShape({s_dim}));

    copy_n(sig.begin(), sig.size(), m_sigma.flat<float>().data());
    copy_n(goal.begin(), goal.size(), m_goal.flat<float>().data());
    copy_n(q_mat.begin(), q_mat.size(), Q.flat<float>().data());

    m_model = ModelBase(1., m_dt, m_s_dim, m_a_dim);
    m_cost = CostBase(lambda, m_sigma, m_goal, Q);
    mBuildGraph();
}


ControllerBase::ControllerBase (Scope root,
                                const int k,
                                const int tau,
                                const float dt,
                                const float mass,
                                const int s_dim,
                                const int a_dim):
                m_root(root),
                m_k(k), m_tau(tau), m_s_dim(s_dim), m_a_dim(a_dim), m_dt(dt),
                m_mass(mass),
                m_sigma(DT_FLOAT, TensorShape({a_dim, a_dim})),
                m_goal(DT_FLOAT, TensorShape({s_dim, 1})) {

    vector<float> sig;
    vector<float> goal;
    vector<float> q_mat;
    float lambda(1.);
    m_lambda = lambda;

    for (int i=0; i < s_dim; i++) {
        goal.push_back(1.);
    }

    for (int i=0; i < a_dim; i++) {
        for (int j=0; j < a_dim; j++) {
            if (i == j) {
                sig.push_back(1.);
            }
            sig.push_back(0.);
        }
    }

    for (int i=0; i < s_dim; i++) {
        q_mat.push_back(1.);
    }

    Tensor Q(DT_FLOAT, TensorShape({s_dim}));

    copy_n(sig.begin(), sig.size(), m_sigma.flat<float>().data());
    copy_n(goal.begin(), goal.size(), m_goal.flat<float>().data());
    copy_n(q_mat.begin(), q_mat.size(), Q.flat<float>().data());

    m_model = ModelBase(1., m_dt, m_s_dim, m_a_dim);
    m_cost = CostBase(m_lambda, m_sigma, m_goal, Q);
    mBuildGraph();
}


ControllerBase::~ControllerBase() {}

bool ControllerBase::setActions(vector<Tensor> actions) {

}

void ControllerBase::next(Tensor x) {
    /*TF_CHECK_OK(sess.Run({{mStateInput, x}, {mActionInput, mU}},
                         {mUpdate, mNext},
                         &out_tensor));*/

}

Output ControllerBase::mBeta(Scope scope, Input cost) {
    return Min(scope.WithOpName("Min"), cost, {0});
}

Output ControllerBase::mExpArg(Scope scope, Input cost, Input beta) {
    return Multiply(scope.WithOpName("Exp_arg"),
                    {{-1.f/m_lambda}},
                    Subtract(scope, cost, beta));
}

Output ControllerBase::mExp(Scope scope, Input arg) {
    return Exp(scope.WithOpName("Exponential"), arg);
}

Output ControllerBase::mNabla(Scope scope, Input exp) {
    return Sum(scope.WithOpName("nabla"), exp, {0});
}

Output ControllerBase::mWeights(Scope scope, Input exp, Input nabla) {
    return RealDiv(scope.WithOpName("weights"), exp, nabla);
}

Output ControllerBase::mWeightedNoise(Scope scope, Input weights, Input noises) {
    return Sum(scope.WithOpName("Sum_WxE"), // Shape [Tau, a_dim]
               Multiply(scope, ExpandDims(scope, weights, {-1}), noises),
               {0});
}

Output ControllerBase::mNoiseGenGraph(Scope scope, Input mean, Input sigma) {
    // Generate Random noise for the rollouts, shape [k, tau, a_dim, 1]
    auto rng = RandomNormal(scope.WithOpName("random_number_generation"),
                            {m_k, m_tau, m_a_dim, 1},
                            DT_FLOAT,
                            RandomNormal::Seed(1.));

    return BatchMatMulV2(scope.WithOpName("Scaling"), m_sigma, rng);

}

Output ControllerBase::mPrepareAction(Scope scope, Input actions, int timestep) {
    auto slice = Slice(scope.WithOpName("Action"), actions, {timestep, 0, 0}, {1, -1, -1});
    return Squeeze(scope, slice, Squeeze::Axis({0}));
}

Output ControllerBase::mPrepareNoise(Scope scope, Input noises, int timestep) {
    auto slice = Slice(scope.WithOpName("Noise"), noises, {0, timestep, 0, 0}, {-1, 1, -1, -1});
    return Squeeze(scope, slice, Squeeze::Axis({1}));
}

Output ControllerBase::mBuildUpdateGraph(Scope scope, Input cost, Input noises) {
    auto beta = mBeta(scope, cost); // Shape: [1, 1]
    auto exp_arg = mExpArg(scope, cost, beta);
    auto exp = mExp(scope, exp_arg); // Shape [K, 1]
    auto nabla = mNabla(scope, exp); // Shape [1, 1]
    auto weights = mWeights(scope, exp, nabla); // Shape [K, 1]
    auto weighted_noise = mWeightedNoise(scope, weights, noises);

    return AddV2(scope, m_a_in, weighted_noise);
}

Output ControllerBase::mBuildModelGraph(Scope model_scope,
                                        Scope cost_scope,
                                        Input init_state,
                                        Input actions,
                                        Input noises) {
    //TODO: REWRITE THIE SECTION.

    // Boot the simulation.

    Scope data_prep_scope(Scope::NewRootScope());
    Scope step_scope(Scope::NewRootScope());
    Scope step_cost_scope(Scope::NewRootScope());
    Scope path_cost_scope(cost_scope.NewSubScope("path_cost"));

    m_cost.setConsts(cost_scope);

    Output action, noise, to_apply, next_state, cost;

    next_state = ExpandDims(model_scope, init_state, {0});
    cost = Identity(cost_scope, Fill(cost_scope, {m_k, 1, 1}, 0.f));


    for (int i = 0; i < m_tau; i++) {
        data_prep_scope = model_scope.NewSubScope("Prepare_Data"+to_string(i));
        step_scope = model_scope.NewSubScope("Step"+to_string(i));
        step_cost_scope = cost_scope.NewSubScope("Step_cost"+to_string(i));

        action = mPrepareAction(data_prep_scope, actions, i);
        noise = mPrepareNoise(data_prep_scope, noises, i);
        to_apply =  AddV2(data_prep_scope, action, noise);

        next_state = m_model.mBuildModelStepGraph(step_scope,
                                                  next_state,
                                                  to_apply);

        auto tmp = m_cost.mBuildStepCostGraph(step_cost_scope,
                                              next_state,
                                              action,
                                              noise);
        cost = AddV2(path_cost_scope, cost, tmp);
    }

    auto final_cost_scope(cost_scope.NewSubScope("Terminal_cost"));
    return AddV2(path_cost_scope, cost, m_cost.mBuildFinalStepCostGraph(final_cost_scope, next_state));
}

Output ControllerBase::mBuildGraph() {
    auto model_scope(m_root.NewSubScope("Model_scope"));
    auto cost_scope(m_root.NewSubScope("Cost_scope"));
    auto weight_scope(m_root.NewSubScope("Weight_scope"));
    auto rand_scop(m_root.NewSubScope("Noise_generation"));

    // Input placeholder for the state and the action sequence.
    m_s_in = Placeholder(m_root.WithOpName("state_input"),
                         DT_FLOAT,
                         Placeholder::Shape({m_s_dim, 1}));

    m_a_in = Placeholder(m_root.WithOpName("action_sequence"),
                         DT_FLOAT,
                         Placeholder::Shape({m_tau, m_a_dim, 1}));
    //auto tmp = ExpandDims(m_root, m_a_in, {-1});
    // TODO: Will be used at some point to see the generated trajectories.
    // auto states = Fill(m_root, {m_k, m_tau, m_s_dim, 1}, 0.f);
    auto noises = mNoiseGenGraph(rand_scop, m_a_in, m_sigma); //shape [k, tau, a_dim, 1]

    // Build the model and cost graph, returns a tensor of shape [K, 1, 1]
    auto cost = mBuildModelGraph(model_scope, cost_scope, m_s_in, m_a_in, noises);
    /* Compute the min of the cost for numerical stability. (I.e at least one
        sample with a none 0 weight. */
    auto update = mBuildUpdateGraph(weight_scope, cost, noises);

    return update;
}

Output ControllerBase::mShift(Scope scope, Input current, Input init) {
    initializer_list<Input> list_init;
    auto remain = Slice(scope.WithOpName("Shift"), current, {1, 0}, {-1, -1});
    list_init = {remain, init};
    InputList newlist(list_init);
    return Concat(scope.WithOpName("Init"), newlist, 0);
}

// Shape [tau, a_dim]
Output ControllerBase::mGetNew(Scope scope, Input current, int nb) {
    return Slice(scope.WithOpName("Next"), current, {0, 0}, {nb, -1});
}

void ControllerBase::test() {
    // Input Tensor.
    Tensor s_tens(DT_FLOAT, TensorShape({m_s_dim, 1}));

    vector<float> s;
    for (int i=0; i < m_s_dim; i++) {
        s.push_back(float(i));
    }
    copy_n(s.begin(), s.size(), s_tens.flat<float>().data());

    Tensor a_tens(DT_FLOAT, TensorShape({m_tau, m_a_dim, 1}));

    vector<float> a;
    for (int i=0; i < m_tau*m_a_dim; i++) {
        a.push_back(float(i));
    }
    copy_n(a.begin(), a.size(), a_tens.flat<float>().data());

    auto in_s = Identity(m_root, m_s_in);
    auto in_a = Identity(m_root, m_a_in);

    ClientSession mSess(m_root);
    vector<Tensor> outTensor;
    TF_CHECK_OK(mSess.Run({{m_s_in, s_tens}, {m_a_in, a_tens}},
                          {in_s, in_a, test_res0},
                          &outTensor));

    cout << "0: " << outTensor[0].DebugString(100) << endl;
    cout << "1: " << outTensor[1].DebugString(100) << endl;
    cout << "2: " << outTensor[2].DebugString(100) << endl;

}

void ControllerBase::run() {
    // Input Tensor.
    Tensor s_tens(DT_FLOAT, TensorShape({m_s_dim, 1}));

    vector<float> s;
    for (int i=0; i < m_s_dim; i++) {
        s.push_back(float(i));
    }
    copy_n(s.begin(), s.size(), s_tens.flat<float>().data());

    Tensor a_tens(DT_FLOAT, TensorShape({m_tau, m_a_dim, 1}));

    vector<float> a;
    for (int i=0; i < m_tau*m_a_dim; i++) {
        a.push_back(float(i));
    }
    copy_n(a.begin(), a.size(), a_tens.flat<float>().data());

    ClientSession mSess(m_root);
    vector<Tensor> outTensor;
    TF_CHECK_OK(mSess.Run({{m_s_in, s_tens}, {m_a_in, a_tens}},
                          {test_res0
                           /*test_res1,
                           test_res2,
                           test_res3,
                           test_res4,
                           test_res5,
                           test_res6
                           test_res7,
                           test_res8,
                           test_res9,
                           test_res10*/
                       },
                          &outTensor));
    cout << "0: " << outTensor[0].DebugString(100) << endl;
    /*cout << "1: " << outTensor[1].DebugString(100) << endl;
    cout << "2: " << outTensor[2].DebugString(100) << endl;
    cout << "3 " << outTensor[3].DebugString(100) << endl;
    cout << "4: " << outTensor[4].DebugString(100) << endl;
    cout << "5: " << outTensor[5].DebugString(100) << endl;
    cout << "6: " << outTensor[6].DebugString(100) << endl;
    cout << "7: " << outTensor[7].DebugString(100) << endl;
    cout << "8: " << outTensor[8].DebugString(100) << endl;
    cout << "9: " << outTensor[9].DebugString(100) << endl;
    cout << "10: " << outTensor[10].DebugString(100) << endl;*/


}

Status ControllerBase::mLogGraph() {
    return utile::logGraph(m_root);
}
