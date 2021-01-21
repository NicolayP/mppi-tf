#include "controller_base.hpp"
#include "cost_base.hpp"
#include <iostream>

#include <chrono>  // for high_resolution_clock

using namespace std;
using namespace tensorflow;

int main(int argc, char const *argv[]) {
    int k(1500), tau(100), sDim(4), aDim(2);
    bool debug(true);
    float dt(0.01);
    ControllerBase ctrl(k, tau, dt, 1., sDim, aDim);
    if (debug) {
        Status log = ctrl.mLogGraph();
        if (log != Status::OK()) {
            cout << "Writing failed: " << log << endl;
        } else {
            cout << "Writing succesful" << endl;
        }
    }
    vector<float> init = {0, 0};

    // Record start time
    auto start = chrono::high_resolution_clock::now();
    for (int i=0; i <100; i++){
        ctrl.next(init);
    }
    // Record end time
    auto finish = chrono::high_resolution_clock::now();

    chrono::duration<double> elapsed = finish - start;
    cout << elapsed.count()/100.f << endl;



    /* Section to test the cost computational Graph.

    // Fake tensor to create cost graph.
    vector<float> goal;
    goal.push_back(1.);
    goal.push_back(0.);
    vector<float> sigma;
    sigma.push_back(1.);
    CostBase cost(1., sigma, goal);
    vector<float> state;
    vector<float> action;
    vector<float> noise;

    state.push_back(3.);
    state.push_back(1.);

    action.push_back(-1.);

    noise.push_back(-0.02);

    Tensor x(DT_FLOAT, TensorShape({1, 2, 1}));
    Tensor u(DT_FLOAT, TensorShape({1, 1, 1}));
    Tensor e(DT_FLOAT, TensorShape({1, 1, 1}));
    cost.mBuildStepCostGraph(x, u, e);
    Status writer = cost.logGraph();
    if (writer != Status::OK()) {
        cout << "Writing failed: " << writer << endl;
    }*/
    return 0;
}
