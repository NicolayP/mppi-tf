#include "controller_base.hpp"
#include "cost_base.hpp"
#include "mj_env.hpp"
#include <iostream>

// Used for config parssing and argument parsing
#include <tclap/CmdLine.h>
#include <yaml-cpp/yaml.h>


#include <chrono>  // for high_resolution_clock

using namespace std;
using namespace tensorflow;

int main(int argc, char const *argv[]) {
    /* parse argument:
    *  -config_file. (Containing all the hyperparameters)
    *  --eval_out (eval mode, need a output dir as well)
    *  -env_file. (model and environment file)
    *  ** For later **
    *  -tpye (string with the type of controller)
    */
    // Store Hpyer parameters in a config file.
    // load the hyper parameters.
    const char* envFile("../envs/point_mass1d.xml");
    const char* mjkey("../lib/contrib/mjkey.txt");
    PointMassEnv env = PointMassEnv(envFile, mjkey, true);

    int k(1500), tau(100), sDim(2), aDim(1);
    bool debug(false), done(false);
    float dt(0.1);
    ControllerBase ctrl(k, tau, dt, 1., sDim, aDim);
    vector<float> state = {0, 0};
    vector<float> action = {0};

    env.get_x(state);
    while (!done) {
        action = ctrl.next(state);
        done = env.simulate(action);
        env.get_x(state);
    }

    if (debug) {
        Status log = ctrl.mLogGraph();
        if (log != Status::OK()) {
            cout << "Writing failed: " << log << endl;
        } else {
            cout << "Writing succesful" << endl;
        }
    }
    // Record start time
    //auto start = chrono::high_resolution_clock::now();
    //for (int i=0; i <100; i++){
    //    ctrl.next(init);
    //}
    // Record end time
    //auto finish = chrono::high_resolution_clock::now();

    //chrono::duration<double> elapsed = finish - start;
    //cout << "Execution time: " << elapsed.count()/100.f << endl;

    return 0;
}
