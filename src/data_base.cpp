#include "data_base.hpp"

#include <iostream>
#include <fstream>


using namespace std;
using namespace tensorflow;

DataBase::DataBase() {}

DataBase::~DataBase() {}

void DataBase::addX(Tensor x) {
    state_input.push_back(x);
}

void DataBase::addU(Tensor u) {
    action_input.push_back(u);
}

void DataBase::addNext(Tensor x_next) {
    output.push_back(x_next);
}

void DataBase::addEl(Tensor x, Tensor u, Tensor x_next) {
    state_input.push_back(x);
    action_input.push_back(u);
    output.push_back(x_next);
    return;
}

string DataBase::tensor2CSV(Tensor t) {
    float* arr = t.flat<float>().data();
    vector<float> vec = {arr, arr + t.NumElements()};
    string format="";
    for (vector<float>::iterator it(vec.begin()); it != vec.end(); it++) {
        format += to_string(*it) + ",";
    }
    return format;
}

string DataBase::csvHeader(Tensor t, string prefix) {
    string format="";
    int n=t.NumElements();
    for (int i=0; i<n; i++) {
        format += prefix + to_string(i) + ",";
    }
    return format;
}

void DataBase::toCSV(string filename) {
    std::ofstream outfile;

    if (state_input.size() != action_input.size() &&
        state_input.size() != output.size()) {
        cerr << "The vector size don't match" << endl;
    }

    outfile.open(filename);

    outfile << csvHeader(state_input[0], "x") <<
               csvHeader(action_input[0], "u") <<
               csvHeader(output[0], "x_next") << endl;

    for (int i=0; i < state_input.size(); i++) {
        outfile << tensor2CSV(state_input[i]) <<
                   tensor2CSV(action_input[i]) <<
                   tensor2CSV(output[i]) << endl;
    }
}
