#ifndef __DATA_BASE_CLASS_HPP__
#define __DATA_BASE_CLASS_HPP__

/* --------- Tensorflow libraries --------- */
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/core/framework/tensor.h"

class DataBase {
public:
    /* ------------ Constructors -------- */

    /*
     * TODO
     * Input:
     * ------
     *  - None
     *
     * Output:
     * -------
     *  - None
     *
     */
    DataBase();

    /* -------------- Destructor -------- */

    ~DataBase();

    void addEl(tensorflow::Tensor x, tensorflow::Tensor u, tensorflow::Tensor x_next);

    void addX(tensorflow::Tensor x);

    void addU(tensorflow::Tensor u);

    void addNext(tensorflow::Tensor x_next);

    void toCSV(std::string filename);

    std::string csvHeader(tensorflow::Tensor t, std::string prefix);

    std::string tensor2CSV(tensorflow::Tensor t);

private:
    // Contains the input for the learning agent [x, u]
    std::vector<tensorflow::Tensor> state_input;
    std::vector<tensorflow::Tensor> action_input;
    std::vector<tensorflow::Tensor> output;
};

#endif
