#ifndef __UTILE_MPPI_TF_HPP__
#define __UTILE_MPPI_TF_HPP__

/* --------- Tensorflow libraries --------- */
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/core/framework/tensor.h"

namespace utile{

    /*
     * Builds a block Diagonal matrix by replicating the input n times.
     * Input:
     * ------
     *  - tensorflow::Scope scope, the scope in the computational graph.
     *  - tensorflow::Input in, the inpute tensor. Shape [x, y].
     *  - int nb, the number of replication.
     *
     * Output:
     * -------
     *  - tensorflow::Output blk, Shape [nb*x, nb*y].
     *
     */
    tensorflow::Output blockDiag(tensorflow::Scope scope,
                                 tensorflow::Input in,
                                 int nb);

    /*
     * Logs the graph to be visualized in Tensorboard.
     * Input:
     * ------
     *  - tensorflow::Scope scope. The scope of the graph to be logged.
     *
     * Output:
     * -------
     *  - tensorflow::Status, the result of the writting.
     *
     */
    tensorflow::Status logGraph(tensorflow::Scope scope);
}

#endif
