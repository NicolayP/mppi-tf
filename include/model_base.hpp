#ifndef __MODEL_BASE_CLASS_HPP__
#define __MODEL_BASE_CLASS_HPP__

/* --------- Tensorflow libraries --------- */
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/core/framework/tensor.h"


class ModelBase {
private:
    /* ------------ Attributes ---------- */
    float m_dt;
    float m_m;
    int m_s_dim, m_a_dim;
    tensorflow::Tensor m_mass;

    /* ------------ Methods ------------- */

public:
    /* ------------ Attributes ---------- */
    // None

    /* ------------ Constructors -------- */
    /*
     * Creates a ModelBase object with mass 1, dt 0.01s.
     * Input:
     * ------
     *  - None
     *
     * Output:
     * -------
     *  - ModelBase object
     *
     */
    ModelBase();

    /*
     * Constructor for the Point Mass Model.
     * Input:
     * ------
     *  - Tensorflow::Scope root scope, the parent scope.
     *  - float mass, the mass of the point mass.
     *  - float dt, the integration timestep.
     *
     * Output:
     * -------
     *  - ModelBase object.
     *
     */
    ModelBase(const float mass,
              const float dt,
              const int s_dim,
              const int a_dim);

    /* ------------ Destructors --------- */
    ~ModelBase();

    /* ------------ Methods ------------- */
    /*
     * Builds the computational graph of one step.
     * Input:
     * ------
     *  - tensorflow::Input state, the input state of shape [k, s_dim, 1] or
     *      [1, s_dim, 1] if it is the first step.
     *  - tensorflow::Input action, the action to be applied to the model.
     *      Shape [k, a_dim, 1]
     *
     * Output:
     * -------
     *  - tensorflow::Output next_state. The next state of the model.
     *      Shape [k, s_dim, 1]
     *
     */
    tensorflow::Output mBuildModelStepGraph(tensorflow::Scope scope,
                                            tensorflow::Input state,
                                            tensorflow::Input action);

    /*
     * Builds the action free update of the model: A*x
     * Input:
     * ------
     *  - tensorflow::Scope scope, the scope in the computational graph.
     *  - tensorflow::Input state, the current state. Shape [k, s_dim, 1]
     *
     * Output:
     * -------
     *  - tensorflow::Output next_state, the next state of the free system. Shape [k, s_dim, 1]
     *
     */
    tensorflow::Output mBuildFreeStepGraph(tensorflow::Scope scope,
                                           tensorflow::Input state);

    /*
     * Builds the action update of the model: B*u
     * Input:
     * ------
     *  - tensorflow::Scope scope, the scope in the computational graph.
     *  - tensorflow::Input state, the current state. Shape [k, a_dim, 1]
     *
     * Output:
     * -------
     *  - tensorflow::Output next_state, the next state of the acted system. Shape [k, s_dim, 1]
     *
     */
    tensorflow::Output mBuildActionStepGraph(tensorflow::Scope scope,
                                             tensorflow::Input action);

    /*
     * Trains the model given the data
     * Input:
     * ------
     *  - None
     *
     * Output:
     * -------
     *  - None
     *
     */
    void train();

};

#endif
