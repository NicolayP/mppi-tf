#ifndef __CONTROLLER_BASE_CLASS_HPP__
#define __CONTROLLER_BASE_CLASS_HPP__

/* ---------- Standard libraries ---------- */

/* --------- Tensorflow libraries --------- */
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/core/framework/tensor.h"

/* ------------- Own libraries ------------ */
#include "cost_base.hpp"
#include "model_base.hpp"



class ControllerBase {

private:
    /* ------------ Attributes ---------- */

    /* ------ Point Mass Variables -------*/
    float m_dt; /* timestep between two calls of the controller */
    float m_mass; /* point mass */

    /* ------ Controller Variables -------*/
    int m_k; /* number of samples */
    int m_tau; /* prediction horizon */
    int m_s_dim; /* state dimension */
    int m_a_dim; /* action dimension */

    float m_lambda; /* lambda variable */

    CostBase m_cost; /* Cost class */
    ModelBase m_model; /* Model class */

    // The current action sequence. i.e. The mean to sample from.
    tensorflow::Tensor m_sigma; /* [a_dim, a_dim]*/
    tensorflow::Tensor m_goal; /* [s_dim, 1]*/
    tensorflow::Tensor m_U; /* [Tau, a_dim, 1]*/

    tensorflow::Scope m_root; /* Root Scope for the algorithm */

    /* Persistant session because we need many calls to tensorflow*/
    tensorflow::ClientSession m_sess;

    tensorflow::Output mStateInput; /* [K, s_dim] */
    tensorflow::Output mActionInput; /* [K, a_dim] */
    tensorflow::Output mUpdate; /* output containing the action sequence [Tau, act_dim, 1] */
    tensorflow::Output mNext; /* output containing the next action [1, act_dim, 1] */

    std::vector<tensorflow::Tensor> out_tensor; // Output tensor from subsequent calls

    /* ------------ Methods ------------- */
public:
    /* ------------ Constructors -------- */
    ControllerBase ();

    ControllerBase (const int k,
                    const int tau,
                    const float dt,
                    const float mass,
                    const int s_dim,
                    const int a_dim);

    ControllerBase (tensorflow::Scope root,
                    const int k,
                    const int tau,
                    const float dt,
                    const float mass,
                    const int s_dim,
                    const int a_dim);

    /* ------------ Destructors --------- */

    ~ControllerBase ();

    /* ------------ Methods ------------- */
    std::vector<float> next (std::vector<float> x);
    bool setActions (std::vector<tensorflow::Tensor> actions);
    tensorflow::Status logGraph ();

    /*
     * Computes the minimal value in the tensor cost
     * Input:
     * ------
     *  - tensorflow::Scope scope, the scope in the computational graph.
     *  - tensorflow::Input cost, the cost vector. Shape: [k, 1, 1]
     *
     * Output:
     * -------
     *  - tensorflow::Output beta, shape: [1, 1]
     *
     */
    tensorflow::Output mBeta (tensorflow::Scope scope, tensorflow::Input cost);

    /*
     * Compute the argument for the exponential $$-\frac{1}{\lambda}(cost - beta)$$
     * Input:
     * ------
     *  - tensorflow::Scope scope, the scope in the computational graph.
     *  - tensorflow::Input cost, the cost vector. Shape: [k, 1, 1]
     *  - tensorflow::Input beta, Shape: [1, 1]
     * Output:
     * -------
     *  - tensorflow::Output exp_arg, Shape [k, 1, 1]
     *
     */
    tensorflow::Output mExpArg (tensorflow::Scope scope,
                                tensorflow::Input cost,
                                tensorflow::Input beta);

    /*
     * Computes the exponential of the input tensor. $$exp(-\frac{1}{\lambda}(cost - beta))$$
     * Input:
     * ------
     *  - tensorflow::Scope scope, the scope in the computational graph.
     *  - tensorflow::Input arg, the argument vector. Shape: [k, 1, 1]
     *
     * Output:
     * -------
     *  - tensorflow::Output, the exponential. Shape: [k, 1, 1]
     *
     */
    tensorflow::Output mExp (tensorflow::Scope scope, tensorflow::Input arg);

    /*
     * Computes nabla, the normalizing factor. $$\nabla = \sum_{k=0}^{K} exp(-\frac{1}{\lambda}(cost - beta))$$
     * Input:
     * ------
     *  - tensorflow::Scope scope, the scope in the computational graph.
     *  - tensorflow::Input exp, the exponential vector. Shape: [k, 1, 1]
     *
     * Output:
     * -------
     *  - tensorflow::Output nabla, Shape: [1, 1]
     *
     */
    tensorflow::Output mNabla (tensorflow::Scope scope, tensorflow::Input exp);

    /*
     * Computes the weights of each sample, $$w(\epsilon^k) = \frac{1}{\nabla} exp(-\frac{1}{\lambda}(cost - beta))
     * Input:
     * ------
     *  - tensorflow::Scope scope, the scope in the computational graph.
     *  - tensorflow::Input exp, the exponential vector. Shape: [k, 1, 1]
     *  - tensorflow::Input nabla, Shape: [1, 1]
     *
     * Output:
     * -------
     *  - tensorflow::Output weights, Shape[k, 1, 1]
     *
     */
    tensorflow::Output mWeights (tensorflow::Scope scope,
                                 tensorflow::Input exp,
                                 tensorflow::Input nabla);

    /*
     * Compute the weighted noise to update the action sequence. $$\sum_{k=0}^{K} w(\epsilon^k) \epsilon_t^k$$
     * Input:
     * ------
     *  - tensorflow::Scope scope, the scope in the computational graph.
     *  - tensorflow::Input weights, the weight vector. Shape: [k, 1, 1]
     *  - tensorflow::Input noises, Shape: [k, tau, a_dim, 1]
     *
     * Output:
     * -------
     *  - tensorflow::Output weighted_noise, Shape: [tau, a_dim, 1]
     *
     */
    tensorflow::Output mWeightedNoise (tensorflow::Scope scope,
                                       tensorflow::Input weights,
                                       tensorflow::Input noises);

    /*
     * Prepares the action tensor for a given timestep.
     * Input:
     * ------
     *  - tensorflow::Scope scope, the scope in the computational graph.
     *  - tensorflow::Input actions, the action sequence vector.
     *      Shape: [tau, a_dim, 1]
     *  - int timestep.
     *
     * Output:
     * -------
     *  - tensorflow::Output weighted_noise, Shape: [a_dim, 1]
     *
     */
    tensorflow::Output mPrepareAction (tensorflow::Scope scope,
                                       tensorflow::Input actions,
                                       int timestep);

    /*
     * Build the update graph. Computes the new action sequence given the cost and the noise
     * Input:
     * ------
     *  - tensorflow::Scope scope, the scope in the computational graph.
     *  - tensorflow::Input noises, the weight vector.
     *      Shape: [k, tau, a_dim, 1]
     *  - int timestep.
     *
     * Output:
     * -------
     *  - tensorflow::Output weighted_noise, Shape: [k, a_dim, 1]
     *
     */
    tensorflow::Output mPrepareNoise (tensorflow::Scope scope,
                                      tensorflow::Input noises,
                                      int timestep);

    /*
     * Build the update graph. Computes the new action sequence given the cost and the noise
     * Input:
     * ------
     *  - None
     *
     * Output:
     * -------
     *  - None
     *
     */
    tensorflow::Output mBuildUpdateGraph (tensorflow::Scope scope,
                                          tensorflow::Input cost,
                                          tensorflow::Input noises);

    tensorflow::Output mShift (tensorflow::Scope scope,
                               tensorflow::Input current,
                               tensorflow::Input init,
                               int nb=1);

    tensorflow::Output mGetNew (tensorflow::Scope scope,
                                tensorflow::Input current,
                                int nb);
     /*
      * Builds the entire computational graph.
      * Input:
      * ------
      *  - None
      *
      * Output:
      * -------
      *  - None
      *
      */
     void mBuildGraph ();

     /*
      * Build the computational Graph for the dynamical model and outputs the
      * cost.
      * Input:
      * ------
      *  -Tensorboard::Input, init_state the inital state tensor,
                 shape [s_dim, 1]
      *  -Tensorboard::Input, actions: the action sequence tensor,
                 shape [tau, a_dim, 1]
      *  -Tensorboard::Input, noises: The perturbation term,
                 shape [K, tau, a_dim, 1]
      *
      * Output:
      * -------
      *  - Tensorboard::Output, the cost array, shape [K, 1, 1]
      *
      */
     tensorflow::Output mBuildModelGraph (tensorflow::Scope model_scope,
                                          tensorflow::Scope cost_scope,
                                          tensorflow::Input init_state,
                                          tensorflow::Input actions,
                                          tensorflow::Input noises);

    /*
     * Builds the noise generation graph.
     * Input:
     * ------
     *  - tensorflow::Scope scope. The scope in the computational graph.
     *  - tensorflow::Input mean. The mean to sample from. Shape [tau, a_dim, 1]
     *  - tensorflow::Input sigma. The std for the samples. Shape [a_dim, a_dim]
     *
     * Output:
     * -------
     *  - tensorflow::Output noise. The noise tensor. Shape: [k, tau, a_dim, 1]
     *
     */
    tensorflow::Output mNoiseGenGraph (tensorflow::Scope scope,
                                       tensorflow::Input sigma);

    tensorflow::Output mInit0 (tensorflow::Scope, int nb);

    tensorflow::Status mLogGraph ();
    /* ------------ Attributes ---------- */
};

#endif
