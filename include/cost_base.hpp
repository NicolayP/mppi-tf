#ifndef __COST_BASE_CLASS_HPP__
#define __COST_BASE_CLASS_HPP__

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/core/framework/tensor.h"

class CostBase {
private:

    /* ------------ Attributes ---------- */
    float in_lambda; /* Shape [1, 1] */
    tensorflow::Tensor in_sigma; /* Shape [a_dim, a_dim] */
    tensorflow::Tensor m_goal; /* Shape [s_dim, 1]*/
    tensorflow::Tensor in_Q; /* Shape [s_dim, s_dim]*/
    tensorflow::Output m_lambda, m_inv_sigma, m_Q;


    /* ------------ Methods ------------- */

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
    CostBase();

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
    CostBase(const float lambda,
             const std::vector<float> sigma,
             const std::vector<float> goal);

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
    CostBase (const float lambda,
              const tensorflow::Tensor sigma,
              const tensorflow::Tensor goal);

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
    CostBase (const float lambda,
              const tensorflow::Tensor sigma,
              const tensorflow::Tensor goal,
              const tensorflow::Tensor Q);

    /* ------------ Destructors --------- */
    ~CostBase();

    /* ------------ Methods ------------- */

    /*
     * Builds the computational graph for one step cost. Effectively:
     *  $$ q(x_t) + \lambda u^{T}_{t-1} \Sigma^{-1} \epsilon^{k}_{t-1} $$
     * Input:
     * ------
     *  - tensorflow::Input state, Current state. Shape [k, s_dim, 1]
     *  - tensorflow::Input action, applied action. Shape [a_dim, 1]
     *  - tensorflow::Input noise, perturbation of each sample.
     *           shape [k, a_dim, 1]
     *
     * Output:
     * -------
     *  - tensorflow::Output the associated cost tensor. Shape [k, 1, 1]
     *
     */

    tensorflow::Output mBuildStepCostGraph(tensorflow::Scope scope,
                                           tensorflow::Input state,
                                           tensorflow::Input action,
                                           tensorflow::Input noise);


    /*
     * Builds the computational graph for the final cost. Effectively:
     * $$ \Phi(x_{\Tau}) $$
     * Input:
     * ------
     *  - tensorflow::Input state, the final state. Shape [k, s_dim, 1]
     *
     * Output:
     * -------
     *  - tensorflow::Output, the final state cost. Shape [k, 1, 1]
     *
     */
    tensorflow::Output mBuildFinalStepCostGraph(tensorflow::Scope scope,
                                                tensorflow::Input state);



    /*
     * Compute the step cost associated with a given state.
     * Input:
     * ------
     *  - tensorflow::Input, state. The evaluated state. [k, s_dim, 1]
     *
     * Output:
     * -------
     *  - tensorflow::Output, The cost of the given state [k, 1, 1]
     *
     */
    tensorflow::Output mStateCost(tensorflow::Scope scope,
                                  tensorflow::Input state);

    /*
     * Compute the step cost associated with a given state.
     * Input:
     * ------
     *  - tensorflow::Input, state. The evaluated state. [k, s_dim, 1]
     *
     * Output:
     * -------
     *  - tensorflow::Output, The cost of the given state [k, 1, 1]
     *
     */
    tensorflow::Output mActionCost(tensorflow::Scope scope,
                                   tensorflow::Input action,
                                   tensorflow::Input noise);


    void setConsts(tensorflow::Scope scope);
    /* ------------ Attributes ---------- */


};

#endif
