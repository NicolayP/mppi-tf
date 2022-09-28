import tensorflow as tf
from utile import assert_shape, dtype

# TODO: compute all constants without tensorflow. Out of the graph
# computation.
class CostBase(tf.Module):
    '''
        Cost base class for the mppi controller.
        This is an abstract class and should be heritated by every
        specific cost.
    '''
    def __init__(self, lam, gamma, upsilon, sigma):
        '''
            Abstract class for the cost function.
            - input:
            --------
                - lam (lambda) the inverse temperature.
                - gamma: decoupling parameter between action and noise.
                - upsilon: covariance augmentation for noise generation.
                - sigma: the noise covariance matrix. shape [aDim, aDim]
                - TODO: add diag arg to feed sigma as the diag element if
                prefered.

            - output:
            ---------
                the new cost object.
        '''

        with tf.name_scope("Cost_setup"):
            self.lam = lam
            self.gamma = gamma
            self.upsilon = upsilon

            s = tf.convert_to_tensor(sigma, dtype=dtype)

            self.sig_shape = s.shape
            if len(self.sig_shape) != 2:
                raise AssertionError("The noise covariance matrix\
                                     needs to be a semi definit positive.")
            self.invSig = tf.linalg.inv(s)

    def step(self, scope, state, action, noise):
        '''
            Computes the step cost of a sample.
            - input:
            --------
                - scope: The tensorflow scope.
                - state: the current state of the system.
                    shape: [k/1, sDim, 1]
                - action: the action applied to reach the current state.
                    shape: [aDim, 1]
                - noise: the noise applied to the sample.
                    shape: [k/1, aDim, 1]

            - output:
            ---------
                - dictionnary with entries:
                    "cost" : the actual step cost.
                    "state_cost": the state related cost $$ q(state) $$
                    "action_cost": the action related cost.
        '''
        if not assert_shape(action, (self.sig_shape[0], 1)):
            raise AssertionError("Bad shape for the action tensor,\
                    should be [{}, 1], got shape {}".format(action.shape))

        if not assert_shape(noise, (-1, self.sig_shape[0], 1)):
            raise AssertionError("Bad shape for the noise tensor,\
                should be [k/1, {}, 1], got shape {}".format(self.sig_shape[0], noise.shape))

        with tf.name_scope("step_cost") as s:
            stateCost = self.state_cost(s, state)
            actionCost = self.action_cost(s, action, noise)
            stepCost = tf.add(stateCost, actionCost,
                               name="add")

        return stepCost

    def final(self, scope, state):
        '''
            terminal cost function

            - input:
            --------
                - state: the current state of the system.
                    shape: [k/1, sDim, 1]

            - output:
            ---------
                - $$psi(state)$$

        '''
        return self.state_cost(scope, state)

    def action_cost(self, scope, action, noise):
        '''
            action related cost part.

            - input: 
            --------
                - action: the action applied to reach the current state.
                    shape: [k/1, aDim, 1]
                - noise: the noise applied to the sample.
                    shape: [k/1, aDim, 1]

            - output:
            ---------
                - Dictionnary with entries:
                    "a_cost": pure action related cost
                    "mix_cost": cost mixing noise and action.
                    "n_cost": the noise related cost.
                    "control_cost": $ \gamma [a_cost + 2mix_cost] $
                    "action_cost": the overall action cost.
        '''
        rhsNcost = tf.linalg.matmul(self.invSig, noise,
                                      name="rhs_noise")

        rhsAcost = tf.linalg.matmul(self.invSig, action,
                                      name="rhs_action")

        # \u^{T}_t \Sigma^{-1} \epsilon_t
        mixCost = tf.linalg.matmul(action, rhsNcost, transpose_a=True,
                                    name="mix")

        mixCost = tf.math.multiply(tf.cast(2., dtype=dtype), mixCost)

        # \epsilon^{T}_t \Sigma^{-1} \epsilon_t
        nCost = tf.linalg.matmul(noise, rhsNcost, transpose_a=True,
                                  name="noise")

        # \u^{T}_t \Sigma^{-1} \u_t
        aCost = tf.linalg.matmul(action, rhsAcost, transpose_a=True,
                                  name="action")

        aCost = tf.math.multiply(tf.cast(self.gamma, dtype=dtype),
                                 aCost)

        mixCost = tf.math.multiply(tf.cast(self.gamma, dtype=dtype),
                                   mixCost)

        nCost = tf.math.multiply(tf.cast(self.lam*(1.-1./self.upsilon),
                                         dtype=dtype),
                                 nCost)

        # \gamma [action_cost + 2mix_cost]
        controlCost = tf.add(aCost, mixCost)
        actionCost = tf.math.multiply(tf.cast(0.5, dtype=dtype),
                                      tf.add(controlCost, nCost))

        return actionCost

    def state_cost(self, scope, state):
        '''
            Computes a step state cost. $q(x)$

            - input:
            --------
                - scope. tensorflow scope name.
                - state. the current state tensor. Shape [k, sDim, 1]
            
            - output:
            ---------
                - the cost for the current state.
        '''
        raise NotImplementedError

    def dist(self, state):
        '''
            computes a distance metric from the state to the goal.
            
            - input:
            --------
                - state: The state tensor. Shape [sDim, 1]

            - output:
            ---------
                - the "distance" from the state to the goal.
        '''
        raise NotImplementedError


class FooCost(CostBase):
    def __init__(self, lam, gamma, upsilon, sigma):
        super(FooCost, self).__init__(
            lam=lam, gamma=gamma, upsilon=upsilon, sigma=sigma
        )
    
    def state_cost(self, scope, state):
        return tf.linalg.norm(state, axis=1)

    def dist(self, state):
        return tf.linalg.norm(state, axis=1)
