import tensorflow as tf
from utile import assert_shape
import numpy as np

# TODO: compute all constants without tensorflow. Out of the graph computation.
class CostBase(object):
    '''
        Cost base class for the mppi controller. 
        This is an abstract class and should be heritated by every specific cost.
    '''
    def __init__(self, lam, gamma, upsilon, sigma):
        '''
            Abstract class for the cost function. 
            - input:
            --------
                - lam (lambda) the inverse temperature. 
                - gamma: decoupling parameter between action and noise.
                - upsilon: covariance augmentation for noise generation.
                - sigma: the noise covariance matrix. shape [a_dim, a_dim]
                - TODO: add diag arg to feed sigma as the diag element if prefered.
            
            - output:
            ---------
                the new cost object.
        '''

        with tf.name_scope("Cost_setup"):
            self.lam = lam
            self.gamma = gamma
            self.upsilon = upsilon
            
            

            s = tf.convert_to_tensor(sigma, dtype=tf.float64)

            self.sig_shape = s.shape
            if len(self.sig_shape) != 2:
                raise AssertionError("The noise covariance matrix needs to be a semi definit positive.")
            self.invSig = tf.linalg.inv(s)

    def build_step_cost_graph(self, scope, state, action, noise):
        '''
            Computes the step cost of a sample.
            - input:
            --------
                - scope: The tensorflow scope.
                - state: the current state of the system. shape: [k/1, s_dim, 1]
                - action: the action applied to reach the current state. shape: [k/1, a_dim, 1]
                - noise: the noise applied to the sample. shape: [k/1, a_dim, 1]

            - output:
            ---------
                - dictionnary with entries:
                    "cost" : the actual step cost.
                    "state_cost": the state related cost $$ q(state) $$
                    "action_cost": the action related cost.
        '''
        return_dict = {}
        if not assert_shape(action, (self.sig_shape[0], 1)):
            raise AssertionError("Bad shape for the action tensor, should be [a_dim, 1], got shape {}".format(action.shape))
        
        if not assert_shape(noise, (-1, self.sig_shape[0], 1)):
            raise AssertionError("Bad shape for the noise tensor, should be [k/1, a_dim, 1], got shape {}".format(noise.shape))

        with tf.name_scope("step_cost") as s:
            state_cost_dict = self.state_cost(s, state)
            action_cost_dict = self.action_cost(s, action, noise)
            step_cost = tf.add(state_cost_dict["state_cost"], action_cost_dict["action_cost"],
                               name="add")

        return_dict["cost"] = step_cost
        return_dict = {**return_dict, **state_cost_dict, **action_cost_dict}
        return return_dict

    def add_cost(self, scope, input_dict, current_dict):
        '''
            Utility funtion to perform += on a dictionnary. 
            If similary keys are found in the two dictionnaries the items are added,
            else: the items stayes unchanged
            
            - input:
            --------
                - input_dict: the first dict.
                - current_dict: the current state of the dict.

            - output:
            ---------
                - current_dict += input_dict.

        '''
        for key in input_dict:
            if key in current_dict.keys():
                current_dict[key] = tf.add(current_dict[key], input_dict[key],
                                           name="tmp_cost")
            else:
                current_dict[key] = input_dict[key]
        return current_dict

    def build_final_step_cost_graph(self, scope, state):
        '''
            terminal cost function

            - input:
            --------
                - state: the current state of the system. shape: [k/1, s_dim, 1]

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
                - action: the action applied to reach the current state. shape: [k/1, a_dim, 1]
                - noise: the noise applied to the sample. shape: [k/1, a_dim, 1]

            - output:
            ---------
                - Dictionnary with entries:
                    "a_cost": pure action related cost
                    "mix_cost": cost mixing noise and action.
                    "n_cost": the noise related cost.
                    "control_cost": $ \gamma [a_cost + 2mix_cost] $
                    "action_cost": the overall action cost. 

        '''
        return_dict = {}

        rhs_n_cost = tf.linalg.matmul(self.invSig, noise,
                                      name="rhs_noise")
        rhs_a_cost = tf.linalg.matmul(self.invSig, action,
                                      name="rhs_action")
        # \u^{T}_t \Sigma^{-1} \epsilon_t
        mix_cost = tf.linalg.matmul(action, rhs_n_cost, transpose_a=True,
                                    name="mix")
        mix_cost = tf.math.multiply(tf.cast(2., dtype=tf.float64), mix_cost)

        # \epsilon^{T}_t \Sigma^{-1} \epsilon_t
        n_cost = tf.linalg.matmul(noise, rhs_n_cost, transpose_a=True,
                                  name="noise")
        # \u^{T}_t \Sigma^{-1} \u_t
        a_cost = tf.linalg.matmul(action, rhs_a_cost, transpose_a=True,
                                  name="action")

        

        return_dict["a_cost"] = tf.math.multiply(tf.cast(self.gamma, dtype=tf.float64), a_cost)
        return_dict["mix_cost"] = tf.math.multiply(tf.cast(self.gamma, dtype=tf.float64), mix_cost)
        return_dict["n_cost"] = tf.math.multiply(tf.cast(self.lam*(1.-1./self.upsilon),
                                             dtype=tf.float64), n_cost)

        # \gamma [action_cost + 2mix_cost]
        return_dict["control_cost"] = tf.add(return_dict["a_cost"], return_dict["mix_cost"])
        return_dict["action_cost"] = tf.math.multiply(tf.cast(0.5, dtype=tf.float64),
                                tf.add(return_dict["control_cost"], return_dict["n_cost"]))

        return return_dict

    def state_cost(self, scope, state):
        '''
            Computes a step state cost. $q(x)$

            - input:
            --------
                - scope. tensorflow scope name.
                - state. the current state tensor. Shape [k, s_dim, 1]
            
            - output:
            ---------
                - the cost for the current state.
        '''
        raise NotImplementedError

    def draw_goal(self):
        '''
            generates a graph representing the goal.
        '''
        raise NotImplementedError

    def dist(self, state):
        '''
            computes a distance metric from the state to the goal.
            
            - input:
            --------
                - state: The state tensor. Shape [s_dim, 1]

            - output:
            ---------
                - the "distance" from the state to the goal.
        '''
        raise NotImplementedError


class WayPointsCost(object):
    '''
        Cost function for reference tracking.
        The class recieves a set of waypoints, interpolates a
        trajectory that will be used as reference to track.

        First implementation. Similar to the static cost, we consider the distance between
        the trajectory and the two first points. Do a weighted average between the two. Revert to
        Static cost if only one waypoint left.
    '''

    def __init__(self, lam, gamma, upsilon, sigma, waypoints=None):
        CostBase.__init__(self, lam, gamma, upsilon, sigma, alpha=0.2)
        if waypoints is not None:
            self.waypoints = waypoints
        else:
            self.waypoints = []

        self.alpha=0.2

    def add_waypoints(self, waypoints):
        for waypoint in waypoints:
            self.waypoints.append(waypoint)

    def add_waypoint(self, waypoint):
        self.waypoints.append(waypoint)

    def pop(self):
        self.waypoints.pop()

    def state_cost(self, scope, state, t):
        '''
            Computes the state cost for the waypoints cost.

            - input:
            --------
                - scope: the tensroflow scope.
                - State: current state of the system, Shape: [k/1, goal_dim, 1]

            - output:
            ---------
                dict with entry:
                    "state_cost"
        '''
        return_dict = {}

        if not assert_shape(state, (-1, self.q_shape[0], 1)):
            raise AssertionError("State tensor shape error, expected: [k/1, {}, 1], got {}".format(self.q_shape[0], state.shape))
        if len(self.waypoints) < 2:
            goal = self.waypoints[0]
            state_cost = self.dist_waypoint(state, goal)
            return_dict["state_cost"] = state_cost
            return return_dict
        first = self.waypoints[0]
        second = self.waypoints[1]
        d_first = self.dist_waypoint(state, first)
        d_sec = self.dist_waypoint(state, second)

        state_cost = (self.alpha-1)*d_first + self.alpha*d_sec

        return_dict["state_cost"] = state_cost
        return return_dict

    def dist_waypoint(self, state, waypoint):

        return_dict = {}

        diff = tf.math.subtract(state, waypoint, name="diff")
        state_cost = tf.linalg.matmul(diff, tf.linalg.matmul(self.Q, diff, name="right"), transpose_a=True, name="left")

        return_dict["state_cost"] = state_cost
        return return_dict

    def dist(self, state):
        raise NotImplementedError
