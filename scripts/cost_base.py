import tensorflow as tf


# TODO: compute all constants without tensorflow. Out of the graph computation.
class CostBase(object):
    def __init__(self, lam, gamma, upsilon, sigma, tau):

        with tf.name_scope("Cost_setup"):
            self.lam = lam
            self.gamma = gamma
            self.upsilon = upsilon
            self.tau = tau
            s = tf.convert_to_tensor(sigma, dtype=tf.float64)
            self.invSig = tf.linalg.inv(s)

    def build_step_cost_graph(self, scope, state, action, noise):
        return_dict = {}

        with tf.name_scope("step_cost") as s:
            state_cost_dict = self.state_cost(s, state)
            action_cost_dict = self.action_cost(s, action, noise)
            step_cost = tf.add(state_cost_dict["state_cost"], action_cost_dict["action_cost"],
                               name="add")

        return_dict["cost"] = step_cost
        return_dict = {**return_dict, **state_cost_dict, **action_cost_dict}
        return return_dict

    def add_cost(self, scope, input_dict, current_dict):
        for key in input_dict:
            if key in current_dict.keys():
                current_dict[key] = tf.add(current_dict[key], input_dict[key],
                                           name="tmp_cost")
            else:
                current_dict[key] = input_dict[key]
        return current_dict

    def build_final_step_cost_graph(self, scope, state):
        return self.state_cost(scope, state)

    def action_cost(self, scope, action, noise):
        #print(action)
        #print(noise)
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
        raise NotImplementedError

    def draw_goal(self):
        raise NotImplementedError

    def dist(self, state):
        raise NotImplementedError
