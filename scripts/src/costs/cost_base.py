import tensorflow as tf
import tensorflow_graphics.geometry.transformation as tfgt
from ..misc.utile import assert_shape, dtype
import numpy as np

# TODO: compute all constants without tensorflow. Out of the graph
# computation.
'''
    Cost base class for the mppi controller.
    This is an abstract class and should be heritated by every
    specific cost.
'''
class CostBase(tf.Module):
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
    def __init__(self, lam, gamma, upsilon, sigma):
        self._observer = None
        self._obstacles = []
        self._mask = tf.convert_to_tensor(np.array([[False]]), dtype=tf.bool)
        self._zero_vec = tf.convert_to_tensor(np.array([[0.]]), dtype=dtype)
        self._inf_vec = tf.convert_to_tensor(np.array([[np.inf]]), dtype=dtype)

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
    def build_step_cost_graph(self, scope, state, action, noise):
        if not assert_shape(action, (self.sig_shape[0], 1)):
            raise AssertionError("Bad shape for the action tensor,\
                    should be [{}, 1], got shape {}".format(action.shape))

        if not assert_shape(noise, (-1, self.sig_shape[0], 1)):
            raise AssertionError("Bad shape for the noise tensor,\
                should be [k/1, {}, 1], got shape {}".format(self.sig_shape[0], noise.shape))

        with tf.name_scope("step_cost") as s:
            self._stateCost = self.state_cost(s, state)
            self._actionCost = self.action_cost(s, action, noise)
            collisionCost = self.collision_cost(s, state)

            stepCost = tf.add(self._stateCost, self._actionCost,
                               name="add")
            stepCost = tf.add(stepCost, collisionCost, name="add_collision")

        return stepCost

    def get_state_cost(self):
        return self._stateCost

    def get_action_cost(self):
        return self._actionCost

    '''
        Utility funtion to perform += on a dictionnary.
        If similary keys are found in the two dictionnaries
        the items are added, else: the items stayes unchanged

        - input:
        --------
            - input_dict: the first dict.
            - current_dict: the current state of the dict.

        - output:
        ---------
            - current_dict += input_dict.

    '''
    def add_cost(self, scope, currentCost, newCost):
        cost = tf.add(currentCost, newCost, name="tmp_cost")
        return cost

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
    def build_final_step_cost_graph(self, scope, state):
        return self.state_cost(scope, state)

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
    def action_cost(self, scope, action, noise):
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
    def state_cost(self, scope, state):
        raise NotImplementedError

    '''
        Detects if there is a collision between the state vector
        and the list of obstacles.

        - input:
        --------
            - state: the current state of the system.
                shape: [k/1, sDim, 1]

        - output:
        ---------
            - $$psi(state)$$

    '''
    def collision_cost(self, scope, state):
        k = tf.shape(state)[0]
        mask = tf.broadcast_to(self._mask, (k, 1, 1))
        zeros_vec = tf.broadcast_to(self._zero_vec, (k, 1, 1))
        inf_vec = tf.broadcast_to(self._inf_vec, (k, 1, 1))
        for obs in self._obstacles:
            mask = tf.logical_or(obs.collide(state), mask)
        mask = tf.where(mask, inf_vec, zeros_vec)
        return mask

    def add_obstacle(self, obs):
        self._obstacles.append(obs)

    def get_obstacles(self):
        return self._obstacles

    '''
        generates a graph representing the goal.
    '''
    def draw_goal(self):
        raise NotImplementedError

    '''
        computes a distance metric from the state to the goal.

        - input:
        --------
            - state: The state tensor. Shape [sDim, 1]

        - output:
        ---------
            - the "distance" from the state to the goal.
    '''
    def dist(self, state):
        raise NotImplementedError

    def set_observer(self, observer):
        self._observer = observer

    '''
    Returns a list of 3D points representing the trajectory of the task. Doesn't handl
    obstacles.

    inputs:
    -------
        - state: the state of the agent.
        - pts: the number of points in the list
    '''
    def get_3D(self, state, pts=100):
        raise NotImplementedError

    '''
        When the cost contains multiple goals.
    '''
    def update_goal(self, state):
        pass

    def angle_error(self, state, split=False):
        pass

    def velocity_error(self, state, split=False):
        pass

    def position_error(self, state, split=False):
        pass


class PrimitiveObstacles(tf.Module):
    def __init__(self, type):
        self.type = type
        pass

    '''
        Checks if "state" is within the obstacles. Returns
        a bool vector whose entry is True if there is a collision
        between state t and the obstacle. False otherwise

        inputs:
        -------
            - state, tf.tensor, shape = [k, sDim, 1]

        outputs:
        --------
            - collision mask, tf.tensor. shape = [k, 1, 1]
    '''
    def collide(self, state):
        raise NotImplementedError

    '''
        Returns the pose of the obstacle.

        outputs:
        --------
            - p, tf.tensor, shape = [3, 1]
            - q, tf.tensor, shape = [4, 1]
    '''
    def get_pose(self):
        raise NotImplementedError


class CylinderObstacle(PrimitiveObstacles):
    '''
        Constructor for cylinder object.
        inputs:
        -------
            - p1: tensor, bottom point of the cylinder.
                shape = [3, 1]
            - p2: tensor, top of the cylinder.
                shape = [3, 1]
            - radius: cylindre radius in meters.
    '''
    def __init__(self, p1, p2, radius):
        super(CylinderObstacle, self).__init__("cylinder")
        self.p1, self.p2 = tf.convert_to_tensor(p1, dtype=dtype), tf.convert_to_tensor(p2, dtype=dtype)
        self.e = self.p2 - self.p1
        self.n = tf.constant([0., 0., 1.], dtype=dtype) # cylinder always up.
        self.r = radius

    '''
        Checks if "state" is within the obstacles. Returns
        a bool vector whose entry is True if there is a collision
        between state t and the obstacle. False otherwise

        inputs:
        -------
            - state, tf.tensor, shape = [k, sDim, 1]

        outputs:
        --------
            - collision mask, tf.tensor. shape = [k, 1, 1]
    '''
    def collide(self, state):
        pos = state[:, :3]
        k = tf.shape(state)[0]
        # First check if state is inbetween the two points.
        f1 = pos - self.p1
        f2 = pos - self.p2
        top = tf.reduce_sum(tf.multiply(self.e, f1), axis=1)
        bot = tf.reduce_sum(tf.multiply(self.e, f2), axis=1)

        e = tf.broadcast_to(self.e, (k, 3, 1))
        cross = tf.linalg.cross(f1[..., 0], e[..., 0])
        cross_norm = tf.linalg.norm(cross[..., None], axis=1)
        d = cross_norm / tf.linalg.norm(e, axis=1)
        mask = tf.logical_and(tf.logical_and(top >= 0., bot <= 0.), d <= self.r)
        return mask[..., None]

    def get_pose(self):
        pc = (self.p1  + (self.e/2)).numpy()
        vec = self.e/tf.linalg.norm(self.e)
        q = tfgt.quaternion.between_two_vectors_3d(self.n, tf.squeeze(vec, axis=-1))
        return pc, q

    def get_param(self):
        h = tf.linalg.norm(self.e)
        return h, self.r


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
    def state_cost(self, scope, state, t):
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
