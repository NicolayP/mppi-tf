import tensorflow as tf
import numpy as np

control_items= {}
control_step = 0


def addItem(**dict):
    global control_items
    for item in dict:
        if item in control_items.keys():
            control_items[item] = tf.add(control_items[item], dict[item])
        else:
            control_items[item] = dict[item]

def log_control(writer, control_items):
    # TODO: Log position error, goal distance, predicted cost.
    # TODO: generate gifs if asked.
    global control_step
    cost = control_items["cost"]
    k = cost.shape[0]
    avg_cost = np.mean(cost)
    best_id = np.argmin(cost)
    best_cost = cost[best_id, 0, 0]

    with writer.as_default():
        for key in control_items:
            if key=="cost":
                tf.summary.scalar("best_cost", best_cost, step=control_step)
                tf.summary.scalar("average_cost", avg_cost, step=control_step)

            elif key=="state_cost":
                state_cost = control_items[key]
                avg_state = np.mean(state_cost)
                best_state = state_cost[best_id, 0, 0]
                tf.summary.scalar("best_state", best_state, step=control_step)
                tf.summary.scalar("avg_state", avg_state, step=control_step)

            elif key=="action_cost":
                action_cost = control_items[key]
                avg_act = np.mean(action_cost)
                best_act = action_cost[best_id, 0, 0]
                tf.summary.scalar("avg_act", avg_act, step=control_step)
                tf.summary.scalar("best_act", best_act, step=control_step)

            elif key=="position_cost":
                position_cost = control_items[key]
                avg_pos = np.mean(position_cost)
                best_pos = position_cost[best_id, 0, 0]
                tf.summary.scalar("avg_pos", avg_pos, step=control_step)
                tf.summary.scalar("best_pos", best_pos, step=control_step)

            elif key=="speed_cost":
                speed_cost = control_items[key]
                avg_speed = np.mean(speed_cost)
                best_speed = speed_cost[best_id, 0, 0]
                tf.summary.scalar("avg_speed", avg_speed, step=control_step)
                tf.summary.scalar("best_speed", best_speed, step=control_step)

            elif key=="nabla":
                nabla = control_items[key]
                # TODO FIX THIS
                tf.summary.scalar("nabla_percent", nabla[0,0]/k, step=control_step)

            elif key=="norm_cost":
                norm_cost = control_items[key]
                tf.summary.histogram("norm_cost", norm_cost, step=control_step)

            elif key=="weighted_noises":
                weighted_noises = control_items[key]
                tf.summary.histogram("Controller_weights", weights, step=control_step)

            elif key=="weights":
                weights = control_items[key]
                tf.summary.histogram("Controller_weights", weights, step=control_step)

    # reset for next summary
    control_items={}
    control_step += 1
