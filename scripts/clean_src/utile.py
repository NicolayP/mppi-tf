import tensorflow as tf
import numpy as np

dtype = tf.float32
npdtype = np.float32

def push_to_numpy(array, entry):
    tmp = array[1:]
    return np.concatenate([tmp, entry[None]], axis=0)

def push_to_tensor(tensor, entry):
    tmp = tensor[1:]
    return tf.concat([tmp, entry[None]], axis=0)