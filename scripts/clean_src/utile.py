import tensorflow as tf
import numpy as np

dtype = tf.float32
npdtype = np.float32

def append_to_tensor(tensor, entry):
    return tf.concat([tensor, entry], axis=1)

def push_to_numpy(array, entry):
    tmp = array[1:]
    return np.concatenate([tmp, entry[None]], axis=0)

def push_to_tensor(tensor, entry):
    tmp = tensor[1:]
    return tf.concat([tmp, entry[None]], axis=0)

def assert_shape(array, shape):
    ashape = array.shape
    if len(ashape) != len(shape):
        return False
    for i, j in zip(ashape, shape):
        if j != -1 and i != j:
            return False
    return True
