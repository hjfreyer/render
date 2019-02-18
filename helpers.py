
import tensorflow as tf
import math

def _cc(a, b):
    return math.cos(a) * math.cos(b)

def _sc(a, b):
    return math.sin(a) * math.cos(b)

def _cs(a, b):
    return math.cos(a) * math.sin(b)

def _css(a, b, c):
    return math.cos(a) * math.sin(b) * math.sin(c)

def _sss(a, b, c):
    return math.sin(a) * math.sin(b) * math.sin(c)

def transform_matrix(yaw, pitch, roll, dx, dy, dz):
    return tf.constant([
        [_cc(yaw, pitch) , _css(yaw, pitch, roll) - _sc(yaw, roll), _csc(yaw, pitch, roll) + _ss(yaw, roll), dx],
        [_sc(yaw, pitch) , _sss(yaw, pitch, roll) + _cc(yaw, roll), _ssc(yaw, pitch, roll) - _cs(yaw, roll), dy],
        [-math.sin(pitch), _cs(pitch, roll)                       , _cc(pitch, roll)                       , dz],
        [               0,                                       0,                                       0,  0],
    ], dtype=tf.float32)
