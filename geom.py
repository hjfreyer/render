"""Library for implict surface functions."""

import tensorflow as tf
import collections

def pt(x, y, z):
    return np.array([x, y, z], dtype=np.float64)

def direction(x, y, z):
    return tf.math.l2_normalize(tf.constant([x, y, z], dtype=tf.float64))

class Sphere:
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

    def distance(self, pts):
        distance_from_center = pts - tf.expand_dims(self.center, 0)
        return tf.norm(distance_from_center, axis=1) - self.radius

class Inverse:
    def __init__(self, geometry):
        self.geometry = geometry

    def distance(self, pts):
        return -self.geometry.distance(pts)

class Plane:
    def __init__(self, point, normal):
        self.point = point
        self.normal = normal

    def distance(self, pts):
        to_ref_point = pts - tf.expand_dims(self.point, axis=0)
        along_normal = tf.reduce_sum(
            to_ref_point * tf.expand_dims(self.normal, axis=0),
            axis=1)
        return along_normal

class Intersection:
    def __init__(self, surfaces):
        self.surfaces = surfaces

    def distance(self, pts):
        stacked = tf.stack([s.distance(pts) for s in self.surfaces], axis=1)
        return tf.reduce_max(stacked, axis=1)

class Box:
    def __init__(self, p1, p2):
        small = tf.minimum(p1, p2)
        big = tf.maximum(p1, p2)

        top = Plane(big, direction(0, 0, 1))
        bottom = Plane(small, direction(0, 0, -1))
        back = Plane(big, direction(1, 0, 0))
        front = Plane(small, direction(-1, 0, 0))
        left = Plane(big, direction(0, 1, 0))
        right = Plane(small, direction(0, -1, 0))

        self.geom = Intersection([front, back, top, bottom, left, right])

    def distance(self, pts):
        return self.geom.distance(pts)
