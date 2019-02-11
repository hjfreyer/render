# Import libraries for simulation
import tensorflow as tf
import collections
import numpy as np

# Imports for visualization
import PIL.Image
from io import BytesIO
import matplotlib.pyplot as plt

def surface(p):
    #p = p - tf.expand_dims(tf.expand_dims(np.array([5.0, 0, 0], dtype=np.float32), 1), 1)
    p = p - match_rank(np.array([5.0, 0, 0], dtype=np.float32), p)
    return tf.norm(p, axis=0) -1

def match_rank(a, b):
    while len(a.shape) < len(b.shape):
        a = tf.expand_dims(a, len(a.shape))
    return a

def sphere_distance(center, radius):
    def distance(pts):
        distance_from_center = pts - match_rank(center, pts)
        return tf.norm(distance_from_center, axis=0) - radius
    return distance

def invert(dist):
    def d(pts):
        return -dist(pts)
    return d

def pt(x, y, z):
    return np.array([x, y, z], dtype=np.float32)


#camera = np.array([0, 0, 0], dtype=np.float32)
focal_length = 1.0
pixels2d = np.stack(np.mgrid[-1:1:0.001, -1:1:0.001])
pixels3d = np.concatenate([np.full((1,)+ pixels2d.shape[1:], focal_length), pixels2d])
rays = pixels3d
#
# rays = np.reshape(pixels3d, (3, -1))
# rays = np.swapaxes(rays, 0, 1)

Surface = collections.namedtuple('Surface', 'distance color')

surfaces = [
    Surface(distance=sphere_distance(pt(5, 0, 0), 1),
            color=pt(1, 0, 0)),
    Surface(distance=invert(sphere_distance(pt(0, 0, 0), 50)),
            color=pt(0, 0, 1)),
]
surface_colors = np.stack([s.color for s in surfaces])

Surface = collections.namedtuple('Surface', 'distance')

def march_op(prop, dist):
    return tf.group(prop.assign_add(dist))

with tf.Session() as sess:
    prop = tf.Variable(np.zeros(rays.shape[1:]), dtype=np.float32)

    ray_ends = rays * tf.expand_dims(prop, 0)
    surface_dists = tf.stack([s.distance(ray_ends) for s in surfaces])
    converged = surface_dists < 3
    point_converged = tf.math.reduce_any(converged, axis=0)
    surface_min_dist = tf.math.reduce_min(surface_dists, axis=0)

    #opt = tf.train.GradientDescentOptimizer(learning_rate=0.1)
#    opt_op = opt.minimize(surface_dist**2, var_list=[prop])
    opt_op = march_op(prop, surface_min_dist)

    tf.global_variables_initializer().run()
        #PROP, RES = sess.run([point_converged, surface_min_dist])
        #RES = np.minimum(RES, 1.0)
        #plt.imshow(RES)
        #plt.imshow(PROP)
        #plt.show()
    all_converged = tf.math.reduce_all(point_converged)
    ball_converged = False
    idx = 0
    while not ball_converged:
        _, ball_converged = sess.run([opt_op, all_converged])
        print(idx)
        idx+=1
    touched_surface = tf.argmin(surface_dists, axis=0)
    color = tf.gather(surface_colors, touched_surface)
    c = sess.run(color)
    plt.imshow(c)
    plt.show()






#
# def DisplayFractal(a, fmt='jpeg'):
#   """Display an array of iteration counts as a
#      colorful picture of a fractal."""
#   a_cyclic = (6.28*a/20.0).reshape(list(a.shape)+[1])
#   img = np.concatenate([10+20*np.cos(a_cyclic),
#                         30+50*np.sin(a_cyclic),
#                         155-80*np.cos(a_cyclic)], 2)
#   img[a==a.max()] = 0
#   a = img
#   a = np.uint8(np.clip(a, 0, 255))
#   plt.imshow(a)
#
#
# sess = tf.InteractiveSession()
# Y, X = np.mgrid[-1.3:1.3:0.005, -2:1:0.005]
# Z = X+1j*Y
# xs = tf.constant(Z.astype(np.complex64))
# zs = tf.Variable(xs)
# ns = tf.Variable(tf.zeros_like(xs, tf.float32))
#
# tf.global_variables_initializer().run()
#
# # Compute the new values of z: z^2 + x
# zs_ = zs*zs + xs
#
# # Have we diverged with this new value?
# not_diverged = tf.abs(zs_) < 4
#
# # Operation to update the zs and the iteration count.
# #
# # Note: We keep computing zs after they diverge! This
# #       is very wasteful! There are better, if a little
# #       less simple, ways to do this.
# #
# step = tf.group(
#   zs.assign(zs_),
#   ns.assign_add(tf.cast(not_diverged, tf.float32))
#   )
#
# for i in range(200): step.run()
