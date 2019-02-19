# Import libraries for simulation
import tensorflow as tf
import collections
import numpy as np

# Imports for visualization
from PIL import Image
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

def direction(x, y, z):
    return tf.math.l2_normalize(tf.constant([x, y, z], dtype=np.float32))


#camera = np.array([0, 0, 0], dtype=np.float32)
WIDTH = 1000
HEIGHT = 1000
focal_length = 1.0

# Camera starts at the origin pointing down the x axis in the positive
# direction.
y_points = tf.range(-1,  1,  2.0/WIDTH)
z_points = tf.range( 1, -1, -2.0/HEIGHT)
y_coords, z_coords = tf.meshgrid(y_points, z_points)
x_coords = tf.fill(y_coords.shape, focal_length)

pixels3d = tf.stack([x_coords, y_coords, z_coords])
rays = tf.reshape(pixels3d, (3, WIDTH*HEIGHT))
#
# rays = np.reshape(pixels3d, (3, -1))
# rays = np.swapaxes(rays, 0, 1)

Surface = collections.namedtuple('Surface', 'distance color')
Emitter = collections.namedtuple('Emitter', 'source color')

surfaces = [
    Surface(distance=sphere_distance(pt(5, 0, 0), 1),
            color=pt(1, 0, 1)),
    Surface(distance=sphere_distance(pt(3, 2, 0), 0.5),
            color=pt(1, 1, 0)),
    Surface(distance=invert(sphere_distance(pt(0, 0, 0), 20)),
            color=pt(0, 0, 1)),
]
surface_colors = np.stack([s.color for s in surfaces])
emitters = [
    Emitter(source=pt(2,2,2), color=pt(1,1,1)),
    Emitter(source=pt(2,-2,-2), color=pt(0.25,0,0)),
]

def march_op(prop, dist):
    return tf.group(prop.assign_add(dist))


def white_balance(colors):
    """Takes (3, N), does white balance"""
    max_per_channel = tf.math.reduce_max(colors, axis=1, keepdims=True)
    colors = colors / max_per_channel
    colors = tf.minimum(colors, 1)  # In case of really small values of
                                    # max_per_channel causing problems.
    return colors

def gamma_correct(colors):
    # Formula from https://mitchellkember.com/blog/post/ray-tracer/
    linear_regime = 12.82 * colors
    exp_regime = 1.055 * colors**(1/2.4) - 0.055
    return tf.where(color <= 0.0031308, linear_regime, exp_regime)


with tf.Session() as sess:
    prop = tf.Variable(np.zeros(rays.shape[1]), dtype=np.float32)

    ray_ends = rays * tf.expand_dims(prop, 0)
    surface_dists = tf.stack([s.distance(ray_ends) for s in surfaces])

    closest_surface_arg = tf.argmin(surface_dists, axis=0)
    closest_surface_dist = tf.math.reduce_min(surface_dists, axis=0)
    all_converged = tf.math.reduce_all(closest_surface_dist < 0.01)

    opt_op = march_op(prop, closest_surface_dist)

    normals, = tf.gradients(closest_surface_dist, ray_ends)
    normals = tf.math.l2_normalize(normals, axis=0)

    emitters_sources = tf.stack([e.source for e in emitters], axis=1)
    emitters_colors = tf.stack([e.color for e in emitters], axis=1)

    pts_to_emitters = (tf.expand_dims(emitters_sources, axis=2)
        - tf.expand_dims(ray_ends, axis=1))
    pts_to_emitters = tf.math.l2_normalize(pts_to_emitters, axis=0)

    emitters_dot_products = tf.reduce_sum(
        pts_to_emitters * tf.expand_dims(normals, axis=1),
        axis=0)
    emitters_dot_products = tf.maximum(emitters_dot_products, 0)

    emitters_colors_per_point = tf.expand_dims(
        emitters_colors, axis=2) * tf.expand_dims(emitters_dot_products, axis=0)
    surface_color = tf.transpose(tf.gather(surface_colors, closest_surface_arg))
    reflected_color_per_emitter = emitters_colors_per_point * tf.expand_dims(surface_color, axis=1)
    color = tf.reduce_sum(reflected_color_per_emitter, axis=1)

    color = white_balance(color)
    color = gamma_correct(color)
    color = tf.reshape(color, (3, WIDTH, HEIGHT))

    tf.global_variables_initializer().run()

    print('Graph built')

    idx = 0
    while True:
        print(idx)
        _, r_all_converged = sess.run([opt_op, all_converged])
        idx += 1
        if r_all_converged:
            break




#    color = normals[0, :]
#    color = tf.reshape(color, (WIDTH, HEIGHT))

    color = tf.transpose(color, perm=[1, 2, 0])
    color = tf.cast(256*color, dtype=tf.uint8)
    # touched_surface = tf.argmin(surface_dists, axis=0)
    # color = tf.gather(surface_colors, touched_surface
    c = sess.run(color)
    Image.fromarray(c).save('/tmp/render.png')
    plt.imshow(c)
    plt.show()

    print('Get normals')






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
