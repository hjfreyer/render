# Import libraries for simulation
import tensorflow as tf
import collections
import numpy as np

# Imports for visualization
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

import geom

def pt(x, y, z):
    return np.array([x, y, z], dtype=np.float64)

def direction(x, y, z):
    return tf.math.l2_normalize(tf.constant([x, y, z], dtype=np.float64))


#camera = np.array([0, 0, 0], dtype=np.float64)
WIDTH = 100
HEIGHT = 100
CONVERGED = 0.001
focal_length = 1.0

# Camera starts at the origin pointing down the x axis in the positive
# direction.
y_points = tf.range(-1,  1,  2.0/WIDTH)
z_points = tf.range( 1, -1, -2.0/HEIGHT)
y_coords, z_coords = tf.meshgrid(y_points, z_points)
x_coords = tf.fill(y_coords.shape, focal_length)

pixels3d = tf.cast(tf.stack([x_coords, y_coords, z_coords]), dtype=tf.float64)
rays = tf.reshape(pixels3d, (3, WIDTH*HEIGHT))
rays = tf.math.l2_normalize(rays, axis=0)
all_rays = tf.concat([tf.zeros_like(rays), rays], axis=0)
all_rays += tf.expand_dims(tf.constant([0, 0, 1, 0, 0, 0], dtype=tf.float64), axis=1)
#
# rays = np.reshape(pixels3d, (3, -1))
# rays = np.swapaxes(rays, 0, 1)

Surface = collections.namedtuple('Surface', 'geometry color')
Emitter = collections.namedtuple('Emitter', 'source color')

surfaces = [
    Surface(geometry=geom.Sphere(pt(5, 0, 0), 1),
            color=pt(1, 0, 1)),
    Surface(geometry=geom.Sphere(pt(3, 2, 0), 0.5),
             color=pt(1, 1, 0)),
    Surface(geometry=geom.Intersection([geom.Plane(pt(10, 0, 0), direction(-1, -1, 0)),
                                    geom.Plane(pt(10, 0, 0), direction(0, 0, 1))]),
           #distance=geom.plane(pt(0, 0, -0.1), direction(0, 0, 1)),
           color=pt(0.7, 0.7, 1)),

    Surface(geometry=geom.Box(pt(4, -2, 0), pt(3, 1, 1)),
          #distance=geom.plane(pt(0, 0, -0.1), direction(0, 0, 1)),
          color=pt(0, 1, 1)),
    Surface(geometry=geom.Inverse(geom.Sphere(pt(0, 0, 0), 100)),
            color=pt(0, 0, 1)),
 ]
surface_colors = np.stack([s.color for s in surfaces])
emitters = [
    Emitter(source=pt(2,2,2), color=pt(1,1,1)),
    Emitter(source=pt(2,-2,-2), color=pt(0.25,0,0)),
]

def size_splits(tensor, split_size, axis=0):
    """Splits the tensor according to chunks of split_sizes.

    Arguments:
        tensor (Tensor): tensor to split.
        split_sizes (list(int)): sizes of chunks
        dim (int): dimension along which to split the tensor.
    """
    res = []
    for start in range(0, tensor.shape[axis], split_size):
        slice_start = [0,] * len(tensor.shape)
        slice_size = list(tensor.shape)
        slice_start[axis] = start
        slice_size[axis] = split_size
        res.append(tf.slice(tensor, slice_start, slice_size))
    return res

def march_op(prop, dist):
    converged = dist < CONVERGED
    delta = tf.where(converged, tf.zeros_like(dist), dist)
    return tf.group(prop.assign_add(delta))


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
    mask = tf.Variable(tf.fill(rays.shape[1:], True))
    all_prop = tf.Variable(np.zeros(rays.shape[1]), dtype=np.float64)

    unconverged = tf.where(mask)[:, 0]

    rays = tf.gather(all_rays, unconverged, axis=1)
    prop = tf.gather(all_prop, unconverged, axis=0)

    ray_ends = rays[0:3] + rays[3:6] * tf.expand_dims(prop, 0)

    surface_dists = tf.stack([s.geometry.distance(ray_ends) for s in surfaces])

    closest_surface_dist = tf.math.reduce_min(surface_dists, axis=0)

    remaining = tf.math.count_nonzero(mask)

    march_op = tf.group(
        tf.scatter_add(all_prop, unconverged, closest_surface_dist),
        tf.scatter_update(mask, unconverged, closest_surface_dist > CONVERGED),
    )

    print('Graph built')
    tf.global_variables_initializer().run()

    idx = 0
    while True:
        _ = sess.run([march_op])
        r_remaining = sess.run(remaining)
        idx += 1
        print('%d: %d' %(idx, r_remaining))
        if r_remaining == 0:
            break

    sess.run(mask.initializer)

    closest_surface_arg = tf.argmin(surface_dists, axis=0)
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
    color = tf.transpose(color, perm=[1, 2, 0])
    color = tf.cast(256*color, dtype=tf.uint8)
    #color = tf.reshape(tf.cast(closest_surface_arg, tf.float64), (WIDTH, HEIGHT))

    c = sess.run(color)
    try:
        Image.fromarray(c).save('/tmp/render.png')
    except Exception as e:
        print(e)
    plt.imshow(c)
    plt.show()



#     closest_surface_arg = tf.argmin(surface_dists, axis=0)
#     all_converged = tf.math.reduce_all(closest_surface_dist < CONVERGED)
#
#     opt_op = march_op(prop, closest_surface_dist)
#

#     #color = tf.reshape(tf.cast(closest_surface_arg, tf.float64), (WIDTH, HEIGHT))
#     #color = tf.reshape(prop, (WIDTH, HEIGHT))
#     #color = tf.reshape(normals[0,:], (WIDTH, HEIGHT))
#
#     xx = tf.reshape(surface_dists, (-1, WIDTH, HEIGHT))[:, 107, 184]
#

#
#
#
#
# #    color = normals[0, :]
# #    color = tf.reshape(color, (WIDTH, HEIGHT))
#
#
#     # touched_surface = tf.argmin(surface_dists, axis=0)
#     # color = tf.gather(surface_colors, touched_surface
#     print(sess.run(xx))
#     c = sess.run(color)
#     plt.imshow(c)
#     plt.show()
#
#     print('Get normals')
