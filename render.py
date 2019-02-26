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
WIDTH = 500
HEIGHT = 500
CONVERGED = 0.001
focal_length = 1.0

# Camera starts at the origin pointing down the x axis in the positive
# direction.
y_points = tf.range(-1,  1,  2.0 / WIDTH, dtype=tf.float64)
z_points = tf.range(1, -1, -2.0 / HEIGHT, dtype=tf.float64)
y_coords, z_coords = tf.meshgrid(y_points, z_points)
x_coords = tf.fill(y_coords.shape, tf.constant(focal_length, tf.float64))

pixels3d = tf.stack([x_coords, y_coords, z_coords], axis=2)
pixels3d = tf.reshape(pixels3d, (WIDTH * HEIGHT, 3))
ray_dirs = tf.math.l2_normalize(pixels3d, axis=1)
ray_coords = tf.zeros_like(
    ray_dirs) + tf.expand_dims(tf.constant([-3, 0, 1], dtype=tf.float64), axis=0)


Ray = collections.namedtuple('Ray', 'root direction')
Surface = collections.namedtuple('Surface', 'geometry color')
Emitter = collections.namedtuple('Emitter', 'source color')
Scene = collections.namedtuple('Scene', 'surfaces emitters')

rays = Ray(root=ray_coords, direction=ray_dirs)


def ray_gather_nd(ray, *args, **kwargs):
    root = tf.gather_nd(ray.root, *args, **kwargs)
    dir = tf.gather_nd(ray.direction, *args, **kwargs)
    return Ray(root, dir)


SCENE = Scene(
    surfaces=[
        Surface(geometry=geom.Sphere(pt(5, 0, 0), 1),
                color=pt(1, 0, 1)),
        Surface(geometry=geom.Sphere(pt(2.5, 2.5, 2.5), 0.1),
                color=pt(1, 1, 0)),
        Surface(geometry=geom.Intersection([geom.Plane(pt(10, 0, 0), direction(-1, -1, 0)),
                                            geom.Plane(pt(10, 0, 0), direction(0, 0, 1))]),
                #distance=geom.plane(pt(0, 0, -0.1), direction(0, 0, 1)),
                color=pt(0.7, 0.7, 1)),

        # Surface(geometry=geom.Box(pt(4, -2, 0), pt(3, 1, 1)),
        #       #distance=geom.plane(pt(0, 0, -0.1), direction(0, 0, 1)),
        #       color=pt(0, 1, 1)),
        Surface(geometry=geom.Inverse(geom.Sphere(pt(0, 0, 0), 100)),
                color=pt(0, 0, 1)),
    ],
    emitters=[
        Emitter(source=pt(0, 5, 5), color=pt(1, 1, 1)),
        Emitter(source=pt(2, -2, -2), color=pt(0.25, 0, 0)),
    ])


def white_balance(colors):
    """Takes (N, 3), does white balance"""
    max_per_channel = tf.math.reduce_max(colors, axis=0, keepdims=True)
    colors = colors / max_per_channel

    # In case of really small values of max_per_channel causing problems.
    colors = tf.minimum(colors, 1)
    return colors


def gamma_correct(colors):
    # Formula from https://mitchellkember.com/blog/post/ray-tracer/
    linear_regime = 12.82 * colors
    exp_regime = 1.055 * colors**(1 / 2.4) - 0.055
    return tf.where(color <= 0.0031308, linear_regime, exp_regime)


def get_closest_surface_distance(rays, lengths):
    ray_ends = rays.root + rays.direction * tf.expand_dims(lengths, 1)
    surface_dists = tf.stack([s.geometry.distance(ray_ends)
                              for s in SCENE.surfaces], axis=1)
    return tf.math.reduce_min(surface_dists, axis=1), tf.argmin(surface_dists, axis=1)


def propagate_rays(surfaces, rays):
    def cond(converged, lengths):
        return tf.math.reduce_any(tf.logical_not(converged))

    def body(converged, lengths):
        unconverged_idxes = tf.where(tf.math.logical_not(converged))
        unconverged_rays = ray_gather_nd(rays, unconverged_idxes)
        unconverged_lengths = tf.gather_nd(lengths, unconverged_idxes)

        ray_ends = unconverged_rays.root + unconverged_rays.direction * \
            tf.expand_dims(unconverged_lengths, 1)

        surface_dists = tf.stack([s.geometry.distance(ray_ends)
                                  for s in SCENE.surfaces], axis=1)
        min_surface_dists = tf.math.reduce_min(surface_dists, axis=1)

        converged = tf.math.logical_or(
            converged,
            tf.scatter_nd(unconverged_idxes, min_surface_dists < CONVERGED, converged.shape))

        lengths += tf.scatter_nd(unconverged_idxes,
                                 min_surface_dists, lengths.shape)

        return converged, lengths

    _, lengths = tf.while_loop(cond, body, (
        tf.zeros(rays.root.shape[:-1], dtype=tf.bool),
        tf.zeros(rays.root.shape[:-1], dtype=tf.float64),
    ))
    return lengths


def get_clearance(surfaces, points):
    def cond(too_close, points):
        return tf.math.reduce_any(too_close)

    def body(too_close, points):
        too_close_idxes = tf.where(too_close)
        too_close_points = tf.gather_nd(points, too_close_idxes)

        surface_dists = tf.stack([s.geometry.distance(too_close_points)
                                  for s in SCENE.surfaces], axis=1)
        min_surface_dists = tf.math.reduce_min(surface_dists, axis=1)

        gradients, = tf.gradients(min_surface_dists, too_close_points)

        # This isn't quite right. Also not quite right in propagate. Each goes one loop too far.
        points_delta = gradients * CONVERGED
        new_too_close = min_surface_dists < CONVERGED

        points += tf.scatter_nd(too_close_idxes, points_delta, points.shape)
        too_close = tf.math.logical_and(
            too_close,
            tf.scatter_nd(too_close_idxes, new_too_close, too_close.shape))

        return too_close, points
    _, points = tf.while_loop(cond, body, (
        tf.fill(points.shape[:1], True),
        points
    ))
    return points


with tf.Session() as sess:
    emitters_sources = tf.stack([e.source for e in SCENE.emitters], axis=0)
    emitters_colors = tf.stack([e.color for e in SCENE.emitters], axis=0)

    rays = rays._replace(root=get_clearance(SCENE.surfaces, rays.root))
    lengths = propagate_rays(SCENE.surfaces, rays)

    reflection_points = rays.root + rays.direction * tf.expand_dims(lengths, 1)
    reflection_points = get_clearance(SCENE.surfaces, reflection_points)

    reflection_points_to_emitters_vector = (tf.expand_dims(emitters_sources, axis=0)
                                            - tf.expand_dims(reflection_points, axis=1))
    reflection_points_to_emitters_dir = tf.math.l2_normalize(
        reflection_points_to_emitters_vector, axis=2)

    reflection_points = tf.broadcast_to(tf.expand_dims(
        reflection_points, 1), tf.shape(reflection_points_to_emitters_dir))

    reflected_rays = Ray(root=reflection_points,
                         direction=reflection_points_to_emitters_dir)
    reflected_lengths = propagate_rays(SCENE.surfaces, reflected_rays)
    reflected_lengths = tf.reshape(
        reflected_lengths, (-1, len(SCENE.emitters)))

    distance_to_emitters = tf.norm(
        reflection_points_to_emitters_vector, axis=2)
    unobstructed = distance_to_emitters < reflected_lengths

    ray_ends = rays.root + rays.direction * tf.expand_dims(lengths, 1)
    surface_dists = tf.stack([s.geometry.distance(ray_ends)
                              for s in SCENE.surfaces], axis=1)
    closest_surface_dist = tf.math.reduce_min(surface_dists, axis=1)
    closest_surface_arg = tf.argmin(surface_dists, axis=1)

    normals, = tf.gradients(closest_surface_dist, ray_ends)
    normals = tf.math.l2_normalize(normals, axis=1)

    emitters_dot_products = tf.reduce_sum(
        reflection_points_to_emitters_dir * tf.expand_dims(normals, axis=1),
        axis=2)
    emitters_dot_products = tf.maximum(emitters_dot_products, 0)
    emitters_dot_products *= tf.cast(unobstructed, tf.float64)

    emitters_colors_per_point = tf.expand_dims(
        emitters_colors, axis=0) * tf.expand_dims(emitters_dot_products, axis=2)
    surface_colors = np.stack([s.color for s in SCENE.surfaces])
    surface_color = tf.gather(surface_colors, closest_surface_arg)
    reflected_color_per_emitter = emitters_colors_per_point * \
        tf.expand_dims(surface_color, axis=1)
    color = tf.reduce_sum(reflected_color_per_emitter, axis=1)

    color = white_balance(color)
    color = gamma_correct(color)
    color = tf.reshape(color, (WIDTH, HEIGHT, 3))
    color = tf.cast(256 * color, dtype=tf.uint8)
    #color = tf.reshape(tf.cast(closest_surface_arg, tf.float64), (WIDTH, HEIGHT))

    print('Graph built')
    tf.global_variables_initializer().run()
    c = sess.run(color)

    try:
        Image.fromarray(c).save('/tmp/render.png')
    except Exception as e:
        print(e)
    plt.imshow(c)
    plt.show()
