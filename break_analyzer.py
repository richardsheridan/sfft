import numpy as np
from os import path
from break_locator import load_breaks
from cvutil import sobel_filter, arg_min_max

from util import quadratic_subpixel_extremum


def find_gap_edges(break_image, subpixel=False):
    filtered_image = sobel_filter(break_image, 1, 0)
    right_edge, left_edge = arg_min_max(filtered_image)

    if subpixel:
        right_edge = quadratic_subpixel_extremum(filtered_image, right_edge)
        left_edge = quadratic_subpixel_extremum(filtered_image, left_edge)

    return left_edge, right_edge


def find_fiber_sides(break_image, left_edge, right_edge):
    left_y, left_x = left_edge
    right_y, right_x = right_edge

    profile = (np.sum(break_image[:, :left_x], axis=1, dtype=np.uint64) +
               np.sum(break_image[:, right_x + 1:], axis=1, dtype=np.uint64))

    y_mid = int(((left_y + right_y ) / 2.0)+0.5)
    top_side = np.argmin(profile[:y_mid])
    bottom_side = np.argmin(profile[y_mid:]) + y_mid

    return top_side, bottom_side


def select_break_image(image, break_centroid, width=.0062371):
    y, x = break_centroid
    rows, cols = image.shape
    width = int(rows * width)
    r, c = int(y * (rows - 1)), int(x * (cols - 1))
    r0, r1 = max(r - width, 0), min(r + width + 1, rows)
    c0, c1 = max(c - width, 0), min(c + width + 1, cols)
    break_image = image[r0:r1, c0:c1]
    return break_image


def analyze_breaks(directory, stabilize_args=()):
    names_breaks = load_breaks(directory, False)
    for name, (break_y, break_x) in names_breaks:
        image_path = path.join(directory, )
        from fiber_locator import load_stab_img
        image = load_stab_img(image_path, *stabilize_args)
        break_y, break_x = np.array(break_y), np.array(break_x)
        sortind = np.argsort(break_x)
        for break_centroid in zip(break_y[sortind], break_x[sortind]):
            break_image = select_break_image(image, break_centroid)
            left_edge, right_edge = find_gap_edges(break_image)
            top_side, bottom_side = find_fiber_sides(break_image, left_edge, right_edge)
            gap_width = right_edge[1] - left_edge[1]
            fiber_diameter = bottom_side - top_side
