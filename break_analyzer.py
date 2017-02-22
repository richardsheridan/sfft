import numpy as np
from os import path
from break_locator import load_breaks
from fiber_locator import load_stab_img
from cvutil import sobel_filter, arg_min_max

from util import quadratic_subpixel_extremum_2d, get_folder, find_zero_crossings, quadratic_subpixel_extremum_1d, PIXEL_SIZE_X


def find_gap_edges(break_image):
    filtered_image = sobel_filter(break_image, 1, 0)
    left_edge, right_edge = arg_min_max(filtered_image)
    right_edge = quadratic_subpixel_extremum_2d(filtered_image, right_edge)
    left_edge = quadratic_subpixel_extremum_2d(filtered_image, left_edge)
    return left_edge, right_edge


def find_fiber_sides(break_image, left_x, right_x):
    profile = (np.sum(break_image[:, :round(left_x)], axis=1, dtype=np.int64) +
               np.sum(break_image[:, round(right_x) + 1:], axis=1, dtype=np.int64))

    crossings = find_zero_crossings(np.gradient(profile), 'upward')
    minima_values = profile[crossings]
    minima_indices = np.where(crossings)[0]
    sorted_minima_indices = minima_indices[np.argsort(minima_values)]
    top_side, bottom_side = sorted_minima_indices[:2]

    top_side = quadratic_subpixel_extremum_1d(profile, top_side)
    bottom_side = quadratic_subpixel_extremum_1d(profile, bottom_side)

    return top_side, bottom_side


def select_break_image(image, break_centroid, width=.0062371):
    y, x = break_centroid
    rows, cols = image.shape
    width = int(cols * width)
    r, c = int(y * (rows - 1)), int(x * (cols - 1))
    r0, r1 = max(r - width - 1, 0), min(r + width + 1, rows)
    c0, c1 = max(c - width - 1, 0), min(c + width + 1, cols)
    break_image = image[r0:r1, c0:c1]
    return break_image


def analyze_breaks(directory, stabilize_args=()):
    names_breaks = load_breaks(directory, False)
    for name, (break_y, break_x) in names_breaks:
        print('\n' + name + '\n')
        image_path = path.join(directory, name + '.tif')
        image = load_stab_img(image_path, *stabilize_args)
        break_y, break_x = np.array(break_y), np.array(break_x)
        sortind = np.argsort(break_x)
        for break_centroid in zip(break_y[sortind], break_x[sortind]):
            print('Centroid: ', break_centroid)
            break_image = select_break_image(image, break_centroid)
            # if '27' in name:
            #     import matplotlib.pyplot as plt
            #     plt.imshow(break_image, cmap='gray')
            #     plt.show(1)
            (_, left_edge), (_, right_edge) = find_gap_edges(break_image)
            print('Left edge: ', left_edge)
            print('Right edge: ', right_edge)
            top_side, bottom_side = find_fiber_sides(break_image, left_edge, right_edge)
            print('Top side: ', top_side)
            print('Bottom side: ', bottom_side)
            gap_width = right_edge - left_edge
            fiber_diameter = top_side - bottom_side
            print('Gap width: ', gap_width)
            print('Fiber diameter', fiber_diameter)


if __name__ == '__main__':
    analyze_breaks(get_folder())
