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


def find_fiber_sides(break_image):
    profile = np.sum(break_image, axis=1, dtype=np.int64)

    crossings = find_zero_crossings(np.gradient(profile), 'upward')
    minima_values = profile[crossings]
    minima_indices = np.where(crossings)[0]

    sorted_minima_indices = minima_indices[np.argsort(minima_values)]
    top_side, bottom_side = sorted(sorted_minima_indices[:2])

    # TODO this seems better in principle but can crash, fix it up?
    # i = np.argmin(minima_values)
    # j = i+1 if minima_values[i+1] < minima_values[i-1] else i-1
    # top_side, bottom_side = sorted(minima_indices[[i,j]])

    # import matplotlib.pyplot as plt
    # a=plt.subplot(2,1,1,)
    # plt.cla()
    # plt.imshow(np.rot90(break_image,1), aspect='auto')
    # plt.subplot(2,1,2, sharex=a)
    # plt.cla()
    # plt.plot(profile)
    # plt.plot(minima_indices,minima_values,'rx')
    # plt.plot([top_side,bottom_side],profile[[top_side,bottom_side]],'bo')
    # plt.show(0)
    # plt.pause(.0001)

    top_side = quadratic_subpixel_extremum_1d(profile, top_side)
    bottom_side = quadratic_subpixel_extremum_1d(profile, bottom_side)
    return top_side, bottom_side


def select_break_image(image, break_centroid, width=0.00311855):
    y, x = break_centroid
    rows, cols = image.shape
    width = int(cols * width)
    r, c = int(y * (rows - 1)), int(x * (cols - 1))
    r0, r1 = max(r - width - 1, 0), min(r + width + 1, rows)
    c0, c1 = max(c - width - 1, 0), min(c + width + 1, cols)
    break_image = image[r0:r1, c0:c1]
    return break_image


def analyze_break(break_image, verbose=False, animate=False):
    top_side, bottom_side = find_fiber_sides(break_image)
    b = int(round(bottom_side))
    t = int(round(top_side))
    (left_edge_y, left_edge), (right_edge_y, right_edge) = find_gap_edges(break_image[t:b, :])
    gap_width = right_edge - left_edge
    fiber_diameter = bottom_side - top_side

    if verbose:
        print('Top side: ', top_side)
        print('Bottom side: ', bottom_side)
        print('Left edge: ', left_edge)
        print('Right edge: ', right_edge)
        print('Gap width: ', gap_width * PIXEL_SIZE_X, ' um')
        print('Fiber diameter', fiber_diameter * PIXEL_SIZE_X, ' um')

    if animate:
        import matplotlib.pyplot as plt
        plt.cla()
        plt.imshow(break_image, cmap='gray')
        plt.plot([left_edge, right_edge], [left_edge_y + top_side, right_edge_y + top_side], 'rx')
        plt.axvline(left_edge, linestyle=':', color='red')
        plt.axvline(right_edge, linestyle=':', color='red')
        plt.axhline(top_side, linestyle=':', color='green')
        plt.axhline(bottom_side, linestyle=':', color='green')
        plt.show(0)
        plt.pause(0.00001)

    return gap_width, fiber_diameter


def sorted_centroids(break_y, break_x):
    break_y, break_x = np.array(break_y), np.array(break_x)
    sortind = np.argsort(break_x)
    return list(zip(break_y[sortind], break_x[sortind]))


def analyze_breaks(directory, stabilize_args=()):
    for name, (break_y, break_x) in load_breaks(directory, 'absolute'):
        print('\n' + name + '\n')
        image_path = path.join(directory, name + '.tif')
        image = load_stab_img(image_path, *stabilize_args)
        for break_centroid in zip(break_y, break_x):
            print('Centroid: ', break_centroid)
            break_image = select_break_image(image, break_centroid)
            gap_width, fiber_diameter = analyze_break(break_image)


if __name__ == '__main__':
    folder = (get_folder())
    analyze_breaks(folder)

    # import cProfile as prof
    # prof.run('analyze_breaks(folder)',sort='cumtime')
