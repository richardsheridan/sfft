import json
from collections import namedtuple, OrderedDict
from os import path

import numpy as np
from .cvutil import sobel_filter, arg_min_max, make_pyramid
from .fiber_locator import load_stab_img
from .gui import GUIPage
from .util import quadratic_subpixel_extremum_2d, find_zero_crossings, quadratic_subpixel_extremum_1d, \
    PIXEL_SIZE_X, batch, basename_without_stab, get_files, dump

from sfft.break_locator import load_breaks

ANALYSIS_FILENAME = 'analysis.json'

class AnalyzerGUI(GUIPage):
    def __init__(self, image_paths, stabilize_args=(), fid_args=(), break_args=(), **kw):
        self.image_paths = image_paths
        self.breaks_dict = load_breaks(path.dirname(image_paths[0]), 'absolute')
        self.load_args = (stabilize_args, fid_args, break_args)
        self.nobreaks = False

        super().__init__(**kw)

    def __str__(self):
        return 'AnalyzerGUI'

    def create_layout(self):
        self.add_axes('image')
        self.add_axes('break', share='')
        self.add_button('save', self.execute_batch, label='Save batch')

        self.add_slider('frame_number', self.full_reload, valmin=0, valmax=len(self.image_paths) - 1,
                        valinit=0,
                        label='Frame Number', isparameter=False, forceint=True)
        self.add_slider('break_position', self.update_vision, valmin=0, valmax=1, valinit=0.5, label='Break Position',
                        isparameter=False)
        self.add_slider('width', self.update_vision, valmin=0.001, valmax=.009, valinit=0.00311855,
                        label='Window Width',
                        valfmt='    %.2e')

    @staticmethod
    def load_image_to_pyramid(image_path, stabilize_args, fid_args, break_args):
        image = load_stab_img(image_path, stabilize_args)
        pyramid = make_pyramid(image)
        return pyramid

    def recalculate_vision(self):
        image = self.pyramid[0]
        name = basename_without_stab(self.image_paths[self.slider_value('frame_number')])
        breaks = np.array(self.breaks_dict[name])
        if not len(breaks[1]):
            self.nobreaks = True
            return
        else:
            self.nobreaks = False
        i = np.argmin(abs(breaks[1] - self.slider_value('break_position')))
        self.centroid = centroid = breaks[:, i]
        print(centroid)
        self.break_image = select_break_image(image, centroid, width=self.slider_value('width'))
        self.edges = analyze_break(self.break_image, verbose=True, edge_output=True)

    def refresh_plot(self):
        image = self.pyramid[2]
        r, c = image.shape
        self.clear('image')
        # TODO: use extent=[0,real_width,0,real_height]
        self.imshow('image', image, extent=[0, 1, 1, 0])
        if self.nobreaks:
            self.clear('break')
            self.draw()
            return
        w = self.slider_value('width')
        h = w / (r - 1) * (c - 1)
        y, x = self.centroid
        left, right = x - w, x + w
        top, bottom = y - h, y + h  # y axis inverted in images!
        self.plot('image', [left, left, right, right, left], [bottom, top, top, bottom, bottom], 'r')

        self.clear('break')
        self.imshow('break', self.break_image, aspect='equal')  # , extent=[0, 1, 1, 0])
        e = self.edges
        self.plot('break', [e.left_edge, e.right_edge], [e.left_edge_y + e.top_side, e.right_edge_y + e.top_side], 'rx')
        self.vline('break', e.left_edge, linestyle=':', color='red')
        self.vline('break', e.right_edge, linestyle=':', color='red')
        self.hline('break', e.top_side, linestyle=':', color='green')
        self.hline('break', e.bottom_side, linestyle=':', color='green')

        self.draw()

    def execute_batch(self, *a, **kw):
        parameters = self.parameters
        analysis = batch(analyze_breaks, self.image_paths, self.breaks_dict, *parameters.values())
        save_analysis(parameters, analysis, self.image_paths)
        # print(analysis)

    def update_vision(self, *a, **kw):
        self.recalculate_vision()
        self.refresh_plot()

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


Edges = namedtuple('Edges', 'top_side, bottom_side, left_edge, right_edge, left_edge_y, right_edge_y')


def analyze_break(break_image, verbose=False, animate=False, edge_output=False):
    top_side, bottom_side = find_fiber_sides(break_image)
    b = int(round(bottom_side))
    t = int(round(top_side))
    (left_edge_y, left_edge), (right_edge_y, right_edge) = find_gap_edges(break_image[t:b, :])
    gap_width = (right_edge - left_edge) * PIXEL_SIZE_X
    fiber_diameter = (bottom_side - top_side) * PIXEL_SIZE_X

    if verbose:
        print('Top side: ', top_side)
        print('Bottom side: ', bottom_side)
        print('Left edge: ', left_edge)
        print('Right edge: ', right_edge)
        print('Gap width: ', gap_width, ' um')
        print('Fiber diameter', fiber_diameter, ' um')

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

    if edge_output:
        return Edges(top_side, bottom_side, left_edge, right_edge, left_edge_y, right_edge_y)

    return gap_width, fiber_diameter


def sorted_centroids(break_y, break_x):
    break_y, break_x = np.array(break_y), np.array(break_x)
    sortind = np.argsort(break_x)
    return list(zip(break_y[sortind], break_x[sortind]))


def analyze_breaks(image_path, breaks_dict, width):
    name = basename_without_stab(image_path)
    print('\n' + name + '\n')
    break_y, break_x = breaks_dict[name]
    image = load_stab_img(image_path)
    result = []
    for break_centroid in zip(break_y, break_x):
        print('Centroid: ', break_centroid)
        break_image = select_break_image(image, break_centroid, width)
        gap_width, fiber_diameter = analyze_break(break_image)
        result.append((gap_width, fiber_diameter))
    return np.transpose(result) if result else (np.array([], dtype=int), np.array([], dtype=int))


def save_analysis(parameters, analysis, images):
    folder = path.dirname(images[0])
    fnames = [basename_without_stab(image) for image in images]

    headers = dict(parameters)
    headers['fields'] = 'name: [gap_width, fiber_diameter]'
    data = dict(zip(fnames, analysis))
    output = [headers, data]

    print('Saving parameters and analysis to:')
    analysis_path = path.join(folder, ANALYSIS_FILENAME)
    print(analysis_path)

    mode = 'w'
    with open(analysis_path, mode) as file:
        dump(output, file)


def load_analysis(directory, output=None):
    analysis_file = path.join(directory, ANALYSIS_FILENAME)
    with open(analysis_file) as fp:
        header, data = json.load(fp)

    if output is None:
        return header, data

    i = {'gap_width': 0,
         'fiber_diameter': 1,
         }.get(output, output)

    return OrderedDict((name, data[name][i]) for name in sorted(data))

if __name__ == '__main__':
    a = AnalyzerGUI(get_files())
    # folder = (get_folder())
    # breaks_dict = load_breaks(folder,'absolute')
    # analyze_breaks(folder)

    # import cProfile as prof
    # prof.run('analyze_breaks(folder)',sort='cumtime')
