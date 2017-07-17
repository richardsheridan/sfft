import json
from collections import OrderedDict
from os import path

import numpy as np
from .cvutil import make_pyramid, make_log
from .fiber_locator import load_stab_img
from .gui import GUIPage
from .util import basename_without_stab, peak_local_max, batch, get_files

BREAK_FILENAME = 'breaks.json'


class BreakGUI(GUIPage):
    def __init__(self, image_paths, stabilize_args=(), fid_args=(), **kw):
        self.image_paths = image_paths
        self.load_args = (stabilize_args, fid_args)
        self.display_type = 'filtered'

        super().__init__(**kw)

    def __str__(self):
        return 'BreakGUI'

    def create_layout(self):
        self.add_axes('image')
        self.add_axes('filtered')
        self.add_button('save', self.execute_batch, label='Save batch')
        self.add_radiobuttons('display_type', self.refresh_plot, labels=('filtered', 'thresholded',))

        self.add_slider('frame_number', self.full_reload, valmin=0, valmax=len(self.image_paths) - 1, valinit=0,
                        label='Frame Number', isparameter=False, forceint=True)
        self.add_slider('p_level', self.update_vision, valmin=0, valmax=7, valinit=3, label='Pyramid Level',
                        forceint=True)
        self.add_slider('filter_width', self.update_vision, valmin=0, valmax=10, valinit=2, label='Filter Width')
        self.add_slider('mask_width', self.update_vision, valmin=0, valmax=.5, valinit=.4, label='Mask Width')
        self.add_slider('cutoff', self.update_vision, valmin=0, valmax=10, valinit=5, label='Amplitude Cutoff')
        self.add_slider('neighborhood', self.update_vision, valmin=1, valmax=100, valinit=10, label='Neighborhood',
                        forceint=True)

    @staticmethod
    def load_image_to_pyramid(image_path, stabilize_args, fid_args):
        image = load_stab_img(image_path, stabilize_args)
        pyramid = make_pyramid(image)
        return pyramid

    def recalculate_vision(self):
        self.recalculate_blobs()
        self.recalculate_locations()

    def recalculate_blobs(self):
        image = self.pyramid[self.slider_value('p_level')]
        self.filtered_image = make_log(image, self.slider_value('filter_width'))

    def recalculate_locations(self):
        filtered_image = self.filtered_image
        rows, cols = filtered_image.shape
        cutoff = self.slider_value('cutoff')
        neighborhood = self.slider_value('neighborhood')
        mask_width = self.slider_value('mask_width')

        row_index, col_index = peak_local_max(filtered_image, cutoff, neighborhood)
        row_index, col_index = mask_stray_peaks(row_index, col_index, mask_width, rows)

        self.locations = row_index / (rows - 1), col_index / (cols - 1)
        print(len(row_index))

    def refresh_plot(self):
        image = self.pyramid[self.slider_value('p_level')]
        self.clear('image')
        # TODO: use extent=[0,real_width,0,real_height]
        self.imshow('image', image, extent=[0, 1, 1, 0])
        self.plot('image', self.locations[1], self.locations[0], 'rx', ms=10)

        filtered = self.filtered_image
        if self.button_value('display_type') == 'thresholded':
            break_amp = self.slider_value('cutoff')
            filtered = (filtered > break_amp).view(np.uint8)
        self.clear('filtered')
        self.imshow('filtered', filtered, extent=[0, 1, 1, 0])
        self.plot('filtered', self.locations[1], self.locations[0], 'rx', ms=10)

        mask_width = self.slider_value('mask_width')
        self.hspan('filtered', 0.001, mask_width, facecolor='r', alpha=0.3)
        self.hspan('filtered', .999 - mask_width, 1, facecolor='r', alpha=0.3)

        self.draw()

    def execute_batch(self, *a, **kw):
        parameters = self.parameters
        breaks = batch(locate_breaks, self.image_paths, *parameters.values())
        save_breaks(parameters, breaks, self.image_paths)

    def update_vision(self, *a, **kw):
        self.recalculate_vision()
        self.refresh_plot()


def mask_stray_peaks(row_index, col_index, mask_width, rows):
    min_row = rows * mask_width
    max_row = rows - 1 - min_row
    keep = (min_row <= row_index) & (row_index <= max_row)
    return row_index[keep], col_index[keep]


def locate_breaks(image_path, p_level, filter_width, mask_width, cutoff, neighborhood, fid_args=(), stabilize_args=()):
    print('Processing: ', path.basename(image_path))
    from .fiber_locator import load_stab_img
    image = load_stab_img(image_path, *stabilize_args)

    from .fiducial_locator import load_fids
    fids = load_fids(image_path, image, *fid_args)

    pyramid = make_pyramid(image, p_level)
    image = pyramid[p_level]
    rows, cols = image.shape
    filtered_image = make_log(image, filter_width)

    peak_y, peak_x = peak_local_max(filtered_image, cutoff, neighborhood)
    peak_y, peak_x = mask_stray_peaks(peak_y, peak_x, mask_width, rows)

    sortind = np.argsort(peak_x)
    peak_y, peak_x = peak_y[sortind], peak_x[sortind]

    locations = peak_y / (rows - 1), peak_x / (cols - 1)
    relative_locations = (locations[1] - fids[0]) / (fids[1] - fids[0])
    fiducial_break_count = np.count_nonzero((0 < relative_locations) & (relative_locations < 1))
    return locations, relative_locations, fiducial_break_count


def save_breaks(parameters, breaks, images):
    folder = path.dirname(images[0])
    fnames = [basename_without_stab(image) for image in images]

    headers = dict(parameters)
    headers['fields'] = 'name: [locations, relative_locations, fiducial_break_count]'
    data = dict(zip(fnames, breaks))
    output = [headers, data]

    print('Saving parameters and locations to:')
    breakpath = path.join(folder, BREAK_FILENAME)
    print(breakpath)

    from .util import dump
    mode = 'w'
    with open(breakpath, mode) as file:
        dump(output, file)


def load_breaks(directory, output=None):
    breakfile = path.join(directory, BREAK_FILENAME)
    with open(breakfile) as fp:
        header, data = json.load(fp)

    if output is None:
        return header, data

    i = {'absolute': 0,
         'relative': 1,
         'count': 2,
         'both': slice(-1)
         }.get(output, output)

    return OrderedDict((name, data[name][i]) for name in sorted(data))


if __name__ == '__main__':
    a = BreakGUI(get_files())

    # images = get_files()

    # from collections import OrderedDict

    # parameters = OrderedDict(
        # [('p_level', 3), ('filter_width', 1.0037878787878789), ('cutoff', 0.00026799242424242417), ('neighborhood', 5)])
    # parameters = a.parameters
    # images = a.images

    # from cProfile import run

    # run('batch(locate_breaks,images, *parameters.values())', sort='time')
