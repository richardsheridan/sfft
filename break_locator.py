import json
from itertools import starmap as _map, repeat
from multiprocessing import freeze_support, pool
from os import path

import cv2
import numpy as np

from cvutil import make_pyramid, make_log, peak_local_max
from util import wavelet_filter, find_crossings, get_files, basename_without_stab, DISPLAY_SIZE
from gui import MPLGUI

BREAK_FILENAME = 'breaks.json'


class BreakGUI(MPLGUI):
    def __init__(self, images, stabilize_args=(), fid_args=()):
        self.images = images
        self.stabilize_args = stabilize_args
        self.fid_args = fid_args

        super().__init__()

    def create_layout(self):
        import matplotlib.pyplot as plt
        self.fig, (self.axes['image'], self.axes['filtered']) = plt.subplots(2, 1, figsize=(8, 10))
        # self.fig, self.axes['image'] = plt.subplots(1, 1, figsize=(8, 10))
        self.fig.subplots_adjust(left=0.1, bottom=0.5)
        # self.artists['profile'] = self.axes['profile'].plot(0)[0]
        # self.artists['cutoff'] = self.axes['profile'].plot(0, 'k:')[0]
        # self.artists['profile_breaks'] = self.axes['profile'].plot([100] * 2, [DISPLAY_SIZE[1] / 2] * 2, 'rx', ms=10)[0]
        self.register_button('save',self.execute_batch,[.4, .95, .2, .03], label='Save batch')

        self.slider_coords = [.3, .40, .55, .03]

        self.register_slider('frame_number',self.update_frame_number,
                             isparameter=False,
                             forceint=True,
                             label='Frame Number',
                             valmin=0,
                             valmax=len(self.images) - 1,
                             valinit=0,
                             )
        self.register_slider('p_level', self.update_p_level,
                             forceint=True,
                             label='Pyramid Level',
                             valmin=0,
                             valmax=7,
                             valinit=3,
                             )
        self.register_slider('filter_width', self.update_filter_width,
                             label='Filter Width',
                             valmin=0,
                             valmax=10,
                             valinit=0.8,
                             )
        self.register_slider('cutoff', self.update_cutoff,
                             label='Amplitude Cutoff',
                             valmin=0,
                             valmax=100,
                             valinit=30,
                             )
        self.register_slider('neighborhood', self.update_neighborhood,
                             label='Neighborhood',
                             forceint=True,
                             valmin=1,
                             valmax=100,
                             valinit=10,
                             )

    def load_frame(self):
        from fiber_locator import load_stab_tif
        image_path = self.images[int(self.sliders['frame_number'].val)]
        image = load_stab_tif(image_path, self.stabilize_args)

        from fiducial_locator import load_fids
        self.fids = load_fids(image_path, image, *self.fid_args)

        self.image = image
        self.pyramid = make_pyramid(image, 7)

    def recalculate_vision(self):
        self.recalculate_blobs()
        self.recalculate_locations()

    def recalculate_blobs(self):
        image = self.pyramid[self.sliders['p_level'].val]
        self.filtered_image = make_log(image, self.sliders['filter_width'].val)

    def recalculate_locations(self):
        image = self.filtered_image
        rows, cols = image.shape
        cutoff = self.sliders['cutoff'].val
        neighborhood = self.sliders['neighborhood'].val

        peaks = peak_local_max(image, cutoff, neighborhood)
        row_index, col_index = np.where(peaks)
        self.locations = row_index / rows, col_index / cols
        print(self.locations)

    def refresh_plot(self):
        ax = self.axes['image']
        ax.clear()
        ax.imshow(cv2.resize(self.pyramid[self.sliders['p_level'].val], DISPLAY_SIZE, interpolation=cv2.INTER_CUBIC),
                  cmap='gray')  # TODO: use extent=[0,real_width,0,real_height] and aspect='auto'
        ax.autoscale_view(tight=True)
        self.artists['image_breaks'] = ax.plot(self.locations[1] * DISPLAY_SIZE[0],
                                               self.locations[0] * DISPLAY_SIZE[1],
                                               'rx', ms=10)[0]

        image = self.filtered_image
        # TODO: add toggle to visualize threshold
        # break_amp = self.sliders['cutoff'].val
        # image = (image>break_amp).view(np.uint8)
        ax = self.axes['filtered']
        ax.clear()
        ax.imshow(cv2.resize(image, DISPLAY_SIZE, interpolation=cv2.INTER_CUBIC), cmap='gray')
        ax.autoscale_view(tight=True)
        self.fig.canvas.draw()

    def execute_batch(self, event=None):
        parameters = self.parameters
        breaks = batch_breaks(self.images, *parameters.values())
        if event is None:
            # called from command line without argument
            return breaks
        else:
            save_breaks(parameters, breaks, self.images)

    def update_frame_number(self, val):
        self.load_frame()
        self.recalculate_vision()
        self.refresh_plot()

    def update_p_level(self, val):
        self.recalculate_vision()
        self.refresh_plot()

    def update_filter_width(self, val):
        self.recalculate_vision()
        self.refresh_plot()

    def update_cutoff(self, val):
        self.recalculate_locations()
        self.refresh_plot()

    def update_neighborhood(self, val):
        self.recalculate_locations()
        self.refresh_plot()


def locate_breaks(image_path, p_level, filter_width, cutoff, neighborhood, fid_args=(), stabilize_args=()):
    print('Processing: ', path.basename(image_path))
    from fiber_locator import load_stab_tif
    image = load_stab_tif(image_path, stabilize_args)

    from fiducial_locator import load_fids
    fids = load_fids(image_path, image, fid_args)

    pyramid = make_pyramid(image, p_level)
    image = pyramid[p_level]
    rows, cols = image.shape
    filtered_image = make_log(image, filter_width)

    peaks = peak_local_max(filtered_image, cutoff, neighborhood)
    row_index, col_index = np.where(peaks)
    locations = row_index / rows, col_index / cols
    relative_locations = (locations[0] - fids[0]) / (fids[1] - fids[0]), locations[1]
    return locations, relative_locations


def batch_breaks(image_paths, p_level, filter_width, cutoff, neighborhood, fid_args=(), stabilize_args=()):
    args = (p_level, filter_width, cutoff, neighborhood, fid_args, stabilize_args)
    args = repeat(args)
    args = [(image_path, *arg) for image_path, arg in zip(image_paths, args)]
    freeze_support()
    p = pool.Pool()
    _map = p.starmap
    locations = list(_map(locate_breaks, args))
    return locations


def save_breaks(parameters, breaks, images):
    folder = path.dirname(images[0])
    fnames = basename_without_stab(images)

    headers = dict(parameters)
    headers['fields'] = 'name: [locations, relative_locations]'
    data = {fname: (locations, relative_locations) for fname, (locations, relative_locations) in zip(fnames, breaks)}
    output = [headers, data]

    print('Saving parameters and locations to:')
    breakpath = path.join(folder, BREAK_FILENAME)
    print(breakpath)

    from util import dump
    mode = 'w'
    with open(breakpath, mode) as file:
        dump(output, file)


def load_breaks(directory):
    breakfile = path.join(directory, BREAK_FILENAME)
    with open(breakfile) as fp:
        header, data = json.load(fp)

    breaks = [data[name][1] for name in sorted(data)]
    return breaks

if __name__ == '__main__':

    a = BreakGUI(get_files(), )

    # import cProfile, pstats, io
    # prof = cProfile.Profile()
    # prof.enable()

    # a = batch_breaks(get_files(), 102, 330, 296)#, fid_args=(7000, 1000), )

    # prof.disable()
    # s = io.StringIO()
    # sortby = 'cumulative'
    # ps = pstats.Stats(prof, stream=s).sort_stats(sortby)
    # ps.print_stats()
    # print(s.getvalue())
