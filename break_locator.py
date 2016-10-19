import json
from collections import OrderedDict
from itertools import starmap as _map, repeat
from multiprocessing import freeze_support, pool
from numbers import Integral as Int
from os import path
from time import perf_counter

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, Button

from util import wavelet_filter, find_crossings, get_files, basename_without_stab

BREAK_FILENAME = 'breaks.json'


class BreakGUI:
    def __init__(self, images, stabilize_args=(), fid_args=(), block=True, ):
        self.images = images
        self.stabilize_args = stabilize_args
        self.fid_args = fid_args
        self.axes = {}
        self.artists = {}
        self.buttons = {}
        self.sliders = {}
        self.t = perf_counter()
        self.cooldown = .1
        self.lines = 1

        self.create_layout()
        self.load_frame()
        self.recalculate_breaks()
        self.refresh_plot()
        # for k, v in stabilize_args.items():
        #     self.sliders[k].set_val(v)

        if block:
            plt.ioff()
        else:
            plt.ion()

        plt.show()

    def create_layout(self):
        self.fig, (self.axes['image'], self.axes['profile']) = plt.subplots(2, 1, figsize=(8, 10))
        self.fig.subplots_adjust(left=0.1, bottom=0.3)
        self.artists['profile'] = self.axes['profile'].plot(0)[0]
        self.artists['cutoff'] = self.axes['profile'].plot(0, 'k:')[0]
        self.axes['save'] = self.fig.add_axes([.4, .95, .2, .03])
        self.buttons['save'] = Button(self.axes['save'], 'Save batch')
        self.buttons['save'].on_clicked(self.execute_batch)

        slider_width = .55
        slider_height = .03
        slider_x_coordinate = .3
        slider_y_step = .05
        slider_y_coord = .20

        self.axes['frame_number'] = self.fig.add_axes(
            [slider_x_coordinate, slider_y_coord, slider_width, slider_height])
        slider_y_coord -= slider_y_step
        self.axes['slice_width'] = self.fig.add_axes(
            [slider_x_coordinate, slider_y_coord, slider_width, slider_height])
        slider_y_coord -= slider_y_step
        self.axes['filter_width'] = self.fig.add_axes(
            [slider_x_coordinate, slider_y_coord, slider_width, slider_height])
        slider_y_coord -= slider_y_step
        self.axes['cutoff'] = self.fig.add_axes(
            [slider_x_coordinate, slider_y_coord, slider_width, slider_height])
        slider_y_coord -= slider_y_step

        self.sliders['frame_number'] = Slider(self.axes['frame_number'], 'Frame Number',
                                              valmin=0, valmax=len(self.images) - 1, valinit=0, valfmt='%d')
        self.sliders['slice_width'] = Slider(self.axes['slice_width'], 'Slice Width',
                                             valmin=0, valmax=400, valinit=200, valfmt='%d')
        self.sliders['filter_width'] = Slider(self.axes['filter_width'], 'Filter Width',
                                              valmin=0, valmax=800, valinit=400, valfmt='%d')
        self.sliders['cutoff'] = Slider(self.axes['cutoff'], 'Amplitude Cutoff',
                                        valmin=0, valmax=400, valinit=100, valfmt='%.4g')

        self.sliders['frame_number'].on_changed(self.update_frame_number)
        self.sliders['slice_width'].on_changed(self.update_slice_width)
        self.sliders['filter_width'].on_changed(self.update_filter_width)
        self.sliders['cutoff'].on_changed(self.update_cutoff)

    def load_frame(self):
        from fiber_locator import load_stab_tif
        image_path = self.images[int(self.sliders['frame_number'].val)]
        image = load_stab_tif(image_path, self.stabilize_args)

        from fiducial_locator import load_fids
        self.fids = load_fids(image_path, image, *self.fid_args)

        ax = self.axes['image']
        self.image = image
        ax.clear()
        ax.imshow(cv2.resize(image, (800, 453), interpolation=cv2.INTER_AREA), cmap='gray')
        self.artists['breaks'] = ax.plot([100] * 2, [453 / 2] * 2, 'rx', ms=10)[0]
        ax.autoscale_view(tight=True)

    def recalculate_breaks(self):
        self.recalculate_profile()
        self.recalculate_locations()

    def recalculate_profile(self):
        # width = self.sliders['slice_width'].val
        self.profile = break_profile_from_image(self.image, self.sliders['slice_width'].val)

    def recalculate_locations(self):
        break_window = self.sliders['filter_width'].val
        break_amp = self.sliders['cutoff'].val
        break_profile = self.profile
        self.filtered_profile = wavelet_filter(break_profile, break_window)
        self.locations = np.array(choose_breaks(self.filtered_profile, break_amp, *self.fids))
        print(self.locations)

    def refresh_plot(self):
        self.artists['breaks'].set_xdata(self.locations / len(self.profile) * 800)
        self.artists['breaks'].set_ydata(np.full_like(self.locations, 453 / 2))
        # TODO: indicate breaks in profile plot as well

        self.artists['profile'].set_xdata(np.arange(len(self.filtered_profile)))
        self.artists['profile'].set_ydata(self.filtered_profile)
        self.artists['cutoff'].set_xdata([0, len(self.profile)])
        self.artists['cutoff'].set_ydata([self.sliders['cutoff'].val] * 2)

        self.axes['profile'].relim()
        self.axes['profile'].autoscale_view()
        # axesimage = self.axes['image'].imshow((self.edges_with_lines),cmap='Spectral')
        # self.cb = self.fig.colorbar(axesimage,cax=self.axes['colorbar'],orientation='horizontal')

        self.fig.canvas.draw()

    def _cooling_down(self):
        t = perf_counter()
        if t - self.t <= self.cooldown:
            return True
        else:
            self.t = t
            return False

    @property
    def parameters(self):
        return OrderedDict((('slice_width', self.sliders['slice_width'].val),
                            ('filter_width', self.sliders['filter_width'].val),
                            ('cutoff', self.sliders['cutoff'].val),
                            ('fid_args', tuple(self.fid_args)),
                            ('stabilize_args', tuple(self.stabilize_args)),
                            )
                           )

    def execute_batch(self, event=None):
        parameters = self.parameters
        breaks = batch_breaks(self.images, *parameters.values())
        if event is None:
            # called from command line without argument
            return breaks
        else:
            save_breaks(parameters, breaks, self.images)

    def update_frame_number(self, val):
        if not isinstance(val, Int):
            self.sliders['frame_number'].set_val(int(val))
            return
        if self._cooling_down():
            return

        # t = perf_counter()
        self.load_frame()
        self.recalculate_breaks()
        self.refresh_plot()
        # print('frame time:', perf_counter() - t)

    def update_slice_width(self, val):
        if not isinstance(val, Int):
            self.sliders['slice_width'].set_val(int(val))
            return
        if self._cooling_down():
            return

        self.recalculate_breaks()
        self.refresh_plot()

    def update_filter_width(self, val):
        if not isinstance(val, Int):
            self.sliders['filter_width'].set_val(int(val))
            return
        if self._cooling_down():
            return

        self.recalculate_breaks()
        self.refresh_plot()

    def update_cutoff(self, val):
        if self._cooling_down():
            return

        self.recalculate_breaks()
        self.refresh_plot()


def break_profile_from_image(image, width=200):
    width = int(width)
    rows, cols = image.shape
    startrow = rows // 2 - width // 2
    stoprow = startrow + width

    fiber_profile = image[startrow:stoprow, :].mean(axis=0)

    break_profile = 255.0 - fiber_profile
    break_profile -= break_profile.mean()

    return break_profile


def choose_breaks(break_filtered, break_amp, left_limit, right_limit):
    break_peaks = find_crossings(np.gradient(break_filtered))
    break_peaks[:left_limit] = False
    break_peaks[right_limit:] = False

    # plt.subplot(211)
    # a = break_profile[3000:-3000]
    # b = break_filtered[3000:-3000]
    # plt.plot(a / a.max() * b.max(), 'b', b, 'r', np.full_like(a, break_amp), ':k')
    # # plt.axis('tight')
    # plt.subplot(212)
    # # derp = cwt(break_profile[3000:-3000],ricker,np.linspace(100,300,20,0))
    # # print()
    # # plt.imshow(derp,cmap='viridis',aspect='auto')
    # # plt.show(1)
    # bp_amps = break_filtered[break_peaks]
    # # bp_amps /= bp_amps.std()
    # (osm, osr), (slope, intercept, r) = stats.probplot(bp_amps, fit=True)
    # plt.plot(osr, 'bo', np.full_like(osm, break_amp), ':k')
    # plt.axis('tight')
    # plt.show(1)

    break_peaks &= (break_filtered >= break_amp)

    break_peaks = break_peaks.nonzero()[0]

    return break_peaks


def locate_breaks(image_path, slice_width, filter_width, cutoff, fid_args=(), stabilize_args=()):
    print('Processing: ', path.basename(image_path))
    from fiber_locator import load_stab_tif
    image = load_stab_tif(image_path, stabilize_args)

    from fiducial_locator import load_fids
    fids = load_fids(image_path, image, fid_args)

    break_profile = break_profile_from_image(image, slice_width)
    filtered_break_profile = wavelet_filter(break_profile, filter_width)
    breaks = choose_breaks(filtered_break_profile, cutoff, *fids)
    rel_breaks = (breaks - fids[0]) / (fids[1] - fids[0])
    return breaks, rel_breaks


def batch_breaks(image_paths, slice_width, filter_width, cutoff, fid_args=(), stabilize_args=()):
    args = (slice_width, filter_width, cutoff, fid_args, stabilize_args)
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
    headers['fields'] = 'name: [locations, fid_relative]'
    data = {fname: (locations, fid_relative) for fname, (locations, fid_relative) in zip(fnames, breaks)}
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

    a = BreakGUI(get_files(), )  # (97.798295454545467, 248.48011363636365, 1), (7000, 1000))

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
