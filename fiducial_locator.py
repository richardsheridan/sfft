from time import perf_counter
import os.path as path

import cv2
import itertools
import numpy as np

from util import wavelet_filter, find_crossings, get_files, basename_without_stab
from gui import MPLGUI
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from multiprocessing import pool, freeze_support

_map = itertools.starmap

FIDUCIAL_FILENAME = 'fiducials.json'


class FidGUI(MPLGUI):
    def __init__(self, images, stabilize_args=(), ):
        self.images = images
        self.stabilize_args = stabilize_args

        super().__init__()

    def create_layout(self):
        self.fig, (self.axes['image'], self.axes['profile']) = plt.subplots(2, 1, figsize=(8, 10))
        self.fig.subplots_adjust(left=0.1, bottom=0.3)
        self.artists['profile'] = self.axes['profile'].plot(0)[0]
        self.artists['cutoff'] = self.axes['profile'].plot(0, 'k:')[0]

        self.register_button('save',self.execute_batch,[.4, .95, .2, .03], label='Save batch')

#        self.axes['save'] = self.fig.add_axes([.4, .95, .2, .03])
#        self.buttons['save'] = Button(self.axes['save'], 'Save batch')
#        self.buttons['save'].on_clicked(self.execute_batch)
#
#        slider_width = .55
#        slider_height = .03
#        slider_x_coordinate = .3
#        slider_y_coord = .20
        self.slider_coords = [.3, .20, .55, .03]

        self.register_slider('frame_number',self.update_frame_number,
                             isparameter=False,
                             forceint=True,
                             label='Frame Number',
                             valmin=0,
                             valmax=len(self.images) - 1,
                             valinit=0,
                             )
        self.register_slider('filter_width', self.update_filter_width,
                             forceint=True,
                             label='Filter Width',
                             valmin=0,
                             valmax=12000,
                             valinit=4000,
                             )
        self.register_slider('cutoff', self.update_cutoff,
                             label='Amplitude Cutoff',
                             valmin=0,
                             valmax=3000,
                             valinit=1000,
                             )
#
#        self.axes['frame_number'] = self.fig.add_axes(
#            [slider_x_coordinate, slider_y_coord, slider_width, slider_height])
#        slider_y_coord -= slider_y_step
#        # self.axes['slice_width'] = self.fig.add_axes(
#        #     [slider_x_coordinate, slider_y_coord, slider_width, slider_height])
#        # slider_y_coord -= slider_y_step
#        self.axes['filter_width'] = self.fig.add_axes(
#            [slider_x_coordinate, slider_y_coord, slider_width, slider_height])
#        slider_y_coord -= slider_y_step
#        self.axes['cutoff'] = self.fig.add_axes(
#            [slider_x_coordinate, slider_y_coord, slider_width, slider_height])
#        slider_y_coord -= slider_y_step
#
#        self.sliders['frame_number'] = Slider(self.axes['frame_number'], 'Frame Number',
#                                              valmin=0, valmax=len(self.images) - 1, valinit=0, valfmt='%d')
#        # self.sliders['slice_width'] = Slider(self.axes['slice_width'], 'Slice Width',
#        #                                       valmin=0, valmax=400, valinit=200, valfmt='%d')
#        self.sliders['filter_width'] = Slider(self.axes['filter_width'], 'Filter Width',
#                                              valmin=0, valmax=12000, valinit=4000, valfmt='%d')
#        self.sliders['cutoff'] = Slider(self.axes['cutoff'], 'Amplitude Cutoff',
#                                        valmin=0, valmax=3000, valinit=1000, valfmt='%.4g')
#
#        self.sliders['frame_number'].on_changed(self.update_frame_number)
#        # self.sliders['slice_width'].on_changed(self.update_slice_width)
#        self.sliders['filter_width'].on_changed(self.update_filter_width)
#        self.sliders['cutoff'].on_changed(self.update_cutoff)

    def load_frame(self):
        image_path = self.images[self.sliders['frame_number'].val]
        from fiber_locator import load_stab_tif
        self.image = image = load_stab_tif(image_path, self.stabilize_args)
        ax = self.axes['image']
        ax.clear()
        ax.imshow(cv2.resize(image, (800, 453), interpolation=cv2.INTER_AREA), cmap='gray')
        self.artists['fids'] = ax.plot([100] * 2, [453 / 2] * 2, 'r.', ms=10)[0]
        ax.autoscale_view(tight=True)

    def recalculate_vision(self):
        self.recalculate_profile()
        self.recalculate_locations()

    def recalculate_profile(self):
        # width = self.sliders['slice_width'].val
        self.profile = fid_profile_from_image(self.image)

    def recalculate_locations(self):
        fid_window = self.sliders['filter_width'].val
        fid_amp = self.sliders['cutoff'].val
        fid_profile = self.profile
        self.filtered_profile = np.empty_like(fid_profile, float)
        try:
            self.locations = np.array(choose_fids(fid_profile, fid_window, fid_amp, self.filtered_profile))
        except NoPeakError:
            print('no peaks found')
            self.locations = np.array([np.nan, np.nan])
        print(self.locations)

    def refresh_plot(self):
        self.artists['fids'].set_xdata(self.locations / len(self.profile) * 800)
        # self.artists['fids'].set_ydata(np.full_like(self.locations,self.image.shape[0]//2))

        self.artists['profile'].set_xdata(np.arange(len(self.filtered_profile)))
        self.artists['profile'].set_ydata(self.filtered_profile)
        self.artists['cutoff'].set_xdata([0, len(self.profile)])
        self.artists['cutoff'].set_ydata([self.sliders['cutoff'].val] * 2)

        self.axes['profile'].relim()
        self.axes['profile'].autoscale_view()
        # axesimage = self.axes['image'].imshow((self.edges_with_lines),cmap='Spectral')
        # self.cb = self.fig.colorbar(axesimage,cax=self.axes['colorbar'],orientation='horizontal')

        self.fig.canvas.draw()
#
#    def _cooling_down(self):
#        t = perf_counter()
#        if t - self.t <= self.cooldown:
#            return True
#        else:
#            self.t = t
#            return False

#    @property
#    def parameters(self):
#        from collections import OrderedDict
#        return OrderedDict((('filter_width', self.sliders['filter_width'].val),
#                            ('cutoff', self.sliders['cutoff'].val),
#                            ('stabilize_args', tuple(self.stabilize_args)),
#                            ))

    def execute_batch(self, event=None):
        parameters = self.parameters
        locations = np.array(batch_fids(self.images, *parameters.values()))
        left_fid, right_fid = locations[:, 0], locations[:, 1]
        if event is None:
            # called from command line without argument
            return left_fid, right_fid
        else:
            save_fids(parameters, self.images, left_fid, right_fid)

    def update_frame_number(self, val):

        # t = perf_counter()
        self.load_frame()
        self.recalculate_vision()
        self.refresh_plot()
        # print('frame time:', perf_counter() - t)

    # def update_slice_width(self, val):
    #     if int(val) != val:
    #         self.sliders['slice_width'].set_val(int(val))
    #         return
    #     val = int(val)
    #     if self._cooling_down():
    #         return
    #
    #     self.recalculate_breaks()
    #     self.refresh_plot()

    def update_filter_width(self, val):

        self.recalculate_vision()
        self.refresh_plot()

    def update_cutoff(self, val):

        self.recalculate_vision()
        self.refresh_plot()


def fid_profile_from_image(image):
    fiducial_profile = image.mean(axis=0)  # FIXME: This is SLOW due to array layout
    fiducial_profile *= -1.0
    fiducial_profile += 255.0
    fiducial_profile -= fiducial_profile.mean()

    return fiducial_profile


def choose_fids(fid_profile, fid_window, fid_amp, filtered_profile=None):
    if filtered_profile is None:
        # WTF is this? I needed a way to get the filtered profile out for the GUI
        # without making other workflows more complicated
        filtered_profile = np.empty_like(fid_profile, float)
    filtered_profile[:] = wavelet_filter(fid_profile, fid_window)

    # only start looking after first zero
    for i, value in enumerate(fid_profile):
        if value <= 0:
            mask_until = i
            break
    else:
        raise NoPeakError('Unfiltered profile never crosses zero!')

    end_mask = np.ones_like(filtered_profile, bool)
    end_mask[:mask_until] = False

    fid_peaks = find_crossings(np.gradient(filtered_profile)) & (filtered_profile >= fid_amp) & end_mask

    fid_ind = fid_peaks.nonzero()[0]

    try:
        left_fid = fid_ind[0]
    except IndexError:
        raise NoPeakError

    for i in fid_ind:
        if i - left_fid > 35000:
            right_fid = i
            break
    else:
        raise NoPeakError(locals())

    return left_fid, right_fid


def batch_fids(image_paths, *args):
    args = itertools.repeat(args)
    args = [(image_path, *arg) for image_path, arg in zip(image_paths, args)]
    freeze_support()
    p = pool.Pool()
    _map = p.starmap
    locations = list(_map(locate_fids, args))
    return locations


def locate_fids(image, filter_width, cutoff, stabilize_args=()):
    if isinstance(image, str):  # TODO: more robust dispatch
        print('Processing: ' + path.basename(image))
        from fiber_locator import load_stab_tif
        image = load_stab_tif(image, *stabilize_args)
    elif isinstance(image, np.ndarray):
        pass
    else:
        raise ValueError('Unknown type', type(image))

    profile = fid_profile_from_image(image)
    return choose_fids(profile, filter_width, cutoff)


class NoPeakError(Exception):
    pass


def load_fids(image_path, image=None, fid_args=()):
    lpath = image_path.lower()
    dirname, basename = path.split(lpath)
    fid_file = path.join(dirname, FIDUCIAL_FILENAME)
    if basename.startswith('stab_'):
        basename = basename[len('stab_'):]
    try:  # if os.path.exists(fid_file):
        with open(fid_file) as fp:
            import json
            headers, data = json.load(fp)

        left_fid, right_fid, strain = data[basename]
    except FileNotFoundError:  # else:
        left_fid, right_fid = locate_fids(image, *fid_args)

    return left_fid, right_fid

def load_strain(dirname):
    fid_file = path.join(dirname, FIDUCIAL_FILENAME)
    with open(fid_file) as fp:
        import json
        headers, data = json.load(fp)

    strains = [data[name][2] for name in sorted(data)]
    toobig = any(strain >= 1 for strain in strains)
    initial_displacement = headers['initial_displacement']
    strains = np.array(strains)
    if toobig:
        strains -= 1.0
    return strains, initial_displacement



def save_fids(parameters, images, left_fids, right_fids):
    initial_displacement = right_fids[0] - left_fids[0]
    strains = (right_fids - left_fids) / initial_displacement - 1
    parameters.update({'initial_displacement': initial_displacement * 19000 / 55060,
                       'fields': 'name: (left, right, strain)'})

    folder = path.dirname(images[0])

    fnames = basename_without_stab(images)
    print('Saving parameters and locations to:')
    fidpath = path.join(folder, FIDUCIAL_FILENAME)
    print(fidpath)

    from util import dump
    data = {fname: (left, right, strain)
            for fname, left, right, strain in zip(fnames, left_fids, right_fids, strains)}
    # data = (fnames, left_fid, right_fid, strains)

    output = [dict(parameters), data]
    with open(fidpath, 'w') as fp:
        dump(output, fp)

        # headerstrings = ['{0}: {1}'.format(*param) for param in parameters.items()]
        # np.savetxt(fidpath,
        #            np.rec.fromarrays((fnames, *locations.T, strain)),
        #            fmt=('%s', '%d', '%d', '%.5e'),
        #            header='\n'.join(headerstrings),
        #            )


if __name__ == '__main__':
    a = FidGUI(get_files())  # , (274,))
    # a = batch_fids(get_files(), 7000, 1000)
