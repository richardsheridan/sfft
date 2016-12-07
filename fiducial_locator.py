from os import path

import cv2
import numpy as np

from cvutil import make_pyramid
from gui import MPLGUI
from util import wavelet_filter, find_zero_crossings, get_files, basename_without_stab, DISPLAY_SIZE, batch, \
    gaussian, convolve


FIDUCIAL_FILENAME = 'fiducials.json'


class FidGUI(MPLGUI):
    def __init__(self, images, stabilize_args=()):
        self.images = images
        self.stabilize_args = stabilize_args

        super().__init__()

    def create_layout(self):
        import matplotlib.pyplot as plt
        self.fig, (self.axes['image'], self.axes['profile']) = plt.subplots(2, 1, figsize=(8, 10))
        self.fig.subplots_adjust(left=0.1, bottom=0.3)
        self.artists['profile'] = self.axes['profile'].plot(0)[0]
        self.artists['cutoff'] = self.axes['profile'].plot(0, 'k:')[0]
        self.artists['profile_fids'] = self.axes['profile'].plot([100] * 2, [DISPLAY_SIZE[1] / 2] * 2, 'r.', ms=10)[0]

        self.register_button('save',self.execute_batch,[.4, .95, .2, .03], label='Save batch')

        self.slider_coords = [.3, .20, .55, .03]

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
                             valinit=4, )
        self.register_slider('filter_width', self.update_filter_width,
                             label='Filter Width',
                             valmin=0,
                             valmax=.035,
                             valinit=.009,
                             )
        self.register_slider('cutoff', self.update_cutoff,
                             label='Amplitude Cutoff',
                             valmin=0,
                             valmax=60,
                             valinit=30,
                             )

    def load_frame(self):
        image_path = self.images[self.sliders['frame_number'].val]
        from fiber_locator import load_stab_tif
        self.image = image = load_stab_tif(image_path, *self.stabilize_args)
        self.pyramid = make_pyramid(image)

    def recalculate_vision(self):
        self.recalculate_profile()
        self.recalculate_locations()

    def recalculate_profile(self):
        image = self.pyramid[self.sliders['p_level'].val]
        self.profile = fid_profile_from_image(image)

    def recalculate_locations(self):
        fid_window = self.sliders['filter_width'].val
        fid_amp = self.sliders['cutoff'].val
        fid_profile = self.profile
        fid_window = fid_window * len(fid_profile)
        # TODO new script to find grips and import those values
        left_grip_index, right_grip_index = find_grips(fid_profile)
        self.filtered_profile = filtered_profile = wavelet_filter(fid_profile, fid_window)
        try:
            self.locations = np.array(choose_fids(filtered_profile, fid_amp, left_grip_index))
        except NoPeakError:
            print('no peaks found')
            self.locations = np.array([np.nan, np.nan])
        print(self.locations)

    def refresh_plot(self):
        ax = self.axes['image']
        ax.clear()
        display_image = self.pyramid[self.sliders['p_level'].val]
        ax.imshow(cv2.resize(display_image, DISPLAY_SIZE, interpolation=cv2.INTER_CUBIC), cmap='gray')
        self.artists['image_fids'] = ax.plot([100] * 2, [DISPLAY_SIZE[1] / 2] * 2, 'r.', ms=10)[0]
        ax.autoscale_view(tight=True)
        self.artists['image_fids'].set_xdata(self.locations * DISPLAY_SIZE[0])

        l = len(self.filtered_profile)
        locations = self.locations * l
        self.artists['profile'].set_xdata(np.arange(l))
        self.artists['profile'].set_ydata(self.filtered_profile)
        self.artists['profile_fids'].set_xdata(locations)
        no_nan = not np.any(np.isnan(locations))
        self.artists['profile_fids'].set_ydata(self.filtered_profile[np.int64(locations) if no_nan else [0, 0]])
        self.artists['cutoff'].set_xdata([0, len(self.profile)])
        self.artists['cutoff'].set_ydata([self.sliders['cutoff'].val] * 2)

        self.axes['profile'].relim()
        self.axes['profile'].autoscale_view()

        self.fig.canvas.draw()

    def execute_batch(self, event=None):
        parameters = self.parameters
        locations = np.array(batch(locate_fids, self.images, *parameters.values()))
        left_fid, right_fid = locations[:, 0], locations[:, 1]
        if event is None:
            # called from command line without argument
            return left_fid, right_fid
        else:
            save_fids(parameters, self.images, left_fid, right_fid)

    def update_frame_number(self, val):
        self.load_frame()
        self.recalculate_vision()
        self.refresh_plot()

    def update_filter_width(self, val):
        self.recalculate_locations()
        self.refresh_plot()

    def update_cutoff(self, val):

        self.recalculate_locations()
        self.refresh_plot()

    def update_p_level(self, val):

        self.recalculate_vision()
        self.refresh_plot()


def fid_profile_from_image(image):
    fiducial_profile = image.mean(axis=0)  # FIXME: This is SLOW due to array layout, proven by Numba replacement
    fiducial_profile *= -1.0
    # fiducial_profile += 255.0
    fiducial_profile -= fiducial_profile.mean()

    return fiducial_profile


def choose_fids(filtered_profile, fid_amp, mask_until):
    l = len(filtered_profile)
    # only start looking after heuristic starting point
    # mask_until = int(.06 * l)

    end_mask = np.ones_like(filtered_profile, bool)
    end_mask[:mask_until] = False

    fid_peaks = find_zero_crossings(np.gradient(filtered_profile)) & (filtered_profile >= fid_amp) & end_mask

    fid_ind = fid_peaks.nonzero()[0]

    try:
        left_fid = fid_ind[0]
    except IndexError:
        raise NoPeakError('No peaks found!')

    for i in fid_ind:
        if i - left_fid > 0.64 * l:
            right_fid = i
            break
    else:
        raise NoPeakError('No right fiducial?', locals())

    return left_fid/l, right_fid/l

def find_grips(profile, threshold = 2):
    """
    Presume that the grips appear as peaks in the filtered slope of the profile i.e. curvature == 0
    left grip is the first one and right grip is the last one
    eliminate false positives with a threshold on the steepness (slope)
    :param profile:
    :return:
    """
    l = len(profile)
    kernel = np.gradient(gaussian(l // 50, l / 500))
    kernel /= np.sum(np.abs(kernel)) # This keeps the values of the slope array stable for thresholding
    slope = convolve(profile - profile[0], kernel, 'same') # shift to eliminate edge effects
    curvature = np.gradient(slope)
    inflections = find_zero_crossings(curvature, 'all')
    # print(np.max(np.abs(slope)))
    inflections &= (np.abs(slope) > threshold)
    inflections = np.where(inflections)[0]
    print(inflections)
    try:
        left_grip = inflections[0]
        right_grip = inflections[-1]
    except:
        left_grip, right_grip = 0, 0
    return left_grip, right_grip

def locate_fids(image, p_level, filter_width, cutoff, stabilize_args=()):
    if isinstance(image, str):  # TODO: more robust dispatch
        print('Processing: ' + path.basename(image))
        from fiber_locator import load_stab_tif
        image = load_stab_tif(image, *stabilize_args)
        pyramid = make_pyramid(image, p_level)
    elif isinstance(image, np.ndarray):
        pass
    else:
        raise ValueError('Unknown type', type(image))

    profile = fid_profile_from_image(pyramid[p_level])
    filter_width = filter_width * len(profile)
    profile = wavelet_filter(profile, filter_width)
    left_grip_index, right_grip_index = find_grips(profile)
    return choose_fids(profile, cutoff, left_grip_index)


class NoPeakError(Exception):
    pass


def load_fids(image_path, image=None, fid_args=()):
    lpath = image_path.lower()
    dirname = path.dirname(lpath)
    fid_file = path.join(dirname, FIDUCIAL_FILENAME)
    basename = basename_without_stab([image_path])[0]
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
    parameters.update({'initial_displacement': initial_displacement *  19000, # TODO: get size info from strain.dat
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
    # a = FidGUI(get_files())
    # a = batch_fids(get_files(), 7000, 1000)


    import cProfile, pstats, io

    prof = cProfile.Profile()
    # images = get_files()
    images = ['c:\\users\\rjs3\\onedrive\\data\\sfft\\10051319\\stab_str000.jpg',
              'c:\\users\\rjs3\\onedrive\\data\\sfft\\10051319\\stab_str01d.jpg',
              'c:\\users\\rjs3\\onedrive\\data\\sfft\\10051319\\stab_str02d.jpg',
              'c:\\users\\rjs3\\onedrive\\data\\sfft\\10051319\\stab_str03d.jpg',
              'c:\\users\\rjs3\\onedrive\\data\\sfft\\10051319\\stab_str04d.jpg',
              'c:\\users\\rjs3\\onedrive\\data\\sfft\\10051319\\stab_str05d.jpg',
              'c:\\users\\rjs3\\onedrive\\data\\sfft\\10051319\\stab_str06d.jpg',
              'c:\\users\\rjs3\\onedrive\\data\\sfft\\10051319\\stab_str07d.jpg',
              'c:\\users\\rjs3\\onedrive\\data\\sfft\\10051319\\stab_str08d.jpg',
              'c:\\users\\rjs3\\onedrive\\data\\sfft\\10051319\\stab_str09d.jpg',
              'c:\\users\\rjs3\\onedrive\\data\\sfft\\10051319\\stab_str10d.jpg',
              'c:\\users\\rjs3\\onedrive\\data\\sfft\\10051319\\stab_str11d.jpg',
              'c:\\users\\rjs3\\onedrive\\data\\sfft\\10051319\\stab_str12d.jpg',
              'c:\\users\\rjs3\\onedrive\\data\\sfft\\10051319\\stab_str13d.jpg',
              'c:\\users\\rjs3\\onedrive\\data\\sfft\\10051319\\stab_str14d.jpg',
              'c:\\users\\rjs3\\onedrive\\data\\sfft\\10051319\\stab_str15d.jpg',
              'c:\\users\\rjs3\\onedrive\\data\\sfft\\10051319\\stab_str16d.jpg',
              'c:\\users\\rjs3\\onedrive\\data\\sfft\\10051319\\stab_str17d.jpg',
              'c:\\users\\rjs3\\onedrive\\data\\sfft\\10051319\\stab_str18d.jpg',
              'c:\\users\\rjs3\\onedrive\\data\\sfft\\10051319\\stab_str19d.jpg',
              'c:\\users\\rjs3\\onedrive\\data\\sfft\\10051319\\stab_str20d.jpg',
              'c:\\users\\rjs3\\onedrive\\data\\sfft\\10051319\\stab_str21d.jpg',
              'c:\\users\\rjs3\\onedrive\\data\\sfft\\10051319\\stab_str22d.jpg',
              'c:\\users\\rjs3\\onedrive\\data\\sfft\\10051319\\stab_str23d.jpg',
              'c:\\users\\rjs3\\onedrive\\data\\sfft\\10051319\\stab_str24d.jpg',
              'c:\\users\\rjs3\\onedrive\\data\\sfft\\10051319\\stab_tdiz.jpg']

    from collections import OrderedDict

    parameters = OrderedDict([('p_level', 4), ('filter_width', 0.010937500000000003), ('cutoff', 8.8636363636363669)])
    prof.enable()
    a = batch(locate_fids, images, *parameters.values())
    # a = FidGUI(images)
    prof.disable()
    s = io.StringIO()
    sortby = 'tottime'
    ps = pstats.Stats(prof, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())
