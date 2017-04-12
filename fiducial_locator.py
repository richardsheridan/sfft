from os import path

import numpy as np

from cvutil import make_pyramid
from gui import GUIPage
from fiber_locator import load_stab_img
from util import wavelet_filter, find_zero_crossings, get_files, basename_without_stab, batch, \
    gaussian, convolve, quadratic_subpixel_extremum_1d

FIDUCIAL_FILENAME = 'fiducials.json'


class FidGUI(GUIPage):
    def __init__(self, image_paths, stabilize_args=(), **kw):
        self.image_paths = image_paths
        self.load_args = (stabilize_args,)

        super().__init__(**kw)

    def __str__(self):
        return 'FidGUI'

    def create_layout(self):
        self.add_axes('image')
        self.add_axes('profile')

        self.add_button('save', self.execute_batch, label='Save batch')

        self.add_slider('frame_number', self.update_frame_number, valmin=0, valmax=len(self.image_paths) - 1,
                        valinit=0,
                        label='Frame Number', isparameter=False, forceint=True)
        self.add_slider('p_level', self.update_p_level, valmin=0, valmax=7, valinit=4, label='Pyramid Level',
                        forceint=True)
        self.add_slider('filter_width', self.update_filter_width, valmin=0, valmax=.035, valinit=.009,
                        label='Filter Width')
        self.add_slider('cutoff', self.update_cutoff, valmin=0, valmax=60, valinit=30, label='Amplitude Cutoff')

    @staticmethod
    def load_image_to_pyramid(image_path, stabilize_args):
        image = load_stab_img(image_path, *stabilize_args)
        pyramid = make_pyramid(image)
        return pyramid

    def recalculate_vision(self):
        self.recalculate_profile()
        self.recalculate_locations()

    def recalculate_profile(self):
        image = self.pyramid[self.slider_value('p_level')]
        self.profile = fid_profile_from_image(image)

    def recalculate_locations(self):
        fid_window = self.slider_value('filter_width')
        fid_amp = self.slider_value('cutoff')
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
        self.clear('image')
        display_image = self.pyramid[self.slider_value('p_level')]
        r, c = display_image.shape

        self.imshow('image', display_image)
        for loc in self.locations:
            self.vline('image', loc * c, color='r')

        l = len(self.filtered_profile)
        locations = self.locations * (l - 1)
        self.clear('profile')
        self.plot('profile', np.arange(l), self.filtered_profile, 'b')
        if not np.any(np.isnan(locations)):
            value_at_locations = np.interp(locations, np.arange(l), self.filtered_profile)
        else:
            value_at_locations = [0, 0]
        self.plot('profile', locations, value_at_locations, 'r.')

        self.hline('profile', self.slider_value('cutoff'), color='black', linestyle=':')

        self.draw()

    def execute_batch(self, *a, **kw):
        parameters = self.parameters
        locations = np.array(batch(locate_fids, self.image_paths, *parameters.values()))
        left_fid, right_fid = locations[:, 0], locations[:, 1]
        save_fids(parameters, self.image_paths, left_fid, right_fid)

    def update_frame_number(self, *a, **kw):
        self.select_frame()
        self.recalculate_vision()
        self.refresh_plot()

    def update_filter_width(self, *a, **kw):
        self.recalculate_locations()
        self.refresh_plot()

    def update_cutoff(self, *a, **kw):

        self.recalculate_locations()
        self.refresh_plot()

    def update_p_level(self, *a, **kw):

        self.recalculate_vision()
        self.refresh_plot()


def fid_profile_from_image(image):
    fiducial_profile = image.mean(axis=0)  # FIXME: This is SLOW due to array layout, proven by Numba replacement
    fiducial_profile *= -1.0
    # fiducial_profile += 255.0
    fiducial_profile -= fiducial_profile.mean()

    return fiducial_profile


def choose_fids(filtered_profile, fid_amp, mask_until):
    """

    Parameters
    ----------
    filtered_profile
    fid_amp
    mask_until

    Returns
    -------
    np.ndarray

    Examples
    -------
    >>> import numpy as np
    >>> x=np.sin(np.linspace(0,3*2*np.pi,100))
    >>> choose_fids(x,.1,1)
    (0.083304472645591932, 0.74997113931225867)

    The exact solution is (1/12, 3/4)
    """
    l = len(filtered_profile) - 1
    # only start looking after heuristic starting point
    # mask_until = int(.06 * l)

    end_mask = np.ones_like(filtered_profile, bool)
    end_mask[:mask_until] = False

    fid_peaks = find_zero_crossings(np.gradient(filtered_profile)) & (filtered_profile >= fid_amp) & end_mask

    fid_ind = np.where(fid_peaks)[0] + 1

    try:
        left_fid = fid_ind[0]
    except IndexError:
        raise NoPeakError('No peaks found!')

    left_fid = quadratic_subpixel_extremum_1d(filtered_profile,left_fid)

    for i in fid_ind:
        if i - left_fid > 0.64 * l:
            right_fid = i
            break
    else:
        raise NoPeakError('No right fiducial?', locals())

    right_fid = quadratic_subpixel_extremum_1d(filtered_profile,right_fid)

    return left_fid/l, right_fid/l

def find_grips(profile, threshold = 2):
    """
    Presume that the grips appear as peaks in the filtered slope of the profile i.e. curvature == 0
    left grip is the first one and right grip is the last one
    eliminate false positives with a threshold on the steepness (slope)

    Parameters
    ----------
    profile
    threshold

    Returns
    -------
    left_grip, right_grip
    """
    l = len(profile)
    kernel = np.gradient(gaussian(l // 50, l / 500).astype(np.float32))
    kernel /= np.sum(np.abs(kernel)) # This keeps the values of the slope array stable for thresholding
    slope = convolve(profile - profile[0], kernel, 'same') # shift to eliminate edge effects
    curvature = np.gradient(slope)
    inflections = find_zero_crossings(curvature, 'all')
    # print(np.max(np.abs(slope)))
    inflections &= (np.abs(slope) > threshold)
    inflections = np.where(inflections)[0]
    # print(inflections)
    try:
        left_grip = inflections[0]
        right_grip = inflections[-1]
    except:
        left_grip, right_grip = 0, 0
    return left_grip, right_grip

def locate_fids(image, p_level, filter_width, cutoff, stabilize_args=()):
    if isinstance(image, str):  # TODO: more robust dispatch
        print('Processing: ' + path.basename(image))
        from fiber_locator import load_stab_img
        image = load_stab_img(image, *stabilize_args)
        pyramid = make_pyramid(image, p_level)
    elif isinstance(image, np.ndarray):
        pass
    else:
        raise ValueError('Unknown type', type(image))

    pyramid_image = pyramid[p_level]
    profile = fid_profile_from_image(pyramid_image)
    filter_width_pixels = filter_width * len(profile)
    left_grip_index, right_grip_index = find_grips(profile)
    filtered_profile = wavelet_filter(profile, filter_width_pixels)
    return choose_fids(filtered_profile, cutoff, left_grip_index)


class NoPeakError(Exception):
    pass


def load_fids(image_path, image=None, fid_args=()):
    lpath = image_path.lower()
    dirname = path.dirname(lpath)
    fid_file = path.join(dirname, FIDUCIAL_FILENAME)
    basename = basename_without_stab(image_path)
    try:  # if os.path.exists(fid_file):
        with open(fid_file) as fp:
            import json
            headers, data = json.load(fp)

        left_fid, right_fid, strain = data[basename]
    except FileNotFoundError:  # else:
        left_fid, right_fid = locate_fids(image, *fid_args)

    return left_fid, right_fid, strain

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
    folder = path.dirname(images[0])
    from util import parse_strain_headers
    tdi_length = parse_strain_headers(folder)[1]
    initial_displacement = right_fids[0] - left_fids[0]
    strains = (right_fids - left_fids) / initial_displacement - 1
    parameters.update({'initial_displacement': initial_displacement * tdi_length * 1e4,
                       'fields': 'name: (left, right, strain)'})


    fnames = [basename_without_stab(image) for image in images]
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
    a = FidGUI(get_files())
    # a = batch_fids(get_files(), 7000, 1000)


    # images = get_files()
    # images = ['c:\\users\\rjs3\\onedrive\\data\\sfft\\10051319\\stab_str000.jpg',
              # 'c:\\users\\rjs3\\onedrive\\data\\sfft\\10051319\\stab_str01d.jpg',
              # 'c:\\users\\rjs3\\onedrive\\data\\sfft\\10051319\\stab_str02d.jpg',
              # 'c:\\users\\rjs3\\onedrive\\data\\sfft\\10051319\\stab_str03d.jpg',
              # 'c:\\users\\rjs3\\onedrive\\data\\sfft\\10051319\\stab_str04d.jpg',
              # 'c:\\users\\rjs3\\onedrive\\data\\sfft\\10051319\\stab_str05d.jpg',
              # 'c:\\users\\rjs3\\onedrive\\data\\sfft\\10051319\\stab_str06d.jpg',
              # 'c:\\users\\rjs3\\onedrive\\data\\sfft\\10051319\\stab_str07d.jpg',
              # 'c:\\users\\rjs3\\onedrive\\data\\sfft\\10051319\\stab_str08d.jpg',
              # 'c:\\users\\rjs3\\onedrive\\data\\sfft\\10051319\\stab_str09d.jpg',
              # 'c:\\users\\rjs3\\onedrive\\data\\sfft\\10051319\\stab_str10d.jpg',
              # 'c:\\users\\rjs3\\onedrive\\data\\sfft\\10051319\\stab_str11d.jpg',
              # 'c:\\users\\rjs3\\onedrive\\data\\sfft\\10051319\\stab_str12d.jpg',
              # 'c:\\users\\rjs3\\onedrive\\data\\sfft\\10051319\\stab_str13d.jpg',
              # 'c:\\users\\rjs3\\onedrive\\data\\sfft\\10051319\\stab_str14d.jpg',
              # 'c:\\users\\rjs3\\onedrive\\data\\sfft\\10051319\\stab_str15d.jpg',
              # 'c:\\users\\rjs3\\onedrive\\data\\sfft\\10051319\\stab_str16d.jpg',
              # 'c:\\users\\rjs3\\onedrive\\data\\sfft\\10051319\\stab_str17d.jpg',
              # 'c:\\users\\rjs3\\onedrive\\data\\sfft\\10051319\\stab_str18d.jpg',
              # 'c:\\users\\rjs3\\onedrive\\data\\sfft\\10051319\\stab_str19d.jpg',
              # 'c:\\users\\rjs3\\onedrive\\data\\sfft\\10051319\\stab_str20d.jpg',
              # 'c:\\users\\rjs3\\onedrive\\data\\sfft\\10051319\\stab_str21d.jpg',
              # 'c:\\users\\rjs3\\onedrive\\data\\sfft\\10051319\\stab_str22d.jpg',
              # 'c:\\users\\rjs3\\onedrive\\data\\sfft\\10051319\\stab_str23d.jpg',
              # 'c:\\users\\rjs3\\onedrive\\data\\sfft\\10051319\\stab_str24d.jpg',
              # 'c:\\users\\rjs3\\onedrive\\data\\sfft\\10051319\\stab_tdiz.jpg']

    # from collections import OrderedDict

    # parameters = OrderedDict([('p_level', 5), ('filter_width', 0.014616477272727278), ('cutoff', 33.06818181818182)])
    # images = a.images
    # parameters = a.parameters

    # from cProfile import run

    # run('batch(locate_fids, images, *parameters.values())', sort='time')
