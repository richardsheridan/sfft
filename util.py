import sys, os, json
from tkinter import Tk
from tkinter.filedialog import askopenfilename, asksaveasfilename
import numpy as np
# from numpy import convolve
from scipy.signal import fftconvolve as convolve, ricker, gaussian
from numbers import Integral as Int, Number

STABILIZE_PREFIX = 'stab_'
DISPLAY_SIZE = (1200, 450)

PIXEL_SIZE_X = .7953179315  # microns per pixel
PIXEL_SIZE_Y = .347386919  # microns per pixel

class NdarrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


_default_encoder = NdarrayEncoder(indent=4, sort_keys=True)


def dumps(obj):
    return _default_encoder.encode(obj)


def dump(obj, fp):
    fp.writelines(_default_encoder.iterencode(obj))

def batch(function,image_paths, *args):
    """
    Pool.map can't deal with lambdas, closures, or functools.partial, so we fake it with itertools
    :param function:
    :param image_paths:
    :param args:
    :return:
    """
    from itertools import starmap, repeat
    args = repeat(args)
    args = [(image_path, *arg) for image_path, arg in zip(image_paths, args)]

    # from multiprocessing import pool, freeze_support
    # freeze_support()
    # p = pool.Pool()
    # starmap = p.starmap

    return list(starmap(function, args))


def get_files():
    """
    Open a dialog and return a set of files to parse.
    """
    # we don't want a full GUI, so keep the root window from appearing
    Tk().withdraw()

    # show an "Open" dialog box and return the paths to the selected files
    fullpaths = askopenfilename(multiple=1, filetypes=(('Images', ('.tif', '.jpg', '.jpeg')),
                                                       ('TIF', '.tif'),
                                                       ('JPEG', ('.jpg', '.jpeg')),
                                                       ('All files', '*')))
    fullpaths = sorted(os.path.normpath(path.lower()) for path in fullpaths)

    if len(fullpaths):
        print('User opened:', *fullpaths, sep='\n')
    else:
        print('No files selected')
        sys.exit()

    return fullpaths


def put_file():
    """
    Open a dialog and return a path for save file.
    """
    # we don't want a full GUI, so keep the root window from appearing
    Tk().withdraw()

    # show an "Open" dialog box and return the paths to the selected files
    fullpath = asksaveasfilename(filetypes=(('DAT', '.dat'), ('All files', '*')),
                                 defaultextension='.dat')

    if fullpath:
        print('Saving data as:', fullpath)
        return fullpath
    else:
        print('No file selected')
        return


def cwt(data, wavelet, widths):
    output = np.zeros([len(widths), len(data)])
    for ind, width in enumerate(widths):
        wavelet_data = wavelet(min(width, len(data)), width / 9)
        output[ind, :] = convolve(data, wavelet_data, mode='same')
    return output


def wavelet_filter(series, sigma, bandwidth=None):
    window_size = int(sigma * 9)
    if window_size <= 3:
        window_array = np.array((-1.0,2.0,-1.0))
    elif window_size <= 5:
        window_array = np.array((-1.0,16.0,-30.0,16.0,-1.0))
    elif window_size <= 7:
        window_array = np.array((2.0,-27.0,270.0,-490.0,270.0,-27.0,2.0))
    elif bandwidth is None or bandwidth <= 0:
        window_array = ricker(window_size, sigma)
    else:
        narrow_window = gaussian(window_size, sigma - bandwidth)
        narrow_window /= narrow_window.sum()
        wide_window = gaussian(window_size, sigma)
        wide_window /= wide_window.sum()
        window_array = narrow_window - wide_window

    window_array /= np.abs(window_array).sum()
    smoothed = convolve(series, window_array, 'same')
    return smoothed


def find_zero_crossings(smooth_series, direction='downward'):
    series_shifted_left = np.roll(smooth_series, shift=-1, axis=-1)
    if direction == 'downward':
        candidate_crossings = (smooth_series >= 0) & (series_shifted_left < 0)
    elif direction == 'upward':
        candidate_crossings = (smooth_series < 0) & (series_shifted_left > 0)
    elif direction == 'all':
        candidate_crossings = (smooth_series * series_shifted_left < 0) | (smooth_series == 0)
    else:
        raise ValueError('Invalid choice of direction:', direction)
    candidate_crossings[-1] = 0

    return candidate_crossings


def path_with_stab(path):
    dirname, filename = os.path.split(path)
    filename, ext = os.path.splitext(filename)
    if filename.lower().startswith(STABILIZE_PREFIX):
        return path
    else:
        return os.path.join(dirname, STABILIZE_PREFIX + filename + '.jpg')


def basename_without_stab(images):
    name_gen = (os.path.splitext(os.path.basename(x))[0].lower() for x in images)
    return [x[len(STABILIZE_PREFIX):] if x.startswith(STABILIZE_PREFIX) else x for x in name_gen]


def parse_strain_dat(straindatpath, max_cycle=None, stress_type='max'):
    if os.path.isdir(straindatpath):
        straindatpath = os.path.join(straindatpath, 'STRAIN.DAT')
    with open(straindatpath) as f:
        for i, line in enumerate(f):
            if i == 0 and not line.startswith(r'C:\IFSS'):
                raise ValueError('Not a proper IFSS strain file')
            if i == 2:
                label = line
            elif i == 13:
                width = float(line.split()[0])
            elif i == 14:
                thickness = float(line.split()[0])
            elif i == 16:
                fid_estimate = float(line.split()[0])
                break
    extension, force, time, s_or_d, cycle = np.loadtxt(straindatpath,
                                                       skiprows=28,
                                                       usecols=(0, 1, 2, 3, 4),
                                                       dtype=[('ex', float),
                                                              ('f', float),
                                                              ('t', float),
                                                              ('sd', '<S6'),
                                                              ('c', int),
                                                              ],
                                                       unpack=True,
                                                       )
    stress = force / (width * thickness)

    if max_cycle is not None:
        cycle = cycle[cycle <= max_cycle]

    if s_or_d[-1] == b'delay':  # detect if cycle ended properly
        cycle[-1] = 0

    cycle_changes = np.diff(cycle)

    before_tdi = np.nonzero(cycle_changes)[0]
    if stress_type == 'before_tdi':
        return stress[before_tdi], label

    after_tdi = np.concatenate(([0], before_tdi[:-1] + 1))
    if stress_type == 'after_tdi':
        return stress[after_tdi], label

    max_stress = np.array([np.argmax(stress[low:high]) + low for low, high in zip(after_tdi, before_tdi)])
    if stress_type == 'max':
        return stress[max_stress], label

    time_at_max_stress = time[max_stress]
    if stress_type == 'all':
        tensec = np.searchsorted(time, time_at_max_stress + 10)

        eightmin = np.searchsorted(time, time_at_max_stress + (8 * 60))

        return stress, (before_tdi, after_tdi, max_stress, tensec, eightmin), label

    if issubclass(stress_type, Int):
        return stress[np.searchsorted(time, time_at_max_stress + stress_type)], label

    raise ValueError('Could not interpret stress type: ' + str(stress_type))

def peak_local_max(image: np.ndarray, threshold=None, neighborhood=1, border=1, subpixel=1):
    if threshold is None:
        maxima = np.ones_like(image, bool)
    elif isinstance(threshold, Number):
        maxima = image > threshold
    else:
        maxima = np.array(threshold, bool)

    # Technically this could replace all the following logic, but it is slower because it works on each pixel
    # maxima &= (image ==  max_filter(image, neighborhood, elliptical))
    # if border:
    #     maxima[:border] = False
    #     maxima[-border:] = False
    #     maxima.T[:border] = False
    #     maxima.T[-border:] = False

    rows, cols = image.shape
    max_indices = []
    for candidate in zip(*np.where(maxima)): # This call to np.where is the bottleneck for large images
        # If we have eliminated a candidate in a previous iteration, we can skip ahead
        if not maxima[candidate]:
            continue

        r, c = candidate
        r0, r1 = max(r - neighborhood, 0), min(r + neighborhood + 1, rows)
        c0, c1 = max(c - neighborhood, 0), min(c + neighborhood + 1, cols)

        # Wipe all maximum candidates in the neighborhood so we don't check others later
        maxima[r0:r1, c0:c1] = False

        # Locate the actual maximum of the neighborhood
        neighborhood_array = image[r0:r1, c0:c1]
        max_index = rm, cm = np.unravel_index(np.argmax(neighborhood_array), neighborhood_array.shape)

        # Shift max_index to image coordinates
        max_index = rm, cm = rm+r0, cm+c0

        # Drop any "maxima" near the border of the image, which are by and large false positives
        if border < rm < rows-border and border < cm < cols-border:
            maxima[max_index] = True
            if subpixel:
                max_index = quadratic_subpixel_maximum(image,max_index)
            max_indices.append(max_index)

    # Emulate np.where behavior for empty max_indices with this special conditional expression
    return np.transpose(max_indices) if max_indices else (np.array([], dtype=int), np.array([], dtype=int))


def quadratic_subpixel_maximum(image, max_index):
    # Image and Vision Computing, vol.20, no.13/14, pp.981-991, December 2002.
    # http://vision-cdc.csiro.au/changs/doc/sun02ivc.pdf
    y, x = max_index

    # optimization: grab the image pixel values as python floats in local variables
    image_item = image.item
    bmm = image_item((y - 1, x - 1))
    b0m = image_item((y - 1, x))
    bpm = image_item((y - 1, x + 1))
    bm0 = image_item((y, x - 1))
    b00 = image_item((y, x))
    bp0 = image_item((y, x + 1))
    bmp = image_item((y + 1, x - 1))
    b0p = image_item((y + 1, x))
    bpp = image_item((y + 1, x + 1))

    # estimate the coefficients of G(x,y)=Axx+Bxy+Cyy+Dx+Ey+F by finite difference
    # the coordinate system origin is implicitly shifted to (x,y) from max_index
    A = (bmm - 2.0 * b0m + bpm + bm0 - 2.0 * b00 + bp0 + bmp - 2.0 * b0p + bpp) / 6.0
    B = (bmm - bpm - bmp + bpp) / 4.0
    C = (bmm + b0m + bpm - 2.0 * bm0 - 2.0 * b00 - 2.0 * bp0 + bmp + b0p + bpp) / 6.0
    D = (-bmm + bpm - bm0 + bp0 - bmp + bpp) / 6.0
    E = (-bmm - b0m - bpm + bmp + b0p + bpp) / 6.0
    # F = (-bmm + 2.0 * b0m - bpm + 2.0 * bm0 + 5.0 * b00 + 2.0 * bp0 - bmp + 2.0 * b0p - bpp) / 9.0

    # Solve the system Gx(x,y) = 0, Gy(x,y) = 0 for the quadratic maximum
    denominator = (4.0 * A * C - B * B)
    if denominator > 0.0: # eliminate saddles or divide by zero problems
        subpixel_x = (B * E - 2.0 * C * D) / denominator
        subpixel_y = (B * D - 2.0 * A * E) / denominator

    # check for badly behaved estimates, possibly due to noise?
    if denominator > 0.0 and -1.0 < subpixel_x < 1.0 and -1.0 < subpixel_y < 1.0:
        # subpx_max_val = A * subpixel_x ** 2 + B * subpixel_y * subpixel_x + C * subpixel_y ** 2 + \
        #                 D * subpixel_x + E * subpixel_y + F
        return y + subpixel_y, x + subpixel_x,  # subpx_max_val
    else:
        return y, x, # b00