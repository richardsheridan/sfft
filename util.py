import sys, os, json
from tkinter import Tk
from tkinter.filedialog import askopenfilename, asksaveasfilename
import numpy as np
# from numpy import convolve
from scipy.signal import fftconvolve as convolve, ricker, gaussian
from numbers import Integral as Int, Number

STABILIZE_PREFIX = 'stab_'

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


def wavelet_filter(series, window_size, bandwidth=None):
    sigma = window_size / 9
    if bandwidth is None or bandwidth <= 0:
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


def find_crossings(smooth_series, slope_cutoff=None):
    candidate_crossings = (smooth_series > 0) & (np.roll(smooth_series, shift=-1, axis=-1) < 0)

    if slope_cutoff is not None:
        sufficient_slope = (np.gradient(smooth_series, axis=-1) < slope_cutoff)
        candidate_crossings &= sufficient_slope

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


def make_pyramid(image, levels=7):
    import cv2
    pyramid = [image]
    x = image
    for i in range(levels):
        x = cv2.pyrDown(x)
        pyramid.append(x)
    return pyramid


def make_log_pyramid(pyramid):
    import cv2
    return [cv2.Laplacian(p, cv2.CV_16S) for p in pyramid]


def display_pyramid(pyramid, cmap='gray'):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(len(pyramid), 1)
    if len(pyramid) <= 1:
        axs = [axs]
    for im, ax in zip(pyramid, axs):
        ax.imshow(im, cmap=cmap)
    plt.show()


def make_dogs(pyramid, output_dtype='int16', intermediate_dtype='int16'):
    import cv2
    output_dtype = np.dtype(output_dtype)
    intermediate_dtype = np.dtype(intermediate_dtype)
    dogs = [cv2.pyrUp(small, dstsize=big.shape[::-1]).astype(intermediate_dtype) - big.astype(intermediate_dtype)
            for big, small in zip(pyramid, pyramid[1:])]
    if output_dtype != intermediate_dtype:
        dogs = [dog.astype(output_dtype) for dog in dogs]
    return dogs






def peak_local_max(image: np.ndarray, threshold=None, neighborhood=1):
    if threshold is None:
        maxima = np.ones_like(image, bool)
    elif isinstance(threshold, Number):
        maxima = image > threshold
    else:
        maxima = np.asanyarray(threshold, bool)

    rows, cols = image.shape

    n = neighborhood
    for candidate in zip(*np.where(maxima)):
        if not maxima[candidate]:
            continue

        r, c = candidate
        rowslice = slice(max(r - n, 0), min(r + n + 1, rows))
        colslice = slice(max(c - n, 0), min(c + n + 1, cols))
        neighborhood_array = image[rowslice, colslice]
        neighborhood_maxima = maxima[rowslice, colslice]
        # TODO: incorporate footprint matrix for round neighborhood
        neighborhood_maxima[:] = False  # there can be only one
        neighborhood_maxima[np.unravel_index(np.argmax(neighborhood_array),
                                             neighborhood_maxima.shape)] = True  # Highlander!
    return maxima

if __name__ == '__main__':
    import cv2
    from time import perf_counter

    image = cv2.imread(r'C:\Users\rjs3\Desktop\test.jpg', cv2.IMREAD_GRAYSCALE)
    import matplotlib.pyplot as plt

    footprint = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    t0 = perf_counter()
    result = peak_local_max(image, 220, 40)
    t1 = perf_counter()
    derp = ((cv2.dilate(image, footprint) == image) & (image > 220))
    t2 = perf_counter()
    result2 = peak_local_max(image, derp, 40)
    t3 = perf_counter()
    print('easy:', t1 - t0, np.count_nonzero(result))
    print('derp:', t2 - t1, np.count_nonzero(derp))
    print('hard:', t3 - t2, np.count_nonzero(result2))

    # im2=cv2.imread(r'C:\Users\rjs3\onedrive\data\sfft\09091321\STR29D.TIF',cv2.IMREAD_GRAYSCALE)
    im2 = image
    pyr = make_pyramid(im2)
    # plt.imshow(im3,'gray');plt.show(1)
    t4 = perf_counter()
    im3 = make_dog(pyr[2], 16, 10)
    result3 = peak_local_max(im3, 15, 10)
    t5 = perf_counter()
    print('dogs:', t5 - t4, np.count_nonzero(result3))
