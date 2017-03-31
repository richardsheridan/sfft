import sys, os, json
from tkinter import Tk
from tkinter.filedialog import askopenfilename, asksaveasfilename, askdirectory
import numpy as np
# from numpy import convolve
from vendored_scipy import fftconvolve as convolve, ricker, gaussian
from numbers import Integral as Int, Number
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Executor, Future

STABILIZE_PREFIX = 'stab_'
VALID_IMAGE_EXTENSIONS = frozenset(('.tif', '.jpg', '.png'))
VALID_ZERO_CROSSING_DIRECTIONS = frozenset(('upward', 'downward', 'all'))
PARALLEL_BATCH_PROCESSING = True

PIXEL_SIZE_X = .7953179315  # microns per pixel
PIXEL_SIZE_Y = .347386919  # microns per pixel


class SynchronousExecutor(Executor):
    def submit(self, fn, *args, **kwargs):
        f = Future()
        result = fn(*args, **kwargs)
        # The future is now
        f.set_result(result)
        return f

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


def batch(function, image_paths, *args, parallel=PARALLEL_BATCH_PROCESSING):
    """
    Pool.map can't deal with lambdas, closures, or functools.partial, so we fake it with itertools
    :param function:
    :param image_paths:
    :param args:
    :param parallel:
    :return:
    """
    from itertools import starmap, repeat
    args = repeat(args)
    args = [(image_path, *arg) for image_path, arg in zip(image_paths, args)]

    if parallel:
        from multiprocessing import pool, freeze_support
        freeze_support()
        p = pool.Pool()  # TODO: one long-lived pool would be better
        starmap = p.starmap

    return list(starmap(function, args))


def get_files():
    """
    Open a dialog and return a set of files to parse.
    """
    # we don't want a full GUI, so keep the root window from appearing
    Tk().withdraw()

    # show an "Open" dialog box and return the paths to the selected files
    fullpaths = askopenfilename(multiple=1, filetypes=(('TIF', '.tif'),
                                                       ('JPEG', ('.jpg', '.jpeg')),
                                                       ('Images', ('.tif', '.jpg', '.jpeg')),
                                                       ('All files', '*')))
    fullpaths = sorted(os.path.normpath(path.lower()) for path in fullpaths)

    if len(fullpaths):
        print('User opened:', *fullpaths, sep='\n')
    else:
        print('No files selected')
        sys.exit()

    return fullpaths


def get_folder():
    """
    Open a dialog and return a folder.
    """
    # we don't want a full GUI, so keep the root window from appearing
    Tk().withdraw()

    # show an "Open" dialog box and return the selected folder
    directory = askdirectory()

    if len(directory):
        print('User opened:', directory, sep='\n')
    else:
        print('No files selected')
        sys.exit()

    return directory


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


def make_future_pyramids(image_paths, pyramid_loader, load_args, executor=None):
    """
    Make a list of futures such that all images are loaded in parallel but one can be requested ASAP
    
    This works almost like Executor.map except we don't iterate over the list and request results for you
    This way it is a bit lazier, but client code needs to be aware of calling .result() on the futures

    Parameters
    ----------
    image_paths
    pyramid_loader
    load_args
    executor

    Returns
    -------
    list of futures containing image pyramids
    """
    # NOTE: as of 3/20/17 0fc7d9d, TPE and PPE are the same speed for normal workloads, so use safer PPE
    if executor is None:
        # executor = ProcessPoolExecutor()
        # executor = ThreadPoolExecutor()
        executor = SynchronousExecutor()
    future_pyramids = [executor.submit(pyramid_loader, image_path, *load_args)
                       for image_path in image_paths]
    executor.shutdown(False)
    return future_pyramids

def wavelet_filter(series, sigma, bandwidth=None):
    window_size = int(sigma * 9)
    if window_size <= 3:
        window_array = np.array((-1.0, 2.0, -1.0))
    elif window_size <= 5:
        window_array = np.array((-1.0, 16.0, -30.0, 16.0, -1.0))
    elif window_size <= 7:
        window_array = np.array((2.0, -27.0, 270.0, -490.0, 270.0, -27.0, 2.0))
    elif bandwidth is None or bandwidth <= 0:
        window_array = ricker(window_size, sigma)
    else:
        narrow_window = gaussian(window_size, sigma - bandwidth)
        narrow_window /= narrow_window.sum()
        wide_window = gaussian(window_size, sigma)
        wide_window /= wide_window.sum()
        window_array = narrow_window - wide_window

    window_array /= np.abs(window_array).sum()  # this keeps the values of the smoothed array stable wrt window
    smoothed = convolve(series, window_array, 'same')
    return smoothed


def find_zero_crossings(smooth_series, direction='downward'):
    """

    Parameters
    ----------
    smooth_series
    direction

    Returns
    -------
    np.ndarray

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(-2,2,10)
    >>> y = x**2-1
    >>> y
    array([ 3.        ,  1.41975309,  0.2345679 , -0.55555556, -0.95061728,
           -0.95061728, -0.55555556,  0.2345679 ,  1.41975309,  3.        ])
    >>> find_zero_crossings(y)
    array([False, False,  True, False, False, False, False, False, False, False], dtype=bool)
    >>> find_zero_crossings(y,'upward')
    array([False, False, False, False, False, False,  True, False, False, False], dtype=bool)
    >>> find_zero_crossings(y, 'all')
    array([False, False,  True, False, False, False,  True, False, False, False], dtype=bool)
    """
    series_shifted_left = np.roll(smooth_series, shift=-1, axis=-1)
    if direction not in VALID_ZERO_CROSSING_DIRECTIONS:
        raise ValueError('Invalid choice of direction:', direction)
    if direction == 'downward':
        candidate_crossings = (smooth_series >= 0) & (series_shifted_left < 0)
    elif direction == 'upward':
        candidate_crossings = (smooth_series < 0) & (series_shifted_left > 0)
    elif direction == 'all':
        candidate_crossings = (smooth_series * series_shifted_left < 0) | (smooth_series == 0)
    else:
        raise AssertionError('should not get here, check to see if you covered every choice for direction')
    candidate_crossings[-1] = 0

    return candidate_crossings


def path_with_stab(path):
    dirname, filename = os.path.split(path)
    filename, ext = os.path.splitext(filename)
    if ext.lower() not in VALID_IMAGE_EXTENSIONS:
        raise ValueError('File has invalid file extension.', ext)
    if '.' in filename:
        import warnings
        warnings.warn('Better not have periods in your filenames, it might break things')

    if filename.lower().startswith(STABILIZE_PREFIX):
        return path
    else:
        return os.path.join(dirname, STABILIZE_PREFIX + filename + '.jpg')


def basename_without_stab(image_path):
    image_path = os.path.splitext(os.path.basename(image_path))[0]

    if image_path.lower().startswith(STABILIZE_PREFIX):
        image_path = image_path[len(STABILIZE_PREFIX):]

    return image_path


def grab_first_number(string):
    return float(string.split()[0])

def parse_strain_headers(straindatpath):
    straindatpath = normalize_straindatpath(straindatpath)
    with open(straindatpath) as f:
        for i, line in enumerate(f):
            if i == 0 and not line.startswith(r'C:\IFSS'):
                raise ValueError('Not a proper IFSS strain file')
            if i == 2:
                label = line
            elif i == 8:
                tdi_length = grab_first_number(line)
            elif i == 13:
                width = grab_first_number(line)
            elif i == 14:
                thickness = grab_first_number(line)
            elif i == 16:
                fid_estimate = grab_first_number(line)
                break
    return label, tdi_length, width, thickness, fid_estimate


def parse_strain_columns(straindatpath):
    straindatpath = normalize_straindatpath(straindatpath)
    extension, force, time, s_or_d, cycle = np.loadtxt(straindatpath,
                                                       skiprows=28,
                                                       usecols=(0, 1, 2, 3, 4),
                                                       dtype=[('ex', 'float'),
                                                              ('f', 'float'),
                                                              ('t', 'float'),
                                                              ('sd', '<S6'),
                                                              ('c', 'int64'),
                                                              ],
                                                       unpack=True,
                                                       )
    return extension, force, time, s_or_d, cycle


def normalize_straindatpath(straindatpath):
    if os.path.isdir(straindatpath):
        straindatpath = os.path.join(straindatpath, 'STRAIN.DAT')
    return straindatpath


def parse_strain_dat(straindatpath, max_cycle=None, stress_type='after_tdi'):
    straindatpath = normalize_straindatpath(straindatpath)

    label, tdi_length, width, thickness, fid_estimate = parse_strain_headers(straindatpath)
    extension, force, time, s_or_d, cycle = parse_strain_columns(straindatpath)
    stress = force / (width * thickness)

    if max_cycle is not None:
        cycle = cycle[cycle <= max_cycle]

    if s_or_d[-1] == b'delay':  # detect if cycle ended properly
        cycle[-1] = 0

    cycle_changes = np.diff(cycle)

    before_tdi = np.where(cycle_changes)[0]
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

    if issubclass(stress_type, Number):
        return stress[np.searchsorted(time, time_at_max_stress + stress_type)], label

    raise ValueError('Could not interpret stress type: ' + str(stress_type))


def image_argmax(array):
    return np.unravel_index(np.argmax(array), array.shape)


def peak_local_max(image: np.ndarray, threshold=None, neighborhood=1, border=1, subpixel=1):
    if threshold is None:
        maxima = np.ones_like(image, bool)
    elif isinstance(threshold, Number):
        maxima = image > threshold
    else:
        maxima = np.array(threshold, bool)

    # Technically this could replace all the following logic, but it is slower because it works on each pixel
    # import cvutil
    # maxima &= (image ==  cvutil.max_filter(image, neighborhood))
    # if border:
    #     maxima[:border] = False
    #     maxima[-border:] = False
    #     maxima.T[:border] = False
    #     maxima.T[-border:] = False
    # return np.transpose(cvutil.argwhere(maxima))

    rows, cols = image.shape
    max_indices = set()
    all_candidates = zip(*np.where(maxima))  # This call to np.where is the bottleneck for large images
    # all_candidates = cvutil.argwhere(maxima)
    for candidate in all_candidates:
        # If we have eliminated a candidate in a previous iteration, we can skip ahead
        if not maxima[candidate]:
            continue

        while True:
            r, c = candidate
            r0, r1 = max(r - neighborhood, 0), min(r + neighborhood + 1, rows)
            c0, c1 = max(c - neighborhood, 0), min(c + neighborhood + 1, cols)

            # Wipe all maximum candidates in the neighborhood so we don't check others later
            maxima[r0:r1, c0:c1] = False

            # Locate the actual maximum of the neighborhood
            max_index = rm, cm = image_argmax(image[r0:r1, c0:c1])

            # Shift max_index to image coordinates
            max_index = rm, cm = rm + r0, cm + c0
            if max_index == candidate:
                break
            else:
                candidate = max_index

        # Drop any "maxima" near the border of the image, which are by and large false positives
        if border <= rm < rows - border and border <= cm < cols - border:
            max_indices.add(max_index)

    if subpixel:
        max_indices = [quadratic_subpixel_extremum_2d(image, max_index) for max_index in max_indices]

    # Emulate np.where behavior for empty max_indices with this special conditional expression
    return np.transpose(max_indices) if max_indices else (np.array([], dtype=int), np.array([], dtype=int))


def quadratic_subpixel_extremum_2d(image, max_index):
    # Image and Vision Computing, vol.20, no.13/14, pp.981-991, December 2002.
    # http://vision-cdc.csiro.au/changs/doc/sun02ivc.pdf
    y, x = max_index

    # optimization: grab the image pixel values as python floats in local variables
    image_item = image.item
    try:
        bmm = image_item((y - 1, x - 1))
        b0m = image_item((y - 1, x))
        bpm = image_item((y - 1, x + 1))
        bm0 = image_item((y, x - 1))
        b00 = image_item((y, x))
        bp0 = image_item((y, x + 1))
        bmp = image_item((y + 1, x - 1))
        b0p = image_item((y + 1, x))
        bpp = image_item((y + 1, x + 1))
    except IndexError:
        return max_index

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
    if denominator > 0.0:  # eliminate saddles or divide by zero problems
        subpixel_x = (B * E - 2.0 * C * D) / denominator
        subpixel_y = (B * D - 2.0 * A * E) / denominator

        # check for badly behaved estimates, possibly due to noise?
        if -1.0 < subpixel_x < 1.0 and -1.0 < subpixel_y < 1.0:
            # subpx_max_val = A * subpixel_x ** 2 + B * subpixel_y * subpixel_x + C * subpixel_y ** 2 + \
            #                 D * subpixel_x + E * subpixel_y + F
            return y + subpixel_y, x + subpixel_x,  # subpx_max_val

    return y, x,  # b00


def quadratic_subpixel_extremum_1d(profile, max_index):
    # optimization: grab the image pixel values as python floats in local variables
    """

    Parameters
    ----------
    profile
    max_index

    Returns
    -------

    Examples
    --------
    >>> import numpy as np
    >>> profile = np.array([ 53.16374747,  53.20836579,  52.61800576])
    >>> quadratic_subpixel_extremum_1d(profile,1)
    0.570267466599464
    """
    profile_item = profile.item
    try:
        bm = profile_item(max_index - 1)
        b0 = profile_item(max_index)
        bp = profile_item(max_index + 1)
    except IndexError:
        return max_index

    # estimate the coefficients of D(x)=Axx+Bx+C by finite difference
    # the coordinate system origin is implicitly shifted to x from max_index
    A = (bp + bm) / 2 - b0
    B = (bp - bm) / 2
    # C = b0

    # Solve the system Gx(x) = 0 for the quadratic extremum
    if A:  # eliminate divide by zero problems
        subpixel_x = -B / 2 / A

        # check for badly behaved estimates, possibly due to noise?
        if -1.0 < subpixel_x < 1.0:
            # subpx_max_val = A * subpixel_x ** 2 + B * subpixel_x + C
            return max_index + subpixel_x # ,subpx_max_val

    return max_index


def vshift_from_si_shape(slope, intercept, shape):
    """
    Calculate a vertical shift factor to place a line in the middle of an image

    Parameters
    ----------
    slope: float
    intercept: float between 0 and 1 (relative intercept)
    shape: Tuple[int,int]

    Returns
    -------
    vshift: float
    """
    r, c = shape
    x, y = c-1, r-1
    x_middle = x // 2
    y_middle = x_middle * slope + intercept * y
    return y // 2 - y_middle


def si_from_ct(centroid, theta, shape):
    slope = np.tan(theta)
    intercept = centroid[1] - slope * centroid[0]
    return slope, intercept/(shape[0]-1)
