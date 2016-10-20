import sys, os, json
from tkinter import Tk
from tkinter.filedialog import askopenfilename, asksaveasfilename
import numpy as np
# from numpy import convolve
from scipy.signal import fftconvolve as convolve, ricker, gaussian


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
    fullpaths = askopenfilename(multiple=1, filetypes=(('TIF', '.tif'), ('All files', '*')))
    fullpaths = [os.path.normpath(path) for path in sorted(fullpaths)]

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
        output[ind, :] = convolve(data, wavelet_data,
                                  mode='same')
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
    if filename.startswith('STAB_'):
        return path
    else:
        return os.path.join(dirname, 'STAB_' + filename)


def basename_without_stab(images):
    name_gen = (os.path.basename(x).lower() for x in images)
    return [x[len('stab_'):] if x.startswith('stab_') else x for x in name_gen]


def parse_strain_dat(straindatpath, max_cycle=None):
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
                                                              ('c', float),
                                                              ],
                                                       unpack=True,
                                                       )
    cycle = cycle.astype(int)
    if max_cycle is not None:
        cycle = cycle[cycle <= max_cycle]
    # fudge cycle number to trigger logging of first and last data points
    cycle[0] = 0
    if s_or_d[-1] == b'delay':  # detect if cycle ended properly
        cycle[-1] = 0
    cycle_changes = np.where(np.diff(cycle))

    force = force[cycle_changes]
    time = time[cycle_changes]
    extension = extension[cycle_changes]

    # TODO: return maximum force in a cycle
    return force / width / thickness, label
