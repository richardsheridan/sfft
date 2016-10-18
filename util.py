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
    """
    Continuous wavelet transform.

    Performs a continuous wavelet transform on `data`,
    using the `wavelet` function. A CWT performs a convolution
    with `data` using the `wavelet` function, which is characterized
    by a width parameter and length parameter.

    Parameters
    ----------
    data : (N,) ndarray
        data on which to perform the transform.
    wavelet : function
        Wavelet function, which should take 2 arguments.
        The first argument is the number of points that the returned vector
        will have (len(wavelet(length,width)) == length).
        The second is a width parameter, defining the size of the wavelet
        (e.g. standard deviation of a gaussian). See `ricker`, which
        satisfies these requirements.
    widths : (M,) sequence
        Widths to use for transform.

    Returns
    -------
    cwt: (M, N) ndarray
        Will have shape of (len(widths), len(data)).

    Notes
    -----
    ::

        length = min(10 * width[ii], len(data))
        cwt[ii,:] = signal.convolve(data, wavelet(length,
                                    width[ii]), mode='same')

    Examples
    --------
    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt
    >>> t = np.linspace(-1, 1, 200, endpoint=False)
    >>> sig  = np.cos(2 * np.pi * 7 * t) + signal.gausspulse(t - 0.4, fc=2)
    >>> widths = np.arange(1, 31)
    >>> cwtmatr = signal.cwt(sig, signal.ricker, widths)
    >>> plt.imshow(cwtmatr, extent=[-1, 1, 31, 1], cmap='PRGn', aspect='auto',
    ...            vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
    >>> plt.show()

    """
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
