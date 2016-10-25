from os import path
from multiprocessing import pool, freeze_support

import cv2
import itertools
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import RadioButtons

from util import get_files, path_with_stab, STABILIZE_PREFIX, PIXEL_SIZE_X, PIXEL_SIZE_Y
from gui import MPLGUI


_map = itertools.starmap

STABILIZE_FILENAME = 'stabilize.json'

class FiberGUI(MPLGUI):
    def __init__(self, images, block=True, downsample=()):
        self.images = images
        self.downsample = downsample
        self.display_original = True
        self.display_rotated = False

        super().__init__()

    def create_layout(self):
        self.fig, self.axes['image'] = plt.subplots(figsize=(8, 10))
        self.fig.subplots_adjust(left=0.1, bottom=0.5)

        self.register_button('display', self.display_external, [.3, .95, .2, .03], label='Display full')
        self.register_button('save', self.execute_batch, [.3, .90, .2, .03], label='Save batch')
        self.register_button('type', self.display_type, [.6, .9, .2, .1], widget=RadioButtons, labels=('original', 'edges', 'rotated'))

        self.slider_coords =[.3, .4, .55, .03 ]
        self.register_slider('frame_number',self.update_frame_number,
                             isparameter=False,
                             forceint=True,
                             label='Frame number',
                             valmin=0,
                             valmax=len(self.images) - 1,
                             valinit=0,)
        self.register_slider('threshold',self.update_edge,
                             label='edge threshold',
                             valmin=0,
                             valmax=2 ** 9 - 1,
                             valinit=70,
                             )

    def load_frame(self):
        image_path = self.images[self.sliders['frame_number'].val]
        self.tdi_array = _load_tdi_corrected(image_path, self.downsample)

    def recalculate_vision(self):
        self.recalculate_edges()
        self.recalculate_lines()

    def recalculate_edges(self):
        threshold = self.sliders['threshold'].val
        self.edges = sobel_edges(self.tdi_array, threshold)

    def recalculate_lines(self):
        processed_image_array = self.edges

        self.slope, self.intercept, self.theta = fit_line_moments(processed_image_array)

        # slope, intercept, theta = fit_line_fitline(processed_image_array)

        self.vshift = _vshift_from_si_shape(self.slope, self.intercept, processed_image_array.shape)

    def draw_line(self, image_array, color=255, thickness=4):
        x_size = image_array.shape[1]
        max_chunk = 2 ** 14
        chunks = x_size // max_chunk
        image_parts = []
        for chunk in range(chunks):
            x0 = max_chunk * chunk
            y0 = int(x0 * self.slope + self.intercept)
            x1 = max_chunk * (chunk + 1)
            y1 = int(x1 * self.slope + self.intercept)
            image_parts.append(cv2.line(image_array[:, x0:x1],
                                        (0, y0),
                                        (max_chunk, y1),
                                        color,
                                        thickness,
                                        ))
        remainder = x_size % max_chunk
        if remainder:
            x0 = max_chunk * chunks
            y0 = int(x0 * self.slope + self.intercept)
            x1 = x_size - 1
            y1 = int(x1 * self.slope + self.intercept)
            image_parts.append(cv2.line(image_array[:, x0:x1],
                                        (0, y0),
                                        (x1 - x0, y1),
                                        color,
                                        thickness,
                                        ))

        output = np.hstack(image_parts)

        return output

    def refresh_plot(self):
        self.axes['image'].clear()

        if self.display_original:
            image = self.tdi_array.copy()
            if self.display_rotated:
                image = rotate_fiber(image, self.vshift, self.theta)
            else:
                image = self.draw_line(image, 0)
        else:
            image = self.edges * 255
            image = self.draw_line(image, 255)

        self.display_image_array = image.copy()
        image = cv2.resize(image, (800, 453), interpolation=cv2.INTER_AREA)
        self.axes['image'].imshow(image, cmap='gray')
        self.fig.canvas.draw()

    def display_external(self, event):
        cv2.namedWindow('display_external', cv2.WINDOW_NORMAL)
        cv2.imshow('display_external', self.display_image_array)
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()

    def display_type(self, label):
        ('original', 'edges', 'rotated')
        if label == 'original':
            self.display_original = True
            self.display_rotated = False
        elif label == 'edges':
            self.display_original = False
            self.display_rotated = False
        elif label == 'rotated':
            self.display_original = True
            self.display_rotated = True
        else:
            print('unknown display type:', label)
            return

        self.refresh_plot()

    def execute_batch(self, event):
        threshold, = self.parameters.values()
        return_image = False
        save_image = True
        images = self.images
        x = batch_stabilize(images, threshold, return_image, save_image)
        save_stab(images, x, threshold)

    def update_frame_number(self, val):
        self.load_frame()
        self.recalculate_vision()
        self.refresh_plot()

    def update_edge(self, val):

        self.recalculate_vision()
        self.refresh_plot()

    def update_open_close(self, val):
        self.update_edge(val)



def _si_from_ct(centroid, theta):
    slope = np.tan(theta)
    intercept = centroid[1] - slope * centroid[0]
    return slope, intercept


def _vshift_from_si_shape(slope, intercept, shape):
    x_middle = shape[1] // 2
    y_middle = x_middle * slope + intercept
    return shape[0] // 2 - y_middle


def _load_tdi_corrected(image_path, downsample=None):
    tdi_array = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if downsample and len(downsample) == 2 and downsample > (2, 2):
        x, y = downsample
    else:
        y, x = tdi_array.shape
        y = int(y * PIXEL_SIZE_Y / PIXEL_SIZE_X)  # shrink y to match x pixel size
        # x = int(x * PIXEL_SIZE_X / PIXEL_SIZE_Y) # grow x to match y pixel size

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(int(x / 2750),) * 2)
    tdi_array = clahe.apply(tdi_array)

    tdi_array = cv2.resize(tdi_array, dsize=(x, y,), interpolation=cv2.INTER_AREA)

    return tdi_array


def fit_line_moments(processed_image_array):
    """
    Use moments to generate whole-image centroid and angle
    works with grayscale data

    :param processed_image_array:
    :return slope, intercept, theta:
    """

    moments = cv2.moments(processed_image_array, True)

    m00 = moments['m00']
    if not m00:
        return 0, processed_image_array.shape[0] / 2, 0

    centroid = np.array((moments['m10'] / m00, moments['m01'] / m00))

    theta = np.arctan2(2 * moments['mu11'], (moments['mu20'] - moments['mu02'])) / 2
    slope, intercept = _si_from_ct(centroid, theta)
    return slope, intercept, theta


def fit_line_fitline(processed_image_array):
    """
    cv2.fitLine internally uses moments, but may iteratively reweight them for robust fit based on DIST_
    but points must be extracted manually

    :param processed_image_array:
    :return slope, intercept, theta:
    """
    points = np.argwhere(processed_image_array)[:, ::-1]  # swap x,y coords
    line = cv2.fitLine(points, cv2.DIST_L2, 0, .01, .01).ravel()
    centroid = line[2:]
    theta = np.arctan2(line[1], line[0])
    slope, intercept = _si_from_ct(centroid, theta)
    return slope, intercept, theta


def sobel_edges(image_array, threshold):
    dy = cv2.Sobel(image_array, cv2.CV_16S, 0, 1, ksize=3)
    edges = (dy > threshold).view('uint8')
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel, iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

    return edges


def rotate_fiber(image, vshift, theta):
    y_size, x_size = image.shape
    mean = int(image.mean())
    translation = np.float32([[1, 0, 0], [0, 1, vshift]])
    # aspect = 2.2894
    # stretch = np.float32([[aspect, 0, 0], [0, 1, 0]])
    # translation = _compose(translation,stretch)
    # theta = np.atan(2*np.tan(theta))
    theta_deg = theta * 180 / np.pi

    rotation = cv2.getRotationMatrix2D((x_size // 2, y_size // 2), theta_deg, 1)
    transform = _compose(rotation, translation)
    return cv2.warpAffine(image, transform, (x_size, y_size), borderValue=mean)

    # rotation = cv2.getRotationMatrix2D((x_size // 2, y_size // 2), theta_deg, 1)
    # transform = _compose(rotation, translation)
    # lefthalf = cv2.warpAffine(image[:, :x_size // 2], transform, (x_size // 2, y_size), borderValue=mean)
    #
    # rotation = cv2.getRotationMatrix2D((0, y_size // 2), theta_deg, 1)
    # transform = _compose(rotation, translation)
    # righthalf = cv2.warpAffine(image[:, x_size // 2:], transform, (x_size // 2, y_size), borderValue=mean)
    #
    # output = np.hstack((lefthalf, righthalf))
    #
    # patchwidth = 10
    # patchslice = slice(x_size // 2 - patchwidth // 2, x_size // 2 + patchwidth // 2)
    # rotation = cv2.getRotationMatrix2D((patchwidth // 2, y_size // 2), theta_deg, 1)
    # transform = _compose(rotation, translation)
    # cv2.warpAffine(image[:, patchslice], transform, (patchwidth, y_size), dst=output[:, patchslice],
    #                borderMode=cv2.BORDER_TRANSPARENT, borderValue=mean)
    #
    # return output


def _compose(a, b):
    return np.dot(np.vstack((a, (0, 0, 1))), np.vstack((b, (0, 0, 1))))[:-1, :]


def batch_stabilize(image_paths, *args):
    """
    Pool.map can't deal with lambdas, closures, or functools.partial, so we fake it with itertools
    :param image_paths:
    :param args:
    :return:
    """
    args = tuple(itertools.repeat(args, len(image_paths)))
    args = [(image_path, *arg) for image_path, arg in zip(image_paths, args)]
    freeze_support()
    p = pool.Pool()
    _map = p.starmap
    return _map(stabilize_file, args)


def load_stab_data(stabilized_image_path):
    dirname, basename = path.split(stabilized_image_path)
    datfile = path.join(dirname, STABILIZE_FILENAME)

    import json
    with open(datfile) as fp:
        header, data = json.load(fp)

    key = basename.lower()
    if key.startswith(STABILIZE_PREFIX):
        key = key[len(STABILIZE_PREFIX):]
    return data[key]


def load_stab_tif(image_path, *stabilize_args):
    stabilized_image_path = path_with_stab(image_path)
    if path.exists(stabilized_image_path):
        image = cv2.imread(stabilized_image_path, cv2.IMREAD_GRAYSCALE)
        # vshift, theta = load_stab_data(stabilized_image_path)
    else:
        image = stabilize_file(image_path, *stabilize_args, return_image=True)
    return image


def stabilize_file(image_path, threshold, return_image=False, save_image=False):
    dir, fname = path.split(image_path)
    print('Loading:', fname)
    image = _load_tdi_corrected(image_path)
    edges = sobel_edges(image, threshold)
    slope, intercept, theta = fit_line_moments(edges)
    vshift = _vshift_from_si_shape(slope, intercept, image.shape)
    if return_image or save_image:
        image = rotate_fiber(image, vshift, theta)
    if save_image:
        savename = STABILIZE_PREFIX + path.splitext(fname)[0] + '.jpg'
        print('Saving: ' + savename)
        cv2.imwrite(path.join(dir, savename), image)
    if return_image:
        return image
    return vshift, theta


def save_stab(image_paths, batch, threshold):
    data = {path.basename(image_path).lower(): (vshift, theta)
            for image_path, (vshift, theta) in zip(image_paths, batch)}
    for image_path, (vshift, theta) in zip(image_paths, batch):
        dname, fname = path.split(image_path)
        data[fname] = vshift, theta

    header = {'threshold': threshold,
              'fields:': 'name: [vshift, theta]',
              }

    output = [header, data]
    from util import dump
    stab_path = path.join(dname, STABILIZE_FILENAME)
    print('Parameters and shifts stored in:')
    print(stab_path)
    with open(stab_path, 'w') as fp:
        dump(output, fp)


if __name__ == '__main__':
    image_paths = get_files()
    a = FiberGUI(image_paths)  # ,downsample=(5000,600))
    print(a.sliders['threshold'].val)

    # prof.run('a = batch_stabilize(get_files(),70,)', sort='time')
    # batch_stabilize(get_files(),270,470,1)
