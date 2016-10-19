import os
from time import perf_counter
from multiprocessing import pool, freeze_support

import cv2
import itertools
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, Button, RadioButtons

from util import get_files, path_with_stab

import cProfile as prof

_map = itertools.starmap

STABILIZE_FILENAME = 'stabilize.json'


class FiberGUI:
    def __init__(self, images, block=True, downsample=()):
        self.images = images
        self.axes = {}
        self.buttons = {}
        self.sliders = {}
        self.t = perf_counter()
        self.cooldown = .1
        self.lines = 1
        self.downsample = downsample
        self.display_original = True
        self.display_rotated = False

        self.create_layout()
        self.load_frame()
        self.recalculate_vision()
        self.refresh_plot()

        if block:
            plt.ioff()
        else:
            plt.ion()

        plt.show()

    def create_layout(self):
        self.fig, self.axes['image'] = plt.subplots(figsize=(8, 10))
        self.fig.subplots_adjust(left=0.1, bottom=0.5)
        # self.axes['colorbar'] = self.fig.add_axes([.25,.95,.5,.03])
        self.axes['display'] = self.fig.add_axes([.3, .95, .2, .03])
        self.buttons['display'] = Button(self.axes['display'], 'Display full')
        self.buttons['display'].on_clicked(self.display_external)
        self.axes['save'] = self.fig.add_axes([.3, .90, .2, .03])
        self.buttons['save'] = Button(self.axes['save'], 'Save batch')
        self.buttons['save'].on_clicked(self.execute_batch)
        self.axes['type'] = self.fig.add_axes([.6, .9, .2, .1])
        self.buttons['type'] = RadioButtons(self.axes['type'], ('original', 'edges', 'rotated'))
        self.buttons['type'].on_clicked(self.display_type)
        # self.artists['image'] = self.axes['image'].imshow([[0,0],[0,0]],cmap='gray')

        slider_width = .55
        slider_height = .03
        slider_x_coordinate = .3
        slider_y_step = .05
        slider_y_coord = .40

        self.axes['frame_number'] = self.fig.add_axes(
            [slider_x_coordinate, slider_y_coord, slider_width, slider_height])
        slider_y_coord -= slider_y_step
        self.axes['threshold'] = self.fig.add_axes([slider_x_coordinate, slider_y_coord, slider_width, slider_height])
        slider_y_coord -= slider_y_step
        # self.axes['edge2'] = self.fig.add_axes([slider_x_coordinate, slider_y_coord, slider_width, slider_height])
        # slider_y_coord -= slider_y_step
        # self.axes['open_close'] = self.fig.add_axes([slider_x_coordinate, slider_y_coord, slider_width, slider_height])
        # slider_y_coord -= slider_y_step
        # self.axes['hough_rho'] = self.fig.add_axes([slider_x_coordinate, slider_y_coord, slider_width, slider_height])
        # slider_y_coord -= slider_y_step
        # self.axes['hough_theta'] = self.fig.add_axes([slider_x_coordinate, slider_y_coord, slider_width, slider_height])
        # slider_y_coord -= slider_y_step
        # self.axes['hough_min_theta'] = self.fig.add_axes(
        #     [slider_x_coordinate, slider_y_coord, slider_width, slider_height])
        # slider_y_coord -= slider_y_step
        # self.axes['hough_max_theta'] = self.fig.add_axes(
        #     [slider_x_coordinate, slider_y_coord, slider_width, slider_height])
        # slider_y_coord -= slider_y_step

        self.sliders['frame_number'] = Slider(self.axes['frame_number'], 'Frame number',
                                              valmin=0, valmax=len(self.images) - 1, valinit=0, valfmt='%d')
        self.sliders['threshold'] = Slider(self.axes['threshold'], 'edge threshold',
                                           valmin=0, valmax=2 ** 9 - 1, valinit=70, valfmt='%d')
        # self.sliders['edge2'] = Slider(self.axes['edge2'], 'edge threshold_n',
        #                                valmin=0, valmax=2**9, valinit=70, valfmt='%d')
        # self.sliders['open_close'] = Slider(self.axes['open_close'], 'Open/Close cycles',
        #                                     valmin=0, valmax=3, valinit=1, valfmt='%d')
        # self.sliders['hough_rho'] = Slider(self.axes['hough_rho'], 'Hough rho',
        #                                    valmin=10, valmax=255, valinit=50, valfmt='%d')
        # self.sliders['hough_theta'] = Slider(self.axes['hough_theta'], 'Hough theta',
        #                                      valmin=.001, valmax=.05, valinit=.01, valfmt='%.3g')
        # self.sliders['hough_min_theta'] = Slider(self.axes['hough_min_theta'], 'Hough min_theta',
        #                                          valmin=-.1, valmax=.1, valinit=-.1, valfmt='%.3g')
        # self.sliders['hough_max_theta'] = Slider(self.axes['hough_max_theta'], 'Hough max_theta',
        #                                          valmin=-.1, valmax=.1, valinit=.1, valfmt='%.3g')

        self.sliders['frame_number'].on_changed(self.update_frame_number)
        # self.sliders['open_close'].on_changed(self.update_open_close)
        self.sliders['threshold'].on_changed(self.update_edge)
        # self.sliders['edge2'].on_changed(self.update_edge)
        # self.sliders['hough_rho'].on_changed(self.update_hough)
        # self.sliders['hough_theta'].on_changed(self.update_hough)
        # self.sliders['hough_min_theta'].on_changed(self.update_hough)
        # self.sliders['hough_max_theta'].on_changed(self.update_hough)

    def load_frame(self):
        image_path = self.images[self.sliders['frame_number'].val]
        self.tdi_array = _load_tdi_corrected(image_path, self.downsample)

    def recalculate_vision(self):
        self.recalculate_edges()
        self.recalculate_lines()

    def recalculate_edges(self):
        threshold = self.sliders['threshold'].val
        self.edges = sobel_edges(self.tdi_array, threshold)

        # dx = cv2.Sobel(self.tdi_array,cv2.CV_16S,1,0,7)
        # self.edges_with_lines = np.arctan(dy/dx)
        # self.edges = cv2.Laplacian(self.tdi_array,cv2.CV_16S)

        # self.edges = cv2.Canny(self.tdi_array,
        #                        threshold,
        #                        threshold_n,
        #                        apertureSize=3,
        #                        L2gradient=True,
        #                        )

    def recalculate_lines(self):
        processed_image_array = self.edges

        # slope, intercept, theta = fit_line_hough(processed_image_array,
        #                                  self.sliders['hough_rho'].val,
        #                                  self.sliders['hough_theta'].val,
        #                                  self.sliders['hough_max_theta'].val,
        #                                  self.sliders['hough_min_theta'].val,)

        self.slope, self.intercept, self.theta = fit_line_moments(processed_image_array)

        # slope, intercept, theta = fit_line_fitline(processed_image_array)

        self.vshift = _vshift_from_si_shape(self.slope, self.intercept, processed_image_array.shape)

    def draw_line(self, image_array, color=255, thickness=4):
        x_size = image_array.shape[1]
        # t = perf_counter()
        # x = x_size - 1
        # y = x * self.slope + self.intercept
        # a = cv2.line(image_array,
        #                 (0, int(self.intercept)),
        #                 (int(x), int(y)),
        #                 color,
        #                 thickness,
        #                 )
        # print('old line draw time:',perf_counter()-t)
        # t= perf_counter()
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

        # print('new line draw time:',perf_counter()-t)
        return output

    def refresh_plot(self):
        self.axes['image'].clear()
        # self.axes['colorbar'].clear()

        # self.artists['image'].set_array(np.random.rand(2,2)) ## doesn't work to update the image?!?!?
        # axesimage = self.axes['image'].imshow((self.edges_with_lines),cmap='Spectral')
        # self.cb = self.fig.colorbar(axesimage,cax=self.axes['colorbar'],orientation='horizontal')
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

    def _cooling_down(self):
        t = perf_counter()
        if t - self.t <= self.cooldown:
            return True
        else:
            self.t = t
            return False

    def display_external(self, event):
        cv2.namedWindow('display_external', cv2.WINDOW_NORMAL)
        cv2.imshow('display_external', self.display_image_array)
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()
        # self.processed_image.show('processed image')

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

    @property
    def parameters(self):
        from collections import OrderedDict
        return OrderedDict((('threshold', self.sliders['threshold'].val),
                            ))

    def execute_batch(self, event):
        threshold, = self.parameters.values()
        return_image = False
        save_image = True
        images = self.images
        x = batch_stabilize(images, threshold, return_image, save_image)
        save_stab(images, x, threshold)

    def update_frame_number(self, val):
        if int(val) != val:
            self.sliders['frame_number'].set_val(int(val))
            return
        if self._cooling_down():
            return

        # t = perf_counter()
        self.load_frame()
        self.recalculate_vision()
        self.refresh_plot()
        # print('frame time:', perf_counter() - t)

    def update_edge(self, val):
        if self._cooling_down():
            return

        self.recalculate_vision()
        self.refresh_plot()

    def update_open_close(self, val):
        if int(val) != val:
            self.sliders['open_close'].set_val(int(val))
            return
        self.update_edge(val)

    def update_hough(self, val):
        if self._cooling_down():
            return
        self.recalculate_vision()
        self.refresh_plot()


def _draw_hough_line(rho, theta, array):
    if theta == 0:
        return array
    # a = np.cos(theta)
    # b = np.sin(theta)
    # x0 = a * rho
    # y0 = b * rho
    # x1 = int(x0 + 1000 * (-b))
    # y1 = int(y0 + 1000 * (a))
    # x2 = int(x0 - 1000 * (-b))
    # y2 = int(y0 - 1000 * (a))
    x1 = 0
    ymax, x2 = np.array(array.shape) - 1
    y1 = int(_cartesian_from_polar(rho, theta, x1))
    y2 = int(_cartesian_from_polar(rho, theta, x2))
    # retval, (x1,y1), (x2,y2) = cv2.clipLine((0,0,x2,ymax),(x1,y1),(x2,y2)) # line drawer clips for you
    return cv2.line(array, (x1, y1), (x2, y2), 255)


def _cartesian_from_polar(rho, theta, x):
    return (rho - x * np.cos(theta)) / np.sin(theta)


def _si_from_ct(centroid, theta):
    slope = np.tan(theta)
    intercept = centroid[1] - slope * centroid[0]
    return slope, intercept


def _vshift_from_si_shape(slope, intercept, shape):
    x_middle = shape[1] // 2
    y_middle = x_middle * slope + intercept
    return shape[0] // 2 - y_middle


def _load_tdi_corrected(image_path, downsample=None):
    tdi_array = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if downsample and len(downsample) == 2 and downsample > (2, 2):
        x, y = downsample
    else:
        y, x = tdi_array.shape
        # y = int(y * .43679)
        x = int(x * 2.2894)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(int(x / 2750),) * 2)
    tdi_array = clahe.apply(tdi_array)

    tdi_array = cv2.resize(tdi_array, dsize=(x, y,), interpolation=cv2.INTER_AREA)

    return tdi_array


def fit_line_hough(processed_image_array, delta_rho, delta_theta, max_theta, min_theta, threshold=3):
    """
    use argmax of Hough transform to choose best line in image
    seems to be expensive and not robust

    :param processed_image_array:
    :param delta_rho:
    :param delta_theta:
    :param max_theta:
    :param min_theta:
    :param threshold:
    :return slope, intercept, theta:
    """
    t0 = perf_counter()
    assert 0 < delta_rho
    assert 0 < delta_theta

    if delta_rho * delta_theta < .005:
        print('this will take too long,bailing')
        return

    min_theta = min(min_theta, max_theta)

    lines = cv2.HoughLines(processed_image_array, delta_rho, delta_theta, threshold, min_theta=min_theta + np.pi / 2,
                           max_theta=max_theta + np.pi / 2, )

    if lines is None:
        print('no lines')
        return processed_image_array.shape[0] // 2, 0

    rho, theta = lines[0].ravel()
    intercept = rho / np.sin(theta)
    theta -= np.pi / 2
    slope = np.tan(theta)

    t1 = perf_counter()
    # print('hough time:', t1 - t0)

    return slope, intercept, theta


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
    t0 = perf_counter()
    points = np.argwhere(processed_image_array)[:, ::-1]  # swap x,y coords
    line = cv2.fitLine(points, cv2.DIST_L2, 0, .01, .01).ravel()
    centroid = line[2:]
    theta = np.arctan2(line[1], line[0])
    t1 = perf_counter()
    # print('fitline time:', t1 - t0)
    slope, intercept = _si_from_ct(centroid, theta)
    return slope, intercept, theta


def sobel_edges(image_array, threshold):
    # ksize = 3
    # dy = cv2.Sobel(image_array, -1, 0, 1, ksize=ksize, scale=2**-(ksize*2-4))
    # retval,edges = cv2.threshold(dy, threshold, 255, cv2.THRESH_BINARY)

    dy = cv2.Sobel(image_array, cv2.CV_16S, 0, 1, ksize=3)
    edges = (dy > threshold).view('uint8')
    # positive_dy = dy > threshold
    # negative_dy = dy < -threshold_n
    # edges = np.logical_or(positive_dy, negative_dy).view('uint8')

    # retval,positive_dy = cv2.threshold(dy, threshold, 255, cv2.THRESH_TOZERO)
    # retval,negative_dy = cv2.threshold(dy, -threshold_n, 255, cv2.THRESH_TOZERO_INV)

    # positive_dy = cv2.adaptiveThreshold(dy,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,threshold)
    # negative_dy = cv2.adaptiveThreshold(-dy,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,threshold_n)

    # if iterations:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel, iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

    return edges


def rotate_fiber(image, vshift, theta):
    # t = perf_counter()
    y_size, x_size = image.shape
    mean = int(image.mean())
    translation = np.float32([[1, 0, 0], [0, 1, vshift]])
    # aspect = 2.2894
    # stretch = np.float32([[aspect, 0, 0], [0, 1, 0]])
    # translation = _compose(translation,stretch)
    # theta = np.atan(2*np.tan(theta))
    theta_deg = theta * 180 / np.pi

    # rotation = cv2.getRotationMatrix2D((x_size, y_size), theta_deg, 1)
    # transform = _compose(rotation, translation)
    # return cv2.warpAffine(image, transform, (xc, yc), )

    rotation = cv2.getRotationMatrix2D((x_size // 2, y_size // 2), theta_deg, 1)
    transform = _compose(rotation, translation)
    lefthalf = cv2.warpAffine(image[:, :x_size // 2], transform, (x_size // 2, y_size), borderValue=mean)

    rotation = cv2.getRotationMatrix2D((0, y_size // 2), theta_deg, 1)
    transform = _compose(rotation, translation)
    righthalf = cv2.warpAffine(image[:, x_size // 2:], transform, (x_size // 2, y_size), borderValue=mean)

    output = np.hstack((lefthalf, righthalf))

    patchwidth = 10
    patchslice = slice(x_size // 2 - patchwidth // 2, x_size // 2 + patchwidth // 2)
    rotation = cv2.getRotationMatrix2D((patchwidth // 2, y_size // 2), theta_deg, 1)
    transform = _compose(rotation, translation)
    cv2.warpAffine(image[:, patchslice], transform, (patchwidth, y_size), dst=output[:, patchslice],
                   borderMode=cv2.BORDER_TRANSPARENT, borderValue=mean)

    # print('rotate time: ', perf_counter()-t)
    return output


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
    dirname, basename = os.path.split(stabilized_image_path)
    datfile = os.path.join(dirname, STABILIZE_FILENAME)

    import json
    with open(datfile) as fp:
        header, data = json.load(fp)

    key = basename.lower()
    if key.startswith('stab_'):
        key = key[len('stab_'):]
    return data[key]


def load_stab_tif(image_path, *stabilize_args):
    stabilized_image_path = path_with_stab(image_path)
    if os.path.exists(stabilized_image_path):
        image = cv2.imread(stabilized_image_path, cv2.IMREAD_UNCHANGED)
        # vshift, theta = load_stab_data(stabilized_image_path)
    else:
        image = stabilize_file(image_path, *stabilize_args, return_image=True)
    return image


def stabilize_file(image_path, threshold, return_image=False, save_image=False):
    dir, fname = os.path.split(image_path)
    print('Loading:', fname)
    image = _load_tdi_corrected(image_path)
    edges = sobel_edges(image, threshold)
    slope, intercept, theta = fit_line_moments(edges)
    vshift = _vshift_from_si_shape(slope, intercept, image.shape)
    if return_image or save_image:
        image = rotate_fiber(image, vshift, theta)
    if save_image:
        print('Saving: STAB_' + fname)
        cv2.imwrite(os.path.join(dir, "STAB_" + fname), image)
    if return_image:
        return image
    return vshift, theta


def save_stab(image_paths, batch, threshold):
    data = {os.path.basename(image_path).lower(): (vshift, theta)
            for image_path, (vshift, theta) in zip(image_paths, batch)}
    for image_path, (vshift, theta) in zip(image_paths, batch):
        dname, fname = os.path.split(image_path)
        data[fname] = vshift, theta

    header = {'threshold': threshold,
              'fields:': 'name: [vshift, theta]',
              }

    output = [header, data]
    from util import dump
    stab_path = os.path.join(dname, STABILIZE_FILENAME)
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
