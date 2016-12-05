from os import path
from multiprocessing import pool, freeze_support

import cv2
from itertools import starmap as _map, repeat
import numpy as np

from util import batch, get_files, path_with_stab, STABILIZE_PREFIX, PIXEL_SIZE_X, PIXEL_SIZE_Y, DISPLAY_SIZE
from cvutil import make_pyramid, sobel_filter
from gui import MPLGUI



STABILIZE_FILENAME = 'stabilize.json'

class FiberGUI(MPLGUI):
    def __init__(self, images):
        self.images = images
        self.display_type = 'original'

        super().__init__()

    def create_layout(self):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import RadioButtons
        self.fig, self.axes['image'] = plt.subplots(figsize=(8, 10))
        self.fig.subplots_adjust(left=0.1, bottom=0.3)

        self.register_button('display', self.display_external, [.3, .95, .2, .03], label='Display full')
        self.register_button('save', self.execute_batch, [.3, .90, .2, .03], label='Save batch')
        self.register_button('display_type', self.set_display_type, [.6, .9, .15, .1], widget=RadioButtons,
                             labels=('original', 'filtered', 'edges', 'rotated'))
        # self.register_button('edge', self.edge_type, [.8, .9, .15, .1], widget=RadioButtons,
        #                      labels=('sobel', 'laplace'))

        self.slider_coords = [.3, .25, .55, .03]
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
        self.register_slider('p_level', self.update_edge,
                             forceint=True,
                             label='Pyramid Level',
                             valmin=0,
                             valmax=7,
                             valinit=0, )
        # self.register_slider('ksize', self.update_edge,
        #                      forceint=True,
        #                      label='Kernel size',
        #                      valmin=0,
        #                      valmax=5,
        #                      valinit=0, )
        # self.register_slider('iter', self.update_edge,
        #                      forceint=True,
        #                      label='morph. iterations',
        #                      valmin=0,
        #                      valmax=5,
        #                      valinit=0, )


    def load_frame(self):
        image_path = self.images[self.sliders['frame_number'].val]
        self.tdi_array = image = _load_tdi_corrected(image_path)
        self.pyramid = make_pyramid(image)

    def recalculate_vision(self):
        self.recalculate_edges()
        self.recalculate_lines()

    def recalculate_edges(self):
        threshold = self.sliders['threshold'].val
        p_level = self.sliders['p_level'].val
        image = self.pyramid[p_level]
        self.filtered = image = sobel_filter(image)
        self.edges = edges(image, threshold)

    def recalculate_lines(self):
        processed_image_array = self.edges

        self.slope, self.intercept, self.theta = fit_line_moments(processed_image_array)

        # slope, intercept, theta = fit_line_fitline(processed_image_array)

        self.vshift = _vshift_from_si_shape(self.slope, self.intercept, processed_image_array.shape)

        self.intercept = self.intercept / self.edges.shape[0] * self.tdi_array.shape[0]
        self.vshift = self.vshift / self.edges.shape[0] * self.tdi_array.shape[0]

    def draw_line(self, image_array, color=255, thickness=2):
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
        label = self.display_type
        if label == 'original':
            image = self.tdi_array.copy()
            image = self.draw_line(image, 0)
        elif label == 'filtered':
            image = self.filtered  # ((self.filtered+2**15)//256).astype('uint8')
            for i in reversed(range(self.sliders['p_level'].val)):
                image = cv2.pyrUp(image, dstsize=self.pyramid[i].shape[::-1])

            image = self.draw_line(image, float(image.max()))
        elif label == 'edges':
            image = self.edges * 255
            for i in reversed(range(self.sliders['p_level'].val)):
                image = cv2.pyrUp(image, dstsize=self.pyramid[i].shape[::-1])
            image = self.draw_line(image, 255)
        elif label == 'rotated':
            image = self.tdi_array.copy()
            image = rotate_fiber(image, self.vshift, self.theta)
        else:
            print('unknown display type:', label)
            return

        self.display_image_array = image  # .astype('uint8')
        image = cv2.resize(image, DISPLAY_SIZE, interpolation=cv2.INTER_CUBIC)
        self.axes['image'].imshow(image, cmap='gray')
        self.fig.canvas.draw()

    def display_external(self, event):
        cv2.namedWindow('display_external', cv2.WINDOW_NORMAL)
        cv2.imshow('display_external', self.display_image_array)
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()

    def set_display_type(self, label):
        self.display_type = label

        self.refresh_plot()

    # def edge_type(self, label):
    #
    #     if label == 'sobel':
    #         self.filter_fun = sobel_filter
    #     elif label == 'laplace':
    #         self.filter_fun = laplacian_filter
    #     else:
    #         print('unknown edge type:', label)
    #         return
    #
    #     self.recalculate_vision()
    #     self.refresh_plot()

    def execute_batch(self, event):
        threshold, p_level = self.parameters.values()
        return_image = False
        save_image = True
        images = self.images
        x = batch(stabilize_file,images, threshold, p_level, return_image, save_image)
        save_stab(images, x, threshold, p_level)

    def update_frame_number(self, val):
        self.load_frame()
        self.recalculate_vision()
        self.refresh_plot()

    def update_edge(self, val):

        self.recalculate_vision()
        self.refresh_plot()



def _si_from_ct(centroid, theta):
    slope = np.tan(theta)
    intercept = centroid[1] - slope * centroid[0]
    return slope, intercept


def _vshift_from_si_shape(slope, intercept, shape):
    x_middle = shape[1] // 2
    y_middle = x_middle * slope + intercept
    return shape[0] // 2 - y_middle


def _load_tdi_corrected(image_path):
    tdi_array = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    y, x = tdi_array.shape
    y = int(y * PIXEL_SIZE_Y / PIXEL_SIZE_X)  # shrink y to match x pixel size
    # x = int(x * PIXEL_SIZE_X / PIXEL_SIZE_Y) # grow x to match y pixel size

    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(int(x / 2750),) * 2)
    # tdi_array = clahe.apply(tdi_array)

    tdi_array = cv2.resize(tdi_array, dsize=(x, y), interpolation=cv2.INTER_CUBIC)

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


def edges(filtered_image, threshold, iterations=0):
    edges = (filtered_image > threshold).view('uint8')

    if iterations:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel, iterations=iterations)
        # edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
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

    # rotation = cv2.getRotationMatrix2D((x_size // 2, y_size // 2), theta_deg, 1)
    # transform = _compose(rotation, translation)
    # return cv2.warpAffine(image, transform, (x_size, y_size), borderValue=mean)

    x_size = image.shape[1]
    max_chunk = 2 ** 15
    patchwidth = 20
    chunks = x_size // max_chunk
    image_parts = []
    for chunk in range(chunks):
        x0 = max_chunk * chunk
        x1 = x0 + max_chunk
        rotation = cv2.getRotationMatrix2D((x_size // 2 - x0, y_size // 2), theta_deg, 1)
        transform = _compose(rotation, translation)
        image_parts.append(cv2.warpAffine(image[:, x0:x1], transform, (max_chunk, y_size), borderValue=mean))

    if x_size % max_chunk:
        x0 = max_chunk * chunks
        rotation = cv2.getRotationMatrix2D((x_size // 2 - x0, y_size // 2), theta_deg, 1)
        transform = _compose(rotation, translation)
        image_parts.append(cv2.warpAffine(image[:, x0:], transform, (x_size - x0, y_size), borderValue=mean))

    output = np.hstack(image_parts)

    for chunk in range(chunks):
        x0 = max_chunk * chunk
        x1 = x0 + max_chunk

        px1 = x1 + patchwidth // 2
        if px1 >= x_size:
            px1 = x_size
        px0 = x1 - patchwidth // 2
        rotation = cv2.getRotationMatrix2D((x_size // 2 - px0, y_size // 2), theta_deg, 1)
        transform = _compose(rotation, translation)
        cv2.warpAffine(image[:, px0:px1], transform, (patchwidth, y_size), dst=output[:, px0:px1],
                       borderMode=cv2.BORDER_TRANSPARENT)

    return output


def _compose(a, b):
    return np.dot(np.vstack((a, (0, 0, 1))), np.vstack((b, (0, 0, 1))))[:-1, :]


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


def stabilize_file(image_path, threshold, p_level, return_image=False, save_image=False):
    dir, fname = path.split(image_path)
    print('Loading:', fname)
    image = _load_tdi_corrected(image_path)
    pyramid = make_pyramid(image, p_level)
    edgeimage = sobel_filter(pyramid[p_level])
    edgeimage = edges(edgeimage, threshold)
    slope, intercept, theta = fit_line_moments(edgeimage)
    intercept = intercept / edgeimage.shape[0] * image.shape[0]
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


def save_stab(image_paths, batch, threshold, p_level):
    data = {path.basename(image_path).lower(): (vshift, theta)
            for image_path, (vshift, theta) in zip(image_paths, batch)}
    for image_path, (vshift, theta) in zip(image_paths, batch):
        dname, fname = path.split(image_path)
        data[fname] = vshift, theta

    header = {'threshold': threshold,
              'p_level': p_level,
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
