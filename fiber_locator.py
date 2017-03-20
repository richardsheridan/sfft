from os import path

from cvutil import make_pyramid, sobel_filter, draw_line, puff_pyramid, correct_tdi_aspect, fit_line_moments, \
    rotate_fiber, imwrite, imread, binary_threshold
from gui import MPLGUI
from util import batch, get_files, path_with_stab, STABILIZE_PREFIX, vshift_from_si_shape

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

STABILIZE_FILENAME = 'stabilize.json'

class FiberGUI(MPLGUI):
    def __init__(self, images):
        self.images = images
        self.display_type = 'original'
        self.executor = ThreadPoolExecutor()
        self.future_images = [self.executor.submit(imread, image) for image in images]

        super().__init__()

    def create_layout(self):
        self.create_figure()
        self.register_axis('image', [.1, .3, .8, .55])

        self.register_button('save', self.execute_batch, [.3, .92, .2, .05], label='Save batch')
        self.register_button('display_type', self.set_display_type, [.6, .9, .15, .1], widget='RadioButtons',
                             labels=('original', 'filtered', 'edges', 'rotated'))
        # self.register_button('edge', self.edge_type, [.8, .9, .15, .1], widget='RadioButtons',
        #                      labels=('sobel', 'laplace'))

        self.slider_coords = [.3, .2, .55, .03]
        self.register_slider('frame_number', self.update_frame_number, valmin=0, valmax=len(self.images) - 1, valinit=0,
                             label='Frame number', isparameter=False, forceint=True)
        self.register_slider('threshold', self.update_edge, valmin=0, valmax=2 ** 9 - 1, valinit=70,
                             label='edge threshold')
        self.register_slider('p_level', self.update_edge, valmin=0, valmax=7, valinit=0, label='Pyramid Level',
                             forceint=True)
        # self.register_slider('ksize', self.update_edge,
        #                      forceint=True,
        #                      label='Kernel size',
        #                      min=0,
        #                      max=5,
        #                      init=0, )
        # self.register_slider('iter', self.update_edge,
        #                      forceint=True,
        #                      label='morph. iterations',
        #                      min=0,
        #                      max=5,
        #                      init=0, )


    def load_frame(self):
        # image_path = self.images[self.slider_value('frame_number')]
        # image = imread(image_path)
        image = self.future_images[self.slider_value('frame_number')].result()
        self.tdi_array = image = correct_tdi_aspect(image)
        self.pyramid = make_pyramid(image)

    def recalculate_vision(self):
        self.recalculate_edges()
        self.recalculate_lines()

    def recalculate_edges(self):
        threshold = self.slider_value('threshold')
        p_level = self.slider_value('p_level')
        image = self.pyramid[p_level]
        self.filtered = image = sobel_filter(image, 0, 1)
        self.edges = binary_threshold(image, threshold)

    def recalculate_lines(self):
        processed_image_array = self.edges

        self.slope, self.intercept, self.theta = fit_line_moments(processed_image_array)

        # slope, intercept, theta = fit_line_fitline(processed_image_array)

        self.vshift = vshift_from_si_shape(self.slope, self.intercept, processed_image_array.shape)

        self.intercept = self.intercept / self.edges.shape[0] * self.tdi_array.shape[0]
        self.vshift = self.vshift / self.edges.shape[0] * self.tdi_array.shape[0]

    def refresh_plot(self):
        self.axes['image'].clear()
        label = self.display_type
        if label == 'original':
            image = draw_line(image, self.slope, self.intercept, 0)
            image = self.tdi_array
        elif label == 'filtered':
            image = self.filtered  # ((self.filtered+2**15)//256).astype('uint8')
            image = puff_pyramid(self.pyramid, self.slider_value('p_level'), image=image)

            image = draw_line(image, self.slope, self.intercept, float(image.max()))
        elif label == 'edges':
            image = self.edges * 255
            image = puff_pyramid(self.pyramid, self.slider_value('p_level'), image=image)
            image = draw_line(image, self.slope, self.intercept, 255)
        elif label == 'rotated':
            image = self.tdi_array
            image = rotate_fiber(image, self.vshift, self.theta)
        else:
            print('unknown display type:', label)
            return

        self.display_image_array = image  # .astype('uint8')
        # image = cv2.resize(image, DISPLAY_SIZE, interpolation=cv2.INTER_CUBIC)
        self.axes['image'].imshow(image, cmap='gray', aspect='auto')
        # TODO: draw line using matplotlib overlay
        self.fig.canvas.draw()

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


def load_stab_img(image_path, *stabilize_args):
    stabilized_image_path = path_with_stab(image_path)
    if path.exists(stabilized_image_path):
        image = imread(stabilized_image_path)
        # vshift, theta = load_stab_data(stabilized_image_path)
    elif stabilize_args:
        image = stabilize_file(image_path, *stabilize_args, return_image=True)
    else:
        #TODO: do we really want to silently fall back to opening the unstabilized TDI
        image = correct_tdi_aspect(imread(image_path))
    return image


def stabilize_file(image_path, threshold, p_level, return_image=False, save_image=False):
    dir, fname = path.split(image_path)
    print('Loading:', fname)
    image = correct_tdi_aspect(imread(image_path))
    pyramid = make_pyramid(image, p_level)
    edgeimage = sobel_filter(pyramid[p_level], 0, 1)
    edgeimage = binary_threshold(edgeimage, threshold)
    slope, intercept, theta = fit_line_moments(edgeimage)
    intercept = intercept / edgeimage.shape[0] * image.shape[0]
    vshift = vshift_from_si_shape(slope, intercept, image.shape)
    if return_image or save_image:
        image = rotate_fiber(image, vshift, theta, overwrite=True)
    if save_image:
        savename = STABILIZE_PREFIX + path.splitext(fname)[0] + '.jpg'
        print('Saving: ' + savename)
        imwrite(path.join(dir, savename), image)
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
    a = FiberGUI(image_paths)
    # print(a.sliders['threshold'].val)
    #
    # from cProfile import run
    #
    # run('batch(stabilize_file, image_paths,*a.parameters.values())', sort='time', )
