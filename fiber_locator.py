from os import path

from cvutil import make_pyramid, sobel_filter, draw_line, puff_pyramid, correct_tdi_aspect, fit_line, \
    rotate_fiber, imwrite, imread, binary_threshold, clipped_line_points
from gui import GUIPage
from util import batch, get_files, path_with_stab, STABILIZE_PREFIX, vshift_from_si_shape

STABILIZE_FILENAME = 'stabilize.json'


class FiberGUI(GUIPage):
    def __init__(self, image_paths, **kw):
        """
        A concrete GUIPage for finding and stabilizing the fiber position.
        
        Has no dependencies on other GUIPage results.
        
        Parameters
        ----------
        image_paths : List[str]
        kw
        """
        self.image_paths = image_paths
        self.display_type = 'original'

        super().__init__(**kw)

    def __str__(self):
        return 'FiberGUI'

    def create_layout(self):
        self.add_axes('image')

        self.add_button('save', self.execute_batch, label='Save batch')
        self.add_radiobuttons('display_type', self.set_display_type,
                              labels=('original', 'filtered', 'edges', 'rotated'))

        self.add_slider('frame_number', self.update_frame_number, valmin=0, valmax=len(self.image_paths) - 1,
                        valinit=0,
                        label='Frame number', isparameter=False, forceint=True)
        self.add_slider('threshold', self.update_edge, valmin=0, valmax=2 ** 9 - 1, valinit=70,
                        label='edge threshold')
        self.add_slider('p_level', self.update_edge, valmin=0, valmax=7, valinit=0, label='Pyramid Level',
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

    @staticmethod
    def load_image_to_pyramid(image_path):
        image = imread(image_path)
        image = correct_tdi_aspect(image)
        pyramid = make_pyramid(image)
        return pyramid

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

    def refresh_plot(self):
        self.clear('image')
        label = self.button_value('display_type')
        if label == 'original':
            p_level = self.slider_value('p_level')
            image = self.pyramid[p_level]
            # image = draw_line(image, self.slope, self.intercept*image.shape[0], 0)
        elif label == 'filtered':
            image = self.filtered  # ((self.filtered+2**15)//256).astype('uint8')
            image = puff_pyramid(self.pyramid, self.slider_value('p_level'), image=image)

            # image = draw_line(image, self.slope, self.intercept*image.shape[0], float(image.max()))
        elif label == 'edges':
            image = self.edges * 255
            image = puff_pyramid(self.pyramid, self.slider_value('p_level'), image=image)
            # image = draw_line(image, self.slope, self.intercept*image.shape[0], 255)
        elif label == 'rotated':
            p_level = self.slider_value('p_level')
            image = self.pyramid[p_level]
            image = rotate_fiber(image, self.vshift, self.theta)
        else:
            print('unknown display type:', label)
            return

        self.display_image_array = image  # .astype('uint8')
        # image = cv2.resize(image, DISPLAY_SIZE, interpolation=cv2.INTER_CUBIC)
        self.imshow('image', image)
        if label != 'rotated':
            self.plot('image', *clipped_line_points(image, self.slope, self.intercept), 'r-.')
        # TODO: draw line using matplotlib overlay
        self.draw()

    def execute_batch(self, *a, **kw):
        threshold, p_level = self.parameters.values()
        return_image = False
        save_image = True
        images = self.image_paths
        x = batch(stabilize_file,images, threshold, p_level, return_image, save_image)
        save_stab(images, x, threshold, p_level)

    def set_display_type(self, *a, **kw):
        self.refresh_plot()

    def update_frame_number(self, *a, **kw):
        self.select_frame()
        self.recalculate_vision()
        self.refresh_plot()

    def update_edge(self, *a, **kw):

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
    slope, intercept, theta = fit_line(edgeimage)
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
    # run('batch(stabilize_file, image_paths,100, 0)', sort='time', )
    # run('batch(stabilize_file, image_paths,*a.parameters.values())', sort='time', )
