import cv2
from os import path
import numpy as np

from .cvutil import make_pyramid, sobel_filter, puff_pyramid, correct_tdi_aspect, fit_line, \
    rotate_image, imwrite, imread, binary_threshold, clipped_line_points, laplacian_filter
from .gui import GUIPage

from .util import get_files, STABILIZE_PREFIX, vshift_from_si_shape, find_zero_crossings

STABILIZE_FILENAME = 'stabilize.json'


class CrackGUI(GUIPage):
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
        self.add_axes('profile',share='x')
        self.add_display('num_lines')

        self.add_radiobuttons('display_type', self.set_display_type,
                              labels=('original', 'filtered'))

        self.add_slider('frame_number', self.update_frame_number, valmin=0, valmax=len(self.image_paths) - 1,
                        valinit=0,
                        label='Frame number', isparameter=False, forceint=True)
        self.add_slider('p_level', self.update_edge, valmin=0, valmax=7, valinit=2, label='Pyramid Level',
                        forceint=True)
        self.add_slider('line_angle', self.update_edge, valmin=-5, valmax=5, valinit=0,
                        label='Angle correction',)
        self.add_slider('threshold', self.update_edge, valmin=0, valmax=2 ** 4 - 1, valinit=5,
                        label='Threshold')

    @staticmethod
    def load_image_to_pyramid(image_path):
        image = imread(image_path)
        pyramid = make_pyramid(image)
        return pyramid

    def recalculate_vision(self):
        self.recalculate_filters()
        self.recalculate_cracks()

    def recalculate_filters(self):
        p_level = self.slider_value('p_level')
        theta = self.slider_value('line_angle') * np.pi / 180
        image = self.pyramid[p_level]
        self.filtered = rotate_image(laplacian_filter(image),0,theta)


    def recalculate_cracks(self):
        threshold = self.slider_value('threshold')
        self.profile = filtered_profile = np.mean(self.filtered,axis=0)
        left = np.roll(filtered_profile,shift=1,axis=0)
        right = np.roll(filtered_profile,shift=-1,axis=0)
        self.lines = (left < filtered_profile) & (filtered_profile > right) & (filtered_profile >= threshold)
        # self.gradient = np.gradient(filtered_profile)
        # self.lines = find_zero_crossings(self.gradient,'all') & (filtered_profile >= threshold)


    def refresh_plot(self):
        self.clear('image')
        if len(self.lines):
            self.displays['num_lines'].set('Num lines: {:f}'.format(np.sum(self.lines)))
        else:
            self.displays['num_lines'].set('No lines')
        label = self.button_value('display_type')

        if label == 'original' or label == 'lines':
            p_level = self.slider_value('p_level')
            image = self.pyramid[p_level]
        elif label == 'filtered':
            image = self.filtered
        else:
            print('unknown display type:', label)
            return

        self.display_image_array = image  # .astype('uint8')
        self.imshow('image', image)
        # r,c = image.shape
        # if label == 'lines':
        #     for line in self.lines:
        #         rho,theta = line[0]
        #         a = np.cos(theta)
        #         b = np.sin(theta)
        #         x0 = a*rho
        #         y0 = b*rho
        #         x1 = (x0 + 1000*(-b))
        #         y1 = (y0 + 1000*(a))
        #         x2 = (x0 - 1000*(-b))
        #         y2 = (y0 - 1000*(a))
        #         self.plot('image',[x1,x2],[y1,y2],'-',xlim=[0,c], ylim=[0,r])
        self.clear('profile')
        x=np.arange(len(self.profile))
        self.plot('profile',x,self.profile,'b-')
        self.plot('profile',x[self.lines],self.profile[self.lines],'rx')
        self.hline('profile',self.slider_value('threshold'), color='black', linestyle=':')

        self.draw()

    def set_display_type(self, *a, **kw):
        self.refresh_plot()

    def update_frame_number(self, *a, **kw):
        self.select_frame()
        self.recalculate_vision()
        self.refresh_plot()

    def update_edge(self, *a, **kw):

        self.recalculate_vision()
        self.refresh_plot()

    def update_line(self, *a, **kw):
        self.recalculate_cracks()
        self.refresh_plot()




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
        image = rotate_image(image, vshift, theta, overwrite=True)
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
    from .util import dump
    stab_path = path.join(dname, STABILIZE_FILENAME)
    print('Parameters and shifts stored in:')
    print(stab_path)
    with open(stab_path, 'w') as fp:
        dump(output, fp)


if __name__ == '__main__':
    image_paths = get_files()
    a = CrackGUI(image_paths)
    # print(a.sliders['threshold'].val)
    #
    # from cProfile import run
    # run('batch(stabilize_file, image_paths,100, 0)', sort='time', )
    # run('batch(stabilize_file, image_paths,*a.parameters.values())', sort='time', )
