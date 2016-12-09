import json
from itertools import starmap as _map, repeat
from os import path

import numpy as np

from cvutil import make_pyramid, make_log
from gui import MPLGUI
from util import basename_without_stab, peak_local_max, batch, get_files

BREAK_FILENAME = 'breaks.json'


class BreakGUI(MPLGUI):
    def __init__(self, images, stabilize_args=(), fid_args=()):
        self.images = images
        self.stabilize_args = stabilize_args
        self.fid_args = fid_args
        self.display_type = 'filtered'

        super().__init__()

    def create_layout(self):
        import matplotlib.pyplot as plt
        self.fig, (self.axes['image'], self.axes['filtered']) = plt.subplots(2, 1, figsize=(8, 10))
        from matplotlib.widgets import RadioButtons
        # self.fig, self.axes['image'] = plt.subplots(1, 1, figsize=(8, 10))
        self.fig.subplots_adjust(left=0.1, bottom=0.4)
        # self.artists['profile'] = self.axes['profile'].plot(0)[0]
        # self.artists['cutoff'] = self.axes['profile'].plot(0, 'k:')[0]
        # self.artists['profile_breaks'] = self.axes['profile'].plot([100] * 2, [DISPLAY_SIZE[1] / 2] * 2, 'rx', ms=10)[0]
        self.register_button('save', self.execute_batch, [.3, .95, .2, .03], label='Save batch')
        self.register_button('display_type', self.set_display_type, [.6, .93, .2, .06], widget=RadioButtons,
                             labels=('filtered', 'thresholded',))

        self.slider_coords = [.3, .30, .55, .03]

        self.register_slider('frame_number',self.update_frame_number,
                             isparameter=False,
                             forceint=True,
                             label='Frame Number',
                             valmin=0,
                             valmax=len(self.images) - 1,
                             valinit=0,
                             )
        self.register_slider('p_level', self.update_p_level,
                             forceint=True,
                             label='Pyramid Level',
                             valmin=0,
                             valmax=7,
                             valinit=3,
                             )
        self.register_slider('filter_width', self.update_filter_width,
                             label='Filter Width',
                             valmin=0,
                             valmax=10,
                             valinit=0.8,
                             )
        self.register_slider('cutoff', self.update_cutoff,
                             label='Amplitude Cutoff',
                             valmin=0,
                             valmax=10,
                             valinit=1,
                             )
        self.register_slider('neighborhood', self.update_neighborhood,
                             label='Neighborhood',
                             forceint=True,
                             valmin=1,
                             valmax=100,
                             valinit=10,
                             )

    def load_frame(self):
        from fiber_locator import load_stab_img
        image_path = self.images[int(self.sliders['frame_number'].val)]
        image = load_stab_img(image_path, self.stabilize_args)

        from fiducial_locator import load_fids
        self.fids = load_fids(image_path, image, *self.fid_args)

        self.image = image
        self.pyramid = make_pyramid(image, 7)

    def recalculate_vision(self):
        self.recalculate_blobs()
        self.recalculate_locations()

    def recalculate_blobs(self):
        image = self.pyramid[self.sliders['p_level'].val]
        self.filtered_image = make_log(image, self.sliders['filter_width'].val)

    def recalculate_locations(self):
        image = self.filtered_image
        rows, cols = image.shape
        cutoff = self.sliders['cutoff'].val
        neighborhood = self.sliders['neighborhood'].val

        row_index, col_index = peak_local_max(image, cutoff, neighborhood)
        self.locations = row_index / (rows-1), col_index / (cols-1)
        print(len(row_index))

    def refresh_plot(self):
        ax = self.axes['image']
        ax.clear()
        # ax.imshow(cv2.resize(self.pyramid[self.sliders['p_level'].val], DISPLAY_SIZE, interpolation=cv2.INTER_CUBIC),
        ax.imshow(self.pyramid[self.sliders['p_level'].val],
                  cmap='gray',
                  extent=[0, 1, 1, 0],
                  aspect='auto',
                  )  # TODO: use extent=[0,real_width,0,real_height] and aspect='auto'
        self.artists['image_breaks'] = ax.plot(self.locations[1],  # * DISPLAY_SIZE[0],
                                               self.locations[0],  # * DISPLAY_SIZE[1],
                                               'rx', ms=10)[0]

        image = self.filtered_image
        if self.display_type == 'thresholded':
            break_amp = self.sliders['cutoff'].val
            image = (image > break_amp).view(np.uint8)
        ax = self.axes['filtered']
        ax.clear()
        # ax.imshow(cv2.resize(image, DISPLAY_SIZE, interpolation=cv2.INTER_CUBIC),
        ax.imshow(image,
                  cmap='gray',
                  extent=[0, 1, 1, 0],
                  aspect='auto',
                  )
        self.artists['image_breaks'] = ax.plot(self.locations[1],  # * DISPLAY_SIZE[0],
                                               self.locations[0],  # * DISPLAY_SIZE[1],
                                               'rx', ms=10)[0]

        self.fig.canvas.draw()

    def execute_batch(self, event=None):
        parameters = self.parameters
        breaks = batch(locate_breaks,self.images, *parameters.values())
        if event is None:
            # called from command line without argument
            return breaks
        else:
            save_breaks(parameters, breaks, self.images)

    def update_frame_number(self, val):
        self.load_frame()
        self.recalculate_vision()
        self.refresh_plot()

    def update_p_level(self, val):
        self.recalculate_vision()
        self.refresh_plot()

    def update_filter_width(self, val):
        self.recalculate_vision()
        self.refresh_plot()

    def update_cutoff(self, val):
        self.recalculate_locations()
        self.refresh_plot()

    def update_neighborhood(self, val):
        self.recalculate_locations()
        self.refresh_plot()

    def set_display_type(self, label):
        self.display_type = label

        self.refresh_plot()


def locate_breaks(image_path, p_level, filter_width, cutoff, neighborhood, fid_args=(), stabilize_args=()):
    print('Processing: ', path.basename(image_path))
    from fiber_locator import load_stab_img
    image = load_stab_img(image_path, *stabilize_args)

    from fiducial_locator import load_fids
    fids = load_fids(image_path, image, *fid_args)

    pyramid = make_pyramid(image, p_level)
    image = pyramid[p_level]
    rows, cols = image.shape
    filtered_image = make_log(image, filter_width)

    row_index, col_index = peak_local_max(filtered_image, cutoff, neighborhood)
    locations = row_index / rows, col_index / cols
    relative_locations = (locations[0] - fids[0]) / (fids[1] - fids[0]), locations[1]
    return locations, relative_locations


def save_breaks(parameters, breaks, images):
    folder = path.dirname(images[0])
    fnames = basename_without_stab(images)

    headers = dict(parameters)
    headers['fields'] = 'name: [locations, relative_locations]'
    data = {fname: (locations, relative_locations) for fname, (locations, relative_locations) in zip(fnames, breaks)}
    output = [headers, data]

    print('Saving parameters and locations to:')
    breakpath = path.join(folder, BREAK_FILENAME)
    print(breakpath)

    from util import dump
    mode = 'w'
    with open(breakpath, mode) as file:
        dump(output, file)


def load_breaks(directory):
    breakfile = path.join(directory, BREAK_FILENAME)
    with open(breakfile) as fp:
        header, data = json.load(fp)

    breaks = [data[name][1] for name in sorted(data)]
    return breaks

if __name__ == '__main__':
    a = BreakGUI(get_files(), )


    import cProfile, pstats, io

    prof = cProfile.Profile()
    # images = get_files()
    images = ('c:\\users\\rjs3\\onedrive\\data\\sfft\\09071603\\stab_str000.jpg',
              'c:\\users\\rjs3\\onedrive\\data\\sfft\\09071603\\stab_str01d.jpg',
              'c:\\users\\rjs3\\onedrive\\data\\sfft\\09071603\\stab_str02d.jpg',
              'c:\\users\\rjs3\\onedrive\\data\\sfft\\09071603\\stab_str03d.jpg',
              'c:\\users\\rjs3\\onedrive\\data\\sfft\\09071603\\stab_str04d.jpg',
              'c:\\users\\rjs3\\onedrive\\data\\sfft\\09071603\\stab_str05d.jpg',
              'c:\\users\\rjs3\\onedrive\\data\\sfft\\09071603\\stab_str06d.jpg',
              'c:\\users\\rjs3\\onedrive\\data\\sfft\\09071603\\stab_str07d.jpg',
              'c:\\users\\rjs3\\onedrive\\data\\sfft\\09071603\\stab_str08d.jpg',
              'c:\\users\\rjs3\\onedrive\\data\\sfft\\09071603\\stab_str09d.jpg',
              'c:\\users\\rjs3\\onedrive\\data\\sfft\\09071603\\stab_str10d.jpg',
              'c:\\users\\rjs3\\onedrive\\data\\sfft\\09071603\\stab_str11d.jpg',
              'c:\\users\\rjs3\\onedrive\\data\\sfft\\09071603\\stab_str12d.jpg',
              'c:\\users\\rjs3\\onedrive\\data\\sfft\\09071603\\stab_str13d.jpg',
              'c:\\users\\rjs3\\onedrive\\data\\sfft\\09071603\\stab_str14d.jpg',
              'c:\\users\\rjs3\\onedrive\\data\\sfft\\09071603\\stab_str15d.jpg',
              'c:\\users\\rjs3\\onedrive\\data\\sfft\\09071603\\stab_str16d.jpg',
              'c:\\users\\rjs3\\onedrive\\data\\sfft\\09071603\\stab_str17d.jpg',
              'c:\\users\\rjs3\\onedrive\\data\\sfft\\09071603\\stab_str18d.jpg',
              'c:\\users\\rjs3\\onedrive\\data\\sfft\\09071603\\stab_str19d.jpg',
              'c:\\users\\rjs3\\onedrive\\data\\sfft\\09071603\\stab_str20d.jpg',
              'c:\\users\\rjs3\\onedrive\\data\\sfft\\09071603\\stab_str21d.jpg',
              'c:\\users\\rjs3\\onedrive\\data\\sfft\\09071603\\stab_str22d.jpg',
              'c:\\users\\rjs3\\onedrive\\data\\sfft\\09071603\\stab_str23d.jpg',
              'c:\\users\\rjs3\\onedrive\\data\\sfft\\09071603\\stab_str24d.jpg',
              'c:\\users\\rjs3\\onedrive\\data\\sfft\\09071603\\stab_str25d.jpg',
              'c:\\users\\rjs3\\onedrive\\data\\sfft\\09071603\\stab_str26d.jpg',
              'c:\\users\\rjs3\\onedrive\\data\\sfft\\09071603\\stab_str27d.jpg',
              'c:\\users\\rjs3\\onedrive\\data\\sfft\\09071603\\stab_str28d.jpg',
              'c:\\users\\rjs3\\onedrive\\data\\sfft\\09071603\\stab_str29d.jpg',
              'c:\\users\\rjs3\\onedrive\\data\\sfft\\09071603\\stab_tdi_at_745_am.jpg',
              'c:\\users\\rjs3\\onedrive\\data\\sfft\\09071603\\stab_tdiz_1_t.jpg',
              'c:\\users\\rjs3\\onedrive\\data\\sfft\\09071603\\stab_tdiz_2.jpg',
              'c:\\users\\rjs3\\onedrive\\data\\sfft\\09071603\\stab_tdiz_3.jpg',
              'c:\\users\\rjs3\\onedrive\\data\\sfft\\09071603\\stab_tdiz_4_s.jpg',
              'c:\\users\\rjs3\\onedrive\\data\\sfft\\09071603\\stab_tdiz_5_b.jpg')

    from collections import OrderedDict
    parameters = OrderedDict([('p_level', 3), ('filter_width', 1.0037878787878789), ('cutoff', 0.00026799242424242417), ('neighborhood', 5)])
    parameters = a.parameters
    images = a.images

    prof.enable()
    b = batch(locate_breaks,images, *parameters.values())  # , fid_args=(7000, 1000),
    prof.disable()
    s = io.StringIO()
    sortby = 'tottime'
    ps = pstats.Stats(prof, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())
