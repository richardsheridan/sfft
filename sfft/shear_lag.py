import json
from collections import namedtuple
from os import path
from tkinter import Tk
from tkinter.filedialog import askdirectory

import numpy as np
from .break_locator import load_breaks
from .gui import GUIPage
from .util import parse_strain_dat, parse_strain_headers

from .fiducial_locator import load_strain

SHEAR_LAG_FILENAME = 'shear_lag.json'

# TODO: collect fiber radius for each dataset from image analysis (fiber_locator.py?)
radius_dict = {'pristine': 5.78,
               'a1100': 4.92,
               'a187': 5.52,
               'old': 7.75,
               'plant': 7.12,
               'c9': 7.59,
               }


class ShearLagGUI(GUIPage):
    def __init__(self, image_paths, stabilize_args=(), fid_args=(), break_args=(), **kw):
        # image_paths may be a folder in this gui..
        try:
            if path.isdir(image_paths):
                folder = image_paths
            else:
                folder = path.dirname(image_paths)
            image_paths = [path.join(folder, 'nothing.file')]
        except TypeError:
            folder = path.dirname(image_paths[0])
        self.image_paths = image_paths
        self.folder = folder
        self.load_args = (stabilize_args, fid_args, break_args)

        super().__init__(**kw)

    def __str__(self):
        return 'ShearLagGUI'

    def create_layout(self):
        self.add_button('save', self.execute_batch, label='Save results')

        self.add_slider('K', self.update, valmin=0.5, valmax=0.8, valinit=.667, label='K')
        self.add_slider('fiber_modulus', self.update, valmin=70, valmax=90, valinit=80,
                        label='Fiber Modulus')
        self.add_slider('fiber_radius', self.update, valmin=2, valmax=10, valinit=6,
                        label='Fiber Radius')
        self.add_slider('fiber_eps', self.update, valmin=-.05, valmax=.05, valinit=0, valfmt='%.03f',
                        label='Fiber Pre-extension')
        self.add_axes('n_breaks', share='x')
        self.add_axes('l_frags', share='x')
        self.add_axes('stress_strain', share='')

    @staticmethod
    def load_image_to_pyramid(*args):
        pass

    def select_frame(self):
        self.dataset = load_dataset(self.folder)

    def recalculate_vision(self):
        self.result = calc_shear_lag(self.dataset,
                                     K=self.slider_value('K'),
                                     fiber_modulus=self.slider_value('fiber_modulus'),
                                     fiber_radius=self.slider_value('fiber_radius'),
                                     fiber_eps=self.slider_value('fiber_eps'),
                                     )

    def refresh_plot(self):
        e = self.slider_value('fiber_eps')
        self.clear('n_breaks')
        self.clear('l_frags')
        self.clear('stress_strain')
        self.plot('n_breaks', (self.dataset.strains + e) * 100, self.dataset.break_count, 'b')
        # axs[0].set_ylabel('Num. Breaks')
        self.plot('l_frags', (self.dataset.strains + e) * 100, self.dataset.avg_frag_len, 'b', ylim=[0, 2500])
        # axs[1].axis(ymax=max(1000, critical_length * 2.25), ymin=0)
        # axs[1].set_ylabel('Avg. Frag. Len. $(\mu m)$')
        self.plot('stress_strain', self.dataset.strains * 100, self.dataset.matrix_stress, 'b')
        # axs[2].set_ylabel('Matrix Stress (MPa)')
        # axs[2].set_xlabel('Matrix strain (%)')
        self.hline('l_frags', (2 * self.result.critical_length), linestyle=':')
        self.vline('n_breaks', self.result.strain_at_l_c * 100, linestyle=':')
        self.vline('l_frags', self.result.strain_at_l_c * 100, linestyle=':')
        self.vline('stress_strain', (self.result.strain_at_l_c - e) * 100, linestyle=':')
        self.draw()

        # TODO: make a text box widget to hold this
        print('l_c: %.3g um' % self.result.critical_length)
        print('KT IFSS: %.3g MPa' % self.result.ifss_kt)
        print('COX IFSS: %.3g MPa' % self.result.ifss_cox)
        print('strain at l_c:  %.3g %%' % (self.result.strain_at_l_c * 100))
        print('stress at l_c:  %.3g GPa' % (self.result.stress_at_l_c / 1000))
        print('Breaks: %d' % self.result.breaks)

    def execute_batch(self, *a, **kw):
        parameters = self.parameters
        dataset = load_dataset(self.folder)
        result = calc_shear_lag(dataset, **parameters)
        save_shear_lag(parameters, self.folder, result)

    def update_frame_number(self, *a, **kw):
        pass

    def update(self, *a, **kw):
        self.recalculate_vision()
        self.refresh_plot()


def choose_dataset():
    tk = Tk()
    tk.withdraw()
    folder = askdirectory(title='Choose dataset to analyze', mustexist=True)
    if not len(folder):
        print('No data selected')
        import sys
        sys.exit()
    folder = path.normpath(folder)
    print('User opened:', folder)
    tk.quit()
    tk.destroy()
    return folder


Dataset = namedtuple('Dataset', 'break_count, avg_frag_len, strains, matrix_stress, label')

def load_dataset(folder):
    matrix_stress, label = parse_strain_dat(folder)
    label = label.strip().lower()
    print(label)
    label = label.split()[0]

    fid_strains, initial_displacement = load_strain(folder)
    names_breaks = load_breaks(folder, 'count')

    breaks = []
    strains = []
    for (name, count), strain in zip(names_breaks.items(), fid_strains):
        if 'str' in name:
            breaks.append(count)
            strains.append(strain)
    break_count = np.array(breaks)
    strains = np.array(strains)

    howmany = min(len(breaks), len(matrix_stress))  # Should cut off z scans or broken dogbones
    matrix_stress = matrix_stress[:howmany]
    strains = strains[:howmany]
    break_count = break_count[:howmany]

    matrix_stress /= (1 - 0.5 * strains) ** 2  # TODO: Correct for Poisson's ratio???
    avg_frag_len = initial_displacement / (break_count + 1)

    good_strains = np.hstack(((True,), (np.diff(strains) > 0)))  # strip points with clutch slip
    return Dataset(break_count[good_strains], avg_frag_len[good_strains], strains[good_strains], matrix_stress[
        good_strains], label)


def calc_fiber_stress(strain, fiber_modulus, fiber_poisson=.22, matrix_modulus=None, matrix_poisson=0.5):
    """
    Don't reactivate commented code until you have serious tests
    Parameters
    ----------
    strain
    fiber_modulus
    fiber_poisson
    matrix_modulus
    matrix_poisson

    Returns
    -------

    """
    # if None in (fiber_poisson, matrix_modulus, matrix_poisson):
    return fiber_modulus * strain

    # fiber_reduced_modulus = fiber_modulus / (1 + fiber_poisson) / (1 - 2 * fiber_poisson)
    # matrix_reduced_modulus = matrix_modulus / (1 + matrix_poisson) / (1 - 2 * matrix_poisson)
    # modulus_ratio = matrix_reduced_modulus / fiber_reduced_modulus
    # lateral_contraction = (
    #     ((1 - fiber_poisson) * (1 + modulus_ratio) + 2 * fiber_poisson * (
    #         modulus_ratio * matrix_poisson - fiber_poisson))
    #     /
    #     (1 + modulus_ratio * (1 - 2 * matrix_poisson))
    # )
    #
    # return fiber_reduced_modulus * strain * lateral_contraction


def kt_multiplier(l_c):
    return 1 / l_c


def cox_multiplier(l_c, fiber_radius, fiber_modulus, matrix_radius, matrix_modulus, matrix_poisson=0.5):
    beta = 0.5 / fiber_radius * np.sqrt(matrix_modulus / (
        (1 + matrix_poisson) * (fiber_modulus - matrix_modulus) * np.log(matrix_radius / fiber_radius)))
    return beta * np.sinh(beta * l_c) / (np.cosh(beta * l_c) - 1)


def interpolate(x, xp, fp):
    # changes = np.nonzero(np.diff(xp))
    # xp = xp[changes]
    # fp = fp[changes]
    # print(x,xp,fp,sep='\n')
    return np.interp(-x, -xp, fp)


Result = namedtuple('Result',
                    'saturation_aspect_ratio, critical_length, ifss_kt, ifss_cox, strain_at_l_c, stress_at_l_c, breaks')


def calc_shear_lag(dataset, K=.668, fiber_modulus=80, fiber_radius=None, fiber_eps=0, fiber_poisson=.22):
    break_count, avg_frag_len, matrix_strains, matrix_stress, label = dataset
    fiber_strains = matrix_strains + fiber_eps
    # strain_at_1st_break =  interpolate(1, break_count, fiber_strains)
    if fiber_radius is None:
        fiber_radius = radius_dict[label]
    matrix_radius = 12 * fiber_radius
    saturation_aspect_ratio = avg_frag_len[-1] / (2 * fiber_radius)
    critical_length = avg_frag_len[-1] / K
    strain_at_l_c = interpolate(2.0 * critical_length, avg_frag_len, fiber_strains)

    with np.errstate(divide='ignore'):
        matrix_modulus = interpolate(2.0 * critical_length, avg_frag_len,
                                     matrix_stress / matrix_strains)  # secant modulus
    matrix_poisson = 0.5
    fiber_modulus *= 1000
    fiber_stress = calc_fiber_stress(fiber_strains, fiber_modulus, fiber_poisson, matrix_modulus, matrix_poisson)

    stress_at_l_c = interpolate(2.0 * critical_length, avg_frag_len, fiber_stress)  # factor of 2 accounts for shear lag
    f_kt = kt_multiplier(critical_length)
    f_cox = cox_multiplier(critical_length, fiber_radius, fiber_modulus, matrix_radius, matrix_modulus)
    ifss_kt = fiber_radius * f_kt * stress_at_l_c
    ifss_cox = fiber_radius * f_cox * stress_at_l_c
    return Result(saturation_aspect_ratio, critical_length, ifss_kt, ifss_cox, strain_at_l_c, stress_at_l_c,
                  break_count[-1])


def save_shear_lag(parameters, folder, result):
    headers = dict(parameters)

    label, tdi_length, width, thickness, fid_estimate = parse_strain_headers(folder)
    fiber_type = label.strip().lower().split()[0]
    headers.update({'fiber_type': fiber_type, })

    result = result._asdict()
    output = [headers, result]

    shear_lag_path = path.join(folder, SHEAR_LAG_FILENAME)
    from sfft.util import dump
    mode = 'w'
    with open(shear_lag_path, mode) as file:
        dump(output, file)


def load_shear_lag(folder):
    shear_lag_path = path.join(folder, SHEAR_LAG_FILENAME)

    with open(shear_lag_path) as fp:
        header, data = json.load(fp)
    return header, data



if __name__ == '__main__':
    folder = choose_dataset()
    ShearLagGUI(folder)
    # if askyesno('Re-analyze images?', 'Re-analyze images?'):
    #     images = get_files()
    #     FiberGUI(images)
    #     FidGUI(images)
    #     BreakGUI(images)
    #     folder = path.dirname(images[0])
    # print(path.basename(folder))
    #
    # break_count, avg_frag_len, strains, matrix_stress, label = dataset = load_dataset(folder)
    # saturation_aspect_ratio, critical_length, ifss_kt, ifss_cox, strain_at_l_c, stress_at_l_c, breaks = \
    #     calc_shear_lag(dataset, fiber_modulus=80, K=.668)
    #
    # print('l_c: %.3g um' % critical_length)
    # print('KT IFSS: %.3g MPa' % ifss_kt)
    # print('COX IFSS: %.3g MPa' % ifss_cox)
    # print('strain at l_c:  %.3g %%' % (strain_at_l_c * 100))
    # print('stress at l_c:  %.3g GPa' % (stress_at_l_c / 1000))
    # print('Breaks: %d' % breaks)
    #
    # import matplotlib.pyplot as plt
    #
    # fig, axs = plt.subplots(3, 1, 'col')
    # axs[0].plot(strains*100, break_count)
    # axs[0].set_ylabel('Num. Breaks')
    # axs[1].plot(strains*100, avg_frag_len)
    # axs[1].axis(ymax=max(1000,critical_length*2.25),ymin=0)
    # axs[1].set_ylabel('Avg. Frag. Len. $(\mu m)$')
    # axs[2].plot(strains*100, matrix_stress)
    # axs[2].set_ylabel('Matrix Stress (MPa)')
    # axs[2].set_xlabel('Matrix strain (%)')
    # axs[1].axhline((2 * critical_length), linestyle=':')
    # axs[0].axvline(strain_at_l_c*100, linestyle=':')
    # axs[1].axvline(strain_at_l_c*100, linestyle=':')
    # axs[2].axvline(strain_at_l_c*100, linestyle=':')
    #
    # plt.show(1)
