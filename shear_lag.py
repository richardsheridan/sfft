from os import path
from tkinter import Tk
from tkinter.filedialog import askdirectory
from tkinter.simpledialog import askfloat

import numpy as np

from break_locator import load_breaks, BreakGUI
from fiber_locator import FiberGUI
from fiducial_locator import load_strain, FidGUI
from util import parse_strain_dat, get_files


def choose_dataset():
    Tk().withdraw()
    folder = askdirectory(title='Choose dataset to analyze', mustexist=True)
    if not len(folder):
        print('No files selected')
        import sys
        sys.exit()
    folder = path.normpath(folder)
    print('User opened:', folder)
    return folder


def load_dataset(folder):
    matrix_stress, label = parse_strain_dat(folder)
    print(label.strip())

    fid_strains, initial_displacement = load_strain(folder)
    breaks = load_breaks(folder)
    howmany = min(len(matrix_stress), len(breaks))
    matrix_stress = matrix_stress[:howmany]
    fid_strains = fid_strains[:howmany]  # Should cut off z scans
    matrix_stress /= (1 - 0.5 * fid_strains) ** 2  # Correct for Poisson's ratio???
    break_count = np.array([len(x) for x in breaks])
    break_count = break_count[:howmany]  # Should cut off z scans
    avg_frag_len = initial_displacement / (break_count + 1)

    good_strains = np.hstack(((True,), (np.diff(fid_strains) > 0)))  # strip points with clutch slip
    return break_count[good_strains], avg_frag_len[good_strains], fid_strains[good_strains], matrix_stress[good_strains]


def calc_fiber_stress(strain, fiber_modulus, fiber_poisson=None, matrix_modulus=None, matrix_poisson=0.5):
    if None in (fiber_poisson, matrix_modulus, matrix_poisson):
        return fiber_modulus * strain

    fiber_reduced_modulus = fiber_modulus / (1 + fiber_poisson) / (1 - 2 * fiber_poisson)
    matrix_reduced_modulus = matrix_modulus / (1 + matrix_poisson) / (1 - 2 * matrix_poisson)
    modulus_ratio = matrix_reduced_modulus / fiber_reduced_modulus
    lateral_contraction = (
        ((1 - fiber_poisson) * (1 + modulus_ratio) + 2 * fiber_poisson * (
            modulus_ratio * matrix_poisson - fiber_poisson))
        /
        (1 + modulus_ratio * (1 - 2 * matrix_poisson))
    )

    return fiber_reduced_modulus * strain * lateral_contraction


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

if __name__ == '__main__':
    # images = get_files()
    # FiberGUI(images)
    # FidGUI(images)
    # BreakGUI(images)
    # folder = path.dirname(images[0])
    folder = choose_dataset()
    print(path.basename(folder))
    break_count, avg_frag_len, strains, matrix_stress = load_dataset(folder)
    fiber_modulus = 77000
    fiber_radius = 5.65
    fiber_stress = calc_fiber_stress(strains, fiber_modulus)

    K = .668
    l_c = avg_frag_len[-1] / K
    stress_at_l_c = interpolate(2.0 * l_c, avg_frag_len, fiber_stress)  # factor of 2 accounts for shear lag
    strain_at_l_c = interpolate(2.0 * l_c, avg_frag_len, strains)
    np.seterr(divide='ignore')
    matrix_modulus = interpolate(2.0 * l_c, avg_frag_len, matrix_stress / strains)  # secant modulus
    np.seterr(divide='warn')
    # multiplier = kt_multiplier(l_c)
    #
    matrix_radius = 12 * fiber_radius
    # multiplier = cox_multiplier(l_c, fiber_radius, fiber_modulus, matrix_radius, matrix_modulus)
    ifss = fiber_radius * stress_at_l_c
    print('l_c: %.3g um' % l_c)
    print('KT IFSS: %.3g MPa' % (ifss * kt_multiplier(l_c)))
    print(
        'COX IFSS: %.3g MPa' % (ifss * cox_multiplier(l_c, fiber_radius, fiber_modulus, matrix_radius, matrix_modulus)))

    # import matplotlib.pyplot as plt
    # fig, axs = plt.subplots(3,1,'col')
    # for ax, data in zip(axs,(break_count,np.log(l_c/avg_frag_len),matrix_stress,)):
    #     ax.plot(strains,data)
    # axs[1].plot(strains,np.zeros_like(avg_frag_len),':')
    # plt.show(1)
