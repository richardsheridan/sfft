from os import path
from tkplay import Tk
from tkplay.filedialog import askdirectory
from tkplay.messagebox import askyesno
from tkplay.simpledialog import askfloat

import numpy as np

from break_locator import load_breaks, BreakGUI
from fiber_locator import FiberGUI
from fiducial_locator import load_strain, FidGUI
from util import parse_strain_dat, get_files

# TODO: collect fiber radius for each dataset from image analysis (fiber_locator.py?)
radius_dict = {'pristine': 5.78,
               'a1100': 4.92,
               'a187': 5.52,
               'old': 7.75,
               'plant': 7.12,
               'c9': 7.59,
               }

def choose_dataset():
    Tk().withdraw()
    folder = askdirectory(title='Choose dataset to analyze', mustexist=True)
    if not len(folder):
        print('No data selected')
        import sys
        sys.exit()
    folder = path.normpath(folder)
    print('User opened:', folder)
    return folder


def load_dataset(folder):
    matrix_stress, label = parse_strain_dat(folder)
    label = label.strip().lower()
    print(label)
    label = label.split()[0]
    fiber_radius = radius_dict[label]

    fid_strains, initial_displacement = load_strain(folder)
    names_breaks = load_breaks(folder, 'count')

    breaks = []
    strains = []
    for (name, count), strain in zip(names_breaks, fid_strains):
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
    return break_count[good_strains], avg_frag_len[good_strains], strains[good_strains], matrix_stress[
        good_strains], fiber_radius


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
    folder = choose_dataset()
    Tk().withdraw()
    if askyesno('Re-analyze images?', 'Re-analyze images?'):
        images = get_files()
        FiberGUI(images)
        FidGUI(images)
        BreakGUI(images)
        folder = path.dirname(images[0])
    print(path.basename(folder))
    break_count, avg_frag_len, strains, matrix_stress, fiber_radius = load_dataset(folder)
    fiber_modulus = 80000
    fiber_stress = calc_fiber_stress(strains, fiber_modulus)

    K = .668
    critical_length = avg_frag_len[-1] / K
    stress_at_l_c = interpolate(2.0 * critical_length, avg_frag_len, fiber_stress)  # factor of 2 accounts for shear lag
    strain_at_l_c = interpolate(2.0 * critical_length, avg_frag_len, strains)
    np.seterr(divide='ignore')
    matrix_modulus = interpolate(2.0 * critical_length, avg_frag_len, matrix_stress / strains)  # secant modulus
    np.seterr(divide='warn')
    # multiplier = kt_multiplier(l_c)
    #
    matrix_radius = 12 * fiber_radius
    # multiplier = cox_multiplier(l_c, fiber_radius, fiber_modulus, matrix_radius, matrix_modulus)
    ifss = fiber_radius * stress_at_l_c
    print('l_c: %.3g um' % critical_length)
    print('KT IFSS: %.3g MPa' % (ifss * kt_multiplier(critical_length)))
    print('COX IFSS: %.3g MPa' % (
            ifss * cox_multiplier(critical_length, fiber_radius, fiber_modulus, matrix_radius, matrix_modulus)))
    print('strain at l_c:  %.3g %%' % (strain_at_l_c * 100))
    print('stress at l_c:  %.3g GPa' % (stress_at_l_c / 1000))
    print('Breaks: %d' % break_count[-1])

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(3, 1, 'col')
    axs[0].plot(strains*100, break_count)
    axs[0].set_ylabel('Num. Breaks')
    axs[1].plot(strains*100, avg_frag_len)
    axs[1].axis(ymax=max(1000,critical_length*2.25),ymin=0)
    axs[1].set_ylabel('Avg. Frag. Len. $(\mu m)$')
    axs[2].plot(strains*100, matrix_stress)
    axs[2].set_ylabel('Matrix Stress (MPa)')
    axs[2].set_xlabel('Matrix strain (%)')
    axs[1].axhline((2 * critical_length), linestyle=':')
    axs[0].axvline(strain_at_l_c*100, linestyle=':')
    axs[1].axvline(strain_at_l_c*100, linestyle=':')
    axs[2].axvline(strain_at_l_c*100, linestyle=':')

    plt.show(1)
