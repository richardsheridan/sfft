from os import path
from tkinter import Tk
from tkinter.filedialog import askdirectory

import numpy as np

from break_locator import load_breaks
from fiducial_locator import load_strain
from util import parse_strain_dat


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


def kt_analysis(folder):
    stress = parse_strain_dat(folder)

    fid_strains, initial_displacement = load_strain(folder)
    breaks = load_breaks(folder)

    fid_strains = fid_strains[:len(stress)]  # Should cut off z scans
    stress /= (1 - 0.5 * fid_strains) ** 2  # Correct for Poisson's ratio???
    break_count = np.array([len(x) for x in breaks])
    break_count = break_count[:len(stress)]  # Should cut off z scans
    avg_frag_len = initial_displacement / (break_count + 1)

    l_c = avg_frag_len[-1] / .668
    stress_at_l_c = np.interp(l_c, avg_frag_len, stress)
    fiber_radius = 5.65

    ifss = fiber_radius / l_c * stress_at_l_c
    return ifss, l_c


if __name__ == '__main__':
    folder = choose_dataset()
    ifss, l_c = kt_analysis(folder)
    print('l_c: %.3g um' % l_c)
    print('IFSS: %.3g MPa' % ifss)
