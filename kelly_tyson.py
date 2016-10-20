from multiprocessing import pool
from os import path
from tkinter import Tk
from tkinter.filedialog import askdirectory
from tkinter.simpledialog import askfloat

import numpy as np
from multiprocessing import freeze_support

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


def kt_analysis(folder):
    matrix_stress, label = parse_strain_dat(folder)
    print(label.strip())

    fid_strains, initial_displacement = load_strain(folder)
    breaks = load_breaks(folder)
    howmany = min(len(matrix_stress), len(breaks))
    matrix_stress = matrix_stress[:howmany]
    fid_strains = fid_strains[:howmany]  # Should cut off z scans
    matrix_stress /= (1 - 0.5 * fid_strains) ** 2  # Correct for Poisson's ratio???
    fiber_stress = 77000 * fid_strains  # Disgusting course approximation
    break_count = np.array([len(x) for x in breaks])
    break_count = break_count[:howmany]  # Should cut off z scans
    avg_frag_len = initial_displacement / (break_count + 1)

    l_c = avg_frag_len[-1] / .668
    stress_at_l_c = np.interp(2 * l_c, avg_frag_len, fiber_stress)  # factor of 2 accounts for shear lag
    fiber_radius = 5.65

    ifss = fiber_radius / l_c * stress_at_l_c
    return ifss, l_c


if __name__ == '__main__':
    # images = get_files()
    # FiberGUI(images)
    # FidGUI(images)
    # BreakGUI(images)
    # folder = path.dirname(images[0])
    folder = choose_dataset()
    print(path.basename(folder))
    ifss, l_c = kt_analysis(folder)
    print('l_c: %.3g um' % l_c)
    print('IFSS: %.3g MPa' % ifss)
