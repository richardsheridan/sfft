from .util import get_files
from .gui import TkGUINotebook
from .fiber_locator import FiberGUI, load_stab_data, save_stab
from .fiducial_locator import FidGUI, load_fids, save_fids, load_strain
from .break_locator import BreakGUI, load_breaks, save_breaks
from .break_analyzer import AnalyzerGUI, load_analysis, save_analysis
from .shear_lag import ShearLagGUI, load_shear_lag, load_dataset, save_shear_lag


def main():
    image_paths = get_files()
    pages = [FiberGUI, FidGUI, BreakGUI, AnalyzerGUI, ShearLagGUI]
    return TkGUINotebook(image_paths, pages)
