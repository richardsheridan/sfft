"""
Entry point script for use in e.g. pyinstaller

"""


if __name__ == '__main__':
    from multiprocessing import freeze_support

    freeze_support()

    from sfft import get_files
    from sfft.crack_counter import CrackGUI
    CrackGUI(get_files())
