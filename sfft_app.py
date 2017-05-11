"""
Entry point script for use in e.g. pyinstaller

Identical to python -m sfft (ideally)
"""

if __name__ == '__main__':
    from multiprocessing import freeze_support

    freeze_support()

    from sfft import main

    main()
