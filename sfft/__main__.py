if __name__ == '__main__':
    from . import main
    from multiprocessing import freeze_support

    freeze_support()
    main()
