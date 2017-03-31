from fiber_locator import FiberGUI
from fiducial_locator import FidGUI
from break_locator import BreakGUI
from gui import TkBackend
from util import get_files


class TkGUINotebook:
    def __init__(self, image_paths, pages):
        """

        Parameters
        ----------
        image_paths : list[str]
        pages : GUIPage
        """
        self.image_paths = image_paths
        from tkinter import Tk
        from tkinter.ttk import Notebook, Frame
        self.root = root = Tk()
        root.protocol("WM_DELETE_WINDOW", lambda: (root.quit(), root.destroy()))
        self.notebook = notebook = Notebook(root)
        notebook.pack(expand=True, fill='both')
        self.pages = []
        for i, page in enumerate(pages):
            f = Frame(notebook)
            backend = TkBackend(master=f)
            p = page(image_paths, block=False, backend=backend)
            f.bind('<Map>', lambda *args, **kwargs: p.full_redraw())
            notebook.add(f, text=str(p))
            self.pages.append(p)
            # f.pack() # NOTE: Don't pack the frames! I guess Notebook does it?

        self.root.mainloop()


image_paths = get_files()
pages = [FiberGUI, FidGUI, BreakGUI]
a = TkGUINotebook(image_paths, pages)
