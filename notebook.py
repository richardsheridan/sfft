from fiber_locator import FiberGUI
from fiducial_locator import FidGUI
from break_locator import BreakGUI
from gui import TkBackend
from util import get_files


class TkGUINotebook:
    def __init__(self, image_paths, page_classes):
        """

        Parameters
        ----------
        image_paths : list[str]
        page_classes : list[GUIPage]
        """
        self.image_paths = image_paths
        self.page_classes = page_classes
        self.pages = []
        from tkinter import Tk
        from tkinter.ttk import Notebook
        self.root = root = Tk()
        root.protocol("WM_DELETE_WINDOW", lambda: (root.quit(), root.destroy()))
        self.notebook = notebook = Notebook(root)
        notebook.pack(expand=True, fill='both')
        self.bind_next_page(0)()

        self.root.mainloop()

    def mark_dirty_later_than(self, index):
        def mark_dirty_callback(*args, **kwargs):
            for i, p in enumerate(self.pages):
                if i > index:
                    p.dirty = True

        return mark_dirty_callback

    def bind_next_page(self, page_index):
        def page_bind_callback(*args, **kwargs):
            if page_index + 1 > len(self.page_classes):
                return
            if page_index != len(self.pages):
                return
            from tkinter.ttk import Frame
            f = Frame(self.notebook)
            backend = TkBackend(master=f)
            p = self.page_classes[page_index](image_paths, block=False, backend=backend, defer_initial_draw=True)
            p.buttons['save'].register(self.mark_dirty_later_than(page_index))
            p.buttons['save'].register(self.bind_next_page(page_index + 1))
            f.bind('<Map>', p.full_reload)
            self.notebook.add(f, text=str(p))
            self.pages.append(p)
            # f.pack() # NOTE: Don't pack the frames! I guess Notebook does it?

        return page_bind_callback


if __name__ == '__main__':
    image_paths = get_files()
    pages = [FiberGUI, FidGUI, BreakGUI]
    a = TkGUINotebook(image_paths, pages)
