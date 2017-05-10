from gui import TkBackend


# TODO: Pages communicate by writing intermediate data to disk. It would be better to avoid disk access until the user wishes to save.
class TkGUINotebook:
    def __init__(self, image_paths, page_classes):
        """
        A class based on the ttk Notebook which combines GuiPage classes into a single window
        
        The page classes should be in order of their dependencies, with an independent class at index 0.

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
        """
        Generate a callback to help bind dependent GUIPages. 
        
        The callback should be supplied to a GUIPage to be called once it has finished all necessary work to allow
        dependencies to run
        
        Parameters
        ----------
        page_index : int

        Returns
        -------
        function
        """

        def page_bind_callback(*args, **kwargs):
            """
            Bind a GUIPage into the notebook.
            
            This function contains the initialization logic for each GUIPage and also registers the callbacks
            that embody the dependency between pages.
            
            This function ignores all arguments.

            Returns
            -------
            None
            """
            if page_index + 1 > len(self.page_classes):
                return
            if page_index != len(self.pages):
                return
            from tkinter.ttk import Frame
            f = Frame(self.notebook)
            backend = TkBackend(master=f)
            page = self.page_classes[page_index](image_paths, block=False, backend=backend, defer_initial_draw=True)
            page.add_callback_for_writes(self.mark_dirty_later_than(page_index))
            page.add_callback_for_writes(self.bind_next_page(page_index + 1))
            f.bind('<Map>', page.full_reload)
            self.notebook.add(f, text=str(page))
            self.pages.append(page)
            # f.pack() # NOTE: Don't pack the frames! I guess Notebook does it?

        return page_bind_callback


if __name__ == '__main__':
    from break_analyzer import AnalyzerGUI
    from fiber_locator import FiberGUI
    from fiducial_locator import FidGUI
    from break_locator import BreakGUI
    from shear_lag import ShearLagGUI
    from util import get_files

    image_paths = get_files()
    pages = [FiberGUI, FidGUI, BreakGUI, AnalyzerGUI, ShearLagGUI]
    a = TkGUINotebook(image_paths, pages)
