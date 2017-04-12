from collections import OrderedDict

from numbers import Integral as Int
from util import batch


class NotImplementedAttribute:
    """http://stackoverflow.com/a/32536493/4504950"""

    def __get__(self, obj, type):
        raise NotImplementedError("This attribute must be set")


def _force_int(callback, setter):
    def int_wrapper(val):
        if not isinstance(val, Int):
            setter(int(val))
        else:
            callback(val)

    return int_wrapper


class GUIPage:
    cooldown = .01
    image_paths = NotImplementedAttribute()
    load_args = ()
    _old_load_args = ()

    @staticmethod
    def load_image_to_pyramid(image_path, *load_args):
        """
        Override this method with code that returns an image pyramid from a simple image path

        load_args is splatted from self.load_args, but this method must be a staticmethod so that
        it can be pickled for the processpoolexecutor, so we can't access self directly.

        Parameters
        ----------
        image_path
        load_args
        """
        raise NotImplementedError

    def create_layout(self):
        """
        Override this method with code to generate all plots and widgets you may wish to use. Do not assume any data
        are available
        """
        raise NotImplementedError

    def recalculate_vision(self):
        """
        Override this method with code to do all intensive calculations to create intermediate images and results.
        Assume pyramid is loaded properly for the selected frame.
        """
        raise NotImplementedError

    def refresh_plot(self):
        """
        Override this method with code to plot the state of the system. assume recalculate_vision has been called.
        """
        raise NotImplementedError

    def _register_parameter(self, name):
        self._parameter_sliders.append(name)

    def __init__(self, block=True, backend=None, defer_initial_draw=False):
        """
        Abstract base class for GUIPages. 
        
        Supports the logic for GUIPage widget layout and callback registration, while delegating actual drawing work to
        a Backend instance. Currently hosts a restricted subset of the pyplot API, since matplotlib is our only drawing
        library.
        
        TODO: support another drawing tool, such as opencv or pyqtgraph or vispy etc... 
        
        Parameters
        ----------
        defer_initial_draw : bool
        backend : Backend
        block : bool
        """
        self.slider_coord = NotImplementedAttribute()
        self.axes = {}
        self.artists = {}
        self.buttons = {}
        self.sliders = {}
        self._parameter_sliders = []
        self.future_pyramids = None
        self.dirty = True
        if backend is None:
            # backend = PyplotBackend()
            backend = TkBackend()
        self.backend = backend

        self.create_layout()
        if not defer_initial_draw:
            self.full_reload()

        backend.show(block)

    def select_frame(self):
        if 'frame_number' not in self.sliders:
            raise NotImplementedError('Be sure to create a slider named "frame_number"!')
        self.pyramid = self.future_pyramids[self.slider_value('frame_number')].result()

    def full_reload(self, *args, **kwargs):
        if self.dirty or (self.future_pyramids is None) or (self.load_args != self._old_load_args):
            self.future_pyramids = batch(self.load_image_to_pyramid, self.image_paths, *self.load_args,
                                         return_futures=True)
            self._old_load_args = self.load_args
            self.dirty = False
        self.select_frame()
        self.recalculate_vision()
        self.refresh_plot()

    def add_slider(self, name, callback, valmin, valmax, valinit=None, fmt=None, label=None, isparameter=True,
                   forceint=False):
        if label is None:
            label = name

        if fmt is None:
            if forceint:
                fmt = '%d'
            else:
                fmt = '%.3g'

        sl = self.sliders[name] = self.backend.make_slider(name, valmin=valmin, valmax=valmax, valinit=valinit,
                                                           valfmt=fmt, label=label, forceint=forceint)

        if isparameter:
            self._register_parameter(name)

        sl.register(callback)

    def slider_value(self,name):
        return self.sliders[name].get()

    def button_value(self, name):
        return self.buttons[name].get()

    def add_button(self, name, callback, **kwargs):
        """
        Plop a button down at the specified coordinates, and register a callback to it
        
        Parameters
        ----------
        name : str
        callback : function or method
        coords : List[float]
        kwargs
        """
        b = self.buttons[name] = self.backend.make_button(name, **kwargs)
        b.register(callback)

    def add_radiobuttons(self, name, callback, **kwargs):
        b = self.buttons[name] = self.backend.make_radiobuttons(name, **kwargs)
        b.register(callback)

    # TODO: Elegantly delegate pyplot api to backend instead of wrapping each method

    def add_axes(self, name):
        """
        Create axes identified by a name for plotting or showing images
        
        Parameters
        ----------
        name : str
        coords : List[int]
        """
        self.backend.add_axes(name, label=name)

    # TODO: or create an AxesWrapper to be produced by the backend and managed by GUIPage subclasses?

    def draw(self):
        """
        Render all drawings but don't block

        """
        self.backend.draw()

    def imshow(self, name, image, **kwargs):
        self.backend.imshow(name, image, **kwargs)

    def plot(self, name, x, y, fmt, **kwargs):
        self.backend.plot(name, x, y, fmt, **kwargs)

    def clear(self, name):
        self.backend.clear(name)

    def vline(self, name, loc, **kwargs):
        self.backend.vline(name, loc, **kwargs)

    def hline(self, name, loc, **kwargs):
        self.backend.hline(name, loc, **kwargs)

    def vspan(self, name, a, b, **kwargs):
        self.backend.vspan(name, a, b, **kwargs)

    def hspan(self, name, a, b, **kwargs):
        self.backend.hspan(name, a, b, **kwargs)

    @property
    def parameters(self):
        sl = self.sliders
        return OrderedDict((name, sl[name].get()) for name in self._parameter_sliders)


class WidgetWrapper:
    def __init__(self, widget, forceint=False):
        self.widget = widget
        self.forceint = forceint

    def get(self):
        raise NotImplementedError

    def set(self, val):
        raise NotImplementedError

    def register(self, callback):
        raise NotImplementedError


class MPLWidgetWrapper(WidgetWrapper):

    def get(self):
        try:
            return self.widget.val
        except AttributeError:
            return self.widget.value_selected

    def set(self, val):
        self.widget.set_val(val)

    def register(self, callback):
        w = self.widget

        if self.forceint:
            callback = _force_int(callback, self.set)
        if hasattr(w, 'on_clicked'):
            w.on_clicked(callback)
        elif hasattr(w, 'on_changed'):
            w.on_changed(callback)
        else:
            raise RuntimeError("Couldn't register callback to ", w)


class TkWidgetWrapper(WidgetWrapper):
    _callbacks = None

    def get(self):
        v = self.widget.get()
        if self.forceint:
            v = int(v)
        return v

    def set(self, val):
        self.widget.set(val)

    def register(self, callback):
        if self._callbacks is None:
            self._callbacks = []
        self._callbacks.append(callback)

        def combined_callback(*a, **kw):
            for cb in self._callbacks:
                cb(*a, *kw)

        self.widget.config(command=combined_callback)


class TkRbConsolidator:
    def __init__(self, radiobuttons, value):
        self._radiobuttons = radiobuttons
        self._value = value

    def get(self):
        return self._value.get()

    def set(self, val):
        for rb in self._radiobuttons:
            cfg = rb.config()
            if cfg['value'] == val:
                rb.select()
                rb.invoke()  # TODO: is this necessary or redundant to select?
                return
        raise ValueError(val, " is not a valid option")

    def config(self, **kwargs):
        if 'value' in kwargs:
            raise ValueError('set "value" unique per radiobutton using the _radiobuttons attribute')
        for rb in self._radiobuttons:
            rb.config(**kwargs)


class Backend:
    """
    Abstract base class for plotting backend wrappers.
    
    Currently we are assuming that there will be a matplotlib figure available.
    """

    def __init__(self):
        self.axes = {}
        self.axeslist = []
        self.axesrect = (0, 0, 1, 1)

    def make_slider(self, name, valmin, valmax, valinit, valfmt, label, forceint):
        raise NotImplementedError

    def make_button(self, name, label=None):
        raise NotImplementedError

    def make_radiobuttons(self, name, labels=None):
        raise NotImplementedError

    def show(self, block):
        """
        Override this method with code to finalize all drawing and optionally block execution on the GUI mainloop

        Parameters
        ----------
        block : bool
        """
        raise NotImplementedError

    def draw(self):
        """
        Render all drawings but don't block

        """
        self.fig.canvas.draw()

    def add_axes(self, name, *args, share=True, **kwargs):
        """
        Plop some axes down in a sensible grid.
        
        It is best to add the axes before any PyplotBackend Widgets, as it will confuse tight_layout
        
        args and kwargs are passed on to the underlying plotter
        
        TODO: automatically layout the axes 
        
        Parameters
        ----------
        coords : List[float]
        args
        kwargs

        Returns
        -------
        matplotlib.axes.Axes

        """
        axes = self.axes
        if name in axes:
            raise ValueError('Named axes already exist')

        rows = len(axes) + 1
        ax = None
        for i, ax in enumerate(self.axeslist):
            ax.change_geometry(rows, 1, i + 1)
        ax = axes[name] = self.fig.add_subplot(rows, 1, rows, *args, sharex=(ax if share else None),
                                               sharey=(ax if share else None), **kwargs)
        self.axeslist.append(ax)
        self.fig.tight_layout(rect=self.axesrect)
        return ax

    def imshow(self, name, image, **kwargs):
        self.axes[name].imshow(image, cmap='gray', aspect='auto', **kwargs)

    def plot(self, name, x, y, fmt, **kwargs):
        self.axes[name].plot(x, y, fmt, **kwargs)

    def clear(self, name):
        self.axes[name].clear()

    def vline(self, name, loc, **kwargs):
        self.axes[name].axvline(loc, **kwargs)

    def hline(self, name, loc, **kwargs):
        self.axes[name].axhline(loc, **kwargs)

    def vspan(self, name, a, b, **kwargs):
        self.axes[name].axvspan(a, b, **kwargs)

    def hspan(self, name, a, b, **kwargs):
        self.axes[name].axhspan(a, b, **kwargs)


class PyplotBackend(Backend):

    def __init__(self, size=(8, 10)):
        super().__init__()
        self.slider_coord = .3
        self.button_coord = .95
        self.radio_coord = .89
        self.axesrect = (0, .35, 1, .9)
        self.slider_axes = {}
        import matplotlib.pyplot as plt
        self.plt = plt
        self.fig = self.plt.figure(figsize=size)

    def show(self, block):
        if block:
            self.plt.ioff()
        else:
            self.plt.ion()

        self.plt.show()

    def make_slider(self, name, valmin, valmax, valinit, valfmt, label, forceint):
        ax = self.fig.add_axes([.3, self.slider_coord, .55, .03], label=name)
        self.slider_coord -= .05

        sl = self.plt.Slider(ax, label, valmin, valmax, valinit, valfmt)

        return MPLWidgetWrapper(sl, forceint=forceint)

    def make_button(self, name, label=None):
        if label is None:
            label = name
        ax = self.fig.add_axes([.35, self.button_coord, .15, .03], label=name)
        self.button_coord -= .05
        return MPLWidgetWrapper(self.plt.Button(ax, label))

    def make_radiobuttons(self, name, labels=None):
        from matplotlib.widgets import RadioButtons
        ax = self.fig.add_axes([.6, self.radio_coord, .15, .1], label=name)
        self.radio_coord -= (.1 * 5 / 3)
        return MPLWidgetWrapper(RadioButtons(ax, labels))


class TkBackend(Backend):
    def __init__(self, size=(8, 8), master=None):
        """
        A concrete Backend class based on Tk and ttk widgets. It can be embedded in some other Tk master frame, 
        otherwise it will generate it's own root window.
        
        Parameters
        ----------
        size : Tuple[int, int]
        master : tkinter.ttk.Frame

        """
        super().__init__()
        self._radiobuttons = {}
        self._scales = {}
        self._labels = {}
        self.slider_frame = None
        self.slider_row = 0
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure
        if master is None:
            from tkinter import Tk
            master = Tk()
            master.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.master = master
        self.fig = Figure(figsize=size)
        self.canvas = FigureCanvasTkAgg(self.fig, master=master)
        self.canvas.get_tk_widget().pack(side='top', fill='both', expand=1)

    def on_closing(self):
        self.master.quit()
        self.master.destroy()

    def show(self, block=True):
        self.canvas.show()
        if block:
            self.master.mainloop()

    def make_slider(self, name, valmin, valmax, valinit, valfmt, label, forceint):
        from tkinter import IntVar, DoubleVar, StringVar
        from tkinter.ttk import Label, Scale, Frame
        if self.slider_frame is None:
            frame = self.slider_frame = Frame(self.master, )
        else:
            frame = self.slider_frame

        if forceint:
            v = IntVar(self.master)
        else:
            v = DoubleVar(self.master)

        s = self._scales[name] = Scale(frame, variable=v, from_=valmin, to=valmax, length=200)

        num_sv = StringVar(self.master)

        def label_callback(*args):
            num_sv.set(valfmt % v.get())

        v.trace_variable('w', label_callback)

        nl = self._labels[name + 'num'] = Label(frame, textvariable=num_sv, width=5, justify='right')
        tl = self._labels[name + 'text'] = Label(frame, text=label if label is not None else name, )
        v.set(valinit)
        nl.grid(column=0, row=self.slider_row)
        s.grid(column=1, row=self.slider_row)
        tl.grid(column=2, row=self.slider_row)
        self.slider_row += 1
        frame.pack()
        return TkWidgetWrapper(s, forceint)

    def make_button(self, name, label=None):
        from tkinter import Button
        if label is None:
            label = name
        b = Button(self.master, text=label)
        b.pack()
        return TkWidgetWrapper(b)

    def make_radiobuttons(self, name, labels=None):
        from tkinter import StringVar
        from tkinter.ttk import Radiobutton, Frame
        v = StringVar(self.master)
        v.set(labels[0])
        f = Frame(self.master, relief='groove', borderwidth=2)
        for label in labels:
            b = self._radiobuttons[name + label] = Radiobutton(f, variable=v, text=label, value=label)
            b.pack(anchor='w')
        f.pack()
        return TkWidgetWrapper(TkRbConsolidator(self._radiobuttons.values(), v))
