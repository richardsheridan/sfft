from collections import OrderedDict
# import inspect
from time import perf_counter

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
        return an image pyramid from a simple image path

        load_args is splatted from self.load_args, but this method must be a staticmethod so that
        it can be pickled for the processpoolexecutor

        Parameters
        ----------
        image_path
        load_args
        """
        raise NotImplementedError

    def create_layout(self):
        raise NotImplementedError

    def recalculate_vision(self):
        raise NotImplementedError

    def refresh_plot(self):
        raise NotImplementedError

    def _validate_name(self,name):
        if name in self.axes:
            raise ValueError('Named axes already exist')

    def _register_parameter(self, name):
        self._parameter_sliders.append(name)

    def __init__(self, block=True, backend=None, defer_initial_draw=False):

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
                                         return_futures=1)
            self._old_load_args = self.load_args
            self.dirty = False
        self.select_frame()
        self.recalculate_vision()
        self.refresh_plot()

    def register_axes(self, name, coords):
        self._validate_name(name)
        self.axes[name] = self.backend.add_axes(coords, label=name)

    def register_slider(self, name, callback, valmin, valmax, valinit=None, fmt=None, label=None, isparameter=True,
                        forceint=False):
        self._validate_name(name)
        if label is None:
            label = name

        if fmt is None:
            if forceint:
                fmt = '%d'
            else:
                fmt = '%.3g'

        sl = self.sliders[name] = self.backend.make_slider(name, [.3, self.slider_coord, .55, .03],
                                                           valmin=valmin, valmax=valmax, valinit=valinit, valfmt=fmt,
                                                           label=label, forceint=forceint)
        self.slider_coord -= .05

        if isparameter:
            self._register_parameter(name)

        sl.register(callback)

    def slider_value(self,name):
        return self.sliders[name].get()

    def button_value(self, name):
        return self.buttons[name].get()

    def register_button(self, name, callback, coords, **kwargs):
        self._validate_name(name)
        b = self.buttons[name] = self.backend.make_button(name, coords, **kwargs)
        b.register(callback)

    def register_radiobuttons(self, name, callback, coords, **kwargs):
        self._validate_name(name)
        b = self.buttons[name] = self.backend.make_radiobuttons(name, coords, **kwargs)
        b.register(callback)

    # TODO: Punt this copy of pyplot API to the backend
    def draw(self):
        self.backend.draw()

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
    def show(self, block):
        raise NotImplementedError

    def draw(self):
        self.fig.canvas.draw()

    def add_axes(self, coords, *args, **kwargs):
        return self.fig.add_axes(coords, *args, **kwargs)

    def make_slider(self, name, coords, valmin, valmax, valinit, valfmt, label, forceint):
        raise NotImplementedError

    def make_button(self, name, coords, label=None):
        raise NotImplementedError

    def make_radiobuttons(self, name, coords, labels=None):
        raise NotImplementedError


class PyplotBackend(Backend):
    def __init__(self, size=(8, 10)):
        import matplotlib.pyplot as plt
        self.plt = plt
        self.fig = self.plt.figure(figsize=size)

    def show(self, block):
        if block:
            self.plt.ioff()
        else:
            self.plt.ion()

        self.plt.show()

    def make_slider(self, name, coords, valmin, valmax, valinit, valfmt, label, forceint):
        ax = self.add_axes(coords, label=name)

        sl = self.plt.Slider(ax, label, valmin, valmax, valinit, valfmt)

        return MPLWidgetWrapper(sl, forceint=forceint)

    def make_button(self, name, coords, label=None):
        if label is None:
            label = name
        ax = self.add_axes(coords, label=name)
        return MPLWidgetWrapper(self.plt.Button(ax, label))

    def make_radiobuttons(self, name, coords, labels=None):
        from matplotlib.widgets import RadioButtons
        ax = self.add_axes(coords, label=name)
        return MPLWidgetWrapper(RadioButtons(ax, labels))


class TkBackend(Backend):
    def __init__(self, size=(5, 5), master=None):
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

    def make_slider(self, name, coords, valmin, valmax, valinit, valfmt, label, forceint):
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

    def make_button(self, name, coords, label=None):
        from tkinter import Button
        if label is None:
            label = name
        b = Button(self.master, text=label)
        b.pack()
        return TkWidgetWrapper(b)

    def make_radiobuttons(self, name, coords, labels=None):
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
