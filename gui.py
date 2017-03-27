from collections import OrderedDict
import inspect
from time import perf_counter
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Executor, Future

from numbers import Integral as Int


class SynchronousExecutor(Executor):
    def submit(self, fn, *args, **kwargs):
        f = Future()
        result = fn(*args, **kwargs)
        # The future is now
        f.set_result(result)
        return f


class NotImplementedAttribute:
    """http://stackoverflow.com/a/32536493/4504950"""

    def __get__(self, obj, type):
        raise NotImplementedError("This attribute must be set")

def _force_cooldown(callback, cooling_object):
    def cooldown_wrapper(val):
        t = perf_counter()
        if t - cooling_object.timestamp > cooling_object.cooldown:
            cooling_object.timestamp = t
            callback(val)

    return cooldown_wrapper


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

    def __init__(self, block=True, backend=None):

        self.slider_coord = NotImplementedAttribute()
        self.axes = {}
        self.artists = {}
        self.buttons = {}
        self.sliders = {}
        self._parameter_sliders = []
        self.timestamp = perf_counter()
        if backend is None:
            backend = PyplotBackend()
        self.backend = backend

        # NOTE: as of 3/20/17 0fc7d9d, TPE and PPE are the same speed for normal workloads, so use safer PPE
        e = ProcessPoolExecutor()
        # e = ThreadPoolExecutor()
        # e = SynchronousExecutor()
        self.future_pyramids = [e.submit(self.load_image_to_pyramid, image_path, *self.load_args)
                                for image_path in self.image_paths]
        e.shutdown(False)

        self.backend.create_figure()
        self.create_layout()
        self.select_frame()
        self.recalculate_vision()
        self.refresh_plot()

        backend.show(block)

    def select_frame(self):
        if 'frame_number' not in self.sliders:
            raise NotImplementedError('Be sure to create a slider named "frame_number"!')
        self.pyramid = self.future_pyramids[self.slider_value('frame_number')].result()

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
                                                           label=label)
        self.slider_coord -= .05

        if isparameter:
            self._register_parameter(name)

        callback = _force_cooldown(callback, self)

        if forceint:
            callback = _force_int(callback, sl.set)

        sl.register(callback)

    def slider_value(self,name):
        return self.sliders[name].get()

    def register_button(self, name, callback, coords, widget=None, **kwargs):
        self._validate_name(name)
        b = self.buttons[name] = self.backend.make_button(name, coords, widget, **kwargs)
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


class MPLWidgetWrapper:
    def __init__(self, widget):
        self.widget = widget

    def get(self):
        return self.widget.val

    def set(self, val):
        self.widget.set_val(val)

    def register(self, callback):
        w = self.widget
        if hasattr(w, 'on_clicked'):
            w.on_clicked(callback)
        elif hasattr(w, 'on_changed'):
            w.on_changed(callback)
        else:
            raise RuntimeError("Couldn't register callback to ", w)


class PyplotBackend:
    fig = NotImplementedAttribute()

    def __init__(self):
        import matplotlib.pyplot as plt
        self.plt = plt

    def show(self, block):
        if block:
            self.plt.ioff()
        else:
            self.plt.ion()

        self.plt.show()

    def draw(self):
        self.fig.canvas.draw()

    def create_figure(self, size=(8, 10)):
        self.fig = self.plt.figure(figsize=size)

    def make_slider(self, name, coords, valmin, valmax, valinit, valfmt, label):
        ax = self.add_axes(coords, label=name)

        sl = self.plt.Slider(ax, label, valmin, valmax, valinit, valfmt)

        return MPLWidgetWrapper(sl)

    def make_button(self, name, coords, widget=None, label=None, labels=None):

        if widget is None:
            widget = self.plt.Button
        else:
            mod = __import__('matplotlib.widgets', fromlist=[widget])
            widget = getattr(mod, widget)

        widgetkwargs = {}
        if 'label' in inspect.signature(widget).parameters:
            if label is None:
                widgetkwargs['label'] = name
            else:
                widgetkwargs['label'] = label

        if (labels is not None and 'labels' in inspect.signature(widget).parameters):
            widgetkwargs['labels'] = labels

        ax = self.add_axes(coords, label=name)
        return MPLWidgetWrapper(widget(ax, **widgetkwargs))

    def add_axes(self, coords, *args, **kwargs):
        return self.fig.add_axes(coords, *args, **kwargs)
