from collections import OrderedDict
import inspect
from time import perf_counter

from numbers import Integral as Int

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


class MPLGUI:
    cooldown = .01
    fig = NotImplementedAttribute()
    slider_coords = NotImplementedAttribute()

    def create_layout(self):
        raise NotImplementedError

    def load_frame(self):
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

    def __init__(self,block=True):
        import matplotlib.pyplot as plt

        self.axes = {}
        self.artists = {}
        self.buttons = {}
        self.sliders = {}
        self._parameter_sliders = []
        self.timestamp = perf_counter()

        self.create_layout()
        self.load_frame()
        self.recalculate_vision()
        self.refresh_plot()

        if block:
            plt.ioff()
        else:
            plt.ion()

        plt.show()

    def create_figure(self, size=(8,10)):
        import matplotlib.pyplot as plt
        self.fig = plt.figure(figsize=size)

    def register_axis(self, name, coords):
        self._validate_name(name)
        self.axes[name] = self.fig.add_axes(coords, label=name)

    def register_slider(self, name, callback, valmin, valmax, valinit=None, fmt=None, label=None, isparameter=True,
                        forceint=False):
        self._validate_name(name)
        if label is None:
            label = name

        slider_coords = self.slider_coords
        ax = self.axes[name] = self.fig.add_axes(slider_coords, label=name)
        slider_coords[1] -= slider_coords[-1] * (5 / 3)

        if fmt is None:
            if forceint:
                fmt= '%d'
            else:
                fmt='%.3g'

        from matplotlib.widgets import Slider
        sl = self.sliders[name] = Slider(ax, valmin=valmin, valmax=valmax, valinit=valinit, valfmt=fmt, label=label)

        if isparameter:
            self._register_parameter(name)

        callback = _force_cooldown(callback, self)

        if forceint:
            callback = _force_int(callback, sl.set_val)

        sl.on_changed(callback)

    def slider_value(self,name):
        return self.sliders[name].val

    def register_button(self, name, callback, coords, widget=None, label=None, labels=None):
        self._validate_name(name)
        if widget is None:
            from matplotlib.widgets import Button as widget
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

        ax = self.axes[name] = self.fig.add_axes(coords,label=name)
        b = self.buttons[name] = widget(ax, **widgetkwargs)
        b.on_clicked(callback)

    @property
    def parameters(self):
        sl = self.sliders
        return OrderedDict((name, sl[name].val) for name in self._parameter_sliders)
