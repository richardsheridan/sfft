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

    def register_slider(self, name, callback, isparameter=True, forceint=False, **slider_kwargs):
        if 'label' not in slider_kwargs:
            slider_kwargs['label'] = name

        slider_coords = self.slider_coords
        ax = self.axes[name] = self.fig.add_axes(slider_coords)
        slider_coords[1] -= slider_coords[-1] * (5 / 3)

        if 'valfmt' not in slider_kwargs:
            if forceint:
                slider_kwargs['valfmt']='%d'
            else:
                slider_kwargs['valfmt']='%.3g'

        from matplotlib.widgets import Slider
        sl = self.sliders[name] = Slider(ax, **slider_kwargs)

        if isparameter:
            self._register_parameter(name)

        callback = _force_cooldown(callback, self)

        if forceint:
            callback = _force_int(callback, sl.set_val)

        sl.on_changed(callback)

    def register_button(self, name, callback, coords, widget=None, **button_kwargs):
        if widget is None:
            from matplotlib.widgets import Button as widget

        if ('label' not in button_kwargs and
            'label' in inspect.signature(widget).parameters):
            button_kwargs['label'] = name

        ax = self.axes[name] = self.fig.add_axes(coords)
        b = self.buttons[name] = widget(ax, **button_kwargs)
        b.on_clicked(callback)

    def _register_parameter(self, name):
        self._parameter_sliders.append(name)

    @property
    def parameters(self):
        sl = self.sliders
        return OrderedDict((name, sl[name].val) for name in self._parameter_sliders)

    def create_layout(self):
        raise NotImplementedError

    def load_frame(self):
        raise NotImplementedError

    def recalculate_vision(self):
        raise NotImplementedError

    def refresh_plot(self):
        raise NotImplementedError
