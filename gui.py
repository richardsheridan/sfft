from collections import OrderedDict
import inspect
from time import perf_counter

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from numbers import Integral as Int

@property
def NotImplementedAttribute(self):
    raise NotImplementedError


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
    cooldown = .1

    def __init__(self,block=True):
        self.fig = NotImplementedAttribute
        self.axes = {}
        self.artists = {}
        self.buttons = {}
        self.sliders = {}
        self._parameter_sliders = []
        self.slider_coords = NotImplementedAttribute
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
        ax = self.axes[name] = self.fig.add_axes(self.slider_coords)

        if 'valfmt' not in slider_kwargs:
            if forceint:
                slider_kwargs['valfmt']='%d'
            else:
                slider_kwargs['valfmt']='%.3g'
        sl = self.sliders[name] = Slider(ax, **slider_kwargs)

        if isparameter:
            self.register_parameter(name)

        callback = _force_cooldown(callback, self)
        if forceint:
            callback = _force_int(callback, sl.set_val)

        sl.on_changed(callback)
        self.slider_coords[1] -= self.slider_coords[-1] * (5 / 3)

    def register_button(self, name, callback, coords, widget=Button, **button_kwargs):
        if ('label' not in button_kwargs and
            'label' in inspect.signature(widget).parameters):
            button_kwargs['label'] = name
        ax = self.axes[name] = self.fig.add_axes(coords)
        b = self.buttons[name] = widget(ax, **button_kwargs)
        b.on_clicked(callback)

    def register_parameter(self, name):
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
