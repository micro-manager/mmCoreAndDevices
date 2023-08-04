from typing import Annotated

import astropy.units as u
import numpy as np
from astropy.units import Quantity

from Pyscanner import single_capture


class GalvoScanner:

    def galvo_scan(self, buffer, invert, resolution, input, output, scanpadding, delay, bidirectional, zoom,
                   input_range, exposure):
        im = single_capture(invert_values=invert, resolution=resolution, input_mapping=input, output_mapping=output,
                            scanpaddingfactor=scanpadding, delay=delay, bidirectional=bidirectional, zoom=zoom,
                            input_range=input_range, duration=exposure, )
        buffer[:, :] = np.reshape(im, resolution)


class Camera:
    """
    Camera implementation that performs laser scanning,
    It is very much under construction, and requires testing
    """

    def __init__(self, left=0, top=0, width=100, height=100, dwelltime: Quantity[u.us] = 6 * u.us,
                 input_mapping='Dev4/ai24',
                 x_mirror_mapping='Dev4/ao2', y_mirror_mapping='Dev4/ao3', input_min=-1, input_max=1, delay=0,
                 measurement_time: Quantity[u.ms] = 600 * u.ms, zoom=1, scan_padding=1, bidirectional=True, invert=0,
                 galvo_scanner=None):
        if galvo_scanner is None:
            galvo_scanner = GalvoScanner()

        self._resized = True
        self._image = None
        self._left = left
        self._top = top
        self._width = width
        self._height = height
        self._galvo_scanner = galvo_scanner
        self._dwelltime = dwelltime
        self._input_mapping = input_mapping
        self._x_mirror_mapping = x_mirror_mapping
        self._y_mirror_mapping = y_mirror_mapping
        self._input_min = input_min
        self._input_max = input_max
        self._delay = delay
        self._measurement_time = measurement_time.to(u.ms)
        self._zoom = zoom
        self._scan_padding = scan_padding
        self._bidirectional = bidirectional
        self._invert = invert

    def read(self):
        return self._image

    def trigger(self):

        if self._resized:
            self._image = np.zeros(self.data_shape, dtype=np.uint16)
            self._resized = False

        self.galvo_scanner.galvo_scan(self._image, [self._invert], [self._height, self._width], [self._input_mapping],
                                      [self._x_mirror_mapping, self._y_mirror_mapping], [self._scan_padding],
                                      [self._delay],
                                      [self._bidirectional], [self._zoom], [self._input_min, self._input_max],
                                      [self._measurement_time.to(u.s).value])

    @property
    def data_shape(self):
        return self._height, self._width

    @property
    def left(self) -> int:
        return self._top

    @left.setter
    def left(self, value: int):
        self._top = value

    @property
    def top(self) -> int:
        return self._top

    @top.setter
    def top(self, value: int):
        self._top = value

    @property
    def width(self) -> Annotated[int, {'min': 1, 'max': 1200}]:
        return self._width

    @width.setter
    def width(self, value: int):
        self._width = value
        self._dwelltime = (self._measurement_time / (self._width * self._height))
        self._resized = True

    @property
    def height(self) -> Annotated[int, {'min': 1, 'max': 960}]:
        return self._height

    @height.setter
    def height(self, value: int):
        self._height = value
        self._dwelltime = (self._measurement_time / (self._width * self._height))
        self._resized = True

    @property
    def galvo_scanner(self) -> object:
        return self._galvo_scanner

    # required

    @property
    def input_min(self) -> Annotated[float, {'min': -1.5, 'max': 1.5}]:
        return self._input_min

    @input_min.setter
    def input_min(self, value: float):
        self._input_min = value

    @property
    def input_max(self) -> Annotated[float, {'min': -1.5, 'max': 1.5}]:
        return self._input_max

    @input_max.setter
    def input_max(self, value: float):
        self._input_max = value

    @property
    def dwelltime(self) -> Quantity[u.us]:
        return self._dwelltime

    @dwelltime.setter
    def dwelltime(self, value):
        self._dwelltime = value.to(u.us)
        if self._measurement_time is not (value) * (self._width * self._height):
            self._measurement_time = (value * (self._width * self._height)).to(u.ms)

    @property
    def measurement_time(self) -> Quantity[u.ms]:
        return self._measurement_time

    @measurement_time.setter
    def measurement_time(self, value):
        self._measurement_time = value.to(u.ms)
        if self._dwelltime is not value / (self._width * self._height):
            self._dwelltime = (value / (self._width * self._height)).to(u.us)

    @property
    def delay(self) -> int:
        return self._delay

    @delay.setter
    def delay(self, value: int):
        self._delay = value

    @property
    def zoom(self) -> float:
        return self._zoom

    @zoom.setter
    def zoom(self, value: float):
        self._zoom = value

    @property
    def scan_padding(self) -> float:
        return self._scan_padding

    @scan_padding.setter
    def scan_padding(self, value: float):
        self._scan_padding = value

    @property
    def bidirectional(self) -> bool:
        return self._bidirectional

    @bidirectional.setter
    def bidirectional(self, value: bool):
        self._bidirectional = value

    @property
    def invert(self) -> bool:
        return self._invert

    @invert.setter
    def invert(self, value: bool):
        self._invert = value


g = GalvoScanner()
devices = {
    'cam': Camera(x_mirror_mapping='Dev2/ao0', y_mirror_mapping='Dev2/ao1', input_mapping='Dev2/ai0', galvo_scanner=g),
    'g': g}

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    devices['cam'].dwelltime = 5 * u.us
    devices['cam'].trigger()
    plt.imshow(devices['cam'].read())
    plt.show()
