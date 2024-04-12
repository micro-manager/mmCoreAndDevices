import numpy as np
from typing import Annotated
import astropy.units as u
from astropy.units import Quantity
from enum import Enum
from concurrent.futures import Future


class NoiseType(Enum):
    UNIFORM = 1
    EXPONENTIAL = 2
    GAUSSIAN = 3


class Camera:
    """Demo camera implementation that returns noise images. To test building device graphs, the random number
    generator is implemented as a separate object with its own properties."""

    def __init__(self, left=0, top=0, width=100, height=100):
        self._rng = np.random.default_rng()
        self._low = 1.0
        self._high = 1000.0
        self._left = left
        self._top = top
        self._width = width
        self._height = height
        self._exposure = 1 * u.ms
        self._noise_type = NoiseType.UNIFORM

    def read(self):
        size = (self._height, self._width)
        if self._noise_type == NoiseType.UNIFORM:
            image = self._rng.uniform(self._low, self._high, size)
        elif self._noise_type == NoiseType.EXPONENTIAL:
            image = self._rng.exponential(self._high - self._low, size) + self._low
        else:
            mean = 0.5 * (self._high + self._low)
            std = 0.5 * (self._high - self._low)
            image = self._rng.normal(mean, std, size)
        return image.astype(np.uint16)

    def busy(self):
        return False

    @property
    def left(self) -> int:
        return self._left

    @left.setter
    def left(self, value: int):
        self._left = value

    @property
    def top(self) -> int:
        return self._top

    @top.setter
    def top(self, value: int):
        self._top = value

    @property
    def width(self) -> int:
        return self._width

    @width.setter
    def width(self, value: int):
        self._width = value

    @property
    def height(self) -> int:
        return self._height

    @height.setter
    def height(self, value: int):
        self._height = value

    @property
    def exposure(self) -> Quantity[u.ms]:
        return self._exposure

    @exposure.setter
    def exposure(self, value):
        self._exposure = value.to(u.ms)

    @property
    def noise_type(self) -> NoiseType:
        return self._noise_type

    @noise_type.setter
    def noise_type(self, value: NoiseType):
        self._noise_type = value

    @property
    def low(self) -> float:
        return self._low

    @low.setter
    def low(self, value: float):
        self._low = value

    @property
    def high(self) -> float:
        return self._high

    @high.setter
    def high(self, value: float):
        self._high = value


devices = {'cam': Camera()}

if __name__ == "__main__":
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from bootstrap import PyDevice

    device = PyDevice(devices['cam'])
    print(device)
    assert device.device_type == 'Camera'
