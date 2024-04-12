"""Collection of device classes for testing"""
from enum import Enum
import astropy.units as u
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Options(Enum):
    A = 'option a'
    B = 'option b'
    C = 'option c'


class GenericDevice:
    def __init__(self):
        self._float_value = 0.0
        self._int_value = 0
        self._string_value = ""
        self._bool_value = False
        self._enum_value = Options.A
        self._meters = 0.0 * u.m
        self._millimeters = 0.0 * u.mm

    @property
    def read_only_float(self) -> float:
        return 1.1

    @property
    def read_only_int(self) -> int:
        return 2

    @property
    def read_only_string(self) -> str:
        return 'str'

    @property
    def read_only_enum(self) -> Options:
        return Options.B

    @property
    def read_only_bool(self) -> bool:
        return True

    @property
    def read_write_float(self) -> float:
        return self._float_value

    @read_write_float.setter
    def read_write_float(self, value):
        self._float_value = value

    @property
    def read_write_int(self) -> int:
        return self._int_value

    @read_write_int.setter
    def read_write_int(self, value):
        self._int_value = value

    @property
    def read_write_string(self) -> str:
        return self._string_value

    @read_write_string.setter
    def read_write_string(self, value):
        self._string_value = value

    @property
    def read_write_enum(self) -> Options:
        return self._enum_value

    @read_write_enum.setter
    def read_write_enum(self, value):
        self._enum_value = value

    @property
    def read_write_bool(self) -> bool:
        return self._bool_value

    @read_write_bool.setter
    def read_write_bool(self, value):
        self._bool_value = value

    @property
    def meters(self) -> u.Quantity[u.m]:
        return self._meters

    @meters.setter
    def meters(self, value: u.Quantity[u.m]):
        self._meters = value.to(u.m)

    @property
    def millimeters(self) -> u.Quantity[u.mm]:
        return self._millimeters

    @millimeters.setter
    def millimeters(self, value: u.Quantity[u.mm]):
        self._millimeters = value.to(u.mm)

    @property
    def not_detected(self):
        """This property should not be detected because it has no type annotation"""
        return 0

    @not_detected.setter
    def not_detected(self, value):
        pass

    def add_one(self, a: int) -> int:
        return a + 1


class Camera1:
    def __init__(self):
        self._shutter_ms = 0.0
        self._top = 0
        self._left = 0
        self._height = 0
        self._width = 0

    @property
    def exposure_ms(self) -> float:
        return self._shutter_ms

    @exposure_ms.setter
    def exposure_ms(self, value):
        self._shutter_ms = value

    @property
    def top(self) -> int:
        return self._top

    @top.setter
    def top(self, value):
        self._top = value

    @property
    def left(self) -> int:
        return self._left

    @left.setter
    def left(self, value):
        self._left = value

    @property
    def height(self) -> int:
        return self._height

    @height.setter
    def height(self, value):
        self._height = value

    @property
    def width(self) -> int:
        return self._width

    @width.setter
    def width(self, value):
        self._width = value

    def read(self) -> np.ndarray:
        return np.zeros((self._height, self._width), dtype=np.uint16)

    def busy(self) -> bool:
        return False


class GenericDeviceDirect:
    float_value: float = 0.0
    int_value: int = 0
    string_value: str = ""
    bool_value: bool = False
    enum_value: Options = Options.A
    meters: u.Quantity[u.m] = 0.0 * u.m
    millimeters: u.Quantity[u.mm] = 0.0 * u.mm
    not_detected = 0  # This property should not be detected because it has no type annotation
