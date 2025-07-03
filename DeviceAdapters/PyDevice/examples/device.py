from enum import Enum
from typing import Annotated

from annotated_types import Ge, Le


class Color(Enum):
    Orange = 1
    Red = 2
    Blue = 3


class GenericDevice:

    def __init__(self, color, floating_point, distance, boolean, integer, command):
        self._color = color
        self._floating_point = floating_point
        self._distance_mm = distance
        self._boolean = boolean
        self._integer = integer
        self._command = command

    @property
    def color(self) -> Color:
        return self._color

    @color.setter
    def color(self, value):
        self._color = value

    @property
    def floating_point(self) -> Annotated[float, Ge(0.0), Le(1.0)]:
        # setting a range, also sets this range in MicroManager
        return self._floating_point

    @floating_point.setter
    def floating_point(self, value):
        self._floating_point = value

    @property
    def distance_mm(self) -> float:
        return self._distance_mm

    @distance_mm.setter
    def distance_mm(self, value):
        self._distance_mm = float(value)

    @property
    def boolean(self) -> bool:
        return self._boolean

    @boolean.setter
    # setting value:bool forces the users to input the correct type. Optional.
    def boolean(self, value: bool):
        self._boolean = value

    @property
    def integer(self) -> int:
        return self._integer

    @integer.setter
    def integer(self, value):
        self._integer = value

    @property
    def command(self) -> str:
        return self._command

    @command.setter
    def command(self, value):
        self._command = str(value)


device = GenericDevice(Color.Blue, 23.7, 0.039, True, 4, 'Something')
devices = {'some_device': device}

if __name__ == "__main__":
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from bootstrap import PyDevice

    device = PyDevice(devices['some_device'])
    print(device)
    assert device.device_type == 'Device'
