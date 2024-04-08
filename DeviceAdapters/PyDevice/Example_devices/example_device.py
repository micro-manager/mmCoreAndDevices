from typing import Annotated
from enum import Enum
import astropy.units as u


class SomeOptions(Enum):
    Orange = 1
    Red = 2
    Blue = 3


class GenericDevice:

    def __init__(self, options, floating_point, distance, boolean, integer, command):
        self._options = options
        self._floating_point = floating_point
        self._distance = distance
        self._boolean = boolean
        self._integer = integer
        self._command = command

    @property
    def options(self) -> SomeOptions:
        return self._options

    @options.setter
    def options(self, value):
        self._options = value

    @property
    def floating_point(self) -> float:
        return self._floating_point

    @floating_point.setter
    def floating_point(self, value):
        self._floating_point = value

    @property
    def distance(self) -> u.Quantity[u.mm]:
        return self._distance

    @distance.setter
    def distance(self, value):
        self._distance = value.to(u.mm)

    @property
    def boolean(self) -> bool:
        return self._boolean

    @boolean.setter
    # setting value:bool forces the users to input the correct type. Optional.
    def boolean(self, value: bool):
        self._boolean = value

    @property
    # setting a range, also sets this range in MicroManager, also optional.
    def integer(self) -> Annotated[int, {'min': 0, 'max': 42}]:
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


device = GenericDevice(SomeOptions.Blue, 23.7, 0.039 * u.m, True, 4, 'Something')
devices = {'some_device': device}

if __name__ == "__main__":
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from bootstrap import PyDevice

    device = PyDevice(devices['some_device'])
    print(device)
    assert device.device_type == 'Device'
