from typing import Annotated
from enum import Enum
import astropy.units as u
from annotated_types import Ge, Le, Gt, Lt, Interval


# Note: this example requires astropy and annotatedtypes to be installed


class Color(Enum):
    Orange = 1
    Red = 2
    Blue = 3


class DeviceWithUnits:

    def __init__(self, color, floating_point, distance, boolean, integer, integer2, voltage1, voltage2):
        self._color = color
        self._floating_point = floating_point
        self._distance = distance
        self._boolean = boolean
        self._integer = integer
        self._integer2 = integer2
        self._voltage1 = voltage1
        self._voltage2 = voltage2

    @property
    def color(self) -> Color:
        return self._color

    @color.setter
    def color(self, value):
        self._color = value

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
    def integer(self) -> Annotated[int, Ge(1), Le(42)]:
        return self._integer

    @integer.setter
    def integer(self, value):
        self._integer = value

    @property
    # setting a range, also sets this range in MicroManager, also optional.
    def integer2(self) -> Annotated[int, Gt(0), Lt(43)]:
        return self._integer2

    @integer2.setter
    def integer2(self, value):
        self._integer2 = value

    @property
    def voltage1(self) -> u.Quantity[u.V]:
        return self._voltage1

    @voltage1.setter
    def voltage1(self, value: u.Quantity[u.V]):
        self._voltage1 = value.to(u.V)

    @property
    def voltage2(self) -> Annotated[u.Quantity[u.V], Interval(ge=0 * u.V, le=0.9 * u.V)]:
        return self._voltage2

    @voltage2.setter
    def voltage2(self, value: u.Quantity[u.V]):
        self._voltage2 = value.to(u.V)


device = DeviceWithUnits(Color.Blue, 23.7, 0.039 * u.m, True, 4, 5, 0.5 * u.V, 0.6 * u.V)
devices = {'some_device': device}

if __name__ == "__main__":
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from bootstrap import PyDevice

    device = PyDevice(devices['some_device'])
    device.properties[-1].get()
    print(device)
    assert device.device_type == 'Device'
