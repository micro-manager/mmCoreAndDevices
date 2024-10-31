from astropy.units import Quantity
import astropy.units as u


class GenericXYStage:
    def __init__(self, step_size_x: Quantity[u.um], step_size_y: Quantity[u.um]):
        super().__init__()
        self._step_size_x = step_size_x.to(u.um)
        self._step_size_y = step_size_y.to(u.um)
        self._y = 0.0 * u.um
        self._x = 0.0 * u.um

    def home(self):
        self._x = 0.0 * u.um
        self._y = 0.0 * u.um

    def busy(self):
        return False

    @property
    def x(self) -> Quantity[u.um]:
        return self._x

    @x.setter
    def x(self, value: Quantity[u.um]):
        former = self._x
        self._x = value.to(u.um) + former

    @property
    def y(self) -> Quantity[u.um]:
        return self._y

    @y.setter
    def y(self, value: Quantity[u.um]):
        former = self._y
        self._y = value.to(u.um) + former

    @property
    def step_size_x(self) -> Quantity[u.um]:
        return self._step_size_x

    @step_size_x.setter
    def step_size_x(self, value: Quantity[u.um]):
        self._step_size_x = value.to(u.um)

    @property
    def step_size_y(self) -> Quantity[u.um]:
        return self._step_size_y

    @step_size_y.setter
    def step_size_y(self, value: Quantity[u.um]):
        self._step_size_y = value.to(u.um)


devices = {'stage': GenericXYStage(1 * u.um, 1 * u.um)}

if __name__ == "__main__":
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from bootstrap import PyDevice

    device = PyDevice(devices['stage'])
    print(device)
    assert device.device_type == 'XYStage'
