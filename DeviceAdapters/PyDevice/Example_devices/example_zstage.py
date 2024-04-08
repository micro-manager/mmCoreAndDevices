from astropy.units import Quantity
import astropy.units as u


class GenericZStage:
    def __init__(self, step_size: Quantity[u.um]):
        super().__init__()
        self._step_size = step_size.to(u.um)

        self._position = 0.0 * u.um

    def home(self):
        self._position = 0.0 * u.um

    def busy(self):
        return False

    @property
    def position(self) -> Quantity[u.um]:
        return self._position

    @position.setter
    def position(self, value: Quantity[u.um]):
        former = self._position
        self._position = value.to(u.um) + former

    @property
    def step_size(self) -> Quantity[u.um]:
        return self._step_size

    @step_size.setter
    def step_size(self, value: Quantity[u.um]):
        self._step_size = value.to(u.um)


devices = {'stage': GenericZStage(1 * u.um)}

if __name__ == "__main__":
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from bootstrap import PyDevice

    device = PyDevice(devices['stage'])
    print(device)
    assert device.device_type == 'Stage'
