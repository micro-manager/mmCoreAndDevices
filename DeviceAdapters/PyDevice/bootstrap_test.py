import numpy as np
from typing import Protocol, runtime_checkable
import numpy as np
from typing import Any, Annotated
from pprint import pprint

class RandomGenerator:
    """Demo device, used to test building device graphs. It generates random numbers for use in the Camera"""

    def __init__(self, min=0, max=1000):
        self._min = min
        self._max = max

    def generate_into(self, buffer):
        buffer[:, :] = np.random.randint(self._min, self._max, buffer.shape, dtype=np.uint16)

    @property
    def min(self) -> Annotated[int, {'min': 0, 'max': 0xFFFF}]:
        return self._min

    @min.setter
    def min(self, value):
        self._min = value

    @property
    def max(self) -> Annotated[int, {'min': 0, 'max': 0xFFFF}]:
        return self._max

    @min.setter
    def max(self, value):
        self._max = value


class Camera:
    """Demo camera implementation that returns noise images. To test building device graphs, the random number generator is implemented as a separate object with its own properties."""

    def __init__(self, left=0, top=0, shape=(100, 100), measurement_time=100, random_generator=None):

        if not len(shape) == 2 or np.prod(shape) == 0:
            raise ValueError("Invalid shape, should be non-zero 2-dimensional.")
        if random_generator is None:
            random_generator = RandomGenerator()

        self._resized = True
        self._image = None
        self._left = left
        self._top = top
        self._shape = shape
        self._measurement_time = measurement_time
        self._random_generator = random_generator

    def trigger(self):
        if self._resized:
            self._image = np.zeros(self.data_shape, dtype=np.uint16)
            self._resized = False
        self.random_generator.generate_into(self._image)

    def read(self):
        return self._image

    @property
    def data_shape(self):
        return self._shape

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
    def width(self) -> int:
        return self._shape[1]

    @width.setter
    def width(self, value: int):
        self._shape[1] = value
        self._resized = true

    @property
    def height(self) -> int:
        return self._shape[0]

    @height.setter
    def height(self, value: int):
        self._shape[0] = value
        self._resized = true

    @property
    def measurement_time(self) -> float:
        return self._measurement_time

    @measurement_time.setter
    def measurement_time(self, value):
        self._measurement_time = value

    @property
    def Binning(self) -> float:
        return 1

    @property
    def random_generator(self):
        return self._random_generator


r = RandomGenerator()
devices = {'cam': Camera(random_generator=r), 'rng': r}

##########################333

@runtime_checkable
class Camera(Protocol):
    data_shape: tuple[int]
    measurement_time: float

    def trigger(self) -> None:
        pass

    def read(self) -> np.ndarray:
        pass

    top: int
    left: int
    height: int
    width: int
    Binning: int


def extract_property_metadata(p):
    if not isinstance(p, property) or not hasattr(p, 'fget') or not hasattr(p.fget, '__annotations__'):
        return None

    return_type = p.fget.__annotations__.get('return', None)
    if return_type is None:
        return None

    if hasattr(return_type, '__metadata__'):
        meta = return_type.__metadata__[0]
        min = meta.get('min', None)
        max = meta.get('max', None)
        return_type = return_type.__origin__
    else:
        min = None
        max = None

    ptype = getattr(return_type, '__name__', 'any')
    return (ptype, min, max)


def set_metadata(obj):
    if isinstance(obj, Camera):
        dtype = "Camera"
    else:
        dtype = "Device"
    properties = [(k, extract_property_metadata(v)) for (k, v) in type(obj).__dict__.items()]
    properties = [(p[0], *p[1]) for p in properties if p[1] is not None]
    obj._MM_dtype = dtype
    obj._MM_properties = properties


for d in devices.values():
    set_metadata(d)

pprint([d._MM_properties for d in devices.values()], indent = 2)