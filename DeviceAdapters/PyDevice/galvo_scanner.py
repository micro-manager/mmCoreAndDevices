import nidaqmx as ni
import numpy as np
from nidaqmx.constants import TaskMode
import pylab as plt
import ast
import os
import sys

# Add the directory to sys.path so Python can find modules in it
sys.path.append('C:\\Users\\Jeroen Doornbos\\Documents\\wfs_current\\micro-manager\\mmCoreAndDevices\\DeviceAdapters\\MM_pydevice')


# Import the module
from Pyscanner import single_capture
from test_cam import float_property, int_property, string_property, object_property, base_property, bool_property, parse_options
# this is all the most recently updated functions


class GalvoScanner:

    def __init__(self, **kwargs):
        parse_options(self, kwargs)
        self.resized = True

    def galvo_scan(self,buffer):
        # we need 2 things build here: a flip values bool, and an  interval bool (0 to 2^16, or -2^16/2 to 2^16/2)
        # but maybe we need that in the scanning function
        im = single_capture()
        buffer[:, :] = im


class Camera:
    """Demo camera implementation that returns noise images. To test building device graphs, the random number generator is implemented as a separate object with its own properties."""

    def __init__(self, **kwargs):
        parse_options(self, kwargs)
        self.resized = True

    def get_image(self):
        if self.resized:
            self._image = np.zeros((self._width, self._height), dtype=np.uint16)
            self.resized = False

        return self._image

    def trigger(self):
        pass

    def wait(self):
        self.random_generator.galvo_scan(self.image)

    def on_resized(self, value):
        self.resized = True
        return value

    top = int_property(min=-1000, max=5000, default=0)
    left = int_property(min=-1000, max=5000, default=0)
    width = int_property(min=1, max=4096, default=512, on_update=on_resized)
    height = int_property(min=1, max=4096, default=512, on_update=on_resized)
    exposure_ms = float_property(min=0.0, default=100)
    random_generator = object_property(default=GalvoScanner())

    # invert = bool_property(default = 0)
    image = property(fget=get_image)