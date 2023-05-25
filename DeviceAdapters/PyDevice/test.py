from base_device_properties import float_property, int_property, string_property, object_property, base_property, bool_property, parse_options
import numpy as np
    
class Device:
    width = int_property(min = 10, max = 11, default = 10)
    height = int_property(min = 0, max = 10, default = 1)
    def __init__(self, **kwargs):
        parse_options(self, kwargs)

class RandomGenerator:
    """Demo device, used to test building device graphs. It generates random numbers for use in the Camera"""
    def __init__(self, **kwargs):
        parse_options(self, kwargs)
        self.resized = True
    def generate_into(self, buffer):
        buffer[:,:] = np.random.randint(self.min, self.max, buffer.shape, dtype=np.uint16)
    min = int_property(default = 0)
    max = int_property(default = 1000)
        

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
        self.random_generator.generate_into(self.image)

    def on_resized(self, value):
        self.resized = True
        return value

    def on_invert(self, value):
        self.resized = True
        return value

            
    top = int_property(min = -1000, max = 5000, default = 0)
    left = int_property(min = -1000, max = 5000, default = 0)
    width = int_property(min = 1, max = 4096, default = 512, on_update = on_resized)
    height = int_property(min = 1, max = 4096, default = 512, on_update = on_resized)
    exposure_ms = float_property(min = 0.0, default = 100)
    random_generator = object_property(default = RandomGenerator())
    Binning = int_property(allowed_values = [1])

    invert = bool_property(default = 0)
    image = property(fget = get_image)
