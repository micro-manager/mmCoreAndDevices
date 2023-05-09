import numpy as np
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

    def galvo_scan(self,buffer,invert,resolution,input,output,scanpadding,delay,bidirectional,zoom,input_range,exposure):

        im = single_capture(invert_values=invert,resolution=resolution,input_mapping=input,output_mapping=output,
                            scanpaddingfactor=scanpadding,delay=delay,bidirectional=bidirectional,zoom=zoom,
                            input_range=input_range,duration=exposure,)
        buffer[:, :] = np.reshape(im, resolution)


class Camera:
    """camera implementation that ."""

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
        self.galvo_scan.galvo_scan(self.image,[self._invert],[self._width, self._height],[self._input_mapping],
                                   [self._xmirror_mapping,self._ymirror_mapping],[self._scan_padding],[self._delay],
                                   [self._bidirectional],[self._zoom],[self._input_min, self._input_max],[self._exposure_ms/1000])

    def on_resized(self, value):
        self.resized = True
#        self._exposure_ms = (self._dwelltime_us / 1000) * (self._width * self._height)
        return value

    def on_dwelltime(self,value):
        self._exposure_ms = (value/1000)*(self._width*self._height)
        return value

    def on_exposure(self,value):
        self._dwelltime_us = (value*1000) / (self._width * self._height)
        return value


    input_mapping = string_property(default='Dev2/ai0')
    xmirror_mapping = string_property(default='Dev2/ao0')
    ymirror_mapping = string_property(default='Dev2/ao1')
    top = int_property(min=-1000, max=5000, default=0)
    left = int_property(min=-1000, max=5000, default=0)
    width = int_property(min=1, max=4096, default=512, on_update=on_resized)
    height = int_property(min=1, max=4096, default=512, on_update=on_resized)

    input_min = float_property(min=-1.5, max=1.5, default=-1)
    input_max = float_property(min=-1.5, max=1.5, default=1)


    dwelltime_us = float_property(min=0.5, default=4, on_update=on_dwelltime)

    delay = int_property(min=0, max=4096, default=0)
    exposure_ms = float_property(min=0.0, default=600, on_update=on_exposure)
    zoom = float_property(min=1, max=1000, default=1)
    scan_padding = float_property(min=1, max=4, default=1)
    galvo_scan = object_property(default=GalvoScanner())

    bidirectional = bool_property(default=1)
    invert = bool_property(default = 0)
    image = property(fget=get_image)
