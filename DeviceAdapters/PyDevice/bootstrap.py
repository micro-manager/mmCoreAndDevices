bootstrap = R"raw("
# This file is formatted so that it can be run by the Python interpreter directly, and that it can also be included
# in the c++ code.
# See RunScript() in PyDevice.cpp for the code that loads this script as a c++ string.

import sys
_original_path = sys.path.copy()
def set_path(path):
    sys.path = [*_original_path, *path.split(';')]
if '_EXTRA_SEARCH_PATH' in locals():
    set_path(_EXTRA_SEARCH_PATH)

import numpy as np
from typing import Protocol, runtime_checkable, get_origin, Annotated
import os
import traceback
import astropy.units as u
import concurrent.futures
import inspect
from enum import Enum
from astropy.units import Quantity

@runtime_checkable
class Camera(Protocol):
    """A camera or other source of 2-D image data"""

    duration: Quantity[u.ms]
    """Shutter time / exposure time of the camera"""

    top: int
    left: int
    height: int
    width: int
    # the binning property is optional. If missing, an attribute with fixed value binning = 1 will be added.
    # binning: int 

    def trigger(self) -> concurrent.futures.Future:
        """Triggers the Camera

        This function returns a `concurrent.future.Future` object that receives the grabbed camera frame as a numpy array. 
        Calling code can call `results()` on the future to retrieve the array.
        The array must have `shape = (height, width)` and hold `uint16` data."""
        pass


@runtime_checkable
class Stage(Protocol):
    """A 1-D translation stage"""

    step_size: Quantity[u.um]
    """Step size in μm"""

    position: Quantity[u.um]
    """Position in μm. Setting the position causes the stage to start moving to that position. Reading it returns the 
    current position (note that the stage may still be moving!). Overwriting this attribute while the stage is moving 
    causes it to start moving to the new position. Also see :func:`~wait`.
    Stages may use the step_size to convert positions in micrometers to positions in steps, using the equation
    `steps = round(position / step_size)`.
    """

    def home(self) -> None:
        """Homes the stage. This function does not wait for homing to complete."""
        pass

    def busy(self) -> bool:
        """Returns False if the stage has finished moving. This should include any time the stage may 'vibrate' after
        stopping."""
        pass


@runtime_checkable
class XYStage(Protocol):
    """A 1-D translation stage"""

    x: Quantity[u.um]
    """Position in μm. Setting the position causes the stage to start moving to that position. Reading it returns the 
    current position (note that the stage may still be moving!). Overwriting this attribute while the stage is moving 
    causes it to start moving to the new position. Also see :func:`~wait`.
    Stages may use the step_size to convert positions in micrometers to positions in steps, using the equation
    `steps = round(position / step_size)`.
    """

    y: Quantity[u.um]
    """Position in μm. Setting the position causes the stage to start moving to that position. Reading it returns the 
    current position (note that the stage may still be moving!). Overwriting this attribute while the stage is moving 
    causes it to start moving to the new position. Also see :func:`~wait`.
    Stages may use the step_size to convert positions in micrometers to positions in steps, using the equation
    `steps = round(position / step_size)`.
    """

    step_size_x: Quantity[u.um]
    """Step size in μm"""

    step_size_y: Quantity[u.um]
    """Step size in μm"""

    def home(self) -> None:
        """Homes the stage. This function does not wait for homing to complete."""
        pass

    def busy(self) -> bool:
        """Returns False if the stage has finished moving. This should include any time the stage may 'vibrate' after
        stopping."""
        pass


def extract_property_metadata(name, p):
    if not isinstance(p, property) or not hasattr(p, 'fget') or not hasattr(p.fget, '__annotations__'):
        return None

    return_type = p.fget.__annotations__.get('return', None)
    if return_type is None:
        return None

    min = None
    max = None
    options = None
    readonly = (not hasattr(p, 'fset')) or p.fset is None
    mm_name = to_title_case(name)

    if get_origin(return_type) is Annotated:  # Annotated
        meta = return_type.__metadata__[0]
        if isinstance(meta, u.Unit):
            ptype = 'quantity'
            options = meta
            mm_name = mm_name + '_' + str(meta)
        else:
            min = meta.get('min', None)
            max = meta.get('max', None)
            ptype = return_type.__origin__.__name__

    elif issubclass(return_type, Enum):  # Enum
        options = {k.lower().capitalize(): v for (k, v) in return_type.__members__.items()}
        ptype = 'enum'

    elif issubclass(return_type, int) or issubclass(return_type, float) or issubclass(return_type, str):
        ptype = return_type.__name__

    else:
        return None  # unsupported type

    return name, mm_name, ptype, readonly, min, max, options


def to_title_case(name):
    # convert attribute name from snake_case to TitleCase
    return name.replace('_', ' ').title().replace(' ', '')


def set_metadata(obj):
    # get a list of all properties, including the ones in base classes
    classes = inspect.getmro(type(obj))
    alldict = {}
    for c in classes[::-1]:
       alldict.update(c.__dict__)  

    properties = [extract_property_metadata(k, v) for (k, v) in alldict.items()]
    properties = [p for p in properties if p is not None]
    if isinstance(obj, Camera):
        dtype = "Camera"
        if not hasattr(obj, 'binning'):
            obj.binning = 1
            properties.append(('binning', 'Binning', 'int', False, 1, 1, None, None, None))
    elif isinstance(obj, XYStage):
        dtype = "XYStage"
    elif isinstance(obj, Stage):
        dtype = "Stage"
    else:
        dtype = "Device"
    obj._MM_dtype = dtype
    obj._MM_properties = properties

unit_um = u.um

def scan_devices(devices):
    # Scans the device dictionary and inserts metadata for each item
    for d in devices.values():
        set_metadata(d)    
    return devices


def traceback_to_string(tb):
    return ''.join(traceback.format_tb(tb))

# )raw";
