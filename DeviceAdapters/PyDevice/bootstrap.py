bootstrap = R"raw("
# This file is formatted so that it can be run by the Python interpreter directly, and that it can also be included
# in the c++ code.
# See RunScript() in PyDevice.cpp for the code that loads this script as a c++ string.

import numpy as np
from typing import Protocol, runtime_checkable, get_origin
import sys
import os
import importlib
import astropy.units as u
from typing import Annotated
from enum import Enum
from astropy.units import Quantity

# When running this code in Python directly (for debugging), set up the SCRIPT_PATH
# Normally, this is done by the c++ code
if 'MODULE_PATH' not in locals():
    MODULE_PATH = os.path.dirname(__file__)
    MODULE_NAME = 'test'
    DEBUGGING = True
else:
    DEBUGGING = False

# Load and execute the script file. This script is expected to set up a dictionary object with the name 'devices',
# which holds name -> device pairs for all devices we want to expose in MM.
# The MODULE_PATH is inserted as the first entry of the sys.path, which typically is the path of the main module.
# To have a module include a submodule in a parent directory, use `sys.path.insert(1, os.path.dirname(sys.path[0]))`
# in order to also add the parent directory to the path.
sys.path.insert(0, MODULE_PATH)
with open(MODULE_NAME+".py") as code:
    exec(code.read())
devices = importlib.import_module(MODULE_NAME).devices

@runtime_checkable
class Camera(Protocol):
    data_shape: tuple[int]
    measurement_time: Quantity[u.ms]
    top: int
    left: int
    height: int
    width: int

    #    binning: int # the binning property is optional. If missing, a property binning = 1 will be added.

    def trigger(self) -> None:
        pass

    def read(self) -> np.ndarray:
        pass


@runtime_checkable
class Stage(Protocol):
    step_size: Quantity[u.um]
    """Step size in μm"""

    position: Quantity[u.um]
    """Position in μm. Setting the position causes the stage to start moving to that position. Reading it returns the 
    current position (note that the stage may still be moving!). Overwriting this attribute while the stage is moving 
    causes it to start moving to the new position. Also see :func:`~wait`.
    Stages should use the step_size to convert positions in micrometers to positions in steps, using the equation
    `steps = round(position / step_size)`. This way, code that uses the stage can also choose to make single steps by 
    moving to a position n * step_size.
    """

    def home(self) -> None:
        """Homes the stage. This function does not wait for homing to complete."""
        pass

    def wait(self) -> None:
        """Wait until the stage has finished moving. This should include any time the stage may 'vibrate' after
        stopping."""
        pass


@runtime_checkable
class XYStage(Protocol):
    position_x: Quantity[u.um]
    position_y: Quantity[u.um]
    step_size_x: Quantity[u.um]
    step_size_y: Quantity[u.um]

    def home(self) -> None:
        pass

    def wait(self) -> None:
        pass


def extract_property_metadata(p):
    if not isinstance(p, property) or not hasattr(p, 'fget') or not hasattr(p.fget, '__annotations__'):
        return None

    return_type = p.fget.__annotations__.get('return', None)
    if return_type is None:
        return None

    min = None
    max = None
    options = None
    readonly = not hasattr(p, 'fset')
    pre_set = None
    post_get = None

    if get_origin(return_type) is Annotated:  # Annotated
        meta = return_type.__metadata__[0]
        if isinstance(meta, u.Unit):
            if meta.name == 'ms':
                ptype = 'quantity'
                pre_set = lambda value: u.Quantity(value, u.ms)
                post_get = lambda value: value.to_value(u.ms)
            elif meta.name == 'um':
                ptype = 'quantity'
                pre_set = lambda value: u.Quantity(value, u.um)
                post_get = lambda value: value.to_value(u.um)
            else:
                raise ValueError('Only milliseconds and microsecond units are supported')
        else:
            min = meta.get('min', None)
            max = meta.get('max', None)
            ptype = return_type.__origin__.__name__

    elif issubclass(return_type, Enum):  # Enum
        options = {k.lower().capitalize(): v for (k, v) in return_type.__members__.items()}
        ptype = 'enum'

    else:
        ptype = return_type.__name__

    return ptype, readonly, min, max, options, pre_set, post_get


def to_title_case(name):
    # convert attribute name from snake_case to TitleCase
    return name.replace('_', ' ').title().replace(' ', '')


def set_metadata(obj):
    properties = [(k, extract_property_metadata(v)) for (k, v) in type(obj).__dict__.items()]
    properties = [(p[0], to_title_case(p[0]), *p[1]) for p in properties if p[1] is not None]
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


for d in devices.values():
    set_metadata(d)
    if DEBUGGING:
        print(d._MM_properties)
# )raw";
