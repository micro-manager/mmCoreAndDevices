bootstrap = R"raw("
# This file is formatted so that it can be run by the Python interpreter directly, and that it can also be included
# in the c++ code.
# See RunScript() in PyDevice.cpp for the code that loads this script as a c++ string.

import numpy as np
from typing import Protocol, runtime_checkable, get_origin
import sys
import os
import astropy.units as u
from typing import Annotated
from enum import Enum
from astropy.units import Quantity

# When running this code in Python directly (for debugging), set up the SCRIPT_PATH
# Normally, this is done by the c++ code
if 'SCRIPT_PATH' not in locals():
    SCRIPT_PATH = os.path.join(os.path.dirname(__file__), 'test.py')
    DEBUGGING = True
else:
    DEBUGGING = False

# Load and execute the script file. This script is expected to set up a dictionary object with the name 'devices',
# which holds name -> device pairs for all devices we want to expose in MM.
# The SCRIPT_PATH is inserted as the first entry of the sys.path.
sys.path.insert(0, os.path.dirname(SCRIPT_PATH))
with open(SCRIPT_PATH) as code:
    exec(code.read())

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
    x: Quantity[u.um]
    y: Quantity[u.um]
    step_size_x: Quantity[u.um]
    step_size_y: Quantity[u.um]

    def home(self) -> None:
        pass

    def wait(self) -> None:
        pass

@runtime_checkable
class SLM(Protocol):
    phases: np.ndarray
    
    def update(self, wait_factor=1.0, wait=True):
        """Refresh the SLM to show the updates phase pattern.

        If the SLM is currently reserved (see `reserve`), this function waits until the reservation is almost (*) over before updating the SLM.
        The SLM waits for the pattern of the SLM to stabilize before returning.

        *) In case of SLMs with an idle time (latency), the image may be sent to the hardware already before the reservation is over, as long as the actual image
        on the SLM is guaranteed not to update until the reservation is over.

        :param wait_factor: time to wait for the image to stabilize. Default = 1.0 should wait for a pre-defined time (the `settle_time`) that guarantees stability
        for most practical cases. Use a higher value to allow for extra stabilization time, or a lower value if you want to trigger a measurement before the SLM is fully stable.

        :param wait: when set to False, do not wait for the image to stabilize but reserve the SLM for this period instead. This can be used to pipeline measurements (see `Feedback`).
        The client code needs to explicilty call `wait` to wait for stabilization of the image. 
        """
        pass

    def wait(self):
        """Wait for the SLM to become available. If there are no current reservations, return immediately."""
        pass

    def reserve(self, time: Quantity[u.ms]):
        """Reserve the SLM for a specified amount of time. During this time, the SLM pattern cannot be changed."""
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

    if get_origin(return_type) is Annotated:  # Annotated
        meta = return_type.__metadata__[0]
        if isinstance(meta, u.Unit):
            ptype = str(meta.physical_type) # 'time','length','mass', etc.
        else:
            min = meta.get('min', None)
            max = meta.get('max', None)
            ptype = return_type.__origin__.__name__

    elif issubclass(return_type, Enum):  # Enum
        options = {k.lower().capitalize(): v for (k, v) in return_type.__members__.items()}
        ptype = 'enum'

    else:
        ptype = return_type.__name__

    return ptype, readonly, min, max, options


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


unit_ms = u.ms
unit_um = u.um

for d in devices.values():
    set_metadata(d)
    if DEBUGGING:
        print(d._MM_properties)
# )raw";
