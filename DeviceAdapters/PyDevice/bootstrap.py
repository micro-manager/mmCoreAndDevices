bootstrap = R"raw("
# This file is formatted so that it can be run by the Python interpreter directly, and that it can also be included
# in the c++ code.
# See RunScript() in PyDevice.cpp for the code that loads this script as a c++ string.
import sys
import importlib
from typing import get_origin, Annotated
from types import MethodType
import traceback
import inspect
from enum import Enum

# the presence of the astropy.units module is optional.
try:
    from astropy.units import UnitBase
except ImportError:
    class UnitBase:
        pass  # dummy class to prevent errors when the astropy.units module is not available


def traceback_to_string(tb):
    """Helper function to convert a traceback object to a string. This function is called from the C++ code."""
    return ''.join(traceback.format_tb(tb))


def load_devices(module_path, script_path, script_name) -> dict:
    """Helper function to load the Python script that defines the devices. This function is called from the C++ code."""
    _original_path = sys.path.copy()
    try:
        sys.path = [*_original_path, *module_path.split(';'), script_path]
        script = importlib.import_module(script_name)
        return {k: PyDevice(v) for k, v in script.devices.items()}
    finally:
        sys.path = _original_path


def _to_title_case(name: str) -> str:
    for suffix in ['_s', '_ms', '_us', '_ns', '_m', '_cm', '_mm', '_um', '_nm', '_A', '_mA', '_uA', '_V', '_mV', '_uV',
                   '_Hz', '_kHz', '_MHz', '_GHz']:
        if name.endswith(suffix):
            return name[:-len(suffix)].title().replace('_', '') + '-' + suffix[1:]

    return name.title().replace('_', '')  # convert to TitleCase


class PyProperty:
    def __init__(self, obj, name: str, p: property):
        """Extracts metadata from a property object and stores it in a format accessible to the C++ code.

        Recognizes properties, as well as attributes.
        """
        self.python_name = name
        self.mm_name = _to_title_case(name)
        self.min = None
        self.max = None
        self.options = None
        self.unit = None

        if isinstance(p, property):  # property
            fset = getattr(p, 'fset', None)
            self.get = lambda: p.fget(obj)
            self.set = (lambda value: fset(obj, value)) if fset is not None else None
            # get the return type from the annotations.
            # Note: this will raise an error if p is not gettable or has no return type annotation
            return_type = p.fget.__annotations__['return']  # noqa: ok to raise an error here
        else:  # attribute
            self.get = lambda: getattr(obj, name)
            self.set = lambda value: setattr(obj, name, value)
            return_type = p

        self.data_type = self._process_return_type(return_type)

    def _process_return_type(self, return_type) -> str:
        if get_origin(return_type) is Annotated:  # Annotated
            meta = return_type.__metadata__[0]
            if isinstance(meta, UnitBase):
                self.unit = meta
                self.mm_name = self.mm_name + '-' + str(self.unit)
                if self.set is not None:
                    set_original = self.set
                    self.set = lambda value: set_original(value * self.unit)
                get_original = self.get
                self.get = lambda: get_original().to_value(self.unit)
                return 'float'
            else:
                self.min = meta.get('min', None)
                self.max = meta.get('max', None)
                return self._process_return_type(return_type.__origin__)  # recursively process the base type

        elif return_type == bool:
            return 'bool'

        elif not inspect.isclass(return_type):  # for example, Optional[...]
            raise ValueError(return_type)  # unsupported property type

        elif issubclass(return_type, Enum):  # Enum
            self.options = {k.lower().capitalize(): v for (k, v) in return_type.__members__.items()}
            return 'enum'

        elif issubclass(return_type, int):
            return 'int'

        elif issubclass(return_type, float):
            return 'float'

        elif issubclass(return_type, str):
            return 'str'

        else:
            raise ValueError(return_type)  # unsupported property type

    def __str__(self):
        return f"{self.mm_name}({self.data_type}{', readonly' if self.set is None else ''})"


class PyDevice:
    """Wrapper class for a device that provides metadata about the device's properties
     and methods in a format accessible to the C++ code."""

    def __init__(self, device):
        # get a list of all properties and methods, including the ones in base classes
        # Also process class annotations, for attributes that are not properties
        class_hierarchy = inspect.getmro(device.__class__)
        all_dict = {}
        for c in class_hierarchy[::-1]:
            all_dict.update(c.__dict__)
            annotations = c.__dict__.get('__annotations__', {})
            all_dict.update(annotations)

        # extract metadata for each property and method
        self.properties = []
        self.methods = {}
        for name, p in all_dict.items():
            if not name.startswith('_'):
                if inspect.isfunction(p):
                    self.methods[name] = MethodType(p, device)
                else:
                    try:
                        self.properties.append(PyProperty(device, name, p))
                    except (AttributeError, KeyError, ValueError):
                        pass  # property does not have a getter or annotations, or is of an incorrect type

        # detect the device type
        if self._init_camera():
            self.device_type = 'Camera'
        elif self._init_xy_stage():
            self.device_type = 'XYStage'
        elif self._init_stage():
            self.device_type = 'Stage'
        else:
            # 'busy' is optional for generic device types
            if not self._has_methods('busy'):
                self.methods['busy'] = lambda: False
            self.device_type = 'Device'

        if not self._has_methods('busy'):
            raise Exception(f"Device type '{self.device_type}' must have a 'busy' method")

    def _init_camera(self) -> bool:
        """Checks if the device corresponds to a Camera, and prepares the camera object if it does"""

        # check required properties and methods
        if not self._has_properties(('Exposure-ms', 'float'), ('Top', 'int'), ('Left', 'int'), ('Height', 'int'),
                                    ('Width', 'int')) or not self._has_methods('read'):
            return False

        # add a read-only binning property if it does not exist
        if not self._has_properties(('Binning', 'int')):
            @property
            def binning(obj) -> int:
                return 1

            self.properties.append(PyProperty(self, 'binning', binning))

        return True

    def _init_xy_stage(self) -> bool:
        """Checks if the device corresponds to an XYStage, and prepares the stage object if it does"""
        return self._has_properties(('X-um', 'float'), ('Y-um', 'float'), ('StepSizeX-um', 'float'),
                                    ('StepSizeY-um', 'float')) and self._has_methods('home')

    def _init_stage(self) -> bool:
        """Checks if the device corresponds to an XYStage, and prepares the stage object if it does"""
        return self._has_properties(('Position-um', 'float'), ('StepSize-um', 'float')) and self._has_methods('home')

    def _has_properties(self, *properties) -> bool:
        """Checks if the device has the specified properties of the specified types

        Returns:
            bool: True if the device has the specified properties, False otherwise

        Raises:
            ValueError: if the device has one of the specified properties, but with a different type
        """
        for name, data_type in properties:
            # find the property with the specified name
            p = next((p for p in self.properties if p.mm_name == name), None)
            if p is None:
                return False
            if p.data_type != data_type:
                raise ValueError(f"Property '{name}' is of type {p.data_type}, expected {data_type}")
        return True

    def _has_methods(self, *required_methods) -> bool:
        """Checks if the device has the specified methods

        Returns:
            bool: True if the device has the specified methods, False otherwise
        """
        return all(m in self.methods.keys() for m in required_methods)

    def __str__(self):
        return f"{self.device_type}({', '.join(str(p) for p in self.properties)})"
# )raw";
