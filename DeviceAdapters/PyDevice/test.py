import numpy as np
import math

def parse_options(obj, values):
    for p in type(obj).__dict__.items():
        name = p[0]
        pp = p[1]
        ppt = type(pp)
        if issubclass(ppt, base_property): # test if this is one of the special property types (must have _name field)
            value = pp.default
            if name in values:
                value = values[name]
            
            setattr(obj, name, value)

class base_property(property):
    def __init__(self, fget=None, fset=None, fdel=None, doc=None, default=0, min=None, max=None, on_update=None):
        """Add default implementations for getter and setter"""
        if on_update == None:
            on_update = lambda obj, value : value
        if fget == None:
            fget = lambda obj : getattr(obj, self._name)
        if fset == None:
            fset = lambda obj, value : setattr(obj, self._name, self.on_update(obj, self.validate(value)))
        
        super().__init__(fget, fset, fdel, doc)
        self.min = min
        self.max = max
        self.on_update = on_update
        self.default = default
        
    def __set_name__(self, owner, name):
        self._name = "_" + name
        
    def _ValueError(self, msg):
        raise ValueError("Invalid value for property " + self._name + "\n" + msg)
        
    def validate(self, value):
        pass

class int_property(base_property):        
    def validate(self, value):
        if not isinstance(value, int): # incorrect type, try converting float to int
            if isinstance(value, float) and value.is_integer():
                value = round(value)
            else:
                self._ValueError(f"Value {value} is not an integer, but has type {type(value)}")
                
        if (self.min != None and value < self.min) or (self.max != None and value > self.max):
            self._ValueError(f"Value {value} not in range [{self.min},{self.max}]")
        return value

class float_property(base_property):        
    def validate(self, value):
        if not (isinstance(value, int) or isinstance(value, float)):
            self._ValueError(self._err() + f"Value {value} is not a number, but has type {type(value)}")
        if (self.min != None and value < self.min) or (self.max != None and value > self.max):
            self._ValueError(f"Value {value} not in range [{self.min},{self.max}]")
        return value
    
class string_property(base_property):        
    def validate(self, value):
        if not isinstance(value, str):
            self._ValueError(f"Value {value} is not a string, but has type {type(value)}")
        return value

class Device:
    width = int_property(min = 10, max = 11, default = 10)
    height = int_property(min = 0, max = 10, default = 1)
    def __init__(self, **kwargs):
        parse_options(self, kwargs)

class Camera:
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
        self.image[:,:] = np.random.randint(0, 1000, (self.width, self.height), dtype=np.uint16)
    
    def on_resized(self, value):
        self.resized = True
        return value
            
    top = int_property(min = -1000, max = 5000, default = 0)
    left = int_property(min = -1000, max = 5000, default = 0)
    width = int_property(min = 1, max = 4096, default = 512, on_update = on_resized)
    height = int_property(min = 1, max = 4096, default = 512, on_update = on_resized)
    exposure_ms = float_property(min = 0.0, max = math.inf, default = 100)
    image = property(fget = get_image)

    
# d = Device(width = 10)