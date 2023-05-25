import numpy as np

def parse_options(obj, values):
    for p in type(obj).__dict__.items():
        name = p[0]
        pp = p[1]
        if isinstance(pp, base_property): # test if this is one of the special property types (must have _name field)
            if name in values:
                value = values[name]
            else:
                value = pp.default
            setattr(obj, name, value)

class base_property(property):
    def __init__(self, fget=None, fset=None, fdel=None, doc=None, default=None, min=None, max=None, on_update=None, allowed_values=None):
        """Add default implementations for getter and setter"""
        if on_update == None:
            on_update = lambda obj, value : value
        if fget == None:
            fget = lambda obj : getattr(obj, self._name)
        if fset == None:
            fset = lambda obj, value : setattr(obj, self._name, self.on_update(obj, self.validate(value)))
        if default == None and allowed_values != None:
            default = allowed_values[0]
        
        super().__init__(fget, fset, fdel, doc)
        self.min = min
        self.max = max
        self.on_update = on_update
        self.default = default
        self.allowed_values = allowed_values
        
    def __set_name__(self, owner, name):
        self._name = "_" + name
        
    def _ValueError(self, msg):
        raise ValueError("Invalid value for property " + self._name + "\n" + msg)
        
    def validate(self, value):
        if self.allowed_values != None and not value in self.allowed_values:
            self._ValueError(f"Value {value} is not one of {self.allowed_values}")
        pass

class int_property(base_property):        
    def validate(self, value):
        super().validate(value)
        if not isinstance(value, int): # incorrect type, try converting float to int
            if isinstance(value, float) and value.is_integer():
                value = round(value)
            else:
                self._ValueError(f"Value {value} is not an integer, but has type {type(value)}")
                
        if (self.min != None and value < self.min) or (self.max != None and value > self.max):
            self._ValueError(f"Value {value} not in range [{self.min},{self.max}]")
        return value

class object_property(base_property):
    def __init__(self, fget=None, fset=None, fdel=None, doc=None, default=None, on_update=None, type=None):
        super().__init__(fget = fget, fset = fset, fdel = fdel, doc = doc, default = default, on_update = on_update)
        self.type = type
    def validate(self, value):
        super().validate(value)
        if self.type != None and not isinstance(value, self.type):
            self._ValueError(f"Value {value} is not of type {self.type}")
        return value


class bool_property(int_property):
    def __init__(self, fget=None, fset=None, fdel=None, doc=None, default=None, on_update=None):
        super().__init__(fget = fget, fset = fset, fdel = fdel, doc = doc, default = default, on_update = on_update, allowed_values = [0, 1], min=0, max=1)

class float_property(base_property):        
    def validate(self, value):
        super().validate(value)
        if not (isinstance(value, int) or isinstance(value, float)):
            self._ValueError(self._err() + f"Value {value} is not a number, but has type {type(value)}")
        if (self.min != None and value < self.min) or (self.max != None and value > self.max):
            self._ValueError(f"Value {value} not in range [{self.min},{self.max}]")
        return value
    
class string_property(base_property):        
    def validate(self, value):
        super().validate(value)
        if not isinstance(value, str):
            self._ValueError(f"Value {value} is not a string, but has type {type(value)}")
        return value
    
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
