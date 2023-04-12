def parse_options(obj, values):
    for p in type(obj).__dict__.items():
        name = p[0]
        pp = p[1]
        ppt = type(pp)
        if issubclass(ppt, base_property): # test if this is one of the special property types (must have _name field)
            print(name)
            value = pp.default
            if name in values:
                value = values[name]
            
            setattr(obj, name, value)

class base_property(property):
    def __init__(self, fget=None, fset=None, fdel=None, doc=None, default=0, min=None, max=None):
        """Add default implementations for getter and setter"""
        if fget == None:
            fget = lambda obj : getattr(obj, self._name)
        if fset == None:
            fset = lambda obj, value : setattr(obj, self._name, self.validate(value))
        
        super().__init__(fget, fset, fdel, doc)
        self.min = min
        self.max = max
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
