def parse_options(obj, values):
    """In this function, we set the properties. 
    Because on_update functions may be interdependent on other properties, the property values are initialised first,
    before actually making the properties"""
    for p in type(obj).__dict__.items():
        name = p[0]
        pp = p[1]
        if isinstance(pp, base_property):
            if name in values:
                value = values[name]
            else:
                value = pp.default

            setattr(obj, "_" + name, value)  # Set the variable as an attribute of the object
            
    for p in type(obj).__dict__.items():
        name = p[0]
        pp = p[1]
        if isinstance(pp, base_property):
            if name in values:
                value = values[name]
            else:
                value = pp.default
            setattr(obj, name, value)  # Set the value using setattr(obj, name, value)

class base_property(property):
    def __init__(self, fget=None, fset=None, fdel=None, doc=None, default=None, min=None, max=None, on_update=None,
                 allowed_values=None):
        """Add default implementations for getter and setter"""
        if on_update == None:
            on_update = lambda obj, value: value
        if fget == None:
            fget = lambda obj: getattr(obj, self._name)
        if fset == None:
            fset = lambda obj, value: setattr(obj, self._name, self.on_update(obj, self.validate(value)))
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
        if not isinstance(value, int):  # incorrect type, try converting float to int
            if isinstance(value, float) and value.is_integer():
                value = round(value)
            else:
                self._ValueError(f"Value {value} is not an integer, but has type {type(value)}")

        if (self.min != None and value < self.min) or (self.max != None and value > self.max):
            self._ValueError(f"Value {value} not in range [{self.min},{self.max}]")
        return value


class object_property(base_property):
    def __init__(self, fget=None, fset=None, fdel=None, doc=None, default=None, on_update=None, type=None):
        super().__init__(fget=fget, fset=fset, fdel=fdel, doc=doc, default=default, on_update=on_update)
        self.type = type

    def validate(self, value):
        super().validate(value)
        if self.type != None and not isinstance(value, self.type):
            self._ValueError(f"Value {value} is not of type {self.type}")
        return value


class bool_property(int_property):
    def __init__(self, fget=None, fset=None, fdel=None, doc=None, default=None, on_update=None):
        super().__init__(fget=fget, fset=fset, fdel=fdel, doc=doc, default=default, on_update=on_update,
                         allowed_values=[0, 1], min=0, max=1)


class float_property(base_property):
    def validate(self, value):
        super().validate(value)
        if not (isinstance(value, int) or isinstance(value, float)):
            self._ValueError(f"Value {value} is not a number, but has type {type(value)}")
        if (self.min != None and value < self.min) or (self.max != None and value > self.max):
            self._ValueError(f"Value {value} not in range [{self.min},{self.max}]")
        return value


class string_property(base_property):
    def validate(self, value):
        super().validate(value)
        if not isinstance(value, str):
            self._ValueError(f"Value {value} is not a string, but has type {type(value)}")
        return value