def parse_options(object, values):
    for o in values # object.options.items():
        setattr(obj, o[0], o[1])

class Device:
    options = {
        "width" : {
            "default" : 10,
            "min" : 0,
            "max" : 100
        }
    }
    def __init__(**kwargs):
        parse_options(self, kwargs)
        
    


d = Device(width = 10)