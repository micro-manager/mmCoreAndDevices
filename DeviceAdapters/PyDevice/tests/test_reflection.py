"""Tests if `bootstrap.py` recognizes device types correctly and extracts all properties"""
from fixture import *
from bootstrap import PyDevice


def test_properties():
    device = GenericDevice()
    pydevice = PyDevice(device)
    properties = pydevice.properties

    assert pydevice.device_type == 'Device'

    assert properties[0].python_name == 'read_only_float'
    assert properties[0].mm_name == 'ReadOnlyFloat'
    assert properties[0].data_type == 'float'
    assert properties[0].set is None
    assert properties[0].min is None
    assert properties[0].max is None
    assert properties[0].options is None
    assert properties[0].unit is None

    assert properties[1].python_name == 'read_only_int'
    assert properties[1].mm_name == 'ReadOnlyInt'
    assert properties[1].data_type == 'int'

    assert properties[2].python_name == 'read_only_string'
    assert properties[2].mm_name == 'ReadOnlyString'
    assert properties[2].data_type == 'str'

    assert properties[3].python_name == 'read_only_enum'
    assert properties[3].mm_name == 'ReadOnlyEnum'
    assert properties[3].data_type == 'enum'
    assert properties[3].options == {'A': Options.A, 'B': Options.B, 'C': Options.C}

    assert properties[4].python_name == 'read_only_bool'
    assert properties[4].mm_name == 'ReadOnlyBool'
    assert properties[4].data_type == 'bool'

    assert properties[5].python_name == 'read_write_float'
    assert properties[5].mm_name == 'ReadWriteFloat'
    assert properties[5].data_type == 'float'
    assert properties[5].set is not None
    device.read_write_float = 0.8
    assert properties[5].get() == 0.8
    properties[5].set(1.1)
    assert properties[5].get() == 1.1
    assert device.read_write_float == 1.1

    # if a property has a unit attached, a _X suffix is added to the name
    # using the getter and setter from the PyProperty object, we can set
    # the value as a float, implicitly converting it to the specified unit.
    # we can also still use the properties on the object directly from Python,
    # where the value is set and returned with the specified astropy unit attached.
    #
    p_meters = properties[10]
    assert p_meters.python_name == 'meters'
    assert p_meters.mm_name == 'Meters-m'
    assert p_meters.data_type == 'float'
    assert p_meters.unit == u.m
    p_meters.set(9.0)
    assert p_meters.get() == 9.0
    assert device.meters == 9 * u.m
    device.meters = 100 * u.mm
    assert p_meters.get() == 0.1

    p_millimeters = properties[11]
    assert p_millimeters.python_name == 'millimeters'
    assert p_millimeters.mm_name == 'Millimeters-mm'
    assert p_millimeters.data_type == 'float'
    assert p_millimeters.unit == u.mm
    p_millimeters.set(12)
    assert p_millimeters.get() == 12
    assert device.millimeters == 12 * u.mm

    assert pydevice.methods['add_one'](5) == 6


def test_camera():
    """Checks if a camera object is detected correctly"""
    cam = Camera1()
    pydevice = PyDevice(cam)
    properties = pydevice.properties

    assert pydevice.device_type == 'Camera'
    assert properties[-1].python_name == 'binning'
    assert properties[-1].mm_name == 'Binning'
    assert properties[0].mm_name == 'Exposure-ms'
