# Copyright (c) 2026, Dr. Volodymyr Cherkas
# SPDX-License-Identifier: BSD-3-Clause

import ctypes as c
import os
import json
from pathlib import Path
import xml.etree.ElementTree as ET

dirs = [os.add_dll_directory(p) for p in [r'C:\Program Files\EVT\eSDK\bin', r'C:\Program Files\EVT\eSDK\genicam\bin\Win64_x64', r'C:\Program Files\EVT\eSDK\ffmpeg\bin']]
sdk = c.CDLL(r'C:\Program Files\EVT\eSDK\bin\EmergentCameraC.dll')
class DeviceSet(c.Structure):
    _fields_ = [('countDevice', c.c_int), ('privateData', c.c_uint64)]
class Attribute(c.Structure):
    _fields_ = [('dataType',c.c_int), ('accessMode',c.c_int)]
devices = DeviceSet()
camera = c.c_uint64()
assert sdk.EVT_ListDevices(c.byref(devices), c.c_bool(True)) == 0
assert devices.countDevice == 1, devices.countDevice
assert sdk.EVT_CameraOpen(c.byref(camera), c.byref(devices), 0) == 0
try:
    xml = c.c_char_p()
    assert sdk.EVT_GetGenICamXml(camera, c.byref(xml)) == 0
    Path(__file__).with_name('camera-features.xml').write_bytes(xml.value)
    root = ET.fromstring(xml.value)
    ns = {'g': root.tag.split('}')[0][1:]}
    result = []
    for element in root.iter():
        name = element.get('Name', '')
        if any(word.lower() in name.lower() for word in ['gpi', 'gpo', 'line', 'link', 'speed', 'phy', 'adc', 'pixelFormat']):
            if element.tag.rsplit('}', 1)[-1] in ['Category', 'EnumEntry', 'IntReg', 'MaskedIntReg', 'StructReg', 'StringReg']:
                continue
            attribute = Attribute()
            error = sdk.EVT_CameraGetParamAttr(camera, name.encode(), c.byref(attribute))
            if error or attribute.accessMode in (0,1): continue
            item = {'name':name, 'type':attribute.dataType,'access':attribute.accessMode}
            if attribute.dataType == 6:
                value=c.create_string_buffer(4096); size=c.c_ulong()
                item['error']=sdk.EVT_CameraGetEnumParamRange(camera,name.encode(),value,len(value),c.byref(size))
                item['choices']=value.value.decode()
                sdk.EVT_CameraGetStringEnumParam(camera,name.encode(),value,len(value),c.byref(size))
                item['value']=value.value.decode()
            elif attribute.dataType == 0:
                value=c.c_uint32()
                item['error']=sdk.EVT_CameraGetUInt32Param(camera,name.encode(),c.byref(value));item['value']=value.value
            result.append(item)
    text=json.dumps(result,indent=2)
    print(text,flush=True)
    Path(__file__).with_name('camera-capabilities.json').write_text(text)
finally:
    sdk.EVT_CameraClose(camera)
    sdk.EVT_DeviceInfoSetCleanup(c.byref(devices))
