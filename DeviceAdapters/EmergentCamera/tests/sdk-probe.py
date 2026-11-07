# Copyright (c) 2026, Dr. Volodymyr Cherkas
# SPDX-License-Identifier: BSD-3-Clause

import ctypes as c
import os
import time

dirs = [os.add_dll_directory(p) for p in [r'C:\Program Files\EVT\eSDK\bin', r'C:\Program Files\EVT\eSDK\genicam\bin\Win64_x64', r'C:\Program Files\EVT\eSDK\ffmpeg\bin']]
sdk = c.CDLL(r'C:\Program Files\EVT\eSDK\bin\EmergentCameraC.dll')
class DeviceSet(c.Structure):
    _fields_ = [('countDevice', c.c_int), ('privateData', c.c_uint64)]
devices = DeviceSet()
camera = c.c_uint64()
assert sdk.EVT_ListDevices(c.byref(devices), c.c_bool(True)) == 0
assert devices.countDevice == 1, devices.countDevice
assert sdk.EVT_CameraOpen(c.byref(camera), c.byref(devices), 0) == 0
try:
    for name in ['Exposure', 'AutoSizeX', 'AutoSizeY', 'AutoOffsetX', 'AutoOffsetY', 'FrameRate']:
        values = {}
        for suffix in ['', 'Min', 'Max', 'Inc']:
            value = c.c_uint32()
            error = getattr(sdk, 'EVT_CameraGetUInt32Param'+suffix)(camera, name.encode(), c.byref(value))
            values[suffix or 'Value'] = (error, value.value)
        print(name, values, flush=True)
    for name in ['TriggerSelector', 'TriggerSource', 'TriggerMode', 'AcquisitionMode', 'Bin', 'PtpMode', 'PtpStatus']:
        value = c.create_string_buffer(4096)
        size = c.c_ulong()
        err = sdk.EVT_CameraGetEnumParamRange(camera, name.encode(), value, len(value), c.byref(size))
        print(name, 'range', err, value.value.decode(), flush=True)
        err = sdk.EVT_CameraGetStringEnumParam(camera, name.encode(), value, len(value), c.byref(size))
        print(name, 'current', err, value.value.decode(), flush=True)
    def enum(name):
        value = c.create_string_buffer(4096)
        size = c.c_ulong()
        err = sdk.EVT_CameraGetStringEnumParam(camera, name.encode(), value, len(value), c.byref(size))
        assert err == 0, (name, err)
        return value.value.decode()
    def uint(name):
        value = c.c_uint32()
        err = sdk.EVT_CameraGetUInt32Param(camera, name.encode(), c.byref(value))
        assert err == 0, (name, err)
        return value.value
    def timestamp():
        assert sdk.EVT_CameraExecuteCommand(camera, b'GevTimestampControlLatch') == 0
        return uint('GevTimestampValueHigh') << 32 | uint('GevTimestampValueLow')
    count = uint('AcquisitionFrameCount')
    try:
        for requested in [7, 255, 256, 257, 1000]:
            error = sdk.EVT_CameraSetUInt32Param(camera, b'AcquisitionFrameCount', c.c_uint32(requested))
            print('FrameCount requested', requested, 'error', error, 'actual', uint('AcquisitionFrameCount'), flush=True)
    finally:
        sdk.EVT_CameraSetUInt32Param(camera, b'AcquisitionFrameCount', c.c_uint32(count))
    mode = enum('PtpMode')
    try:
        for test_mode in ['OneStep', 'TwoStep']:
            err = sdk.EVT_CameraSetEnumParam(camera, b'PtpMode', test_mode.encode())
            print('PTP set', test_mode, 'error', err, flush=True)
            if err: continue
            for i in range(6):
                offset = c.c_int32()
                error = sdk.EVT_CameraGetInt32Param(camera, b'PtpOffset', c.byref(offset))
                print('PTP', test_mode, i, 'status', enum('PtpStatus'), 'offset_ns', (error, offset.value), 'timestamp_ns', timestamp(), flush=True)
                time.sleep(2)
    finally:
        assert sdk.EVT_CameraSetEnumParam(camera, b'PtpMode', mode.encode()) == 0
        print('PTP restored', mode, flush=True)
finally:
    sdk.EVT_CameraClose(camera)
