///////////////////////////////////////////////////////////////////////////////
// FILE:          ModuleInterface.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     MMDevice - Device adapter kit
//-----------------------------------------------------------------------------
// DESCRIPTION:   The implementation for the common plugin functions
// AUTHOR:        Nenad Amodaj, nenad@amodaj.com, 08/08/2005
// NOTE:          Change the implementation of module interface methods in
//                this file with caution, since the Micro-Manager plugin
//                mechanism relies on specific functionality as implemented
//                here.
// COPYRIGHT:     University of California, San Francisco, 2006
// LICENSE:       This file is distributed under the BSD license.
//                License text is included with the source distribution.
//
//                This file is distributed in the hope that it will be useful,
//                but WITHOUT ANY WARRANTY; without even the implied warranty
//                of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
//
//                IN NO EVENT SHALL THE COPYRIGHT OWNER OR
//                CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
//                INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES.

#include "ModuleInterface.h"
#include "RegisteredDeviceCollection.h"

#ifndef MMDEVICE_CLIENT_BUILD

#include <algorithm>
#include <string>
#include <vector>

namespace {

// Registered devices in this module (device adapter library)
MM::internal::RegisteredDeviceCollection& TheRegisteredDeviceCollection()
{
   static MM::internal::RegisteredDeviceCollection devices;
   return devices;
}

} // anonymous namespace


MODULE_API long GetModuleVersion()
{
   return MODULE_INTERFACE_VERSION;   
}

MODULE_API long GetDeviceInterfaceVersion()
{
   return DEVICE_INTERFACE_VERSION;   
}

MODULE_API unsigned GetNumberOfDevices()
{
   return TheRegisteredDeviceCollection().GetNumberOfDevices();
}

MODULE_API bool GetDeviceName(unsigned deviceIndex, char* name, unsigned bufLen)
{
   return TheRegisteredDeviceCollection().GetDeviceName(deviceIndex, name, bufLen);
}

MODULE_API bool GetDeviceType(const char* deviceName, int* type)
{
   return TheRegisteredDeviceCollection().GetDeviceType(deviceName, type);
}

MODULE_API bool GetDeviceDescription(const char* deviceName, char* description, unsigned bufLen)
{
   return TheRegisteredDeviceCollection().GetDeviceDescription(deviceName, description, bufLen);
}

void RegisterDevice(const char* deviceName, MM::DeviceType deviceType, const char* deviceDescription)
{
   TheRegisteredDeviceCollection().RegisterDevice(deviceName, deviceType, deviceDescription);
}

#endif // MMDEVICE_CLIENT_BUILD
