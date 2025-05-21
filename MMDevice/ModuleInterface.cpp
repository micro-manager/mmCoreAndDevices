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

#ifndef MMDEVICE_CLIENT_BUILD

#include <algorithm>
#include <string>
#include <vector>

namespace {

class RegisteredDeviceCollection
{
   struct DeviceInfo
   {
      std::string name;
      MM::DeviceType type = MM::DeviceType::UnknownType;
      std::string description;
   };

   std::vector<DeviceInfo> devices_;

public:
   void RegisterDevice(const char* deviceName, MM::DeviceType deviceType, const char* deviceDescription)
   {
      if (!deviceName)
         return;

      if (!deviceDescription)
         // This is a bug; let the programmer know by displaying an ugly string
         deviceDescription = "(Null description)";

      auto it = std::find_if(devices_.begin(), devices_.end(),
         [&](const DeviceInfo& dev) { return dev.name == deviceName; });
      if (it != devices_.end())
      {
         // Device with this name already registered
         // TODO This should be an error
         return;
      }

      devices_.push_back(DeviceInfo{deviceName, deviceType, deviceDescription});

   }

   unsigned GetNumberOfDevices() const
   {
      return static_cast<unsigned>(devices_.size());
   }

   bool GetDeviceName(unsigned deviceIndex, char* name, unsigned bufLen) const
   {
      if (deviceIndex >= devices_.size())
         return false;

      const std::string& deviceName = devices_[deviceIndex].name;

      if (deviceName.size() >= bufLen)
         return false; // buffer too small, can't truncate the name

      std::snprintf(name, bufLen, "%s", deviceName.c_str());
      return true;
   }

   bool GetDeviceType(const char* deviceName, int* type)
   {
      auto it = std::find_if(devices_.begin(), devices_.end(),
         [&](const DeviceInfo& dev) { return dev.name == deviceName; });
      if (it == devices_.end())
      {
         *type = MM::UnknownType;
         return false;
      }

      // Prefer int over enum across DLL boundary so that the module ABI does not
      // change (somewhat pedantic, but let's be safe).
      *type = static_cast<int>(it->type);

      return true;
   }

   bool GetDeviceDescription(const char* deviceName, char* description, unsigned bufLen)
   {
      auto it = std::find_if(devices_.begin(), devices_.end(),
         [&](const DeviceInfo& dev) { return dev.name == deviceName; });
      if (it == devices_.end())
         return false;

      std::snprintf(description, bufLen, "%s", it->description.c_str());
      return true;
   }
};

// Registered devices in this module (device adapter library)
RegisteredDeviceCollection& TheRegisteredDeviceCollection()
{
   static RegisteredDeviceCollection devices;
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
