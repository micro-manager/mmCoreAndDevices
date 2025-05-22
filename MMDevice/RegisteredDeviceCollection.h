///////////////////////////////////////////////////////////////////////////////
// FILE:          RegisteredDeviceCollection.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     MMDevice - Device adapter kit
//-----------------------------------------------------------------------------
// COPYRIGHT:     University of California, San Francisco, 2006
//                Board of Regents of the University of Wisconsin System, 2025
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

// This header contains internal implementation of device registration.
// Device adapter code must not use these definitions directly.

#pragma once

#include "MMDeviceConstants.h"

#include <algorithm>
#include <string>
#include <vector>

namespace MM {
namespace internal {

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

      devices_.push_back(DeviceInfo{ deviceName, deviceType, deviceDescription });

   }

   unsigned GetNumberOfDevices() const
   {
      return static_cast<unsigned>(devices_.size());
   }

   bool GetDeviceName(unsigned deviceIndex, char* name, unsigned bufSize) const
   {
      if (deviceIndex >= devices_.size())
         return false;

      const std::string& deviceName = devices_[deviceIndex].name;

      if (deviceName.size() >= bufSize)
         return false; // buffer too small, can't truncate the name

      std::snprintf(name, bufSize, "%s", deviceName.c_str());
      return true;
   }

   bool GetDeviceType(const char* deviceName, int* type) const
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

   bool GetDeviceDescription(const char* deviceName, char* description, unsigned bufSize) const
   {
      auto it = std::find_if(devices_.begin(), devices_.end(),
         [&](const DeviceInfo& dev) { return dev.name == deviceName; });
      if (it == devices_.end())
         return false;

      std::snprintf(description, bufSize, "%s", it->description.c_str());
      return true;
   }
};

} // namespace internal
} // namespace MM
