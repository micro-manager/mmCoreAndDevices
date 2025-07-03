// PROJECT:       Micro-Manager
// SUBSYSTEM:     MMCore
//
// COPYRIGHT:     University of California, San Francisco, 2013-2014
//                2025, Board of Regents of the University of Wisconsin System
//
// LICENSE:       This file is distributed under the "Lesser GPL" (LGPL) license.
//                License text is included with the source distribution.
//
//                This file is distributed in the hope that it will be useful,
//                but WITHOUT ANY WARRANTY; without even the implied warranty
//                of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
//
//                IN NO EVENT SHALL THE COPYRIGHT OWNER OR
//                CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
//                INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES.
//
// AUTHOR:        Mark Tsuchida

#pragma once

#include "../../MMDevice/MMDevice.h"


class LoadedDeviceAdapterImpl
{
public:
   virtual ~LoadedDeviceAdapterImpl() = default;

   virtual void Unload() = 0;

   // Wrappers around raw module interface functions
   virtual void InitializeModuleData() = 0;
   virtual long GetModuleVersion() const = 0;
   virtual long GetDeviceInterfaceVersion() const = 0;
   virtual unsigned GetNumberOfDevices() const = 0;
   virtual bool GetDeviceName(unsigned index, char* buf, unsigned bufLen) const = 0;
   virtual bool GetDeviceDescription(const char* deviceName,
      char* buf, unsigned bufLen) const = 0;
   virtual bool GetDeviceType(const char* deviceName, int* type) const = 0;
   virtual MM::Device* CreateDevice(const char* deviceName) = 0;
   virtual void DeleteDevice(MM::Device* device) = 0;
};
