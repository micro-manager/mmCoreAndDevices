// PROJECT:       Micro-Manager
// SUBSYSTEM:     MMCore
//
// COPYRIGHT:     2025 Board of Regents of the University of Wisconsin System
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

#include "../MMDevice/MMDevice.h"

#include <functional>

// Derive from this class to create a mock device adapter implementation.
// (For use by MMCore unit tests.)
struct MockDeviceAdapter {
   using RegisterDeviceFunc = std::function<void(const char*, MM::DeviceType, const char*)>;

   virtual void InitializeModuleData(RegisterDeviceFunc registerDevice) = 0;
   virtual MM::Device* CreateDevice(const char* name) = 0;
   virtual void DeleteDevice(MM::Device* device) = 0;
};
