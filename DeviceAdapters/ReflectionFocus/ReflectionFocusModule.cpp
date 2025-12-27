///////////////////////////////////////////////////////////////////////////////
// FILE:          ReflectionFocusModule.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Hardware-based reflection spot tracking autofocus using a
//                camera and optional shutter.
//
// AUTHOR:        Nico Stuurman, nico@cmp.ucsf.edu, 11/07/2008
//                Nico Stuurman, nstuurman@altoslabs.com, 4/22/2022
// COPYRIGHT:     University of California, San Francisco, 2008
//                2015 Open Imaging, Inc.
//                Altos Labs, 2022
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
//

#include "ReflectionFocus.h"
#include "ModuleInterface.h"
#include <sstream>

///////////////////////////////////////////////////////////////////////////////
// Device name constants
//
const char* g_DeviceNameReflectionFocus = "ReflectionFocus";
const char* g_DeviceNameReflectionFocusStage = "ReflectionFocus-Stage";

///////////////////////////////////////////////////////////////////////////////
// Exported MMDevice API
///////////////////////////////////////////////////////////////////////////////

MODULE_API void InitializeModuleData()
{
   RegisterDevice(g_DeviceNameReflectionFocus, MM::AutoFocusDevice,
      "Hardware-based reflection spot tracking autofocus using a camera and optional shutter");
   RegisterDevice(g_DeviceNameReflectionFocusStage, MM::StageDevice,
      "Treats ReflectionFocus offset as a Z-stage device");
}

MODULE_API MM::Device* CreateDevice(const char* deviceName)
{
   if (deviceName == 0)
      return 0;

   if (strcmp(deviceName, g_DeviceNameReflectionFocus) == 0)
   {
      return new ReflectionFocus();
   }
   else if (strcmp(deviceName, g_DeviceNameReflectionFocusStage) == 0)
   {
      return new ReflectionFocusStage();
   }

   return 0;
}

MODULE_API void DeleteDevice(MM::Device* pDevice)
{
   delete pDevice;
}
