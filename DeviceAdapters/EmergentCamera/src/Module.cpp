// Copyright (c) 2026, Dr. Volodymyr Cherkas
// SPDX-License-Identifier: BSD-3-Clause

#include "EmergentCamera.h"
#include "ModuleInterface.h"

#include <cstring>

const char* g_EmergentCameraName = "EmergentCamera";

MODULE_API void InitializeModuleData()
{
   RegisterDevice(g_EmergentCameraName, MM::CameraDevice,
                  "Emergent camera (eSDK), Advanced Optical Devices, Dr. Volodymyr Cherkas");
}

MODULE_API MM::Device* CreateDevice(const char* deviceName)
{
   if (deviceName != nullptr && std::strcmp(deviceName, g_EmergentCameraName) == 0)
      return new EmergentCamera();
   return nullptr;
}

MODULE_API void DeleteDevice(MM::Device* device)
{
   delete device;
}
