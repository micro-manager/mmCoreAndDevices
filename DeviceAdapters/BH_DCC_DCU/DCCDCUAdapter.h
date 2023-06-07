// Micro-Manager Device Adapter for Backer & Hickl DCC/DCU
// Author: Mark A. Tsuchida
//
// Copyright 2023 Board of Regents of the University of Wisconsin System
//
// This file is distributed under the BSD license. License text is included
// with the source distribution.
//
// This file is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE.
//
// IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES.

// This file to be included after defining Model to
// DCCOrDCU::DCC or DCCOrDCU::DCU.

#include "ModuleInterface.h"

MODULE_API void InitializeModuleData() {
   RegisterDevice(HubDeviceName<Model>().c_str(), MM::HubDevice,
                  ModelDescription<Model>().c_str());
}

MODULE_API auto CreateDevice(const char* deviceName) -> MM::Device* {
   if (deviceName == nullptr)
      return nullptr;
   const std::string name = deviceName;
   if (name == HubDeviceName<Model>())
      return new DCCDCUHubDevice<Model>(name);
   const short moduleNo = DeviceNameToModuleNo<Model>(name);
   if (moduleNo >= 0)
      return new DCCDCUModuleDevice<Model>(name, moduleNo);
   return nullptr;
}

MODULE_API void DeleteDevice(MM::Device* pDevice) { delete pDevice; }
