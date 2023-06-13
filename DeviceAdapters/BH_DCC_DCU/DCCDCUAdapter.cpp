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

#include "DCCDCUDevices.h"

#include "ModuleInterface.h"

MODULE_API void InitializeModuleData() {
   RegisterDevice(HubDeviceName<DCCOrDCU::DCC>().c_str(), MM::HubDevice,
                  ModelDescription<DCCOrDCU::DCC>().c_str());
   RegisterDevice(HubDeviceName<DCCOrDCU::DCU>().c_str(), MM::HubDevice,
                  ModelDescription<DCCOrDCU::DCU>().c_str());
}

MODULE_API auto CreateDevice(const char* deviceName) -> MM::Device* {
   if (deviceName == nullptr)
      return nullptr;

   const std::string name = deviceName;

   if (name == HubDeviceName<DCCOrDCU::DCC>())
      return new DCCDCUHubDevice<DCCOrDCU::DCC>(name);

   if (name == HubDeviceName<DCCOrDCU::DCU>())
      return new DCCDCUHubDevice<DCCOrDCU::DCU>(name);

   const short dccModuleNo = DeviceNameToModuleNo<DCCOrDCU::DCC>(name);
   if (dccModuleNo >= 0)
      return new DCCDCUModuleDevice<DCCOrDCU::DCC>(name, dccModuleNo);

   const short dcuModuleNo = DeviceNameToModuleNo<DCCOrDCU::DCU>(name);
   if (dcuModuleNo >= 0)
      return new DCCDCUModuleDevice<DCCOrDCU::DCU>(name, dcuModuleNo);

   return nullptr;
}

MODULE_API void DeleteDevice(MM::Device* pDevice) { delete pDevice; }
