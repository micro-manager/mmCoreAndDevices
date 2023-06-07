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
