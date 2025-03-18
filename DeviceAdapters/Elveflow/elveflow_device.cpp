#include "ob1_mk4.h"
#include "mux_wire_v3.h"
#include "mux_distrib.h"

MODULE_API void InitializeModuleData() {
   RegisterDevice("OB1_MK4", MM::GenericDevice, "OB1 MK4 Device");
   RegisterDevice("MUX_WIRE", MM::GenericDevice, "MUX WIRE Device");
   RegisterDevice("MUX_DISTRIB", MM::GenericDevice, "MUX DISTRIB Device");
}

MODULE_API MM::Device* CreateDevice(const char* deviceName) {
   if (deviceName == nullptr) {
      return nullptr;
   }

   if (strcmp(deviceName, "OB1_MK4") == 0) {
      return new Ob1Mk4();
   }

   if (strcmp(deviceName, "MUX_WIRE") == 0) {
      return new MuxWireV3();
   }

   if (strcmp(deviceName, "MUX_DISTRIB") == 0) {
      return new MuxDistrib();
   }

   return nullptr;
}

MODULE_API void DeleteDevice(MM::Device* pDevice) {
   delete pDevice;
}