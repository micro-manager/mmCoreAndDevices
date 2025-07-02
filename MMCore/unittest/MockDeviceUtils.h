#include "MMCore.h"

#include <initializer_list>
#include <utility>
#include <string>

// A mock device adapter that provides the device(s) passed in at construction.
// This is intended for tests where we're not testing device creation and
// destruction, and each device is only loaded once. This class does not own
// the devices; the caller is responsible for that (usually, the devices should
// just be created on the stack).
class MockAdapterWithDevices : public MockDeviceAdapter {
   std::string adapter_name = "mock_adapter";
   std::vector<std::pair<std::string, MM::Device*>> devices;

public:
   explicit MockAdapterWithDevices(
      std::initializer_list<std::pair<std::string, MM::Device*>> il)
      : devices(il) {}

   void InitializeModuleData(RegisterDeviceFunc registerDevice) override {
      for (auto name_device : devices) {
         const auto name = name_device.first;
         const auto device = name_device.second;
         const auto desc = "description for " + name;
         registerDevice(name.c_str(), device->GetType(), desc.c_str());
      }
   }

   MM::Device *CreateDevice(const char *name) override {
      for (auto name_device : devices) {
         if (name_device.first == name) {
            return name_device.second;
         }
      }
      return nullptr;
   }

   void DeleteDevice(MM::Device *device) override { (void)device; }

   // Convenience for loading all the devices
   void LoadIntoCore(CMMCore &core) {
      core.loadMockDeviceAdapter(adapter_name.c_str(), this);
      for (auto name_device : devices) {
         const auto name = name_device.first;
         core.loadDevice(name.c_str(), adapter_name.c_str(), name.c_str());
      }
      for (auto name_device : devices) {
         const auto name = name_device.first;
         core.initializeDevice(name.c_str());
      }
   }
};