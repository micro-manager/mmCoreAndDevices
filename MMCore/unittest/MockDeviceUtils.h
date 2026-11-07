#include "MMCore.h"

#include <initializer_list>
#include <memory>
#include <utility>
#include <string>

// A mock device adapter that provides the device(s) passed in at construction.
// This is intended for tests where we're not testing device creation and
// destruction, and each device is only loaded once. This class does not own
// the devices; the caller is responsible for that (usually, the devices should
// just be created on the stack). The adapter itself must outlive the Core.
class MockAdapterWithDevices : public ClientDeviceAdapter {
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

   // Load the adapter only (non-owning)
   void LoadAdapterIntoCore(CMMCore &core, const char *name) {
      core.loadClientDeviceAdapter(name,
         std::shared_ptr<ClientDeviceAdapter>(this, [](ClientDeviceAdapter*) {}));
   }

   // Convenience for loading all the devices
   void LoadIntoCore(CMMCore &core) {
      LoadAdapterIntoCore(core, adapter_name.c_str());
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