#include <catch2/catch_all.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "MMCore.h"
#include "../../MMDevice/DeviceBase.h"
#include "MockDeviceUtils.h"

class MyMockDevice : public CGenericBase<MyMockDevice> {
public:
   int Initialize() override { return DEVICE_OK; }
   int Shutdown() override { return DEVICE_OK; }
   bool Busy() override { return false; }
   void GetName(char* name) const override {
      snprintf(name, MM::MaxStrLength, "name-returned-by-device");
   }
};

class MyMockAdapter : public MockDeviceAdapter {
public:
   void InitializeModuleData(RegisterDeviceFunc registerDevice) override {
      registerDevice("mydevice", MM::GenericDevice, "my device description");
   }

   MM::Device* CreateDevice(const char* name) override {
      CHECK(std::string(name) == "mydevice");
      return new MyMockDevice();
   }

   void DeleteDevice(MM::Device* device) override {
      delete device;
   }
};

TEST_CASE("Register and load a mock device")
{
   MyMockAdapter adapter;

   CMMCore c;
   c.loadMockDeviceAdapter("myadapter", &adapter);
   c.loadDevice("mylabel", "myadapter", "mydevice");
   c.initializeDevice("mylabel");
   c.waitForDevice("mylabel");
   CHECK(c.getDeviceName("mylabel") == "name-returned-by-device");
   c.unloadDevice("mylabel");
}

TEST_CASE("Create mock devices using MockAdapterWithDevices convenience class") {
   MyMockDevice dev1;
   MyMockDevice dev2;
   MockAdapterWithDevices adapter{
      {"dev1", &dev1},
      {"dev2", &dev2},
   };

   CMMCore c;
   adapter.LoadIntoCore(c);

   using Catch::Matchers::ContainsSubstring;
   CHECK_THAT(c.getDeviceDescription("dev1"), ContainsSubstring("dev1"));
   CHECK_THAT(c.getDeviceDescription("dev2"), ContainsSubstring("dev2"));
}