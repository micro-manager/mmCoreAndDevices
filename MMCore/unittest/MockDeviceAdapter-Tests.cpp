#include <catch2/catch_all.hpp>

#include "MMCore.h"
#include "../../MMDevice/DeviceBase.h"

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
