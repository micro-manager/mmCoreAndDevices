#include <catch2/catch_all.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "MMCore.h"
#include "MockDeviceUtils.h"

#include "DeviceBase.h"

#include <memory>
#include <string>
#include <vector>

class MyMockDevice : public CGenericBase<MyMockDevice> {
public:
   int Initialize() override { return DEVICE_OK; }
   int Shutdown() override { return DEVICE_OK; }
   bool Busy() override { return false; }
   void GetName(char* name) const override {
      snprintf(name, MM::MaxStrLength, "name-returned-by-device");
   }
};

class MyMockAdapter : public ClientDeviceAdapter {
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

class EventRecordingAdapter : public MyMockAdapter {
   std::vector<std::string>& events_;

public:
   explicit EventRecordingAdapter(std::vector<std::string>& events)
      : events_(events) {}

   void DeleteDevice(MM::Device* device) override {
      events_.push_back("DeleteDevice");
      MyMockAdapter::DeleteDevice(device);
   }
};

std::shared_ptr<ClientDeviceAdapter>
MakeRecordingAdapter(std::vector<std::string>& events) {
   return std::shared_ptr<ClientDeviceAdapter>(
      new EventRecordingAdapter(events),
      [&events](ClientDeviceAdapter* p) {
         events.push_back("released");
         delete p;
      });
}

TEST_CASE("Register and load a client device")
{
   CMMCore c;
   c.loadClientDeviceAdapter("myadapter", std::make_shared<MyMockAdapter>());
   c.loadDevice("mylabel", "myadapter", "mydevice");
   c.initializeDevice("mylabel");
   c.waitForDevice("mylabel");
   CHECK(c.getDeviceName("mylabel") == "mydevice");
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

TEST_CASE("Unloaded client adapter can be loaded again under the same name") {
   CMMCore c;
   c.loadClientDeviceAdapter("myadapter", std::make_shared<MyMockAdapter>());
   c.unloadLibrary("myadapter");
   c.loadClientDeviceAdapter("myadapter", std::make_shared<MyMockAdapter>());
   c.loadDevice("mylabel", "myadapter", "mydevice");
   c.initializeDevice("mylabel");
   CHECK(c.getDeviceName("mylabel") == "mydevice");
   c.unloadDevice("mylabel");
}

TEST_CASE("Null client adapter is rejected") {
   CMMCore c;
   CHECK_THROWS_AS(c.loadClientDeviceAdapter("myadapter", nullptr), CMMError);
}

TEST_CASE("Client adapter name must not already be loaded") {
   CMMCore c;
   c.loadClientDeviceAdapter("myadapter", std::make_shared<MyMockAdapter>());
   CHECK_THROWS_AS(c.loadClientDeviceAdapter("myadapter",
                                             std::make_shared<MyMockAdapter>()),
                   CMMError);
}

TEST_CASE("Client adapter is released by unloadLibrary after its devices") {
   std::vector<std::string> events;
   CMMCore c;
   c.loadClientDeviceAdapter("myadapter", MakeRecordingAdapter(events));
   c.loadDevice("mylabel", "myadapter", "mydevice");
   c.initializeDevice("mylabel");
   CHECK(events.empty());
   c.unloadLibrary("myadapter");
   CHECK(events == std::vector<std::string>{"DeleteDevice", "released"});
}

TEST_CASE("Client adapter is released by Core destruction after its devices") {
   std::vector<std::string> events;
   {
      CMMCore c;
      c.loadClientDeviceAdapter("myadapter", MakeRecordingAdapter(events));
      c.loadDevice("mylabel", "myadapter", "mydevice");
      c.initializeDevice("mylabel");
      CHECK(events.empty());
   }
   CHECK(events == std::vector<std::string>{"DeleteDevice", "released"});
}

TEST_CASE("Client adapter is not released while caller holds a reference") {
   std::vector<std::string> events;
   auto adapter = MakeRecordingAdapter(events);
   {
      CMMCore c;
      c.loadClientDeviceAdapter("myadapter", adapter);
   }
   CHECK(events.empty());
   adapter.reset();
   CHECK(events == std::vector<std::string>{"released"});
}
