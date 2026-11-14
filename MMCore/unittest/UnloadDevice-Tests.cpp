#include <catch2/catch_all.hpp>

#include "MMCore.h"
#include "MockDeviceUtils.h"
#include "StubDevices.h"

TEST_CASE("Unload the current camera", "[regression]") {
   StubCamera cam;
   MockAdapterWithDevices adapter{{"cam", &cam}};
   CMMCore c;
   adapter.LoadIntoCore(c);

   c.setCameraDevice("cam");
   c.unloadDevice("cam");
}

TEST_CASE("Unload all devices with current camera set", "[regression]") {
   StubCamera cam;
   MockAdapterWithDevices adapter{{"cam", &cam}};
   CMMCore c;
   adapter.LoadIntoCore(c);

   c.setCameraDevice("cam");
   c.unloadAllDevices();
}

TEST_CASE("Unload the current shutter", "[regression]") {
   StubShutter shutter;
   MockAdapterWithDevices adapter{{"shutter", &shutter}};
   CMMCore c;
   adapter.LoadIntoCore(c);

   c.setShutterDevice("shutter");
   c.unloadDevice("shutter");
}

TEST_CASE("Unload all devices with current shutter set", "[regression]") {
   StubShutter shutter;
   MockAdapterWithDevices adapter{{"shutter", &shutter}};
   CMMCore c;
   adapter.LoadIntoCore(c);

   c.setShutterDevice("shutter");
   c.unloadAllDevices();
}

TEST_CASE("Unload the focus stage", "[regression]") {
   StubStage stage;
   MockAdapterWithDevices adapter{{"stage", &stage}};
   CMMCore c;
   adapter.LoadIntoCore(c);

   c.setFocusDevice("stage");
   c.unloadDevice("stage");
}

TEST_CASE("Unload all devices with focus stage set", "[regression]") {
   StubStage stage;
   MockAdapterWithDevices adapter{{"stage", &stage}};
   CMMCore c;
   adapter.LoadIntoCore(c);

   c.setFocusDevice("stage");
   c.unloadAllDevices();
}

TEST_CASE("Unload the XY stage", "[regression]") {
   StubXYStage xy;
   MockAdapterWithDevices adapter{{"xy", &xy}};
   CMMCore c;
   adapter.LoadIntoCore(c);

   c.setXYStageDevice("xy");
   c.unloadDevice("xy");
}

TEST_CASE("Unload all devices with XY stage set", "[regression]") {
   StubXYStage xy;
   MockAdapterWithDevices adapter{{"xy", &xy}};
   CMMCore c;
   adapter.LoadIntoCore(c);

   c.setXYStageDevice("xy");
   c.unloadAllDevices();
}

TEST_CASE("Unload a state device", "[regression]") {
   StubStateDevice state;
   MockAdapterWithDevices adapter{{"state", &state}};
   CMMCore c;
   adapter.LoadIntoCore(c);

   c.unloadDevice("state");
}

TEST_CASE("Unload all devices with a state device loaded", "[regression]") {
   StubStateDevice state;
   MockAdapterWithDevices adapter{{"state", &state}};
   CMMCore c;
   adapter.LoadIntoCore(c);

   c.unloadAllDevices();
}

TEST_CASE("Unload a magnifier", "[regression]") {
   StubMagnifier mag;
   MockAdapterWithDevices adapter{{"mag", &mag}};
   CMMCore c;
   adapter.LoadIntoCore(c);

   c.unloadDevice("mag");
}

TEST_CASE("Unload all devices with a magnifier loaded", "[regression]") {
   StubMagnifier mag;
   MockAdapterWithDevices adapter{{"mag", &mag}};
   CMMCore c;
   adapter.LoadIntoCore(c);

   c.unloadAllDevices();
}

namespace {

struct ShutdownTrackingDevice : StubGeneric {
   int shutdownResult = DEVICE_OK;
   bool shutdownCalled = false;
   int Shutdown() override {
      shutdownCalled = true;
      return shutdownResult;
   }
};

} // namespace

TEST_CASE("Unload all devices shuts down every device even if one fails",
      "[regression]") {
   ShutdownTrackingDevice first;
   ShutdownTrackingDevice failing;
   ShutdownTrackingDevice last;
   failing.shutdownResult = DEVICE_ERR;
   MockAdapterWithDevices adapter{
      {"first", &first}, {"failing", &failing}, {"last", &last}};
   CMMCore c;
   adapter.LoadIntoCore(c);

   using Catch::Matchers::ContainsSubstring;
   CHECK_THROWS_WITH(c.unloadAllDevices(), ContainsSubstring("\"failing\""));
   CHECK(first.shutdownCalled);
   CHECK(failing.shutdownCalled);
   CHECK(last.shutdownCalled);
   CHECK(c.getLoadedDevices() == std::vector<std::string>{"Core"});
}

TEST_CASE("Unload all devices reports every device that failed to shut down",
      "[regression]") {
   ShutdownTrackingDevice ok;
   ShutdownTrackingDevice failing1;
   ShutdownTrackingDevice failing2;
   failing1.shutdownResult = DEVICE_ERR;
   failing2.shutdownResult = DEVICE_NOT_CONNECTED;
   MockAdapterWithDevices adapter{
      {"failing1", &failing1}, {"ok", &ok}, {"failing2", &failing2}};
   CMMCore c;
   adapter.LoadIntoCore(c);

   using Catch::Matchers::ContainsSubstring;
   CHECK_THROWS_WITH(c.unloadAllDevices(),
         ContainsSubstring("\"failing1\"") &&
         ContainsSubstring("\"failing2\"") &&
         !ContainsSubstring("\"ok\""));
   CHECK(ok.shutdownCalled);
   CHECK(failing1.shutdownCalled);
   CHECK(failing2.shutdownCalled);
   CHECK(c.getLoadedDevices() == std::vector<std::string>{"Core"});
}
