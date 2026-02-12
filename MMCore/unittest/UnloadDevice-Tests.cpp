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
