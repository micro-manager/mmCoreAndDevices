#include <catch2/catch_all.hpp>

#include "MMCore.h"
#include "MockDeviceUtils.h"
#include "StubDevices.h"

#include <string>
#include <vector>

using Catch::Matchers::UnorderedEquals;
using Catch::Matchers::VectorContains;

// --- Core properties ---

TEST_CASE("Core has expected properties") {
   CMMCore c;

   SECTION("getDevicePropertyNames returns all 12 properties") {
      auto names = c.getDevicePropertyNames("Core");
      CHECK(names.size() == 12);
      CHECK_THAT(names, UnorderedEquals(std::vector<std::string>{
         "Initialize", "Camera", "Shutter", "Focus", "XYStage",
         "AutoFocus", "AutoShutter", "ChannelGroup",
         "ImageProcessor", "SLM", "Galvo", "TimeoutMs",
      }));
   }

   SECTION("hasProperty returns true for known properties") {
      CHECK(c.hasProperty("Core", "Camera"));
      CHECK(c.hasProperty("Core", "AutoShutter"));
      CHECK(c.hasProperty("Core", "TimeoutMs"));
   }

   SECTION("hasProperty returns false for unknown properties") {
      CHECK_FALSE(c.hasProperty("Core", "Nonexistent"));
   }
}

// --- AutoShutter property ---

TEST_CASE("Core AutoShutter property") {
   CMMCore c;

   SECTION("default value is 1") {
      CHECK(c.getProperty("Core", "AutoShutter") == "1");
      CHECK(c.getAutoShutter() == true);
   }

   SECTION("set via property") {
      c.setProperty("Core", "AutoShutter", "0");
      CHECK(c.getProperty("Core", "AutoShutter") == "0");
      CHECK(c.getAutoShutter() == false);

      c.setProperty("Core", "AutoShutter", "1");
      CHECK(c.getProperty("Core", "AutoShutter") == "1");
      CHECK(c.getAutoShutter() == true);
   }

   SECTION("set via dedicated function updates property") {
      c.setAutoShutter(false);
      CHECK(c.getProperty("Core", "AutoShutter") == "0");
      c.setAutoShutter(true);
      CHECK(c.getProperty("Core", "AutoShutter") == "1");
   }

   SECTION("invalid value is rejected") {
      CHECK_THROWS(c.setProperty("Core", "AutoShutter", "2"));
      CHECK_THROWS(c.setProperty("Core", "AutoShutter", "abc"));
   }

   SECTION("state cache is updated") {
      c.setAutoShutter(false);
      auto cache = c.getSystemStateCache();
      CHECK(cache.isPropertyIncluded("Core", "AutoShutter"));
      auto setting = cache.getSetting("Core", "AutoShutter");
      CHECK(setting.getPropertyValue() == "0");
   }
}

// --- TimeoutMs property ---

TEST_CASE("Core TimeoutMs property") {
   CMMCore c;

   SECTION("default value is 5000") {
      CHECK(c.getProperty("Core", "TimeoutMs") == "5000");
      CHECK(c.getTimeoutMs() == 5000);
   }

   SECTION("set via property updates getTimeoutMs") {
      c.setProperty("Core", "TimeoutMs", "10000");
      CHECK(c.getTimeoutMs() == 10000);
      CHECK(c.getProperty("Core", "TimeoutMs") == "10000");
   }

   SECTION("set via dedicated function") {
      c.setTimeoutMs(3000);
      CHECK(c.getTimeoutMs() == 3000);
   }

   SECTION("setTimeoutMs ignores zero") {
      c.setTimeoutMs(0);
      CHECK(c.getTimeoutMs() == 5000);
      CHECK(c.getProperty("Core", "TimeoutMs") == "5000");
   }

   SECTION("setTimeoutMs ignores negative") {
      c.setTimeoutMs(-1);
      CHECK(c.getTimeoutMs() == 5000);
      CHECK(c.getProperty("Core", "TimeoutMs") == "5000");
   }
}

// Bug: setTimeoutMs() is a simple inline setter that does not sync the
// "TimeoutMs" Core property. The property retains the old value until
// the next Refresh.
TEST_CASE("setTimeoutMs updates Core property", "[!shouldfail]") {
   CMMCore c;
   c.setTimeoutMs(3000);
   CHECK(c.getProperty("Core", "TimeoutMs") == "3000");
}

// Bug: setProperty("Core", "TimeoutMs", ...) does not reject non-positive
// values. The property string is stored before Execute() calls setTimeoutMs(),
// which silently ignores the value, leaving the property out of sync.
TEST_CASE("setProperty Core-TimeoutMs with zero is ignored", "[!shouldfail]") {
   CMMCore c;
   c.setProperty("Core", "TimeoutMs", "0");
   CHECK(c.getTimeoutMs() == 5000);
   CHECK(c.getProperty("Core", "TimeoutMs") == "5000");
}

TEST_CASE("setProperty Core-TimeoutMs with negative is ignored",
      "[!shouldfail]") {
   CMMCore c;
   c.setProperty("Core", "TimeoutMs", "-1");
   CHECK(c.getTimeoutMs() == 5000);
   CHECK(c.getProperty("Core", "TimeoutMs") == "5000");
}

// --- Device role properties ---

TEST_CASE("Core Camera property") {
   StubCamera cam;
   cam.name = "MyCam";
   StubShutter shutter;
   MockAdapterWithDevices adapter{{"cam", &cam}, {"sh", &shutter}};
   CMMCore c;
   adapter.LoadIntoCore(c);

   SECTION("initially empty") {
      CHECK(c.getCameraDevice().empty());
      CHECK(c.getProperty("Core", "Camera") == "");
   }

   SECTION("set via property updates getter") {
      c.setProperty("Core", "Camera", "cam");
      CHECK(c.getCameraDevice() == "cam");
   }

   SECTION("set via dedicated function updates property") {
      c.setCameraDevice("cam");
      CHECK(c.getProperty("Core", "Camera") == "cam");
   }

   SECTION("setting to empty unsets") {
      c.setCameraDevice("cam");
      c.setCameraDevice("");
      CHECK(c.getCameraDevice().empty());
   }

   SECTION("setting to nonexistent label throws") {
      CHECK_THROWS(c.setCameraDevice("nosuchdevice"));
   }

   SECTION("setting to wrong device type throws") {
      CHECK_THROWS(c.setCameraDevice("sh"));
   }

   SECTION("allowed values include loaded devices") {
      auto allowed = c.getAllowedPropertyValues("Core", "Camera");
      CHECK_THAT(allowed, VectorContains(std::string("cam")));
      CHECK_THAT(allowed, VectorContains(std::string("")));
   }

   SECTION("state cache updated on change") {
      c.setCameraDevice("cam");
      auto cache = c.getSystemStateCache();
      CHECK(cache.getSetting("Core", "Camera").getPropertyValue() == "cam");
   }
}

TEST_CASE("Core Shutter property") {
   StubShutter sh;
   StubCamera cam;
   MockAdapterWithDevices adapter{{"sh", &sh}, {"cam", &cam}};
   CMMCore c;
   adapter.LoadIntoCore(c);

   SECTION("initially empty") {
      CHECK(c.getShutterDevice().empty());
   }

   SECTION("set via property updates getter") {
      c.setProperty("Core", "Shutter", "sh");
      CHECK(c.getShutterDevice() == "sh");
   }

   SECTION("set via dedicated function updates property") {
      c.setShutterDevice("sh");
      CHECK(c.getProperty("Core", "Shutter") == "sh");
   }

   SECTION("setting to empty unsets") {
      c.setShutterDevice("sh");
      c.setShutterDevice("");
      CHECK(c.getShutterDevice().empty());
   }

   SECTION("setting to nonexistent label throws") {
      CHECK_THROWS(c.setShutterDevice("nosuchdevice"));
   }

   SECTION("setting to wrong device type throws") {
      CHECK_THROWS(c.setShutterDevice("cam"));
   }
}

TEST_CASE("Core Focus property") {
   StubStage stage;
   StubCamera cam;
   MockAdapterWithDevices adapter{{"stage", &stage}, {"cam", &cam}};
   CMMCore c;
   adapter.LoadIntoCore(c);

   SECTION("initially empty") {
      CHECK(c.getFocusDevice().empty());
   }

   SECTION("set via property updates getter") {
      c.setProperty("Core", "Focus", "stage");
      CHECK(c.getFocusDevice() == "stage");
   }

   SECTION("set via dedicated function updates property") {
      c.setFocusDevice("stage");
      CHECK(c.getProperty("Core", "Focus") == "stage");
   }

   SECTION("setting to empty unsets") {
      c.setFocusDevice("stage");
      c.setFocusDevice("");
      CHECK(c.getFocusDevice().empty());
   }

   SECTION("setting to wrong device type throws") {
      CHECK_THROWS(c.setFocusDevice("cam"));
   }
}

TEST_CASE("Core XYStage property") {
   StubXYStage xy;
   StubCamera cam;
   MockAdapterWithDevices adapter{{"xy", &xy}, {"cam", &cam}};
   CMMCore c;
   adapter.LoadIntoCore(c);

   SECTION("initially empty") {
      CHECK(c.getXYStageDevice().empty());
   }

   SECTION("set via property updates getter") {
      c.setProperty("Core", "XYStage", "xy");
      CHECK(c.getXYStageDevice() == "xy");
   }

   SECTION("set via dedicated function updates property") {
      c.setXYStageDevice("xy");
      CHECK(c.getProperty("Core", "XYStage") == "xy");
   }

   SECTION("setting to empty unsets") {
      c.setXYStageDevice("xy");
      c.setXYStageDevice("");
      CHECK(c.getXYStageDevice().empty());
   }

   SECTION("setting to wrong device type throws") {
      CHECK_THROWS(c.setXYStageDevice("cam"));
   }
}

TEST_CASE("Core AutoFocus property") {
   StubAutoFocus af;
   StubCamera cam;
   MockAdapterWithDevices adapter{{"af", &af}, {"cam", &cam}};
   CMMCore c;
   adapter.LoadIntoCore(c);

   SECTION("initially empty") {
      CHECK(c.getAutoFocusDevice().empty());
   }

   SECTION("set via property updates getter") {
      c.setProperty("Core", "AutoFocus", "af");
      CHECK(c.getAutoFocusDevice() == "af");
   }

   SECTION("set via dedicated function updates property") {
      c.setAutoFocusDevice("af");
      CHECK(c.getProperty("Core", "AutoFocus") == "af");
   }

   SECTION("setting to empty unsets") {
      c.setAutoFocusDevice("af");
      c.setAutoFocusDevice("");
      CHECK(c.getAutoFocusDevice().empty());
   }

   SECTION("setting to wrong device type throws") {
      CHECK_THROWS(c.setAutoFocusDevice("cam"));
   }
}

TEST_CASE("Core ImageProcessor property") {
   StubImageProcessor ip;
   StubCamera cam;
   MockAdapterWithDevices adapter{{"ip", &ip}, {"cam", &cam}};
   CMMCore c;
   adapter.LoadIntoCore(c);

   SECTION("initially empty") {
      CHECK(c.getImageProcessorDevice().empty());
   }

   SECTION("set via property updates getter") {
      c.setProperty("Core", "ImageProcessor", "ip");
      CHECK(c.getImageProcessorDevice() == "ip");
   }

   SECTION("set via dedicated function updates property") {
      c.setImageProcessorDevice("ip");
      CHECK(c.getProperty("Core", "ImageProcessor") == "ip");
   }

   SECTION("setting to empty unsets") {
      c.setImageProcessorDevice("ip");
      c.setImageProcessorDevice("");
      CHECK(c.getImageProcessorDevice().empty());
   }

   SECTION("setting to wrong device type throws") {
      CHECK_THROWS(c.setImageProcessorDevice("cam"));
   }
}

TEST_CASE("Core SLM property") {
   StubSLM slm;
   StubCamera cam;
   MockAdapterWithDevices adapter{{"slm", &slm}, {"cam", &cam}};
   CMMCore c;
   adapter.LoadIntoCore(c);

   SECTION("initially empty") {
      CHECK(c.getSLMDevice().empty());
   }

   SECTION("set via property updates getter") {
      c.setProperty("Core", "SLM", "slm");
      CHECK(c.getSLMDevice() == "slm");
   }

   SECTION("set via dedicated function updates property") {
      c.setSLMDevice("slm");
      CHECK(c.getProperty("Core", "SLM") == "slm");
   }

   SECTION("setting to empty unsets") {
      c.setSLMDevice("slm");
      c.setSLMDevice("");
      CHECK(c.getSLMDevice().empty());
   }

   SECTION("setting to wrong device type throws") {
      CHECK_THROWS(c.setSLMDevice("cam"));
   }
}

TEST_CASE("Core Galvo property") {
   StubGalvo galvo;
   StubCamera cam;
   MockAdapterWithDevices adapter{{"galvo", &galvo}, {"cam", &cam}};
   CMMCore c;
   adapter.LoadIntoCore(c);

   SECTION("initially empty") {
      CHECK(c.getGalvoDevice().empty());
   }

   SECTION("set via property updates getter") {
      c.setProperty("Core", "Galvo", "galvo");
      CHECK(c.getGalvoDevice() == "galvo");
   }

   SECTION("set via dedicated function updates property") {
      c.setGalvoDevice("galvo");
      CHECK(c.getProperty("Core", "Galvo") == "galvo");
   }

   SECTION("setting to empty unsets") {
      c.setGalvoDevice("galvo");
      c.setGalvoDevice("");
      CHECK(c.getGalvoDevice().empty());
   }

   SECTION("setting to wrong device type throws") {
      CHECK_THROWS(c.setGalvoDevice("cam"));
   }
}

// --- Shutter-specific behavior ---

TEST_CASE("Shutter open state transfers when switching shutters") {
   StubShutter sh1;
   sh1.name = "Shutter1";
   StubShutter sh2;
   sh2.name = "Shutter2";
   MockAdapterWithDevices adapter{{"sh1", &sh1}, {"sh2", &sh2}};
   CMMCore c;
   adapter.LoadIntoCore(c);

   SECTION("open state transfers to new shutter") {
      c.setShutterDevice("sh1");
      c.setShutterOpen(true);
      CHECK(sh1.open == true);

      c.setShutterDevice("sh2");
      CHECK(sh1.open == false);
      CHECK(sh2.open == true);
   }

   SECTION("closed state stays closed") {
      c.setShutterDevice("sh1");
      CHECK(sh1.open == false);

      c.setShutterDevice("sh2");
      CHECK(sh1.open == false);
      CHECK(sh2.open == false);
   }

   SECTION("setting same shutter is a no-op") {
      c.setShutterDevice("sh1");
      c.setShutterOpen(true);
      c.setShutterDevice("sh1");
      CHECK(sh1.open == true);
   }
}

// --- Camera-specific behavior ---

TEST_CASE("Cannot switch camera while sequence is running") {
   StubCamera cam1;
   cam1.name = "Cam1";
   StubCamera cam2;
   cam2.name = "Cam2";
   MockAdapterWithDevices adapter{{"cam1", &cam1}, {"cam2", &cam2}};
   CMMCore c;
   adapter.LoadIntoCore(c);

   c.setCameraDevice("cam1");
   cam1.capturing = true;
   CHECK_THROWS(c.setCameraDevice("cam2"));
   cam1.capturing = false;
}

// --- ChannelGroup property ---

TEST_CASE("Core ChannelGroup property") {
   CMMCore c;

   SECTION("initially empty") {
      CHECK(c.getChannelGroup().empty());
      CHECK(c.getProperty("Core", "ChannelGroup") == "");
   }

   SECTION("set via property calls setChannelGroup") {
      c.defineConfigGroup("MyChannels");
      c.setProperty("Core", "ChannelGroup", "MyChannels");
      CHECK(c.getChannelGroup() == "MyChannels");
   }

   SECTION("set via dedicated function updates property") {
      c.defineConfigGroup("MyChannels");
      c.setChannelGroup("MyChannels");
      CHECK(c.getProperty("Core", "ChannelGroup") == "MyChannels");
   }

   SECTION("allowed values updated when groups are defined") {
      c.defineConfigGroup("Group1");
      auto allowed = c.getAllowedPropertyValues("Core", "ChannelGroup");
      CHECK_THAT(allowed, VectorContains(std::string("Group1")));
      CHECK_THAT(allowed, VectorContains(std::string("")));
   }

   SECTION("setting to undefined group is rejected") {
      CHECK_THROWS(c.setProperty("Core", "ChannelGroup", "NoSuchGroup"));
   }

   SECTION("renaming active channel group resets to empty") {
      // TODO: Is this the desired behavior? We could keep the channel group.
      c.defineConfigGroup("OldName");
      c.setChannelGroup("OldName");
      c.renameConfigGroup("OldName", "NewName");
      CHECK(c.getChannelGroup().empty());
   }

   SECTION("deleting active channel group resets to empty") {
      c.defineConfigGroup("ToDelete");
      c.setChannelGroup("ToDelete");
      c.deleteConfigGroup("ToDelete");
      CHECK(c.getChannelGroup().empty());
   }
}

// --- Initialize property ---

TEST_CASE("Core Initialize property") {
   SECTION("default value and allowed values") {
      CMMCore c;
      CHECK(c.getProperty("Core", "Initialize") == "0");
      auto allowed = c.getAllowedPropertyValues("Core", "Initialize");
      CHECK_THAT(allowed, UnorderedEquals(std::vector<std::string>{"0", "1"}));
   }

   SECTION("setting to 1 initializes all loaded devices") {
      StubCamera cam;
      StubShutter sh;
      MockAdapterWithDevices adapter{{"cam", &cam}, {"sh", &sh}};
      CMMCore c;
      // Load without initializing (manually)
      c.loadMockDeviceAdapter("mock", &adapter);
      c.loadDevice("cam", "mock", "cam");
      c.loadDevice("sh", "mock", "sh");

      c.setProperty("Core", "Initialize", "1");
      // Devices should now be initialized; we can set them
      c.setCameraDevice("cam");
      CHECK(c.getCameraDevice() == "cam");
   }

   SECTION("setting to 0 unloads all devices") {
      StubCamera cam;
      StubShutter sh;
      MockAdapterWithDevices adapter{{"cam", &cam}, {"sh", &sh}};
      CMMCore c;
      adapter.LoadIntoCore(c);
      c.setCameraDevice("cam");

      c.setProperty("Core", "Initialize", "0");
      auto devices = c.getLoadedDevices();
      CHECK(devices.size() == 1);
      CHECK(devices[0] == "Core");
   }

   SECTION("after setting to 0, device role properties return to defaults") {
      StubCamera cam;
      MockAdapterWithDevices adapter{{"cam", &cam}};
      CMMCore c;
      adapter.LoadIntoCore(c);
      c.setCameraDevice("cam");

      c.setProperty("Core", "Initialize", "0");
      CHECK(c.getCameraDevice().empty());
      CHECK(c.getProperty("Core", "Camera") == "");
      CHECK(c.getShutterDevice().empty());
   }
}

// --- Config group interaction with Core properties ---

TEST_CASE("Config groups can contain Core property settings") {
   StubCamera cam1;
   cam1.name = "Cam1";
   StubCamera cam2;
   cam2.name = "Cam2";
   MockAdapterWithDevices adapter{{"cam1", &cam1}, {"cam2", &cam2}};
   CMMCore c;
   adapter.LoadIntoCore(c);

   c.defineConfig("MyGroup", "preset1", "Core", "Camera", "cam1");
   c.defineConfig("MyGroup", "preset2", "Core", "Camera", "cam2");

   SECTION("getCurrentConfig reflects current Core property state") {
      c.setCameraDevice("cam1");
      CHECK(c.getCurrentConfig("MyGroup") == "preset1");
      c.setCameraDevice("cam2");
      CHECK(c.getCurrentConfig("MyGroup") == "preset2");
   }

   SECTION("getCurrentConfigFromCache updates with Core property changes") {
      c.setCameraDevice("cam1");
      CHECK(c.getCurrentConfigFromCache("MyGroup") == "preset1");
      c.setCameraDevice("cam2");
      CHECK(c.getCurrentConfigFromCache("MyGroup") == "preset2");
   }

   SECTION("getConfigGroupState returns current Core property values") {
      c.setCameraDevice("cam1");
      auto state = c.getConfigGroupState("MyGroup");
      CHECK(state.getSetting("Core", "Camera").getPropertyValue() == "cam1");
   }
}

// --- Property metadata ---

TEST_CASE("Core property metadata") {
   CMMCore c;

   SECTION("AutoShutter is Integer type") {
      CHECK(c.getPropertyType("Core", "AutoShutter") == MM::Integer);
   }

   SECTION("Camera is String type") {
      CHECK(c.getPropertyType("Core", "Camera") == MM::String);
   }

   SECTION("Core properties are not read-only") {
      CHECK_FALSE(c.isPropertyReadOnly("Core", "Camera"));
      CHECK_FALSE(c.isPropertyReadOnly("Core", "AutoShutter"));
   }

   SECTION("AutoShutter allowed values are 0 and 1") {
      auto allowed = c.getAllowedPropertyValues("Core", "AutoShutter");
      CHECK_THAT(allowed, UnorderedEquals(std::vector<std::string>{"0", "1"}));
   }
}

// --- System state includes Core properties ---

TEST_CASE("System state includes Core properties") {
   StubCamera cam;
   MockAdapterWithDevices adapter{{"cam", &cam}};
   CMMCore c;
   adapter.LoadIntoCore(c);

   SECTION("getSystemState includes Core properties") {
      c.setCameraDevice("cam");
      auto state = c.getSystemState();
      CHECK(state.isPropertyIncluded("Core", "Camera"));
      CHECK(state.getSetting("Core", "Camera").getPropertyValue() == "cam");
      CHECK(state.isPropertyIncluded("Core", "AutoShutter"));
   }

   SECTION("getSystemStateCache includes Core properties after changes") {
      c.setCameraDevice("cam");
      auto cache = c.getSystemStateCache();
      CHECK(cache.isPropertyIncluded("Core", "Camera"));
      CHECK(cache.getSetting("Core", "Camera").getPropertyValue() == "cam");
   }
}
