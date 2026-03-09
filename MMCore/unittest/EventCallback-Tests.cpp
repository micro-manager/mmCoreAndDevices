#include <catch2/catch_all.hpp>

#include "MMCore.h"
#include "MMEventCallback.h"
#include "MockDeviceUtils.h"
#include "StubDevices.h"
#include "TempFile.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <string>
#include <vector>

// --- Test infrastructure ---

namespace {

enum class CBType {
   PropertiesChanged,
   PropertyChanged,
   ChannelGroupChanged,
   ConfigGroupChanged,
   SystemConfigurationLoaded,
   PixelSizeChanged,
   PixelSizeAffineChanged,
   StagePositionChanged,
   XYStagePositionChanged,
   ExposureChanged,
   ShutterOpenChanged,
   SLMExposureChanged,
   ImageSnapped,
   SequenceAcquisitionStarted,
   SequenceAcquisitionStopped,
};

struct CallbackRecord {
   CBType type;
   std::string s1;
   std::string s2;
   std::string s3;
   double d1 = 0.0;
   double d2 = 0.0;
   double d3 = 0.0;
   double d4 = 0.0;
   double d5 = 0.0;
   double d6 = 0.0;
   bool b1 = false;

   CallbackRecord(CBType t) : type(t) {}
};

class RecordingCallback : public MMEventCallback {
public:
   void onPropertiesChanged() override {
      push({CBType::PropertiesChanged});
   }

   void onPropertyChanged(const char* name, const char* propName,
                           const char* propValue) override {
      CallbackRecord r{CBType::PropertyChanged};
      r.s1 = name;
      r.s2 = propName;
      r.s3 = propValue;
      push(r);
   }

   void onChannelGroupChanged(const char* newChannelGroupName) override {
      CallbackRecord r{CBType::ChannelGroupChanged};
      r.s1 = newChannelGroupName;
      push(r);
   }

   void onConfigGroupChanged(const char* groupName,
                              const char* newConfigName) override {
      CallbackRecord r{CBType::ConfigGroupChanged};
      r.s1 = groupName;
      r.s2 = newConfigName;
      push(r);
   }

   void onSystemConfigurationLoaded() override {
      push({CBType::SystemConfigurationLoaded});
   }

   void onPixelSizeChanged(double newPixelSizeUm) override {
      CallbackRecord r{CBType::PixelSizeChanged};
      r.d1 = newPixelSizeUm;
      push(r);
   }

   void onPixelSizeAffineChanged(double v0, double v1, double v2,
                                  double v3, double v4,
                                  double v5) override {
      CallbackRecord r{CBType::PixelSizeAffineChanged};
      r.d1 = v0;
      r.d2 = v1;
      r.d3 = v2;
      r.d4 = v3;
      r.d5 = v4;
      r.d6 = v5;
      push(r);
   }

   void onStagePositionChanged(const char* name, double pos) override {
      CallbackRecord r{CBType::StagePositionChanged};
      r.s1 = name;
      r.d1 = pos;
      push(r);
   }

   void onXYStagePositionChanged(const char* name, double xpos,
                                  double ypos) override {
      CallbackRecord r{CBType::XYStagePositionChanged};
      r.s1 = name;
      r.d1 = xpos;
      r.d2 = ypos;
      push(r);
   }

   void onExposureChanged(const char* name, double newExposure) override {
      CallbackRecord r{CBType::ExposureChanged};
      r.s1 = name;
      r.d1 = newExposure;
      push(r);
   }

   void onShutterOpenChanged(const char* name, bool open) override {
      CallbackRecord r{CBType::ShutterOpenChanged};
      r.s1 = name;
      r.b1 = open;
      push(r);
   }

   void onSLMExposureChanged(const char* name,
                              double newExposure) override {
      CallbackRecord r{CBType::SLMExposureChanged};
      r.s1 = name;
      r.d1 = newExposure;
      push(r);
   }

   void onImageSnapped(const char* cameraLabel) override {
      CallbackRecord r{CBType::ImageSnapped};
      r.s1 = cameraLabel;
      push(r);
   }

   void onSequenceAcquisitionStarted(
         const char* cameraLabel) override {
      CallbackRecord r{CBType::SequenceAcquisitionStarted};
      r.s1 = cameraLabel;
      push(r);
   }

   void onSequenceAcquisitionStopped(
         const char* cameraLabel) override {
      CallbackRecord r{CBType::SequenceAcquisitionStopped};
      r.s1 = cameraLabel;
      push(r);
   }

   bool waitFor(CBType type,
                std::chrono::milliseconds timeout =
                   std::chrono::milliseconds(5000)) {
      std::unique_lock<std::mutex> lk(mu_);
      return cv_.wait_for(lk, timeout, [&] { return hasType(type); });
   }

   bool waitForCount(CBType type, std::size_t count,
                     std::chrono::milliseconds timeout =
                        std::chrono::milliseconds(5000)) {
      std::unique_lock<std::mutex> lk(mu_);
      return cv_.wait_for(lk, timeout,
                          [&] { return countType(type) >= count; });
   }

   std::vector<CallbackRecord> records(CBType type) {
      std::lock_guard<std::mutex> lk(mu_);
      std::vector<CallbackRecord> result;
      for (auto& r : records_) {
         if (r.type == type)
            result.push_back(r);
      }
      return result;
   }

   void clear() {
      std::lock_guard<std::mutex> lk(mu_);
      records_.clear();
   }

private:
   void push(CallbackRecord r) {
      {
         std::lock_guard<std::mutex> lk(mu_);
         records_.push_back(std::move(r));
      }
      cv_.notify_all();
   }

   bool hasType(CBType type) const {
      return std::any_of(records_.begin(), records_.end(),
                         [type](auto& r) { return r.type == type; });
   }

   std::size_t countType(CBType type) const {
      return std::count_if(records_.begin(), records_.end(),
                           [type](auto& r) { return r.type == type; });
   }

   std::mutex mu_;
   std::condition_variable cv_;
   std::vector<CallbackRecord> records_;
};

// A device with a read-write string property, needed for
// OnPropertyChanged tests (CoreCallback calls GetPropertyReadOnly).
struct StubWithProperty : CGenericBase<StubWithProperty> {
   std::string name = "StubWithProperty";
   using CGenericBase::OnPropertyChanged;

   int Initialize() override {
      CreateStringProperty("TestProp", "initial", false);
      return DEVICE_OK;
   }
   int Shutdown() override { return DEVICE_OK; }
   bool Busy() override { return false; }
   void GetName(char* buf) const override {
      CDeviceUtils::CopyLimitedString(buf, name.c_str());
   }
};

} // namespace

// --- Device-originated callback tests ---

TEST_CASE("onPropertiesChanged from device", "[EventCallback]") {
   StubGeneric dev;
   MockAdapterWithDevices adapter{{"dev", &dev}};
   RecordingCallback cb;
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.registerCallback(&cb);

   dev.OnPropertiesChanged();

   REQUIRE(cb.waitFor(CBType::PropertiesChanged));
}

TEST_CASE("onPropertyChanged from device", "[EventCallback]") {
   StubWithProperty dev;
   MockAdapterWithDevices adapter{{"dev", &dev}};
   RecordingCallback cb;
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.registerCallback(&cb);

   dev.OnPropertyChanged("TestProp", "newValue");

   REQUIRE(cb.waitFor(CBType::PropertyChanged));
   auto recs = cb.records(CBType::PropertyChanged);
   REQUIRE(recs.size() >= 1);
   CHECK(recs[0].s1 == "dev");
   CHECK(recs[0].s2 == "TestProp");
   CHECK(recs[0].s3 == "newValue");
}

TEST_CASE("onStagePositionChanged from device", "[EventCallback]") {
   StubStage stage;
   MockAdapterWithDevices adapter{{"stage", &stage}};
   RecordingCallback cb;
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.registerCallback(&cb);

   stage.OnStagePositionChanged(42.0);

   REQUIRE(cb.waitFor(CBType::StagePositionChanged));
   auto recs = cb.records(CBType::StagePositionChanged);
   REQUIRE(recs.size() >= 1);
   CHECK(recs[0].s1 == "stage");
   CHECK(recs[0].d1 == 42.0);
}

TEST_CASE("onXYStagePositionChanged from device", "[EventCallback]") {
   StubXYStage xy;
   MockAdapterWithDevices adapter{{"xy", &xy}};
   RecordingCallback cb;
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.registerCallback(&cb);

   xy.OnXYStagePositionChanged(1.0, 2.0);

   REQUIRE(cb.waitFor(CBType::XYStagePositionChanged));
   auto recs = cb.records(CBType::XYStagePositionChanged);
   REQUIRE(recs.size() >= 1);
   CHECK(recs[0].s1 == "xy");
   CHECK(recs[0].d1 == 1.0);
   CHECK(recs[0].d2 == 2.0);
}

TEST_CASE("onExposureChanged from device", "[EventCallback]") {
   StubCamera cam;
   MockAdapterWithDevices adapter{{"cam", &cam}};
   RecordingCallback cb;
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.registerCallback(&cb);

   cam.OnExposureChanged(25.0);

   REQUIRE(cb.waitFor(CBType::ExposureChanged));
   auto recs = cb.records(CBType::ExposureChanged);
   REQUIRE(recs.size() >= 1);
   CHECK(recs[0].s1 == "cam");
   CHECK(recs[0].d1 == 25.0);
}

TEST_CASE("onSLMExposureChanged from device", "[EventCallback]") {
   StubSLM slm;
   MockAdapterWithDevices adapter{{"slm", &slm}};
   RecordingCallback cb;
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.registerCallback(&cb);

   slm.OnSLMExposureChanged(100.0);

   REQUIRE(cb.waitFor(CBType::SLMExposureChanged));
   auto recs = cb.records(CBType::SLMExposureChanged);
   REQUIRE(recs.size() >= 1);
   CHECK(recs[0].s1 == "slm");
   CHECK(recs[0].d1 == 100.0);
}

TEST_CASE("onShutterOpenChanged from device", "[EventCallback]") {
   StubShutter shutter;
   MockAdapterWithDevices adapter{{"shutter", &shutter}};
   RecordingCallback cb;
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.registerCallback(&cb);

   shutter.GetCoreCallback()->OnShutterOpenChanged(&shutter, true);

   REQUIRE(cb.waitFor(CBType::ShutterOpenChanged));
   auto recs = cb.records(CBType::ShutterOpenChanged);
   REQUIRE(recs.size() >= 1);
   CHECK(recs[0].s1 == "shutter");
   CHECK(recs[0].b1 == true);
}

TEST_CASE("onSequenceAcquisitionStarted from device",
          "[EventCallback]") {
   StubCamera cam;
   MockAdapterWithDevices adapter{{"cam", &cam}};
   RecordingCallback cb;
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("cam");
   c.registerCallback(&cb);

   cam.PrepareForAcq();

   REQUIRE(cb.waitFor(CBType::SequenceAcquisitionStarted));
   auto recs = cb.records(CBType::SequenceAcquisitionStarted);
   REQUIRE(recs.size() >= 1);
   CHECK(recs[0].s1 == "cam");
}

TEST_CASE("onSequenceAcquisitionStopped from device",
          "[EventCallback]") {
   StubCamera cam;
   MockAdapterWithDevices adapter{{"cam", &cam}};
   RecordingCallback cb;
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("cam");
   c.registerCallback(&cb);

   cam.AcqFinished();

   REQUIRE(cb.waitFor(CBType::SequenceAcquisitionStopped));
   auto recs = cb.records(CBType::SequenceAcquisitionStopped);
   REQUIRE(recs.size() >= 1);
   CHECK(recs[0].s1 == "cam");
}

// --- Derived callbacks (via device property changes) ---

TEST_CASE("onConfigGroupChanged from device property change",
          "[EventCallback]") {
   StubWithProperty dev;
   MockAdapterWithDevices adapter{{"dev", &dev}};
   RecordingCallback cb;
   CMMCore c;
   adapter.LoadIntoCore(c);

   c.defineConfig("Group1", "Config1", "dev", "TestProp", "val1");
   c.registerCallback(&cb);

   dev.OnPropertyChanged("TestProp", "val1");

   REQUIRE(cb.waitFor(CBType::PropertyChanged));
   REQUIRE(cb.waitFor(CBType::ConfigGroupChanged));
   auto recs = cb.records(CBType::ConfigGroupChanged);
   REQUIRE(recs.size() >= 1);
   CHECK(recs[0].s1 == "Group1");
}

TEST_CASE("onPixelSizeChanged from device property change",
          "[EventCallback]") {
   StubWithProperty dev;
   MockAdapterWithDevices adapter{{"dev", &dev}};
   RecordingCallback cb;
   CMMCore c;
   adapter.LoadIntoCore(c);

   c.definePixelSizeConfig("Res1", "dev", "TestProp", "val1");
   c.setPixelSizeUm("Res1", 0.5);
   c.registerCallback(&cb);

   dev.OnPropertyChanged("TestProp", "val1");

   REQUIRE(cb.waitFor(CBType::PixelSizeChanged));
   auto recs = cb.records(CBType::PixelSizeChanged);
   REQUIRE(recs.size() >= 1);
   CHECK(recs[0].d1 == 0.5);
}

TEST_CASE("onPixelSizeAffineChanged from device property change",
          "[EventCallback]") {
   StubWithProperty dev;
   MockAdapterWithDevices adapter{{"dev", &dev}};
   RecordingCallback cb;
   CMMCore c;
   adapter.LoadIntoCore(c);

   c.definePixelSizeConfig("Res1", "dev", "TestProp", "val1");
   c.setPixelSizeUm("Res1", 0.5);
   std::vector<double> matrix = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0};
   c.setPixelSizeAffine("Res1", matrix);
   c.registerCallback(&cb);

   dev.OnPropertyChanged("TestProp", "val1");

   REQUIRE(cb.waitFor(CBType::PixelSizeAffineChanged));
   auto recs = cb.records(CBType::PixelSizeAffineChanged);
   REQUIRE(recs.size() >= 1);
   CHECK(recs[0].d1 == 1.0);
   CHECK(recs[0].d2 == 0.0);
   CHECK(recs[0].d3 == 0.0);
   CHECK(recs[0].d4 == 0.0);
   CHECK(recs[0].d5 == 1.0);
   CHECK(recs[0].d6 == 0.0);
}

TEST_CASE("onPixelSizeChanged from magnifier change",
          "[EventCallback]") {
   StubMagnifier mag;
   MockAdapterWithDevices adapter{{"mag", &mag}};
   RecordingCallback cb;
   CMMCore c;
   adapter.LoadIntoCore(c);

   c.definePixelSizeConfig("Res1");
   c.setPixelSizeUm("Res1", 0.5);
   c.setPixelSizeConfig("Res1");
   c.registerCallback(&cb);

   mag.OnMagnifierChanged();

   REQUIRE(cb.waitFor(CBType::PixelSizeChanged));
   auto recs = cb.records(CBType::PixelSizeChanged);
   REQUIRE(recs.size() >= 1);
   CHECK(recs[0].d1 == 0.5);
}

// --- Core-originated callback tests ---

TEST_CASE("onShutterOpenChanged from setShutterOpen", "[EventCallback]") {
   StubShutter shutter;
   MockAdapterWithDevices adapter{{"shutter", &shutter}};
   RecordingCallback cb;
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.registerCallback(&cb);

   c.setShutterOpen("shutter", true);

   REQUIRE(cb.waitFor(CBType::ShutterOpenChanged));
   auto recs = cb.records(CBType::ShutterOpenChanged);
   REQUIRE(recs.size() >= 1);
   CHECK(recs[0].s1 == "shutter");
   CHECK(recs[0].b1 == true);
}

TEST_CASE("onShutterOpenChanged from snapImage with auto-shutter",
          "[EventCallback]") {
   StubCamera cam;
   StubShutter shutter;
   MockAdapterWithDevices adapter{{"cam", &cam}, {"shutter", &shutter}};
   RecordingCallback cb;
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("cam");
   c.setShutterDevice("shutter");
   c.setAutoShutter(true);
   c.registerCallback(&cb);

   c.snapImage();

   REQUIRE(cb.waitForCount(CBType::ShutterOpenChanged, 2));
   auto recs = cb.records(CBType::ShutterOpenChanged);
   REQUIRE(recs.size() >= 2);
   CHECK(recs[0].s1 == "shutter");
   CHECK(recs[0].b1 == true);
   CHECK(recs[1].s1 == "shutter");
   CHECK(recs[1].b1 == false);
}

TEST_CASE("onShutterOpenChanged from AcqFinished", "[EventCallback]") {
   StubCamera cam;
   StubShutter shutter;
   MockAdapterWithDevices adapter{{"cam", &cam}, {"shutter", &shutter}};
   RecordingCallback cb;
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("cam");
   c.setShutterDevice("shutter");
   c.setAutoShutter(true);
   c.registerCallback(&cb);

   cam.AcqFinished();

   REQUIRE(cb.waitFor(CBType::ShutterOpenChanged));
   auto recs = cb.records(CBType::ShutterOpenChanged);
   REQUIRE(recs.size() >= 1);
   CHECK(recs[0].s1 == "shutter");
   CHECK(recs[0].b1 == false);
}

TEST_CASE("onShutterOpenChanged from PrepareForAcq", "[EventCallback]") {
   StubCamera cam;
   StubShutter shutter;
   MockAdapterWithDevices adapter{{"cam", &cam}, {"shutter", &shutter}};
   RecordingCallback cb;
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("cam");
   c.setShutterDevice("shutter");
   c.setAutoShutter(true);
   c.registerCallback(&cb);

   cam.PrepareForAcq();

   REQUIRE(cb.waitFor(CBType::ShutterOpenChanged));
   auto recs = cb.records(CBType::ShutterOpenChanged);
   REQUIRE(recs.size() >= 1);
   CHECK(recs[0].s1 == "shutter");
   CHECK(recs[0].b1 == true);
}

TEST_CASE("onImageSnapped from snapImage", "[EventCallback]") {
   StubCamera cam;
   MockAdapterWithDevices adapter{{"cam", &cam}};
   RecordingCallback cb;
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("cam");
   c.registerCallback(&cb);

   c.snapImage();

   REQUIRE(cb.waitFor(CBType::ImageSnapped));
   auto recs = cb.records(CBType::ImageSnapped);
   REQUIRE(recs.size() >= 1);
   CHECK(recs[0].s1 == "cam");
}

TEST_CASE("onChannelGroupChanged from setChannelGroup",
          "[EventCallback]") {
   RecordingCallback cb;
   CMMCore c;
   c.defineConfig("Channel", "Ch1");
   c.registerCallback(&cb);

   c.setChannelGroup("Channel");

   REQUIRE(cb.waitFor(CBType::ChannelGroupChanged));
   auto recs = cb.records(CBType::ChannelGroupChanged);
   REQUIRE(recs.size() >= 1);
   CHECK(recs[0].s1 == "Channel");
}

TEST_CASE("onSystemConfigurationLoaded from unloadAllDevices",
          "[EventCallback]") {
   StubGeneric dev;
   MockAdapterWithDevices adapter{{"dev", &dev}};
   RecordingCallback cb;
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.registerCallback(&cb);

   c.unloadAllDevices();

   REQUIRE(cb.waitFor(CBType::SystemConfigurationLoaded));
}

TEST_CASE(
   "onSystemConfigurationLoaded from loadSystemConfiguration success",
   "[EventCallback]") {
   RecordingCallback cb;
   CMMCore c;
   c.registerCallback(&cb);

   TempFile tmp("# empty config\n");
   c.loadSystemConfiguration(tmp.getPath().c_str());

   REQUIRE(cb.waitFor(CBType::SystemConfigurationLoaded));
}

TEST_CASE(
   "onSystemConfigurationLoaded from loadSystemConfiguration failure",
   "[EventCallback]") {
   RecordingCallback cb;
   CMMCore c;
   c.registerCallback(&cb);

   TempFile tmp("Device,NoSuchLabel,NoSuchAdapter,NoSuchDevice\n");
   CHECK_THROWS(c.loadSystemConfiguration(tmp.getPath().c_str()));

   // The callback fires via unloadAllDevices in the error path
   REQUIRE(cb.waitFor(CBType::SystemConfigurationLoaded));
}

TEST_CASE("onPropertyChanged from AutoShutter", "[EventCallback]") {
   RecordingCallback cb;
   CMMCore c;
   c.registerCallback(&cb);

   SECTION("via setProperty") {
      c.setProperty("Core", "AutoShutter", "0");

      REQUIRE(cb.waitFor(CBType::PropertyChanged));
      auto recs = cb.records(CBType::PropertyChanged);
      REQUIRE(recs.size() >= 1);
      CHECK(recs[0].s1 == "Core");
      CHECK(recs[0].s2 == "AutoShutter");
      CHECK(recs[0].s3 == "0");
   }

   SECTION("via setAutoShutter") {
      c.setAutoShutter(false);

      REQUIRE(cb.waitFor(CBType::PropertyChanged));
      auto recs = cb.records(CBType::PropertyChanged);
      REQUIRE(recs.size() >= 1);
      CHECK(recs[0].s1 == "Core");
      CHECK(recs[0].s2 == "AutoShutter");
      CHECK(recs[0].s3 == "0");
   }
}

TEST_CASE("onPropertyChanged from TimeoutMs", "[EventCallback]") {
   RecordingCallback cb;
   CMMCore c;
   c.registerCallback(&cb);

   SECTION("via setProperty") {
      c.setProperty("Core", "TimeoutMs", "10000");

      REQUIRE(cb.waitFor(CBType::PropertyChanged));
      auto recs = cb.records(CBType::PropertyChanged);
      REQUIRE(recs.size() >= 1);
      CHECK(recs[0].s1 == "Core");
      CHECK(recs[0].s2 == "TimeoutMs");
      CHECK(recs[0].s3 == "10000");
   }

   SECTION("via setTimeoutMs") {
      c.setTimeoutMs(10000);

      REQUIRE(cb.waitFor(CBType::PropertyChanged));
      auto recs = cb.records(CBType::PropertyChanged);
      REQUIRE(recs.size() >= 1);
      CHECK(recs[0].s1 == "Core");
      CHECK(recs[0].s2 == "TimeoutMs");
      CHECK(recs[0].s3 == "10000");
   }
}

TEST_CASE("onPropertyChanged from ChannelGroup", "[EventCallback]") {
   RecordingCallback cb;
   CMMCore c;
   c.defineConfig("Channel", "Ch1");
   c.registerCallback(&cb);

   SECTION("via setProperty") {
      c.setProperty("Core", "ChannelGroup", "Channel");

      REQUIRE(cb.waitFor(CBType::PropertyChanged));
      auto recs = cb.records(CBType::PropertyChanged);
      REQUIRE(recs.size() >= 1);
      CHECK(recs[0].s1 == "Core");
      CHECK(recs[0].s2 == "ChannelGroup");
      CHECK(recs[0].s3 == "Channel");
   }

   SECTION("via setChannelGroup") {
      c.setChannelGroup("Channel");

      REQUIRE(cb.waitFor(CBType::PropertyChanged));
      auto recs = cb.records(CBType::PropertyChanged);
      REQUIRE(recs.size() >= 1);
      CHECK(recs[0].s1 == "Core");
      CHECK(recs[0].s2 == "ChannelGroup");
      CHECK(recs[0].s3 == "Channel");
   }
}

TEST_CASE("onPropertyChanged from device role properties",
          "[EventCallback]") {
   SECTION("Camera") {
      StubCamera cam;
      MockAdapterWithDevices adapter{{"cam", &cam}};
      RecordingCallback cb;
      CMMCore c;
      adapter.LoadIntoCore(c);
      c.registerCallback(&cb);

      c.setProperty("Core", "Camera", "cam");

      REQUIRE(cb.waitFor(CBType::PropertyChanged));
      auto recs = cb.records(CBType::PropertyChanged);
      REQUIRE(recs.size() >= 1);
      CHECK(recs[0].s1 == "Core");
      CHECK(recs[0].s2 == "Camera");
      CHECK(recs[0].s3 == "cam");

      cb.clear();
      c.setCameraDevice("");

      REQUIRE(cb.waitFor(CBType::PropertyChanged));
      recs = cb.records(CBType::PropertyChanged);
      REQUIRE(recs.size() >= 1);
      CHECK(recs[0].s1 == "Core");
      CHECK(recs[0].s2 == "Camera");
      CHECK(recs[0].s3 == "");
   }

   SECTION("Shutter") {
      StubShutter sh;
      MockAdapterWithDevices adapter{{"sh", &sh}};
      RecordingCallback cb;
      CMMCore c;
      adapter.LoadIntoCore(c);
      c.registerCallback(&cb);

      c.setProperty("Core", "Shutter", "sh");

      REQUIRE(cb.waitFor(CBType::PropertyChanged));
      auto recs = cb.records(CBType::PropertyChanged);
      REQUIRE(recs.size() >= 1);
      CHECK(recs[0].s1 == "Core");
      CHECK(recs[0].s2 == "Shutter");
      CHECK(recs[0].s3 == "sh");

      cb.clear();
      c.setShutterDevice("");

      REQUIRE(cb.waitFor(CBType::PropertyChanged));
      recs = cb.records(CBType::PropertyChanged);
      REQUIRE(recs.size() >= 1);
      CHECK(recs[0].s1 == "Core");
      CHECK(recs[0].s2 == "Shutter");
      CHECK(recs[0].s3 == "");
   }

   SECTION("Focus") {
      StubStage stage;
      MockAdapterWithDevices adapter{{"stage", &stage}};
      RecordingCallback cb;
      CMMCore c;
      adapter.LoadIntoCore(c);
      c.registerCallback(&cb);

      c.setProperty("Core", "Focus", "stage");

      REQUIRE(cb.waitFor(CBType::PropertyChanged));
      auto recs = cb.records(CBType::PropertyChanged);
      REQUIRE(recs.size() >= 1);
      CHECK(recs[0].s1 == "Core");
      CHECK(recs[0].s2 == "Focus");
      CHECK(recs[0].s3 == "stage");

      cb.clear();
      c.setFocusDevice("");

      REQUIRE(cb.waitFor(CBType::PropertyChanged));
      recs = cb.records(CBType::PropertyChanged);
      REQUIRE(recs.size() >= 1);
      CHECK(recs[0].s1 == "Core");
      CHECK(recs[0].s2 == "Focus");
      CHECK(recs[0].s3 == "");
   }

   SECTION("XYStage") {
      StubXYStage xy;
      MockAdapterWithDevices adapter{{"xy", &xy}};
      RecordingCallback cb;
      CMMCore c;
      adapter.LoadIntoCore(c);
      c.registerCallback(&cb);

      c.setProperty("Core", "XYStage", "xy");

      REQUIRE(cb.waitFor(CBType::PropertyChanged));
      auto recs = cb.records(CBType::PropertyChanged);
      REQUIRE(recs.size() >= 1);
      CHECK(recs[0].s1 == "Core");
      CHECK(recs[0].s2 == "XYStage");
      CHECK(recs[0].s3 == "xy");

      cb.clear();
      c.setXYStageDevice("");

      REQUIRE(cb.waitFor(CBType::PropertyChanged));
      recs = cb.records(CBType::PropertyChanged);
      REQUIRE(recs.size() >= 1);
      CHECK(recs[0].s1 == "Core");
      CHECK(recs[0].s2 == "XYStage");
      CHECK(recs[0].s3 == "");
   }

   SECTION("AutoFocus") {
      StubAutoFocus af;
      MockAdapterWithDevices adapter{{"af", &af}};
      RecordingCallback cb;
      CMMCore c;
      adapter.LoadIntoCore(c);
      c.registerCallback(&cb);

      c.setProperty("Core", "AutoFocus", "af");

      REQUIRE(cb.waitFor(CBType::PropertyChanged));
      auto recs = cb.records(CBType::PropertyChanged);
      REQUIRE(recs.size() >= 1);
      CHECK(recs[0].s1 == "Core");
      CHECK(recs[0].s2 == "AutoFocus");
      CHECK(recs[0].s3 == "af");

      cb.clear();
      c.setAutoFocusDevice("");

      REQUIRE(cb.waitFor(CBType::PropertyChanged));
      recs = cb.records(CBType::PropertyChanged);
      REQUIRE(recs.size() >= 1);
      CHECK(recs[0].s1 == "Core");
      CHECK(recs[0].s2 == "AutoFocus");
      CHECK(recs[0].s3 == "");
   }

   SECTION("ImageProcessor") {
      StubImageProcessor ip;
      MockAdapterWithDevices adapter{{"ip", &ip}};
      RecordingCallback cb;
      CMMCore c;
      adapter.LoadIntoCore(c);
      c.registerCallback(&cb);

      c.setProperty("Core", "ImageProcessor", "ip");

      REQUIRE(cb.waitFor(CBType::PropertyChanged));
      auto recs = cb.records(CBType::PropertyChanged);
      REQUIRE(recs.size() >= 1);
      CHECK(recs[0].s1 == "Core");
      CHECK(recs[0].s2 == "ImageProcessor");
      CHECK(recs[0].s3 == "ip");

      cb.clear();
      c.setImageProcessorDevice("");

      REQUIRE(cb.waitFor(CBType::PropertyChanged));
      recs = cb.records(CBType::PropertyChanged);
      REQUIRE(recs.size() >= 1);
      CHECK(recs[0].s1 == "Core");
      CHECK(recs[0].s2 == "ImageProcessor");
      CHECK(recs[0].s3 == "");
   }

   SECTION("SLM") {
      StubSLM slm;
      MockAdapterWithDevices adapter{{"slm", &slm}};
      RecordingCallback cb;
      CMMCore c;
      adapter.LoadIntoCore(c);
      c.registerCallback(&cb);

      c.setProperty("Core", "SLM", "slm");

      REQUIRE(cb.waitFor(CBType::PropertyChanged));
      auto recs = cb.records(CBType::PropertyChanged);
      REQUIRE(recs.size() >= 1);
      CHECK(recs[0].s1 == "Core");
      CHECK(recs[0].s2 == "SLM");
      CHECK(recs[0].s3 == "slm");

      cb.clear();
      c.setSLMDevice("");

      REQUIRE(cb.waitFor(CBType::PropertyChanged));
      recs = cb.records(CBType::PropertyChanged);
      REQUIRE(recs.size() >= 1);
      CHECK(recs[0].s1 == "Core");
      CHECK(recs[0].s2 == "SLM");
      CHECK(recs[0].s3 == "");
   }

   SECTION("Galvo") {
      StubGalvo galvo;
      MockAdapterWithDevices adapter{{"galvo", &galvo}};
      RecordingCallback cb;
      CMMCore c;
      adapter.LoadIntoCore(c);
      c.registerCallback(&cb);

      c.setProperty("Core", "Galvo", "galvo");

      REQUIRE(cb.waitFor(CBType::PropertyChanged));
      auto recs = cb.records(CBType::PropertyChanged);
      REQUIRE(recs.size() >= 1);
      CHECK(recs[0].s1 == "Core");
      CHECK(recs[0].s2 == "Galvo");
      CHECK(recs[0].s3 == "galvo");

      cb.clear();
      c.setGalvoDevice("");

      REQUIRE(cb.waitFor(CBType::PropertyChanged));
      recs = cb.records(CBType::PropertyChanged);
      REQUIRE(recs.size() >= 1);
      CHECK(recs[0].s1 == "Core");
      CHECK(recs[0].s2 == "Galvo");
      CHECK(recs[0].s3 == "");
   }
}

TEST_CASE("onPropertyChanged from Initialize", "[EventCallback]") {
   StubGeneric dev;
   MockAdapterWithDevices adapter{{"dev", &dev}};
   RecordingCallback cb;
   CMMCore c;
   c.loadMockDeviceAdapter("mock", &adapter);
   c.loadDevice("dev", "mock", "dev");
   c.registerCallback(&cb);

   c.setProperty("Core", "Initialize", "1");

   REQUIRE(cb.waitFor(CBType::PropertyChanged));
   auto recs = cb.records(CBType::PropertyChanged);
   REQUIRE(recs.size() >= 1);
   CHECK(recs[0].s1 == "Core");
   CHECK(recs[0].s2 == "Initialize");
   CHECK(recs[0].s3 == "1");
}

// --- Negative tests ---

TEST_CASE("registerCallback throws when called from callback handler",
          "[EventCallback]") {
   StubGeneric dev;
   MockAdapterWithDevices adapter{{"dev", &dev}};
   std::mutex mu;
   std::condition_variable cv;
   bool threw = false;
   bool done = false;
   CMMCore c;
   adapter.LoadIntoCore(c);

   struct ReentrantCallback : MMEventCallback {
      CMMCore& core;
      std::mutex& mu;
      std::condition_variable& cv;
      bool& threw;
      bool& done;

      ReentrantCallback(CMMCore& c, std::mutex& m,
                        std::condition_variable& v, bool& t, bool& d)
          : core(c), mu(m), cv(v), threw(t), done(d) {}

      void onPropertiesChanged() override {
         bool caught = false;
         try {
            core.registerCallback(nullptr);
         } catch (const CMMError&) {
            caught = true;
         }
         {
            std::lock_guard<std::mutex> lk(mu);
            threw = caught;
            done = true;
         }
         cv.notify_all();
      }
   };

   ReentrantCallback cb(c, mu, cv, threw, done);
   c.registerCallback(&cb);

   dev.OnPropertiesChanged();

   {
      std::unique_lock<std::mutex> lk(mu);
      REQUIRE(cv.wait_for(lk, std::chrono::milliseconds(5000),
                          [&] { return done; }));
   }
   c.registerCallback(nullptr);
   REQUIRE(threw);
}

TEST_CASE("No crash when no callback is registered", "[EventCallback]") {
   StubCamera cam;
   StubGeneric dev;
   StubStage stage;
   MockAdapterWithDevices adapter{
      {"cam", &cam}, {"dev", &dev}, {"stage", &stage}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("cam");

   // Trigger various actions without registering a callback
   c.snapImage();
   dev.OnPropertiesChanged();
   stage.OnStagePositionChanged(1.0);
   cam.OnExposureChanged(10.0);
   c.unloadAllDevices();
   // If we get here without crashing, the test passes.
}
