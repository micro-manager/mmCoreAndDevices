#include <catch2/catch_all.hpp>

#include "MMCore.h"
#include "DeviceBase.h"
#include "ImageMetadata.h"
#include "MMDeviceConstants.h"
#include "MockDeviceUtils.h"
#include "StubDevices.h"

#include <chrono>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

// Minimal mock of an intrinsic multi-channel camera (in the style of the
// TwoPhoton adapter): a single device that itself emits N channels by
// calling InsertImage once per channel per frame, embedding only
// CameraChannelIndex in the serialized metadata. (CameraChannelName is
// stamped by MMCore from its start-time snapshot of GetChannelName, so
// intrinsic devices do not — and should not — emit it.) Tests drive
// InsertTestImage() manually after StartSequenceAcquisition.
struct MockIntrinsicMultiChannelCamera :
   CCameraBase<MockIntrinsicMultiChannelCamera> {
   std::string name = "MockIntrinsicMultiChannelCamera";
   std::vector<std::string> channelNames;
   unsigned width = 64;
   unsigned height = 64;
   unsigned bytesPerPixel = 1;
   unsigned bitDepth = 8;
   int binning = 1;
   double exposure = 10.0;
   bool capturing_ = false;

   explicit MockIntrinsicMultiChannelCamera(std::vector<std::string> chNames)
      : channelNames(std::move(chNames)) {}

   int Initialize() override { return DEVICE_OK; }
   int Shutdown() override { return DEVICE_OK; }
   bool Busy() override { return false; }
   void GetName(char* buf) const override {
      CDeviceUtils::CopyLimitedString(buf, name.c_str());
   }

   int SnapImage() override {
      imgBuf_.assign(static_cast<size_t>(width) * height * bytesPerPixel, 0);
      return DEVICE_OK;
   }
   const unsigned char* GetImageBuffer() override { return imgBuf_.data(); }
   long GetImageBufferSize() const override {
      return static_cast<long>(width) * height * bytesPerPixel;
   }
   unsigned GetImageWidth() const override { return width; }
   unsigned GetImageHeight() const override { return height; }
   unsigned GetImageBytesPerPixel() const override { return bytesPerPixel; }
   unsigned GetNumberOfComponents() const override { return 1; }
   unsigned GetNumberOfChannels() const override {
      return static_cast<unsigned>(channelNames.size());
   }
   int GetChannelName(unsigned channel, char* chName) override {
      if (channel < channelNames.size()) {
         CDeviceUtils::CopyLimitedString(chName,
            channelNames[channel].c_str());
      } else {
         CDeviceUtils::CopyLimitedString(chName, "");
      }
      return DEVICE_OK;
   }
   unsigned GetBitDepth() const override { return bitDepth; }
   int GetBinning() const override { return binning; }
   int SetBinning(int b) override { binning = b; return DEVICE_OK; }
   void SetExposure(double e) override { exposure = e; }
   double GetExposure() const override { return exposure; }
   int SetROI(unsigned, unsigned, unsigned, unsigned) override {
      return DEVICE_OK;
   }
   int GetROI(unsigned& x, unsigned& y, unsigned& w, unsigned& h) override {
      x = 0; y = 0; w = width; h = height;
      return DEVICE_OK;
   }
   int ClearROI() override { return DEVICE_OK; }
   int IsExposureSequenceable(bool& seq) const override {
      seq = false;
      return DEVICE_OK;
   }

   int StartSequenceAcquisition(long, double, bool) override {
      capturing_ = true;
      GetCoreCallback()->PrepareForAcq(this);
      return DEVICE_OK;
   }
   int StartSequenceAcquisition(double) override {
      capturing_ = true;
      GetCoreCallback()->PrepareForAcq(this);
      return DEVICE_OK;
   }
   int StopSequenceAcquisition() override {
      Finish();
      return DEVICE_OK;
   }
   bool IsCapturing() override { return capturing_; }

   // Simulates the camera deciding to finish on its own (finite-count
   // completion or error path).
   void TriggerSelfFinish() { Finish(); }

   // Inserts a single channel of a multi-channel frame, with the channel
   // index embedded in the serialized metadata. (No channel name: MMCore
   // stamps it from its start-time snapshot.) Tests call this once per
   // channel per frame; the order is up to the test (real intrinsic
   // adapters may interleave channels).
   int InsertTestImage(unsigned channel) {
      std::vector<unsigned char> buf(
         static_cast<size_t>(width) * height * bytesPerPixel, 0);
      MM::CameraImageMetadata md;
      md.AddTag(MM::g_Keyword_CameraChannelIndex,
         std::to_string(channel).c_str());
      return GetCoreCallback()->InsertImage(this, buf.data(),
         width, height, bytesPerPixel, md.Serialize());
   }

private:
   // A real camera calls AcqFinished exactly once per acquisition; the
   // capturing_ guard models that.
   void Finish() {
      if (capturing_) {
         capturing_ = false;
         GetCoreCallback()->AcqFinished(this, 0);
      }
   }

   std::vector<unsigned char> imgBuf_;
};

} // namespace

// --- Headline end-to-end ---

TEST_CASE("Sequence acquisition with 2-channel composite camera tags each "
          "frame with its physical's CameraChannelName/Index",
          "[MultiChannelSequenceAcquisition]") {
   SyncCamera p0("p0");
   SyncCamera p1("p1");
   MockCompositeCamera composite({&p0, &p1});
   MockAdapterWithDevices adapter{
      {"p0", &p0}, {"p1", &p1}, {"composite", &composite}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("composite");

   c.startSequenceAcquisition(4, 0.0, true);
   REQUIRE(p0.InsertTestImage() == DEVICE_OK);
   REQUIRE(p1.InsertTestImage() == DEVICE_OK);
   REQUIRE(p0.InsertTestImage() == DEVICE_OK);
   REQUIRE(p1.InsertTestImage() == DEVICE_OK);
   c.stopSequenceAcquisition();

   CHECK(c.getRemainingImageCount() == 4);

   struct Expected {
      const char* cameraLabel;
      const char* channelName;
      const char* channelIndex;
   };
   const Expected expected[] = {
      {"p0", "p0", "0"},
      {"p1", "p1", "1"},
      {"p0", "p0", "0"},
      {"p1", "p1", "1"},
   };
   for (const auto& e : expected) {
      Metadata md;
      c.popNextImageMD(md);
      CHECK(md.GetSingleTag(MM::g_Keyword_Metadata_CameraLabel).GetValue() ==
            e.cameraLabel);
      CHECK(md.GetSingleTag("composite-CameraChannelName").GetValue() ==
            e.channelName);
      CHECK(md.GetSingleTag("composite-CameraChannelIndex").GetValue() ==
            e.channelIndex);
   }
}

// --- Lifecycle ---

TEST_CASE("isSequenceRunning is true while composite camera is acquiring "
          "and false after stop",
          "[MultiChannelSequenceAcquisition]") {
   SyncCamera p0("p0");
   SyncCamera p1("p1");
   MockCompositeCamera composite({&p0, &p1});
   MockAdapterWithDevices adapter{
      {"p0", &p0}, {"p1", &p1}, {"composite", &composite}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("composite");

   CHECK_FALSE(c.isSequenceRunning());
   c.startSequenceAcquisition(10, 0.0, true);
   CHECK(c.isSequenceRunning());
   c.stopSequenceAcquisition();
   CHECK_FALSE(c.isSequenceRunning());
}

// --- Tag scoping ---

TEST_CASE("Composite phys cam used standalone after composite acquisition has "
          "no composite-prefixed channel tags",
          "[MultiChannelSequenceAcquisition]") {
   SyncCamera p0("p0");
   SyncCamera p1("p1");
   MockCompositeCamera composite({&p0, &p1});
   MockAdapterWithDevices adapter{
      {"p0", &p0}, {"p1", &p1}, {"composite", &composite}};
   CMMCore c;
   adapter.LoadIntoCore(c);

   c.setCameraDevice("composite");
   c.startSequenceAcquisition(2, 0.0, true);
   REQUIRE(p0.InsertTestImage() == DEVICE_OK);
   REQUIRE(p1.InsertTestImage() == DEVICE_OK);
   c.stopSequenceAcquisition();
   // Drain composite frames.
   while (c.getRemainingImageCount() > 0) {
      Metadata md;
      c.popNextImageMD(md);
   }

   // Now use p0 standalone — frames should not carry composite tags.
   c.setCameraDevice("p0");
   c.startSequenceAcquisition(1, 0.0, true);
   REQUIRE(p0.InsertTestImage() == DEVICE_OK);
   c.stopSequenceAcquisition();

   REQUIRE(c.getRemainingImageCount() == 1);
   Metadata md;
   c.popNextImageMD(md);
   CHECK_THROWS(md.GetSingleTag("composite-CameraChannelIndex"));
   CHECK_THROWS(md.GetSingleTag("composite-CameraChannelName"));
}

TEST_CASE("Composite circular buffer holds frames-times-channels",
          "[MultiChannelSequenceAcquisition]") {
   SyncCamera p0("p0");
   SyncCamera p1("p1");
   MockCompositeCamera composite({&p0, &p1});
   MockAdapterWithDevices adapter{
      {"p0", &p0}, {"p1", &p1}, {"composite", &composite}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("composite");

   c.startSequenceAcquisition(6, 0.0, true);
   for (int i = 0; i < 3; ++i) {
      REQUIRE(p0.InsertTestImage() == DEVICE_OK);
      REQUIRE(p1.InsertTestImage() == DEVICE_OK);
   }
   c.stopSequenceAcquisition();

   CHECK(c.getRemainingImageCount() == 6);
}

// --- Core channel APIs ---

TEST_CASE("Core reports composite camera's channel count and names",
          "[MultiChannelSequenceAcquisition]") {
   SyncCamera p0("p0");
   SyncCamera p1("p1");
   SyncCamera p2("p2");
   MockCompositeCamera composite({&p0, &p1, &p2});
   MockAdapterWithDevices adapter{
      {"p0", &p0}, {"p1", &p1}, {"p2", &p2}, {"composite", &composite}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("composite");

   CHECK(c.getNumberOfCameraChannels() == 3);
   CHECK(c.getCameraChannelName(0) == "p0");
   CHECK(c.getCameraChannelName(1) == "p1");
   CHECK(c.getCameraChannelName(2) == "p2");
}

// --- Continuous and named-camera entry points ---

TEST_CASE("startContinuousSequenceAcquisition with composite camera "
          "tags frames and runs until stopped",
          "[MultiChannelSequenceAcquisition]") {
   SyncCamera p0("p0");
   SyncCamera p1("p1");
   MockCompositeCamera composite({&p0, &p1});
   MockAdapterWithDevices adapter{
      {"p0", &p0}, {"p1", &p1}, {"composite", &composite}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("composite");

   c.startContinuousSequenceAcquisition(0.0);
   CHECK(c.isSequenceRunning());
   REQUIRE(p0.InsertTestImage() == DEVICE_OK);
   REQUIRE(p1.InsertTestImage() == DEVICE_OK);
   c.stopSequenceAcquisition();
   CHECK_FALSE(c.isSequenceRunning());

   CHECK(c.getRemainingImageCount() == 2);
   {
      Metadata md;
      c.popNextImageMD(md);
      CHECK(md.GetSingleTag("composite-CameraChannelIndex").GetValue() == "0");
   }
   {
      Metadata md;
      c.popNextImageMD(md);
      CHECK(md.GetSingleTag("composite-CameraChannelIndex").GetValue() == "1");
   }
}

TEST_CASE("Named-camera startSequenceAcquisition on composite camera "
          "drives the same path as the default overload",
          "[MultiChannelSequenceAcquisition]") {
   SyncCamera p0("p0");
   SyncCamera p1("p1");
   MockCompositeCamera composite({&p0, &p1});
   MockAdapterWithDevices adapter{
      {"p0", &p0}, {"p1", &p1}, {"composite", &composite}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("composite");

   c.startSequenceAcquisition("composite", 4, 0.0, true);
   CHECK(c.isSequenceRunning("composite"));
   REQUIRE(p0.InsertTestImage() == DEVICE_OK);
   REQUIRE(p1.InsertTestImage() == DEVICE_OK);
   c.stopSequenceAcquisition("composite");
   CHECK_FALSE(c.isSequenceRunning("composite"));
   CHECK(c.getRemainingImageCount() == 2);
}

// --- Cleanup ---

TEST_CASE("Composite: only the last physical's AcqFinished closes the "
          "auto-shutter",
          "[MultiChannelSequenceAcquisition]") {
   SyncCamera p0("p0");
   SyncCamera p1("p1");
   MockCompositeCamera composite({&p0, &p1});
   StubShutter shutter;
   MockAdapterWithDevices adapter{
      {"p0", &p0}, {"p1", &p1}, {"composite", &composite},
      {"shutter", &shutter}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("composite");
   c.setShutterDevice("shutter");
   c.setAutoShutter(true);

   c.startSequenceAcquisition(10, 0.0, true);
   REQUIRE(shutter.open == true);

   p0.TriggerSelfFinish();
   // First physical to finish should not close the shutter.
   CHECK(shutter.open == true);

   p1.TriggerSelfFinish();
   // Last physical to finish closes it.
   CHECK(shutter.open == false);

   c.stopSequenceAcquisition();
}

TEST_CASE("Composite: physical's AcqFinished does not touch shutter "
          "when autoShutter is off",
          "[MultiChannelSequenceAcquisition]") {
   SyncCamera p0("p0");
   SyncCamera p1("p1");
   MockCompositeCamera composite({&p0, &p1});
   StubShutter shutter;
   MockAdapterWithDevices adapter{
      {"p0", &p0}, {"p1", &p1}, {"composite", &composite},
      {"shutter", &shutter}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("composite");
   c.setShutterDevice("shutter");
   c.setAutoShutter(false);

   c.startSequenceAcquisition(10, 0.0, true);
   shutter.open = true;
   p0.TriggerSelfFinish();
   CHECK(shutter.open == true);
   p1.TriggerSelfFinish();
   CHECK(shutter.open == true);

   c.stopSequenceAcquisition();
}

TEST_CASE("Composite with async physicals: same-module shutter closes on stop "
          "without deadlock",
          "[MultiChannelSequenceAcquisition]") {
   AsyncCamera p0;
   p0.name = "p0";
   AsyncCamera p1;
   p1.name = "p1";
   MockCompositeCamera composite({&p0, &p1});
   StubShutter shutter;
   MockAdapterWithDevices adapter{
      {"p0", &p0}, {"p1", &p1}, {"composite", &composite},
      {"shutter", &shutter}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("composite");
   c.setShutterDevice("shutter");
   c.setAutoShutter(true);

   c.startSequenceAcquisition(1000000, 0.0, true);
   REQUIRE(shutter.open == true);
   c.stopSequenceAcquisition();
   CHECK(shutter.open == false);
}

TEST_CASE("Composite in separate module from phys cam and shutter: shutter "
          "closes via blocking lock without deadlock",
          "[MultiChannelSequenceAcquisition]") {
   AsyncCamera p0;
   p0.name = "p0";
   MockCompositeCamera composite({&p0});
   StubShutter shutter;
   MockAdapterWithDevices compositeAdapter{"composite_adapter",
      {{"composite", &composite}}};
   MockAdapterWithDevices physAdapter{"phys_adapter",
      {{"p0", &p0}, {"shutter", &shutter}}};
   CMMCore c;
   compositeAdapter.LoadIntoCore(c);
   physAdapter.LoadIntoCore(c);
   c.setCameraDevice("composite");
   c.setShutterDevice("shutter");
   c.setAutoShutter(true);

   c.startSequenceAcquisition(1000000, 0.0, true);
   REQUIRE(shutter.open == true);
   c.stopSequenceAcquisition();
   CHECK(shutter.open == false);
}

TEST_CASE("Composite with async physicals: same-module shutter not touched "
          "when autoShutter is off",
          "[MultiChannelSequenceAcquisition]") {
   AsyncCamera p0;
   p0.name = "p0";
   AsyncCamera p1;
   p1.name = "p1";
   MockCompositeCamera composite({&p0, &p1});
   StubShutter shutter;
   MockAdapterWithDevices adapter{
      {"p0", &p0}, {"p1", &p1}, {"composite", &composite},
      {"shutter", &shutter}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("composite");
   c.setShutterDevice("shutter");
   c.setAutoShutter(false);

   c.startSequenceAcquisition(1000000, 0.0, true);
   shutter.open = true;
   c.stopSequenceAcquisition();
   CHECK(shutter.open == true);
}

TEST_CASE("Composite: startSequenceAcquisition after all physicals self-finish "
          "without intervening stop succeeds",
          "[MultiChannelSequenceAcquisition]") {
   SyncCamera p0("p0");
   SyncCamera p1("p1");
   MockCompositeCamera composite({&p0, &p1});
   MockAdapterWithDevices adapter{
      {"p0", &p0}, {"p1", &p1}, {"composite", &composite}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("composite");

   c.startSequenceAcquisition(10, 0.0, true);
   p0.TriggerSelfFinish();
   p1.TriggerSelfFinish();
   REQUIRE(c.isSequenceRunning() == false);

   c.startSequenceAcquisition(10, 0.0, true);
   CHECK(c.isSequenceRunning() == true);
   REQUIRE(p0.InsertTestImage() == DEVICE_OK);
   REQUIRE(p1.InsertTestImage() == DEVICE_OK);
   CHECK(c.getRemainingImageCount() == 2);
   c.stopSequenceAcquisition();
}

// === Intrinsic multi-channel camera (TwoPhoton-style) ===

// --- Headline end-to-end ---

TEST_CASE("Sequence acquisition with 2-channel intrinsic camera tags each "
          "frame with CameraChannelName/Index from serialized metadata",
          "[MultiChannelSequenceAcquisition]") {
   MockIntrinsicMultiChannelCamera cam({"chA", "chB"});
   MockAdapterWithDevices adapter{{"intrinsic", &cam}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("intrinsic");

   c.startSequenceAcquisition(4, 0.0, true);
   REQUIRE(cam.InsertTestImage(0) == DEVICE_OK);
   REQUIRE(cam.InsertTestImage(1) == DEVICE_OK);
   REQUIRE(cam.InsertTestImage(0) == DEVICE_OK);
   REQUIRE(cam.InsertTestImage(1) == DEVICE_OK);
   c.stopSequenceAcquisition();

   CHECK(c.getRemainingImageCount() == 4);

   struct Expected {
      const char* channelName;
      const char* channelIndex;
   };
   const Expected expected[] = {
      {"chA", "0"},
      {"chB", "1"},
      {"chA", "0"},
      {"chB", "1"},
   };
   for (const auto& e : expected) {
      Metadata md;
      c.popNextImageMD(md);
      CHECK(md.GetSingleTag(MM::g_Keyword_Metadata_CameraLabel).GetValue() ==
            "intrinsic");
      CHECK(md.GetSingleTag("intrinsic-CameraChannelIndex").GetValue() ==
            e.channelIndex);
      // CameraChannelName is stamped by MMCore from its start-time snapshot
      // even though the device only emits CameraChannelIndex.
      CHECK(md.GetSingleTag("intrinsic-CameraChannelName").GetValue() ==
            e.channelName);
   }
}

// --- Lifecycle ---

TEST_CASE("isSequenceRunning is true while intrinsic multi-channel camera "
          "is acquiring and false after stop",
          "[MultiChannelSequenceAcquisition]") {
   MockIntrinsicMultiChannelCamera cam({"chA", "chB"});
   MockAdapterWithDevices adapter{{"intrinsic", &cam}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("intrinsic");

   CHECK_FALSE(c.isSequenceRunning());
   c.startSequenceAcquisition(10, 0.0, true);
   CHECK(c.isSequenceRunning());
   c.stopSequenceAcquisition();
   CHECK_FALSE(c.isSequenceRunning());
}

TEST_CASE("Stopping intrinsic multi-channel camera clears IsCapturing",
          "[MultiChannelSequenceAcquisition]") {
   MockIntrinsicMultiChannelCamera cam({"chA", "chB"});
   MockAdapterWithDevices adapter{{"intrinsic", &cam}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("intrinsic");

   c.startSequenceAcquisition(10, 0.0, true);
   CHECK(cam.IsCapturing());
   c.stopSequenceAcquisition();
   CHECK_FALSE(cam.IsCapturing());
}

// --- Core channel APIs ---

TEST_CASE("Core reports intrinsic camera's channel count and names",
          "[MultiChannelSequenceAcquisition]") {
   MockIntrinsicMultiChannelCamera cam({"chA", "chB", "chC"});
   MockAdapterWithDevices adapter{{"intrinsic", &cam}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("intrinsic");

   CHECK(c.getNumberOfCameraChannels() == 3);
   CHECK(c.getCameraChannelName(0) == "chA");
   CHECK(c.getCameraChannelName(1) == "chB");
   CHECK(c.getCameraChannelName(2) == "chC");
}

TEST_CASE("Intrinsic multi-channel circular buffer holds frames-times-channels",
          "[MultiChannelSequenceAcquisition]") {
   MockIntrinsicMultiChannelCamera cam({"chA", "chB"});
   MockAdapterWithDevices adapter{{"intrinsic", &cam}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("intrinsic");

   c.startSequenceAcquisition(6, 0.0, true);
   for (int i = 0; i < 3; ++i) {
      REQUIRE(cam.InsertTestImage(0) == DEVICE_OK);
      REQUIRE(cam.InsertTestImage(1) == DEVICE_OK);
   }
   c.stopSequenceAcquisition();

   CHECK(c.getRemainingImageCount() == 6);
}

// --- Continuous and named-camera entry points ---

TEST_CASE("startContinuousSequenceAcquisition with intrinsic multi-channel "
          "camera tags frames and runs until stopped",
          "[MultiChannelSequenceAcquisition]") {
   MockIntrinsicMultiChannelCamera cam({"chA", "chB"});
   MockAdapterWithDevices adapter{{"intrinsic", &cam}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("intrinsic");

   c.startContinuousSequenceAcquisition(0.0);
   CHECK(c.isSequenceRunning());
   REQUIRE(cam.InsertTestImage(0) == DEVICE_OK);
   REQUIRE(cam.InsertTestImage(1) == DEVICE_OK);
   c.stopSequenceAcquisition();
   CHECK_FALSE(c.isSequenceRunning());

   CHECK(c.getRemainingImageCount() == 2);
   {
      Metadata md;
      c.popNextImageMD(md);
      CHECK(md.GetSingleTag("intrinsic-CameraChannelIndex").GetValue() == "0");
   }
   {
      Metadata md;
      c.popNextImageMD(md);
      CHECK(md.GetSingleTag("intrinsic-CameraChannelIndex").GetValue() == "1");
   }
}

TEST_CASE("Named-camera startSequenceAcquisition on intrinsic multi-channel "
          "camera drives the same path as the default overload",
          "[MultiChannelSequenceAcquisition]") {
   MockIntrinsicMultiChannelCamera cam({"chA", "chB"});
   MockAdapterWithDevices adapter{{"intrinsic", &cam}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("intrinsic");

   c.startSequenceAcquisition("intrinsic", 4, 0.0, true);
   CHECK(c.isSequenceRunning("intrinsic"));
   REQUIRE(cam.InsertTestImage(0) == DEVICE_OK);
   REQUIRE(cam.InsertTestImage(1) == DEVICE_OK);
   c.stopSequenceAcquisition("intrinsic");
   CHECK_FALSE(c.isSequenceRunning("intrinsic"));
   CHECK(c.getRemainingImageCount() == 2);
}

// --- Cleanup ---

TEST_CASE("Intrinsic: AcqFinished closes the auto-shutter when "
          "autoShutter is on",
          "[MultiChannelSequenceAcquisition]") {
   MockIntrinsicMultiChannelCamera cam({"chA", "chB"});
   StubShutter shutter;
   MockAdapterWithDevices adapter{
      {"intrinsic", &cam}, {"shutter", &shutter}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("intrinsic");
   c.setShutterDevice("shutter");
   c.setAutoShutter(true);

   c.startSequenceAcquisition(10, 0.0, true);
   REQUIRE(shutter.open == true);
   cam.TriggerSelfFinish();
   CHECK(shutter.open == false);
   c.stopSequenceAcquisition();
}

TEST_CASE("Intrinsic: AcqFinished does not touch shutter when "
          "autoShutter is off",
          "[MultiChannelSequenceAcquisition]") {
   MockIntrinsicMultiChannelCamera cam({"chA", "chB"});
   StubShutter shutter;
   MockAdapterWithDevices adapter{
      {"intrinsic", &cam}, {"shutter", &shutter}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("intrinsic");
   c.setShutterDevice("shutter");
   c.setAutoShutter(false);

   c.startSequenceAcquisition(10, 0.0, true);
   shutter.open = true;
   cam.TriggerSelfFinish();
   CHECK(shutter.open == true);
   c.stopSequenceAcquisition();
}

TEST_CASE("Intrinsic: startSequenceAcquisition after self-finish "
          "without intervening stop succeeds",
          "[MultiChannelSequenceAcquisition]") {
   MockIntrinsicMultiChannelCamera cam({"chA", "chB"});
   MockAdapterWithDevices adapter{{"intrinsic", &cam}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("intrinsic");

   c.startSequenceAcquisition(10, 0.0, true);
   cam.TriggerSelfFinish();
   REQUIRE(c.isSequenceRunning() == false);

   c.startSequenceAcquisition(10, 0.0, true);
   CHECK(c.isSequenceRunning() == true);
   REQUIRE(cam.InsertTestImage(0) == DEVICE_OK);
   REQUIRE(cam.InsertTestImage(1) == DEVICE_OK);
   CHECK(c.getRemainingImageCount() == 2);
   c.stopSequenceAcquisition();
}

// --- Document current MMCore limitations (refactor will revise) ---
//
// These limitations are independent of camera shape (composite vs intrinsic),
// so each is exercised once against each shape via Catch2 SECTIONs.

TEST_CASE("popNextImageMD with channel != 0 throws after multi-channel "
          "acquisition",
          "[MultiChannelSequenceAcquisition]") {
   SECTION("composite camera") {
      SyncCamera p0("p0");
      SyncCamera p1("p1");
      MockCompositeCamera composite({&p0, &p1});
      MockAdapterWithDevices adapter{
         {"p0", &p0}, {"p1", &p1}, {"composite", &composite}};
      CMMCore c;
      adapter.LoadIntoCore(c);
      c.setCameraDevice("composite");

      c.startSequenceAcquisition(2, 0.0, true);
      REQUIRE(p0.InsertTestImage() == DEVICE_OK);
      REQUIRE(p1.InsertTestImage() == DEVICE_OK);
      c.stopSequenceAcquisition();

      Metadata md;
      CHECK_THROWS_AS(c.popNextImageMD(1, 0, md), CMMError);
   }
   SECTION("intrinsic camera") {
      MockIntrinsicMultiChannelCamera cam({"chA", "chB"});
      MockAdapterWithDevices adapter{{"intrinsic", &cam}};
      CMMCore c;
      adapter.LoadIntoCore(c);
      c.setCameraDevice("intrinsic");

      c.startSequenceAcquisition(2, 0.0, true);
      REQUIRE(cam.InsertTestImage(0) == DEVICE_OK);
      REQUIRE(cam.InsertTestImage(1) == DEVICE_OK);
      c.stopSequenceAcquisition();

      Metadata md;
      CHECK_THROWS_AS(c.popNextImageMD(1, 0, md), CMMError);
   }
}

TEST_CASE("getLastImageMD with channel != 0 throws after multi-channel "
          "acquisition",
          "[MultiChannelSequenceAcquisition]") {
   SECTION("composite camera") {
      SyncCamera p0("p0");
      SyncCamera p1("p1");
      MockCompositeCamera composite({&p0, &p1});
      MockAdapterWithDevices adapter{
         {"p0", &p0}, {"p1", &p1}, {"composite", &composite}};
      CMMCore c;
      adapter.LoadIntoCore(c);
      c.setCameraDevice("composite");

      c.startSequenceAcquisition(2, 0.0, true);
      REQUIRE(p0.InsertTestImage() == DEVICE_OK);
      REQUIRE(p1.InsertTestImage() == DEVICE_OK);
      c.stopSequenceAcquisition();

      Metadata md;
      CHECK_THROWS_AS(c.getLastImageMD(1, 0, md), CMMError);
   }
   SECTION("intrinsic camera") {
      MockIntrinsicMultiChannelCamera cam({"chA", "chB"});
      MockAdapterWithDevices adapter{{"intrinsic", &cam}};
      CMMCore c;
      adapter.LoadIntoCore(c);
      c.setCameraDevice("intrinsic");

      c.startSequenceAcquisition(2, 0.0, true);
      REQUIRE(cam.InsertTestImage(0) == DEVICE_OK);
      REQUIRE(cam.InsertTestImage(1) == DEVICE_OK);
      c.stopSequenceAcquisition();

      Metadata md;
      CHECK_THROWS_AS(c.getLastImageMD(1, 0, md), CMMError);
   }
}

// --- Default-channel image retrieval after multi-channel acquisition ---

TEST_CASE("popNextImage (default) returns one pointer per inserted frame "
          "after multi-channel acquisition",
          "[MultiChannelSequenceAcquisition]") {
   SECTION("composite camera") {
      SyncCamera p0("p0");
      SyncCamera p1("p1");
      MockCompositeCamera composite({&p0, &p1});
      MockAdapterWithDevices adapter{
         {"p0", &p0}, {"p1", &p1}, {"composite", &composite}};
      CMMCore c;
      adapter.LoadIntoCore(c);
      c.setCameraDevice("composite");

      c.startSequenceAcquisition(4, 0.0, true);
      REQUIRE(p0.InsertTestImage() == DEVICE_OK);
      REQUIRE(p1.InsertTestImage() == DEVICE_OK);
      REQUIRE(p0.InsertTestImage() == DEVICE_OK);
      REQUIRE(p1.InsertTestImage() == DEVICE_OK);
      c.stopSequenceAcquisition();

      REQUIRE(c.getRemainingImageCount() == 4);
      for (int i = 0; i < 4; ++i)
         CHECK(c.popNextImage() != nullptr);
      CHECK(c.getRemainingImageCount() == 0);
   }
   SECTION("intrinsic camera") {
      MockIntrinsicMultiChannelCamera cam({"chA", "chB"});
      MockAdapterWithDevices adapter{{"intrinsic", &cam}};
      CMMCore c;
      adapter.LoadIntoCore(c);
      c.setCameraDevice("intrinsic");

      c.startSequenceAcquisition(4, 0.0, true);
      REQUIRE(cam.InsertTestImage(0) == DEVICE_OK);
      REQUIRE(cam.InsertTestImage(1) == DEVICE_OK);
      REQUIRE(cam.InsertTestImage(0) == DEVICE_OK);
      REQUIRE(cam.InsertTestImage(1) == DEVICE_OK);
      c.stopSequenceAcquisition();

      REQUIRE(c.getRemainingImageCount() == 4);
      for (int i = 0; i < 4; ++i)
         CHECK(c.popNextImage() != nullptr);
      CHECK(c.getRemainingImageCount() == 0);
   }
}

TEST_CASE("getLastImage returns a non-null pointer "
          "after multi-channel acquisition",
          "[MultiChannelSequenceAcquisition]") {
   SECTION("composite camera") {
      SyncCamera p0("p0");
      SyncCamera p1("p1");
      MockCompositeCamera composite({&p0, &p1});
      MockAdapterWithDevices adapter{
         {"p0", &p0}, {"p1", &p1}, {"composite", &composite}};
      CMMCore c;
      adapter.LoadIntoCore(c);
      c.setCameraDevice("composite");

      c.startSequenceAcquisition(2, 0.0, true);
      REQUIRE(p0.InsertTestImage() == DEVICE_OK);
      REQUIRE(p1.InsertTestImage() == DEVICE_OK);
      c.stopSequenceAcquisition();

      CHECK(c.getLastImage() != nullptr);
   }
   SECTION("intrinsic camera") {
      MockIntrinsicMultiChannelCamera cam({"chA", "chB"});
      MockAdapterWithDevices adapter{{"intrinsic", &cam}};
      CMMCore c;
      adapter.LoadIntoCore(c);
      c.setCameraDevice("intrinsic");

      c.startSequenceAcquisition(2, 0.0, true);
      REQUIRE(cam.InsertTestImage(0) == DEVICE_OK);
      REQUIRE(cam.InsertTestImage(1) == DEVICE_OK);
      c.stopSequenceAcquisition();

      CHECK(c.getLastImage() != nullptr);
   }
}

// --- Buffer overflow during a multi-channel frame set ---

TEST_CASE("Mid-frame buffer overflow with stopOnOverflow=true returns "
          "DEVICE_BUFFER_OVERFLOW from the next channel insert",
          "[MultiChannelSequenceAcquisition]") {
   SECTION("composite camera") {
      SyncCamera p0("p0");
      SyncCamera p1("p1");
      MockCompositeCamera composite({&p0, &p1});
      MockAdapterWithDevices adapter{
         {"p0", &p0}, {"p1", &p1}, {"composite", &composite}};
      CMMCore c;
      adapter.LoadIntoCore(c);
      c.setCameraDevice("composite");
      c.setCircularBufferMemoryFootprint(1);
      c.startSequenceAcquisition(1000, 0.0, true);

      const long total = c.getBufferTotalCapacity();
      // Fill all but the last slot with full frames (alternating physicals).
      for (long i = 0; i < total - 1; ++i) {
         auto& cam = (i % 2 == 0) ? p0 : p1;
         REQUIRE(cam.InsertTestImage() == DEVICE_OK);
      }
      // First channel of next frame fits.
      auto& nextFirst = ((total - 1) % 2 == 0) ? p0 : p1;
      auto& nextSecond = ((total - 1) % 2 == 0) ? p1 : p0;
      REQUIRE(nextFirst.InsertTestImage() == DEVICE_OK);
      // Second channel overflows mid-frame.
      CHECK(nextSecond.InsertTestImage() == DEVICE_BUFFER_OVERFLOW);
      c.stopSequenceAcquisition();
   }
   SECTION("intrinsic camera") {
      MockIntrinsicMultiChannelCamera cam({"chA", "chB"});
      MockAdapterWithDevices adapter{{"intrinsic", &cam}};
      CMMCore c;
      adapter.LoadIntoCore(c);
      c.setCameraDevice("intrinsic");
      c.setCircularBufferMemoryFootprint(1);
      c.startSequenceAcquisition(1000, 0.0, true);

      const long total = c.getBufferTotalCapacity();
      for (long i = 0; i < total - 1; ++i) {
         REQUIRE(cam.InsertTestImage(static_cast<unsigned>(i % 2)) ==
                 DEVICE_OK);
      }
      REQUIRE(cam.InsertTestImage(0) == DEVICE_OK);
      CHECK(cam.InsertTestImage(1) == DEVICE_BUFFER_OVERFLOW);
      c.stopSequenceAcquisition();
   }
}

TEST_CASE("Mid-frame buffer overflow with stopOnOverflow=false overwrites "
          "without error",
          "[MultiChannelSequenceAcquisition]") {
   SECTION("composite camera") {
      SyncCamera p0("p0");
      SyncCamera p1("p1");
      MockCompositeCamera composite({&p0, &p1});
      MockAdapterWithDevices adapter{
         {"p0", &p0}, {"p1", &p1}, {"composite", &composite}};
      CMMCore c;
      adapter.LoadIntoCore(c);
      c.setCameraDevice("composite");
      c.setCircularBufferMemoryFootprint(1);
      c.startSequenceAcquisition(1000, 0.0, false);

      const long total = c.getBufferTotalCapacity();
      for (long i = 0; i < total; ++i) {
         auto& cam = (i % 2 == 0) ? p0 : p1;
         REQUIRE(cam.InsertTestImage() == DEVICE_OK);
      }
      // Past capacity: should still succeed (overwrite).
      CHECK(p0.InsertTestImage() == DEVICE_OK);
      CHECK(p1.InsertTestImage() == DEVICE_OK);
      c.stopSequenceAcquisition();
   }
   SECTION("intrinsic camera") {
      MockIntrinsicMultiChannelCamera cam({"chA", "chB"});
      MockAdapterWithDevices adapter{{"intrinsic", &cam}};
      CMMCore c;
      adapter.LoadIntoCore(c);
      c.setCameraDevice("intrinsic");
      c.setCircularBufferMemoryFootprint(1);
      c.startSequenceAcquisition(1000, 0.0, false);

      const long total = c.getBufferTotalCapacity();
      for (long i = 0; i < total; ++i) {
         REQUIRE(cam.InsertTestImage(static_cast<unsigned>(i % 2)) ==
                 DEVICE_OK);
      }
      CHECK(cam.InsertTestImage(0) == DEVICE_OK);
      CHECK(cam.InsertTestImage(1) == DEVICE_OK);
      c.stopSequenceAcquisition();
   }
}

// --- Nested-multi-channel rejection ---

TEST_CASE("Composite camera whose phys cam is itself multi-channel is "
          "rejected at start time",
          "[MultiChannelSequenceAcquisition]") {
   SyncCamera p0("p0");
   SyncCamera p1("p1");
   MockCompositeCamera inner({&p0, &p1});
   MockCompositeCamera outer({&inner});
   MockAdapterWithDevices adapter{
      {"p0", &p0}, {"p1", &p1}, {"inner", &inner}, {"outer", &outer}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("outer");

   CHECK_THROWS_AS(c.startSequenceAcquisition(10, 0.0, true), CMMError);
   CHECK_FALSE(inner.IsCapturing());
}

// --- Single-channel composite ---

TEST_CASE("Single-channel composite (one phys cam) tags its frames with "
          "channel index 0 and the phys cam's channel name",
          "[MultiChannelSequenceAcquisition]") {
   SyncCamera p0("p0");
   MockCompositeCamera composite({&p0});
   MockAdapterWithDevices adapter{
      {"p0", &p0}, {"composite", &composite}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("composite");

   c.startSequenceAcquisition(1, 0.0, true);
   REQUIRE(p0.InsertTestImage() == DEVICE_OK);
   c.stopSequenceAcquisition();

   REQUIRE(c.getRemainingImageCount() == 1);
   Metadata md;
   c.popNextImageMD(md);
   CHECK(md.GetSingleTag("composite-CameraChannelIndex").GetValue() == "0");
   CHECK(md.GetSingleTag("composite-CameraChannelName").GetValue() == "p0");
}

// --- Plain single-channel camera, no opt-in ---

TEST_CASE("Plain single-channel camera without CameraChannelIndex tag in "
          "device metadata gets no composite-prefixed tags",
          "[MultiChannelSequenceAcquisition]") {
   SyncCamera cam("cam");
   MockAdapterWithDevices adapter{{"cam", &cam}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("cam");

   c.startSequenceAcquisition(1, 0.0, true);
   REQUIRE(cam.InsertTestImage() == DEVICE_OK);
   c.stopSequenceAcquisition();

   REQUIRE(c.getRemainingImageCount() == 1);
   Metadata md;
   c.popNextImageMD(md);
   CHECK_THROWS(md.GetSingleTag("cam-CameraChannelIndex"));
   CHECK_THROWS(md.GetSingleTag("cam-CameraChannelName"));
}

// --- Intrinsic emits invalid index ---

TEST_CASE("Intrinsic multi-channel device emitting out-of-range "
          "CameraChannelIndex has its image rejected",
          "[MultiChannelSequenceAcquisition]") {
   MockIntrinsicMultiChannelCamera cam({"chA", "chB"});
   MockAdapterWithDevices adapter{{"intrinsic", &cam}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("intrinsic");

   c.startSequenceAcquisition(10, 0.0, true);
   CHECK(cam.InsertTestImage(99) == DEVICE_INCOMPATIBLE_IMAGE);
   c.stopSequenceAcquisition();
}
