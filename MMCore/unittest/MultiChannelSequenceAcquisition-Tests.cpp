#include <catch2/catch_all.hpp>

#include "MMCore.h"
#include "DeviceBase.h"
#include "ImageMetadata.h"
#include "MMDeviceConstants.h"
#include "MockDeviceUtils.h"
#include "StubDevices.h"

#include <string>
#include <utility>
#include <vector>

namespace {

// Minimal mock of a composite multi-channel camera (in the style of the
// Multi Camera (Utilities) adapter, where multiple physical cameras are
// presented as channels of a single device — as opposed to an intrinsic
// multi-channel camera, which is a single device that itself emits multiple
// channels). Holds its physical cameras as direct pointers given at
// construction.
// Mirrors the real adapter's behaviors that matter for MMCore-side testing:
//   - Reports GetNumberOfChannels() == number of physicals.
//   - Calls AddTag on each physical with its own label as deviceLabel.
//   - Forwards Start/StopSequenceAcquisition to each physical.
//   - IsCapturing() reflects whether any physical is capturing.
struct MockCompositeCamera : CCameraBase<MockCompositeCamera> {
   std::string name = "MockCompositeCamera";
   std::vector<SyncCamera*> physicals;

   explicit MockCompositeCamera(std::vector<SyncCamera*> p)
      : physicals(std::move(p)) {}

   int Initialize() override {
      char myLabel[MM::MaxStrLength];
      GetLabel(myLabel);
      for (size_t i = 0; i < physicals.size(); ++i) {
         physicals[i]->AddTag(MM::g_Keyword_CameraChannelName,
            myLabel, physicals[i]->name.c_str());
         physicals[i]->AddTag(MM::g_Keyword_CameraChannelIndex,
            myLabel, std::to_string(i).c_str());
      }
      return DEVICE_OK;
   }
   int Shutdown() override { return DEVICE_OK; }
   bool Busy() override { return false; }
   void GetName(char* buf) const override {
      CDeviceUtils::CopyLimitedString(buf, name.c_str());
   }

   int SnapImage() override { return DEVICE_OK; }
   const unsigned char* GetImageBuffer() override {
      return physicals.empty() ? nullptr : physicals[0]->GetImageBuffer();
   }
   long GetImageBufferSize() const override {
      return physicals.empty() ? 0 : physicals[0]->GetImageBufferSize();
   }
   unsigned GetImageWidth() const override {
      return physicals.empty() ? 0 : physicals[0]->GetImageWidth();
   }
   unsigned GetImageHeight() const override {
      return physicals.empty() ? 0 : physicals[0]->GetImageHeight();
   }
   unsigned GetImageBytesPerPixel() const override {
      return physicals.empty() ? 0 : physicals[0]->GetImageBytesPerPixel();
   }
   unsigned GetNumberOfComponents() const override { return 1; }
   unsigned GetNumberOfChannels() const override {
      return static_cast<unsigned>(physicals.size());
   }
   int GetChannelName(unsigned channel, char* chName) override {
      if (channel < physicals.size()) {
         CDeviceUtils::CopyLimitedString(chName,
            physicals[channel]->name.c_str());
      } else {
         CDeviceUtils::CopyLimitedString(chName, "");
      }
      return DEVICE_OK;
   }
   unsigned GetBitDepth() const override {
      return physicals.empty() ? 8 : physicals[0]->GetBitDepth();
   }
   int GetBinning() const override {
      return physicals.empty() ? 1 : physicals[0]->GetBinning();
   }
   int SetBinning(int b) override {
      for (auto* p : physicals) p->SetBinning(b);
      return DEVICE_OK;
   }
   void SetExposure(double e) override {
      for (auto* p : physicals) p->SetExposure(e);
   }
   double GetExposure() const override {
      return physicals.empty() ? 0.0 : physicals[0]->GetExposure();
   }
   int SetROI(unsigned, unsigned, unsigned, unsigned) override {
      return DEVICE_OK;
   }
   int GetROI(unsigned& x, unsigned& y, unsigned& w, unsigned& h) override {
      x = 0; y = 0;
      w = GetImageWidth();
      h = GetImageHeight();
      return DEVICE_OK;
   }
   int ClearROI() override { return DEVICE_OK; }
   int IsExposureSequenceable(bool& seq) const override {
      seq = false;
      return DEVICE_OK;
   }

   int StartSequenceAcquisition(long n, double i, bool s) override {
      for (auto* p : physicals) {
         int ret = p->StartSequenceAcquisition(n, i, s);
         if (ret != DEVICE_OK) return ret;
      }
      return DEVICE_OK;
   }
   int StartSequenceAcquisition(double i) override {
      for (auto* p : physicals) {
         int ret = p->StartSequenceAcquisition(i);
         if (ret != DEVICE_OK) return ret;
      }
      return DEVICE_OK;
   }
   int StopSequenceAcquisition() override {
      for (auto* p : physicals) {
         int ret = p->StopSequenceAcquisition();
         if (ret != DEVICE_OK) return ret;
      }
      return DEVICE_OK;
   }
   bool IsCapturing() override {
      for (auto* p : physicals) {
         if (p->IsCapturing()) return true;
      }
      return false;
   }
};

// Minimal mock of an intrinsic multi-channel camera (in the style of the
// TwoPhoton adapter): a single device that itself emits N channels by calling
// InsertImage once per channel per frame. Channel-identifying tags are
// embedded in the serialized metadata passed to InsertImage, not stored on
// the device via AddTag (which is the composite pattern). Tests drive
// InsertTestFrame() manually after StartSequenceAcquisition.
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

   // Inserts a single channel of a multi-channel frame, with channel-
   // identifying tags in the serialized metadata. Tests call this once per
   // channel per frame; the order is up to the test (real intrinsic adapters
   // may interleave channels).
   //
   // The dual unprefixed + label-prefixed tag format mirrors the current
   // behavior of the only in-tree intrinsic multi-chan camera (TwoPhoton,
   // which is unmaintained) and may need updating if/when the tag rules are
   // clarified.
   int InsertTestImage(unsigned channel) {
      char label[MM::MaxStrLength];
      GetLabel(label);
      const std::string labelStr = label;
      std::vector<unsigned char> buf(
         static_cast<size_t>(width) * height * bytesPerPixel, 0);
      MM::CameraImageMetadata md;
      const std::string idx = std::to_string(channel);
      md.AddTag(MM::g_Keyword_CameraChannelIndex, idx);
      md.AddTag(labelStr + '-' + MM::g_Keyword_CameraChannelIndex, idx);
      md.AddTag(MM::g_Keyword_CameraChannelName, channelNames[channel]);
      md.AddTag(labelStr + '-' + MM::g_Keyword_CameraChannelName,
         channelNames[channel]);
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

// --- Tag attachment ---

TEST_CASE("Composite camera Initialize attaches CameraChannelName/Index tags "
          "to physicals",
          "[MultiChannelSequenceAcquisition]") {
   SyncCamera p0("p0");
   SyncCamera p1("p1");
   MockCompositeCamera composite({&p0, &p1});
   MockAdapterWithDevices adapter{
      {"p0", &p0}, {"p1", &p1}, {"composite", &composite}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("composite");
   c.initializeCircularBuffer();

   REQUIRE(p0.InsertTestImage() == DEVICE_OK);
   {
      Metadata md;
      c.popNextImageMD(md);
      CHECK(md.GetSingleTag("composite-CameraChannelName").GetValue() == "p0");
      CHECK(md.GetSingleTag("composite-CameraChannelIndex").GetValue() == "0");
   }

   REQUIRE(p1.InsertTestImage() == DEVICE_OK);
   {
      Metadata md;
      c.popNextImageMD(md);
      CHECK(md.GetSingleTag("composite-CameraChannelName").GetValue() == "p1");
      CHECK(md.GetSingleTag("composite-CameraChannelIndex").GetValue() == "1");
   }
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

TEST_CASE("Composite: each physical's AcqFinished re-closes the auto-shutter",
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
   CHECK(shutter.open == false);

   // This might be a little over-constraining; the important thing is that the
   // shutter gets closed at least once. But we can't easily write the correct
   // test until we have the correct behavior (close shutter when the _last_
   // phys cam finishes).
   shutter.open = true;
   p1.TriggerSelfFinish();
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
      CHECK(md.GetSingleTag(MM::g_Keyword_CameraChannelName).GetValue() ==
            e.channelName);
      CHECK(md.GetSingleTag(MM::g_Keyword_CameraChannelIndex).GetValue() ==
            e.channelIndex);
      CHECK(md.GetSingleTag("intrinsic-CameraChannelName").GetValue() ==
            e.channelName);
      CHECK(md.GetSingleTag("intrinsic-CameraChannelIndex").GetValue() ==
            e.channelIndex);
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
