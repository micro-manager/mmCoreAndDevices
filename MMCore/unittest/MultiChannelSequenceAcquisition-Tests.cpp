#include <catch2/catch_all.hpp>

#include "MMCore.h"
#include "DeviceBase.h"
#include "ImageMetadata.h"
#include "MMDeviceConstants.h"
#include "MockDeviceUtils.h"

#include <string>
#include <utility>
#include <vector>

namespace {

// Grayscale stub camera modeled on SyncCamera in SequenceAcquisition-Tests.cpp.
// Single-channel via the inherited default GetNumberOfChannels() == 1.
// Tests drive InsertTestImage() manually after StartSequenceAcquisition.
struct SyncPhysicalCamera : CCameraBase<SyncPhysicalCamera> {
   std::string name;
   unsigned width = 64;
   unsigned height = 64;
   unsigned bytesPerPixel = 1;
   unsigned nComponents = 1;
   unsigned bitDepth = 8;
   int binning = 1;
   double exposure = 10.0;
   bool capturing = false;

   explicit SyncPhysicalCamera(std::string n) : name(std::move(n)) {}

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
   unsigned GetNumberOfComponents() const override { return nComponents; }
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
      capturing = true;
      GetCoreCallback()->PrepareForAcq(this);
      return DEVICE_OK;
   }
   int StartSequenceAcquisition(double) override {
      capturing = true;
      GetCoreCallback()->PrepareForAcq(this);
      return DEVICE_OK;
   }
   int StopSequenceAcquisition() override {
      capturing = false;
      GetCoreCallback()->AcqFinished(this, 0);
      return DEVICE_OK;
   }
   bool IsCapturing() override { return capturing; }

   int InsertTestImage() {
      std::vector<unsigned char> buf(
         static_cast<size_t>(width) * height * bytesPerPixel, 0);
      return GetCoreCallback()->InsertImage(this, buf.data(),
         width, height, bytesPerPixel, nComponents, "{}");
   }

private:
   std::vector<unsigned char> imgBuf_;
};

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
   std::vector<SyncPhysicalCamera*> physicals;

   explicit MockCompositeCamera(std::vector<SyncPhysicalCamera*> p)
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

} // namespace

// --- Headline end-to-end ---

TEST_CASE("Sequence acquisition with 2-channel composite camera tags each "
          "frame with its physical's CameraChannelName/Index",
          "[MultiChannelSequenceAcquisition]") {
   SyncPhysicalCamera p0("p0");
   SyncPhysicalCamera p1("p1");
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
   SyncPhysicalCamera p0("p0");
   SyncPhysicalCamera p1("p1");
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
   SyncPhysicalCamera p0("p0");
   SyncPhysicalCamera p1("p1");
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
   SyncPhysicalCamera p0("p0");
   SyncPhysicalCamera p1("p1");
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
