#include <catch2/catch_all.hpp>

#include "MMCore.h"
#include "MockDeviceUtils.h"
#include "StubDevices.h"

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

namespace {

// Camera that produces images on its own thread between Start and Stop.
struct AsyncCamera : CCameraBase<AsyncCamera> {
   std::string name = "AsyncCamera";
   unsigned width = 64;
   unsigned height = 64;
   unsigned bytesPerPixel = 1;
   unsigned nComponents = 1;
   unsigned bitDepth = 8;
   int binning = 1;
   double exposure = 10.0;

   int Initialize() override { return DEVICE_OK; }
   int Shutdown() override { return DEVICE_OK; }
   bool Busy() override { return false; }
   void GetName(char* buf) const override {
      CDeviceUtils::CopyLimitedString(buf, name.c_str());
   }

   int SnapImage() override {
      imgBuf_.assign(
         static_cast<size_t>(width) * height * bytesPerPixel, 0);
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

   int StartSequenceAcquisition(long numImages, double, bool) override {
      GetCoreCallback()->PrepareForAcq(this);
      {
         std::lock_guard<std::mutex> lk(mu_);
         running_ = true;
         stopRequested_ = false;
      }
      thread_ = std::thread([this, numImages] { AcqThread(numImages); });
      return DEVICE_OK;
   }

   int StartSequenceAcquisition(double) override {
      GetCoreCallback()->PrepareForAcq(this);
      {
         std::lock_guard<std::mutex> lk(mu_);
         running_ = true;
         stopRequested_ = false;
      }
      thread_ = std::thread([this] { AcqThread(-1); });
      return DEVICE_OK;
   }

   ~AsyncCamera() {
      {
         std::lock_guard<std::mutex> lk(mu_);
         stopRequested_ = true;
      }
      cv_.notify_one();
      if (thread_.joinable())
         thread_.join();
   }

   int StopSequenceAcquisition() override {
      {
         std::lock_guard<std::mutex> lk(mu_);
         stopRequested_ = true;
      }
      cv_.notify_one();
      if (thread_.joinable())
         thread_.join();
      return DEVICE_OK;
   }

   bool IsCapturing() override {
      std::lock_guard<std::mutex> lk(mu_);
      return running_;
   }

private:
   void AcqThread(long numImages) {
      std::vector<unsigned char> buf(
         static_cast<size_t>(width) * height * bytesPerPixel, 0);
      long count = 0;
      while (numImages < 0 || count < numImages) {
         {
            std::lock_guard<std::mutex> lk(mu_);
            if (stopRequested_)
               break;
         }
         GetCoreCallback()->InsertImage(this, buf.data(),
            width, height, bytesPerPixel, nComponents, "{}");
         ++count;
         {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait_for(lk, std::chrono::microseconds(100),
               [this] { return stopRequested_; });
            if (stopRequested_)
               break;
         }
      }
      GetCoreCallback()->AcqFinished(this, 0);
      {
         std::lock_guard<std::mutex> lk(mu_);
         running_ = false;
      }
   }

   bool running_ = false;
   bool stopRequested_ = false;
   std::mutex mu_;
   std::condition_variable cv_;
   std::thread thread_;
   std::vector<unsigned char> imgBuf_;
};

} // namespace

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

TEST_CASE("unloadAllDevices during sequence acquisition does not crash",
          "[Unload]") {
   AsyncCamera cam;
   MockAdapterWithDevices adapter{{"cam", &cam}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("cam");
   c.startContinuousSequenceAcquisition(0.0);
   CHECK_NOTHROW(c.unloadAllDevices());
}

TEST_CASE("unloadDevice during sequence acquisition does not crash",
          "[Unload]") {
   AsyncCamera cam;
   MockAdapterWithDevices adapter{{"cam", &cam}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("cam");
   c.startContinuousSequenceAcquisition(0.0);
   CHECK_NOTHROW(c.unloadDevice("cam"));
}
