#include <catch2/catch_all.hpp>

#include "DeviceBase.h"
#include "MMCore.h"
#include "MMDeviceConstants.h"
#include "MockDeviceUtils.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

bool TestPassed(const nlohmann::json& results, const std::string& testName) {
   for (const auto& t : results["tests"]) {
      if (t["name"] == testName)
         return t["passed"].get<bool>();
   }
   throw std::runtime_error("Test not found: " + testName);
}

struct ConfigurableAsyncCamera : CCameraBase<ConfigurableAsyncCamera> {
   std::string name = "ConfigurableAsyncCamera";
   unsigned width = 64;
   unsigned height = 64;
   unsigned bytesPerPixel = 1;
   unsigned nComponents = 1;
   unsigned bitDepth = 8;
   int binning = 1;
   double exposure = 10.0;

   bool callPrepareForAcq = true;
   bool callAcqFinished = true;
   bool checkInsertImageReturn = true;

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
   const unsigned char* GetImageBuffer() override {
      return imgBuf_.data();
   }
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
      // Thread may be left over from previous (unstopped) run
      if (thread_.joinable())
         thread_.join();
      if (callPrepareForAcq)
         GetCoreCallback()->PrepareForAcq(this);
      {
         std::lock_guard<std::mutex> lk(mu_);
         running_ = true;
         stopRequested_ = false;
      }
      thread_ = std::thread([this, numImages] {
         AcqThread(numImages);
      });
      return DEVICE_OK;
   }

   int StartSequenceAcquisition(double intervalMs) override {
      return StartSequenceAcquisition(-1, intervalMs, false);
   }

   ~ConfigurableAsyncCamera() {
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
         int ret = GetCoreCallback()->InsertImage(this, buf.data(),
            width, height, bytesPerPixel, nComponents);
         if (checkInsertImageReturn && ret != DEVICE_OK)
            break;
         ++count;
         {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait_for(lk, std::chrono::microseconds(100),
               [this] { return stopRequested_; });
            if (stopRequested_)
               break;
         }
      }
      if (callAcqFinished)
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

} // anonymous namespace

TEST_CASE("Conformant camera passes all conformance tests",
          "[CameraConformance]") {
   ConfigurableAsyncCamera cam;
   MockAdapterWithDevices adapter{{"cam", &cam}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("cam");

   auto results = nlohmann::json::parse(c.runCameraConformanceTests("cam"));
   for (const auto& test : results["tests"]) {
      INFO("Test: " << test["name"].get<std::string>());
      CHECK(test["passed"].get<bool>());
   }
   CHECK(results["summary"]["passed"].get<int>() ==
         results["summary"]["total"].get<int>());
}

TEST_CASE("Missing PrepareForAcq is detected by conformance test",
          "[CameraConformance]") {
   ConfigurableAsyncCamera cam;
   cam.callPrepareForAcq = false;
   MockAdapterWithDevices adapter{{"cam", &cam}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("cam");

   auto results = nlohmann::json::parse(
      c.runCameraConformanceTests("cam", "seq-prepare-before-insert"));
   CHECK_FALSE(TestPassed(results, "seq-prepare-before-insert"));
}

TEST_CASE("Missing AcqFinished is detected by conformance test",
          "[CameraConformance]") {
   ConfigurableAsyncCamera cam;
   cam.callAcqFinished = false;
   MockAdapterWithDevices adapter{{"cam", &cam}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("cam");

   auto results = nlohmann::json::parse(
      c.runCameraConformanceTests("cam", "seq-finished-after-count"));
   CHECK_FALSE(TestPassed(results, "seq-finished-after-count"));
}

TEST_CASE("Ignoring InsertImage error return is detected by conformance test",
          "[CameraConformance]") {
   ConfigurableAsyncCamera cam;
   cam.checkInsertImageReturn = false;
   MockAdapterWithDevices adapter{{"cam", &cam}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("cam");

   auto results = nlohmann::json::parse(
      c.runCameraConformanceTests("cam", "seq-finished-on-error-finite"));
   CHECK_FALSE(TestPassed(results, "seq-finished-on-error-finite"));
}
