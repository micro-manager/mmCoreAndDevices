#include <catch2/catch_all.hpp>

#include "MMCore.h"
#include "MMDeviceConstants.h"
#include "MockDeviceUtils.h"
#include "StubDevices.h"

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

// A camera that tracks its capturing state synchronously. The device never
// produces images on its own; the test drives InsertTestImage() manually.
struct SyncCamera : CCameraBase<SyncCamera> {
   std::string name = "SyncCamera";
   unsigned width = 512;
   unsigned height = 512;
   unsigned bytesPerPixel = 1;
   unsigned nComponents = 1;
   unsigned bitDepth = 8;
   int binning = 1;
   double exposure = 10.0;
   bool capturing_ = false;

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
      capturing_ = false;
      GetCoreCallback()->AcqFinished(this, 0);
      return DEVICE_OK;
   }
   bool IsCapturing() override { return capturing_; }

   int InsertTestImage(const unsigned char* pixels = nullptr) {
      std::vector<unsigned char> defaultBuf;
      const unsigned char* buf = pixels;
      if (!buf) {
         defaultBuf.assign(
            static_cast<size_t>(width) * height * bytesPerPixel, 0);
         buf = defaultBuf.data();
      }
      return GetCoreCallback()->InsertImage(this, buf,
         width, height, bytesPerPixel, nComponents, "{}");
   }

private:
   std::vector<unsigned char> imgBuf_;
};

// A camera that produces images asynchronously on its own thread.
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

   int StartSequenceAcquisition(double) override {
      GetCoreCallback()->PrepareForAcq(this);
      {
         std::lock_guard<std::mutex> lk(mu_);
         running_ = true;
         stopRequested_ = false;
      }
      thread_ = std::thread([this] {
         AcqThread(-1);
      });
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

// --- Lifecycle error handling ---

TEST_CASE("startSequenceAcquisition throws when no camera set",
          "[SequenceAcquisition]") {
   CMMCore c;
   CHECK_THROWS_AS(c.startSequenceAcquisition(10, 0.0, true), CMMError);
}

TEST_CASE("startSequenceAcquisition throws when already capturing",
          "[SequenceAcquisition]") {
   SyncCamera cam;
   MockAdapterWithDevices adapter{{"cam", &cam}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("cam");
   c.startSequenceAcquisition(10, 0.0, true);
   REQUIRE(c.isSequenceRunning());
   CHECK_THROWS_AS(c.startSequenceAcquisition(10, 0.0, true), CMMError);
   c.stopSequenceAcquisition();
}

TEST_CASE("startContinuousSequenceAcquisition throws when no camera set",
          "[SequenceAcquisition]") {
   CMMCore c;
   CHECK_THROWS_AS(c.startContinuousSequenceAcquisition(0.0), CMMError);
}

TEST_CASE("startContinuousSequenceAcquisition throws when already capturing",
          "[SequenceAcquisition]") {
   SyncCamera cam;
   MockAdapterWithDevices adapter{{"cam", &cam}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("cam");
   c.startContinuousSequenceAcquisition(0.0);
   REQUIRE(c.isSequenceRunning());
   CHECK_THROWS_AS(c.startContinuousSequenceAcquisition(0.0), CMMError);
   c.stopSequenceAcquisition();
}

TEST_CASE("stopSequenceAcquisition (default) throws when no camera set",
          "[SequenceAcquisition]") {
   CMMCore c;
   CHECK_THROWS_AS(c.stopSequenceAcquisition(), CMMError);
}

TEST_CASE("stopSequenceAcquisition (by label) on non-existent label throws",
          "[SequenceAcquisition]") {
   CMMCore c;
   CHECK_THROWS_AS(c.stopSequenceAcquisition("noSuchCamera"), CMMError);
}

TEST_CASE("isSequenceRunning (default) returns false when no camera set",
          "[SequenceAcquisition]") {
   CMMCore c;
   CHECK(c.isSequenceRunning() == false);
}

TEST_CASE("isSequenceRunning (default) tracks acquisition lifecycle",
          "[SequenceAcquisition]") {
   SyncCamera cam;
   MockAdapterWithDevices adapter{{"cam", &cam}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("cam");
   CHECK(c.isSequenceRunning() == false);
   c.startSequenceAcquisition(10, 0.0, true);
   CHECK(c.isSequenceRunning() == true);
   c.stopSequenceAcquisition();
   CHECK(c.isSequenceRunning() == false);
}

TEST_CASE("isSequenceRunning (by label) tracks acquisition lifecycle",
          "[SequenceAcquisition]") {
   SyncCamera cam;
   MockAdapterWithDevices adapter{{"cam", &cam}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("cam");
   CHECK(c.isSequenceRunning("cam") == false);
   c.startSequenceAcquisition(10, 0.0, true);
   CHECK(c.isSequenceRunning("cam") == true);
   c.stopSequenceAcquisition();
   CHECK(c.isSequenceRunning("cam") == false);
}

// --- Buffer initialization side effects ---

TEST_CASE("startSequenceAcquisition clears pre-existing images from buffer",
          "[SequenceAcquisition]") {
   SyncCamera cam;
   MockAdapterWithDevices adapter{{"cam", &cam}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("cam");
   c.initializeCircularBuffer();
   REQUIRE(cam.InsertTestImage() == DEVICE_OK);
   REQUIRE(cam.InsertTestImage() == DEVICE_OK);
   REQUIRE(c.getRemainingImageCount() == 2);
   c.startSequenceAcquisition(10, 0.0, true);
   CHECK(c.getRemainingImageCount() == 0);
   c.stopSequenceAcquisition();
}

TEST_CASE("startSequenceAcquisition with stopOnOverflow=true disables overwrite",
          "[SequenceAcquisition]") {
   SyncCamera cam;
   MockAdapterWithDevices adapter{{"cam", &cam}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("cam");
   c.setCircularBufferMemoryFootprint(1);
   c.startSequenceAcquisition(100, 0.0, true);

   long total = c.getBufferTotalCapacity();
   for (long i = 0; i < total; ++i)
      REQUIRE(cam.InsertTestImage() == DEVICE_OK);
   CHECK(cam.InsertTestImage() == DEVICE_BUFFER_OVERFLOW);
   c.stopSequenceAcquisition();
}

TEST_CASE("startSequenceAcquisition with stopOnOverflow=false enables overwrite",
          "[SequenceAcquisition]") {
   SyncCamera cam;
   MockAdapterWithDevices adapter{{"cam", &cam}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("cam");
   c.setCircularBufferMemoryFootprint(1);
   c.startSequenceAcquisition(100, 0.0, false);

   long total = c.getBufferTotalCapacity();
   for (long i = 0; i < total; ++i)
      REQUIRE(cam.InsertTestImage() == DEVICE_OK);
   CHECK(cam.InsertTestImage() == DEVICE_OK);
   c.stopSequenceAcquisition();
}

TEST_CASE("startContinuousSequenceAcquisition always enables overwrite",
          "[SequenceAcquisition]") {
   SyncCamera cam;
   MockAdapterWithDevices adapter{{"cam", &cam}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("cam");
   c.setCircularBufferMemoryFootprint(1);
   c.startContinuousSequenceAcquisition(0.0);

   long total = c.getBufferTotalCapacity();
   for (long i = 0; i < total; ++i)
      REQUIRE(cam.InsertTestImage() == DEVICE_OK);
   CHECK(cam.InsertTestImage() == DEVICE_OK);
   c.stopSequenceAcquisition();
}

TEST_CASE("Named-camera startSequenceAcquisition initializes and clears buffer",
          "[SequenceAcquisition]") {
   SyncCamera cam;
   MockAdapterWithDevices adapter{{"cam", &cam}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("cam");
   c.initializeCircularBuffer();
   REQUIRE(cam.InsertTestImage() == DEVICE_OK);
   REQUIRE(c.getRemainingImageCount() == 1);
   c.startSequenceAcquisition("cam", 10, 0.0, true);
   CHECK(c.getRemainingImageCount() == 0);
   c.stopSequenceAcquisition("cam");
}

// --- Auto-shutter ---

TEST_CASE("Shutter opens on startSequenceAcquisition when autoShutter is on",
          "[SequenceAcquisition]") {
   SyncCamera cam;
   StubShutter shutter;
   MockAdapterWithDevices adapter{{"cam", &cam}, {"shutter", &shutter}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("cam");
   c.setShutterDevice("shutter");
   c.setAutoShutter(true);
   REQUIRE(shutter.open == false);
   c.startSequenceAcquisition(10, 0.0, true);
   CHECK(shutter.open == true);
   c.stopSequenceAcquisition();
}

TEST_CASE("Shutter closes on stopSequenceAcquisition when autoShutter is on",
          "[SequenceAcquisition]") {
   SyncCamera cam;
   StubShutter shutter;
   MockAdapterWithDevices adapter{{"cam", &cam}, {"shutter", &shutter}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("cam");
   c.setShutterDevice("shutter");
   c.setAutoShutter(true);
   c.startSequenceAcquisition(10, 0.0, true);
   REQUIRE(shutter.open == true);
   c.stopSequenceAcquisition();
   CHECK(shutter.open == false);
}

TEST_CASE("Shutter not opened on start when autoShutter is off",
          "[SequenceAcquisition]") {
   SyncCamera cam;
   StubShutter shutter;
   MockAdapterWithDevices adapter{{"cam", &cam}, {"shutter", &shutter}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("cam");
   c.setShutterDevice("shutter");
   c.setAutoShutter(false);
   REQUIRE(shutter.open == false);
   c.startSequenceAcquisition(10, 0.0, true);
   CHECK(shutter.open == false);
   c.stopSequenceAcquisition();
}

TEST_CASE("Shutter not closed on stop when autoShutter is off",
          "[SequenceAcquisition]") {
   SyncCamera cam;
   StubShutter shutter;
   MockAdapterWithDevices adapter{{"cam", &cam}, {"shutter", &shutter}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("cam");
   c.setShutterDevice("shutter");
   c.setAutoShutter(false);
   c.startSequenceAcquisition(10, 0.0, true);
   shutter.open = true;
   c.stopSequenceAcquisition();
   CHECK(shutter.open == true);
}

// --- End-to-end async acquisition ---

TEST_CASE("Finite acquisition produces expected number of images",
          "[SequenceAcquisition]") {
   AsyncCamera cam;
   MockAdapterWithDevices adapter{{"cam", &cam}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("cam");
   c.setCircularBufferMemoryFootprint(1);

   const long numImages = 5;
   c.startSequenceAcquisition(numImages, 0.0, true);
   CHECK(c.isSequenceRunning() == true);

   auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
   while (c.isSequenceRunning() &&
          std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
   }
   CHECK(c.isSequenceRunning() == false);
   CHECK(c.getRemainingImageCount() == numImages);
   c.stopSequenceAcquisition();
}

TEST_CASE("Continuous acquisition accumulates images and stops on request",
          "[SequenceAcquisition]") {
   AsyncCamera cam;
   MockAdapterWithDevices adapter{{"cam", &cam}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("cam");

   c.startContinuousSequenceAcquisition(0.0);
   CHECK(c.isSequenceRunning() == true);

   auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
   while (c.getRemainingImageCount() < 3 &&
          std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
   }

   c.stopSequenceAcquisition();
   CHECK(c.isSequenceRunning() == false);
   CHECK(c.getRemainingImageCount() >= 3);
}

TEST_CASE("Images retrieved via popNextImage after acquisition completes",
          "[SequenceAcquisition]") {
   AsyncCamera cam;
   MockAdapterWithDevices adapter{{"cam", &cam}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("cam");

   const long numImages = 3;
   c.startSequenceAcquisition(numImages, 0.0, true);

   auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
   while (c.isSequenceRunning() &&
          std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
   }
   REQUIRE(c.isSequenceRunning() == false);

   for (long i = 0; i < numImages; ++i) {
      CHECK(c.popNextImage() != nullptr);
   }
   CHECK(c.getRemainingImageCount() == 0);
   c.stopSequenceAcquisition();
}

TEST_CASE("stopSequenceAcquisition on finite acquisition stops it early",
          "[SequenceAcquisition]") {
   AsyncCamera cam;
   MockAdapterWithDevices adapter{{"cam", &cam}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("cam");

   c.startSequenceAcquisition(1000000, 0.0, true);
   REQUIRE(c.isSequenceRunning() == true);
   c.stopSequenceAcquisition();
   CHECK(c.isSequenceRunning() == false);
   CHECK(c.getRemainingImageCount() < 1000000);
}
