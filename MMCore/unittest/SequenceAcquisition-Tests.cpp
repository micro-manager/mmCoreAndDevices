#include <catch2/catch_all.hpp>

#include "MMCore.h"
#include "MMDeviceConstants.h"
#include "MockDeviceUtils.h"
#include "StubDevices.h"

#include <chrono>
#include <thread>
#include <vector>

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

   c.startSequenceAcquisition(10, 0.0, true);
   REQUIRE(cam.InsertTestImage() == DEVICE_OK);
   REQUIRE(cam.InsertTestImage() == DEVICE_OK);
   REQUIRE(c.getRemainingImageCount() == 2);
   c.stopSequenceAcquisition();
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

   c.startSequenceAcquisition("cam", 10, 0.0, true);
   REQUIRE(cam.InsertTestImage() == DEVICE_OK);
   REQUIRE(c.getRemainingImageCount() == 1);
   c.stopSequenceAcquisition("cam");

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
   c.stopSequenceAcquisition();
}

// --- Cleanup ---

TEST_CASE("popNextImage after stopSequenceAcquisition returns inserted images",
          "[SequenceAcquisition]") {
   SyncCamera cam;
   MockAdapterWithDevices adapter{{"cam", &cam}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("cam");

   const int numImages = 3;
   c.startSequenceAcquisition(numImages, 0.0, true);
   for (int i = 0; i < numImages; ++i)
      REQUIRE(cam.InsertTestImage() == DEVICE_OK);
   REQUIRE(c.getRemainingImageCount() == numImages);
   c.stopSequenceAcquisition();
   CHECK(c.getRemainingImageCount() == numImages);
   for (int i = 0; i < numImages; ++i)
      CHECK(c.popNextImage() != nullptr);
   CHECK(c.getRemainingImageCount() == 0);
}

TEST_CASE("Camera self-finish closes shutter when autoShutter is on",
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
   cam.TriggerSelfFinish();
   CHECK(shutter.open == false);
   c.stopSequenceAcquisition();
}

TEST_CASE("Camera self-finish does not touch shutter when autoShutter is off",
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
   cam.TriggerSelfFinish();
   CHECK(shutter.open == true);
   c.stopSequenceAcquisition();
}

TEST_CASE("startSequenceAcquisition after camera self-finish without "
          "intervening stop succeeds",
          "[SequenceAcquisition]") {
   SyncCamera cam;
   MockAdapterWithDevices adapter{{"cam", &cam}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("cam");

   c.startSequenceAcquisition(10, 0.0, true);
   cam.TriggerSelfFinish();
   REQUIRE(c.isSequenceRunning() == false);

   c.startSequenceAcquisition(10, 0.0, true);
   CHECK(c.isSequenceRunning() == true);
   REQUIRE(cam.InsertTestImage() == DEVICE_OK);
   CHECK(c.getRemainingImageCount() == 1);
   CHECK(c.popNextImage() != nullptr);
   c.stopSequenceAcquisition();
}
