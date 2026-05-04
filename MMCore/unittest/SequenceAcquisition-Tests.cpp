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

// --- Participant-level conflict detection ---

TEST_CASE("Standalone camera conflicts with running composite that uses it",
          "[SequenceAcquisition]") {
   SyncCamera p1("p1");
   SyncCamera p2("p2");
   MockCompositeCamera composite({&p1, &p2});
   MockAdapterWithDevices adapter{
      {"p1", &p1}, {"p2", &p2}, {"composite", &composite}};
   CMMCore c;
   adapter.LoadIntoCore(c);

   c.startSequenceAcquisition("composite", 10, 0.0, true);
   REQUIRE(c.isSequenceRunning("composite"));

   // Hide IsCapturing() so the acquisitions_ check is the one that fires.
   p1.reportCapturing = false;
   CHECK_THROWS_AS(
      c.startSequenceAcquisition("p1", 10, 0.0, true), CMMError);

   CHECK(c.isSequenceRunning("composite"));
   c.stopSequenceAcquisition("composite");
}

TEST_CASE("Composite conflicts with running standalone that shares a physical",
          "[SequenceAcquisition]") {
   SyncCamera p1("p1");
   SyncCamera p2("p2");
   MockCompositeCamera composite({&p1, &p2});
   MockAdapterWithDevices adapter{
      {"p1", &p1}, {"p2", &p2}, {"composite", &composite}};
   CMMCore c;
   adapter.LoadIntoCore(c);

   c.startSequenceAcquisition("p1", 10, 0.0, true);
   REQUIRE(c.isSequenceRunning("p1"));

   CHECK_THROWS_AS(
      c.startSequenceAcquisition("composite", 10, 0.0, true), CMMError);

   CHECK(c.isSequenceRunning("p1"));
   c.stopSequenceAcquisition("p1");
}

TEST_CASE("Two composites sharing a physical camera conflict",
          "[SequenceAcquisition]") {
   SyncCamera p1("p1");
   SyncCamera p2("p2");
   SyncCamera p3("p3");
   MockCompositeCamera compositeA({&p1, &p2});
   MockCompositeCamera compositeB({&p1, &p3});
   compositeB.name = "compositeB";
   MockAdapterWithDevices adapter{
      {"p1", &p1}, {"p2", &p2}, {"p3", &p3},
      {"compositeA", &compositeA}, {"compositeB", &compositeB}};
   CMMCore c;
   adapter.LoadIntoCore(c);

   c.startSequenceAcquisition("compositeA", 10, 0.0, true);
   REQUIRE(c.isSequenceRunning("compositeA"));

   // Hide IsCapturing() on the physicals that compositeB would check.
   p1.reportCapturing = false;
   p3.reportCapturing = false;
   CHECK_THROWS_AS(
      c.startSequenceAcquisition("compositeB", 10, 0.0, true), CMMError);

   CHECK(c.isSequenceRunning("compositeA"));
   c.stopSequenceAcquisition("compositeA");
}

TEST_CASE("Independent cameras do not conflict",
          "[SequenceAcquisition]") {
   SyncCamera p1("p1");
   SyncCamera p2("p2");
   MockAdapterWithDevices adapter{{"p1", &p1}, {"p2", &p2}};
   CMMCore c;
   adapter.LoadIntoCore(c);

   c.startSequenceAcquisition("p1", 10, 0.0, true);
   REQUIRE(c.isSequenceRunning("p1"));

   c.startSequenceAcquisition("p2", 10, 0.0, true);
   CHECK(c.isSequenceRunning("p1"));
   CHECK(c.isSequenceRunning("p2"));

   c.stopSequenceAcquisition("p1");
   c.stopSequenceAcquisition("p2");
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

// --- Async shutter close paths ---

TEST_CASE("Async same-module shutter closes on stopSequenceAcquisition without "
          "deadlock",
          "[SequenceAcquisition]") {
   AsyncCamera cam;
   StubShutter shutter;
   MockAdapterWithDevices adapter{{"cam", &cam}, {"shutter", &shutter}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("cam");
   c.setShutterDevice("shutter");
   c.setAutoShutter(true);

   c.startSequenceAcquisition(1000000, 0.0, true);
   REQUIRE(shutter.open == true);
   c.stopSequenceAcquisition();
   CHECK(shutter.open == false);
}

TEST_CASE("Async same-module shutter closes on camera self-finish",
          "[SequenceAcquisition]") {
   AsyncCamera cam;
   StubShutter shutter;
   MockAdapterWithDevices adapter{{"cam", &cam}, {"shutter", &shutter}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("cam");
   c.setShutterDevice("shutter");
   c.setAutoShutter(true);

   c.startSequenceAcquisition(3, 0.0, true);
   REQUIRE(shutter.open == true);

   auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
   while (c.isSequenceRunning() &&
          std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
   }
   REQUIRE_FALSE(c.isSequenceRunning());
   CHECK(shutter.open == false);
   c.stopSequenceAcquisition();
}

TEST_CASE("Async same-module shutter not touched when autoShutter is off",
          "[SequenceAcquisition]") {
   AsyncCamera cam;
   StubShutter shutter;
   MockAdapterWithDevices adapter{{"cam", &cam}, {"shutter", &shutter}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("cam");
   c.setShutterDevice("shutter");
   c.setAutoShutter(false);

   c.startSequenceAcquisition(1000000, 0.0, true);
   shutter.open = true;
   c.stopSequenceAcquisition();
   CHECK(shutter.open == true);
}

TEST_CASE("Async different-module shutter closes on stopSequenceAcquisition "
          "without deadlock",
          "[SequenceAcquisition]") {
   AsyncCamera cam;
   StubShutter shutter;
   MockAdapterWithDevices camAdapter{"cam_adapter", {{"cam", &cam}}};
   MockAdapterWithDevices shutterAdapter{"shutter_adapter",
      {{"shutter", &shutter}}};
   CMMCore c;
   camAdapter.LoadIntoCore(c);
   shutterAdapter.LoadIntoCore(c);
   c.setCameraDevice("cam");
   c.setShutterDevice("shutter");
   c.setAutoShutter(true);

   c.startSequenceAcquisition(1000000, 0.0, true);
   REQUIRE(shutter.open == true);
   c.stopSequenceAcquisition();
   CHECK(shutter.open == false);
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

// --- Open-side autoshutter: inline (synchronous) PrepareForAcq ---

TEST_CASE("Inline open: shutter opens before startSequenceAcquisition returns "
          "(same adapter as camera)",
          "[SequenceAcquisition][Autoshutter]") {
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
   CHECK(shutter.setOpenTrueCount == 1);

   c.stopSequenceAcquisition();
   CHECK(shutter.open == false);
}

TEST_CASE("Inline open: shutter opens when shutter is in a different adapter",
          "[SequenceAcquisition][Autoshutter]") {
   SyncCamera cam;
   StubShutter shutter;
   MockAdapterWithDevices camAdapter{"cam_adapter", {{"cam", &cam}}};
   MockAdapterWithDevices shutterAdapter{"shutter_adapter",
      {{"shutter", &shutter}}};
   CMMCore c;
   camAdapter.LoadIntoCore(c);
   shutterAdapter.LoadIntoCore(c);
   c.setCameraDevice("cam");
   c.setShutterDevice("shutter");
   c.setAutoShutter(true);

   c.startSequenceAcquisition(10, 0.0, true);
   CHECK(shutter.open == true);
   CHECK(shutter.setOpenTrueCount == 1);
   c.stopSequenceAcquisition();
   CHECK(shutter.open == false);
}

// --- Open-side autoshutter: deferred (async) PrepareForAcq ---

TEST_CASE("Deferred open: shutter opens before worker's PrepareForAcq returns "
          "(same adapter as camera)",
          "[SequenceAcquisition][Autoshutter]") {
   WorkerThreadCamera cam;
   StubShutter shutter;
   MockAdapterWithDevices adapter{{"cam", &cam}, {"shutter", &shutter}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("cam");
   c.setShutterDevice("shutter");
   c.setAutoShutter(true);
   REQUIRE(shutter.open == false);

   c.startSequenceAcquisition(10, 0.0, true);
   // After startSequenceAcquisition returns, the deferred open must have
   // executed and the shutter is open.
   CHECK(shutter.open == true);
   CHECK(shutter.setOpenTrueCount == 1);

   // The worker thread's PrepareForAcq returned with DEVICE_OK after the
   // shutter was opened.
   REQUIRE(cam.WaitForPrepareReturned());
   CHECK(cam.PrepareReturnValue() == DEVICE_OK);

   c.stopSequenceAcquisition();
}

TEST_CASE("Deferred open: ShutterOpenChanged + SequenceAcquisitionStarted "
          "fire exactly once",
          "[SequenceAcquisition][Autoshutter]") {
   WorkerThreadCamera cam;
   StubShutter shutter;
   MockAdapterWithDevices adapter{{"cam", &cam}, {"shutter", &shutter}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("cam");
   c.setShutterDevice("shutter");
   c.setAutoShutter(true);

   c.startSequenceAcquisition(10, 0.0, true);
   REQUIRE(cam.WaitForPrepareReturned());
   CHECK(shutter.setOpenTrueCount == 1);
   c.stopSequenceAcquisition();
   CHECK(shutter.setOpenTrueCount == 1);
}

// --- Composite autoshutter ---

TEST_CASE("Composite (sync physicals): only the FirstOpener calls SetOpen(true)",
          "[SequenceAcquisition][Autoshutter]") {
   SyncCamera p1("p1");
   SyncCamera p2("p2");
   MockCompositeCamera composite({&p1, &p2});
   StubShutter shutter;
   MockAdapterWithDevices adapter{
      {"p1", &p1}, {"p2", &p2}, {"composite", &composite},
      {"shutter", &shutter}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("composite");
   c.setShutterDevice("shutter");
   c.setAutoShutter(true);

   c.startSequenceAcquisition(10, 0.0, true);
   CHECK(shutter.open == true);
   CHECK(shutter.setOpenTrueCount == 1);
   c.stopSequenceAcquisition("composite");
   CHECK(shutter.open == false);
}

TEST_CASE("Composite (async physicals): exactly one SetOpen(true), all "
          "participants released",
          "[SequenceAcquisition][Autoshutter]") {
   WorkerThreadCamera p1("p1");
   WorkerThreadCamera p2("p2");
   MockCompositeCamera composite({&p1, &p2});
   StubShutter shutter;
   MockAdapterWithDevices adapter{
      {"p1", &p1}, {"p2", &p2}, {"composite", &composite},
      {"shutter", &shutter}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("composite");
   c.setShutterDevice("shutter");
   c.setAutoShutter(true);

   c.startSequenceAcquisition(10, 0.0, true);
   REQUIRE(p1.WaitForPrepareReturned());
   REQUIRE(p2.WaitForPrepareReturned());
   CHECK(p1.PrepareReturnValue() == DEVICE_OK);
   CHECK(p2.PrepareReturnValue() == DEVICE_OK);
   CHECK(shutter.open == true);
   CHECK(shutter.setOpenTrueCount == 1);

   c.stopSequenceAcquisition("composite");
}

TEST_CASE("Composite (mixed sync + async): inline FirstOpener wins, async "
          "sibling sees AlreadyOpened",
          "[SequenceAcquisition][Autoshutter]") {
   SyncCamera pSync("pSync");
   WorkerThreadCamera pAsync("pAsync");
   // Composite starts physicals in order; pSync first ensures inline
   // FirstOpener opens the shutter on the calling thread before pAsync's
   // worker reaches PrepareForAcq.
   MockCompositeCamera composite({&pSync, &pAsync});
   StubShutter shutter;
   MockAdapterWithDevices adapter{
      {"pSync", &pSync}, {"pAsync", &pAsync},
      {"composite", &composite}, {"shutter", &shutter}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("composite");
   c.setShutterDevice("shutter");
   c.setAutoShutter(true);

   c.startSequenceAcquisition(10, 0.0, true);
   REQUIRE(pAsync.WaitForPrepareReturned());
   CHECK(pAsync.PrepareReturnValue() == DEVICE_OK);
   CHECK(shutter.open == true);
   CHECK(shutter.setOpenTrueCount == 1);

   c.stopSequenceAcquisition("composite");
}

// --- Failure paths ---

TEST_CASE("Inline open: SetOpen failure propagates as start error",
          "[SequenceAcquisition][Autoshutter]") {
   SyncCamera cam;
   StubShutter shutter;
   shutter.setOpenTrueReturnValue = DEVICE_ERR;
   MockAdapterWithDevices adapter{{"cam", &cam}, {"shutter", &shutter}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("cam");
   c.setShutterDevice("shutter");
   c.setAutoShutter(true);

   CHECK_THROWS_AS(c.startSequenceAcquisition(10, 0.0, true), CMMError);
   CHECK(shutter.open == false);
   CHECK(shutter.setOpenTrueCount == 1);
   CHECK(shutter.setOpenFalseCount == 0);
}

TEST_CASE("Deferred open: startDevice failure releases parked worker without "
          "opening shutter",
          "[SequenceAcquisition][Autoshutter]") {
   WorkerThreadCamera cam;
   cam.startReturnValue = DEVICE_ERR;
   cam.waitForPrepareCalledBeforeStartReturns = true;
   // Give the worker time to enter PrepareForAcq, defer, and park on the CV
   // before StartSequenceAcquisition returns failure.
   cam.extraSleepBeforeStartReturns = std::chrono::milliseconds(50);
   StubShutter shutter;
   MockAdapterWithDevices adapter{{"cam", &cam}, {"shutter", &shutter}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("cam");
   c.setShutterDevice("shutter");
   c.setAutoShutter(true);

   CHECK_THROWS_AS(c.startSequenceAcquisition(10, 0.0, true), CMMError);
   // Worker must have been released; PrepareForAcq returned DEVICE_ERR.
   REQUIRE(cam.WaitForPrepareReturned());
   CHECK(cam.PrepareReturnValue() == DEVICE_ERR);
   // Shutter never opened, so neither SetOpen(true) nor SetOpen(false) ran.
   CHECK(shutter.open == false);
   CHECK(shutter.setOpenTrueCount == 0);
   CHECK(shutter.setOpenFalseCount == 0);

   // Drive the worker thread to exit. cam dtor would also do this, but be
   // explicit.
   cam.StopSequenceAcquisition();
}

// --- Negative cases (no autoshutter, no shutter) ---

TEST_CASE("Deferred-open machinery is inert when autoShutter is off",
          "[SequenceAcquisition][Autoshutter]") {
   WorkerThreadCamera cam;
   StubShutter shutter;
   MockAdapterWithDevices adapter{{"cam", &cam}, {"shutter", &shutter}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("cam");
   c.setShutterDevice("shutter");
   c.setAutoShutter(false);

   c.startSequenceAcquisition(10, 0.0, true);
   REQUIRE(cam.WaitForPrepareReturned());
   CHECK(cam.PrepareReturnValue() == DEVICE_OK);
   CHECK(shutter.open == false);
   CHECK(shutter.setOpenTrueCount == 0);
   c.stopSequenceAcquisition();
}

TEST_CASE("Deferred-open machinery is inert when no shutter is set",
          "[SequenceAcquisition][Autoshutter]") {
   WorkerThreadCamera cam;
   MockAdapterWithDevices adapter{{"cam", &cam}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("cam");
   c.setAutoShutter(true);

   c.startSequenceAcquisition(10, 0.0, true);
   REQUIRE(cam.WaitForPrepareReturned());
   CHECK(cam.PrepareReturnValue() == DEVICE_OK);
   c.stopSequenceAcquisition();
}
