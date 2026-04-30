#include <catch2/catch_all.hpp>

#include "MMCore.h"
#include "ImageMetadata.h"
#include "MMDeviceConstants.h"
#include "MockDeviceUtils.h"
#include "StubDevices.h"

#include <cstddef>
#include <vector>

// Initialization

TEST_CASE("initializeCircularBuffer succeeds with camera set",
          "[CircularBuffer]") {
   StubCamera cam;
   MockAdapterWithDevices adapter{{"cam", &cam}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("cam");
   c.initializeCircularBuffer();
   CHECK(c.getBufferTotalCapacity() > 0);
}

TEST_CASE("initializeCircularBuffer without camera throws",
          "[CircularBuffer]") {
   CMMCore c;
   CHECK_THROWS_AS(c.initializeCircularBuffer(), CMMError);
}

TEST_CASE("Buffer is empty after init", "[CircularBuffer]") {
   StubCamera cam;
   MockAdapterWithDevices adapter{{"cam", &cam}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("cam");
   c.initializeCircularBuffer();
   CHECK(c.getRemainingImageCount() == 0);
   CHECK(c.isBufferOverflowed() == false);
}

TEST_CASE("Starting a fresh sequence acquisition clears existing images",
          "[CircularBuffer]") {
   StubCamera cam;
   MockAdapterWithDevices adapter{{"cam", &cam}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("cam");
   c.startSequenceAcquisition(1'000'000, 0.0, true);
   REQUIRE(cam.InsertTestImage() == DEVICE_OK);
   REQUIRE(cam.InsertTestImage() == DEVICE_OK);
   c.stopSequenceAcquisition();
   c.startSequenceAcquisition(1'000'000, 0.0, true);
   CHECK(c.getRemainingImageCount() == 0);
   CHECK(c.isBufferOverflowed() == false);
   c.stopSequenceAcquisition();
}

// Insert

TEST_CASE("getRemainingImageCount is 1 after one insert",
          "[CircularBuffer]") {
   StubCamera cam;
   MockAdapterWithDevices adapter{{"cam", &cam}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("cam");
   c.startSequenceAcquisition(1'000'000, 0.0, true);
   CHECK(cam.InsertTestImage() == DEVICE_OK);
   CHECK(c.getRemainingImageCount() == 1);
   c.stopSequenceAcquisition();
}

TEST_CASE("Insert with mismatched width returns DEVICE_INCOMPATIBLE_IMAGE",
          "[CircularBuffer]") {
   StubCamera cam;
   MockAdapterWithDevices adapter{{"cam", &cam}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("cam");
   c.startSequenceAcquisition(1'000'000, 0.0, true);
   cam.width = 256;
   CHECK(cam.InsertTestImage() == DEVICE_INCOMPATIBLE_IMAGE);
   CHECK(c.getRemainingImageCount() == 0);
   c.stopSequenceAcquisition();
}

TEST_CASE("Insert with mismatched height returns DEVICE_INCOMPATIBLE_IMAGE",
          "[CircularBuffer]") {
   StubCamera cam;
   MockAdapterWithDevices adapter{{"cam", &cam}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("cam");
   c.startSequenceAcquisition(1'000'000, 0.0, true);
   cam.height = 256;
   CHECK(cam.InsertTestImage() == DEVICE_INCOMPATIBLE_IMAGE);
   CHECK(c.getRemainingImageCount() == 0);
   c.stopSequenceAcquisition();
}

TEST_CASE(
      "Insert with mismatched byte depth returns DEVICE_INCOMPATIBLE_IMAGE",
      "[CircularBuffer]") {
   StubCamera cam;
   MockAdapterWithDevices adapter{{"cam", &cam}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("cam");
   c.startSequenceAcquisition(1'000'000, 0.0, true);
   cam.bytesPerPixel = 2;
   CHECK(cam.InsertTestImage() == DEVICE_INCOMPATIBLE_IMAGE);
   CHECK(c.getRemainingImageCount() == 0);
   c.stopSequenceAcquisition();
}

TEST_CASE("Insert with mismatched nComponents succeeds",
          "[CircularBuffer]") {
   StubCamera cam;
   MockAdapterWithDevices adapter{{"cam", &cam}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("cam");
   c.startSequenceAcquisition(1'000'000, 0.0, true);
   cam.nComponents = 4;
   CHECK(cam.InsertTestImage() == DEVICE_OK);
   CHECK(c.getRemainingImageCount() == 1);
   c.stopSequenceAcquisition();
}

// getLastImage

TEST_CASE("getLastImage returns non-null after insert", "[CircularBuffer]") {
   StubCamera cam;
   MockAdapterWithDevices adapter{{"cam", &cam}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("cam");
   c.startSequenceAcquisition(1'000'000, 0.0, true);
   REQUIRE(cam.InsertTestImage() == DEVICE_OK);
   CHECK(c.getLastImage() != nullptr);
   c.stopSequenceAcquisition();
}

TEST_CASE("getLastImage on empty buffer throws", "[CircularBuffer]") {
   StubCamera cam;
   MockAdapterWithDevices adapter{{"cam", &cam}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("cam");
   c.initializeCircularBuffer();
   CHECK_THROWS_AS(c.getLastImage(), CMMError);
}

TEST_CASE("getLastImage returns most recently inserted image",
          "[CircularBuffer]") {
   StubCamera cam;
   MockAdapterWithDevices adapter{{"cam", &cam}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("cam");
   c.startSequenceAcquisition(1'000'000, 0.0, true);

   const std::size_t imgSize =
       static_cast<std::size_t>(cam.width) * cam.height * cam.bytesPerPixel;
   for (unsigned char fill = 10; fill <= 30; fill += 10) {
      std::vector<unsigned char> pixels(imgSize, fill);
      REQUIRE(cam.InsertTestImage({}, pixels.data()) == DEVICE_OK);
   }

   auto* img = static_cast<const unsigned char*>(c.getLastImage());
   REQUIRE(img != nullptr);
   CHECK(img[0] == 30);
   c.stopSequenceAcquisition();
}

// popNextImage

TEST_CASE("popNextImage returns non-null and decrements count",
          "[CircularBuffer]") {
   StubCamera cam;
   MockAdapterWithDevices adapter{{"cam", &cam}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("cam");
   c.startSequenceAcquisition(1'000'000, 0.0, true);
   REQUIRE(cam.InsertTestImage() == DEVICE_OK);
   CHECK(c.popNextImage() != nullptr);
   CHECK(c.getRemainingImageCount() == 0);
   c.stopSequenceAcquisition();
}

TEST_CASE("popNextImage on empty buffer throws", "[CircularBuffer]") {
   StubCamera cam;
   MockAdapterWithDevices adapter{{"cam", &cam}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("cam");
   c.initializeCircularBuffer();
   CHECK_THROWS_AS(c.popNextImage(), CMMError);
}

TEST_CASE("popNextImage returns images in insertion order",
          "[CircularBuffer]") {
   StubCamera cam;
   MockAdapterWithDevices adapter{{"cam", &cam}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("cam");
   c.startSequenceAcquisition(1'000'000, 0.0, true);

   const std::size_t imgSize =
       static_cast<std::size_t>(cam.width) * cam.height * cam.bytesPerPixel;
   for (unsigned char fill = 1; fill <= 3; ++fill) {
      std::vector<unsigned char> pixels(imgSize, fill);
      CHECK(cam.InsertTestImage({}, pixels.data()) == DEVICE_OK);
   }

   for (unsigned char expected = 1; expected <= 3; ++expected) {
      auto* img = static_cast<unsigned char*>(c.popNextImage());
      REQUIRE(img != nullptr);
      CHECK(img[0] == expected);
   }
   c.stopSequenceAcquisition();
}

TEST_CASE("FIFO ordering is maintained across interleaved pops and inserts",
          "[CircularBuffer]") {
   StubCamera cam;
   MockAdapterWithDevices adapter{{"cam", &cam}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("cam");
   c.startSequenceAcquisition(1'000'000, 0.0, true);

   const std::size_t imgSize =
       static_cast<std::size_t>(cam.width) * cam.height * cam.bytesPerPixel;

   for (unsigned char fill = 1; fill <= 3; ++fill) {
      std::vector<unsigned char> pixels(imgSize, fill);
      REQUIRE(cam.InsertTestImage({}, pixels.data()) == DEVICE_OK);
   }

   auto* img = static_cast<unsigned char*>(c.popNextImage());
   REQUIRE(img != nullptr);
   CHECK(img[0] == 1);

   std::vector<unsigned char> pixels(imgSize, 4);
   REQUIRE(cam.InsertTestImage({}, pixels.data()) == DEVICE_OK);

   for (unsigned char expected = 2; expected <= 4; ++expected) {
      img = static_cast<unsigned char*>(c.popNextImage());
      REQUIRE(img != nullptr);
      CHECK(img[0] == expected);
   }
   c.stopSequenceAcquisition();
}

// getNBeforeLastImageMD

TEST_CASE("getNBeforeLastImageMD returns images by reverse offset",
          "[CircularBuffer]") {
   StubCamera cam;
   MockAdapterWithDevices adapter{{"cam", &cam}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("cam");
   c.startSequenceAcquisition(1'000'000, 0.0, true);

   for (int i = 0; i < 3; ++i)
      REQUIRE(cam.InsertTestImage() == DEVICE_OK);

   Metadata md;
   c.getNBeforeLastImageMD(0, md);
   CHECK(md.GetSingleTag(MM::g_Keyword_Metadata_ImageNumber).GetValue() ==
         "2");

   c.getNBeforeLastImageMD(2, md);
   CHECK(md.GetSingleTag(MM::g_Keyword_Metadata_ImageNumber).GetValue() ==
         "0");
   c.stopSequenceAcquisition();
}

TEST_CASE("getNBeforeLastImageMD throws when offset exceeds available images",
          "[CircularBuffer]") {
   StubCamera cam;
   MockAdapterWithDevices adapter{{"cam", &cam}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("cam");
   c.startSequenceAcquisition(1'000'000, 0.0, true);
   REQUIRE(cam.InsertTestImage() == DEVICE_OK);
   REQUIRE(cam.InsertTestImage() == DEVICE_OK);

   Metadata md;
   CHECK_NOTHROW(c.getNBeforeLastImageMD(0, md));
   CHECK_NOTHROW(c.getNBeforeLastImageMD(1, md));
   CHECK_THROWS_AS(c.getNBeforeLastImageMD(2, md), CMMError);
   c.stopSequenceAcquisition();
}

TEST_CASE("getNBeforeLastImageMD on empty buffer throws",
          "[CircularBuffer]") {
   StubCamera cam;
   MockAdapterWithDevices adapter{{"cam", &cam}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("cam");
   c.initializeCircularBuffer();

   Metadata md;
   CHECK_THROWS_AS(c.getNBeforeLastImageMD(0, md), CMMError);
}

// Capacity

TEST_CASE("Free plus remaining equals total after each insert",
          "[CircularBuffer]") {
   StubCamera cam;
   MockAdapterWithDevices adapter{{"cam", &cam}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("cam");
   c.setCircularBufferMemoryFootprint(1);
   c.startSequenceAcquisition(1'000'000, 0.0, true);

   long total = c.getBufferTotalCapacity();
   REQUIRE(total == 4);

   for (long i = 0; i < total; ++i) {
      REQUIRE(cam.InsertTestImage() == DEVICE_OK);
      CHECK(c.getBufferFreeCapacity() + c.getRemainingImageCount() == total);
   }
   c.stopSequenceAcquisition();
}

TEST_CASE("setCircularBufferMemoryFootprint round-trips", "[CircularBuffer]") {
   CMMCore c;
   c.setCircularBufferMemoryFootprint(32);
   CHECK(c.getCircularBufferMemoryFootprint() == 32);
}

TEST_CASE("Changing memory footprint changes total capacity",
          "[CircularBuffer]") {
   StubCamera cam;
   MockAdapterWithDevices adapter{{"cam", &cam}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("cam");

   c.setCircularBufferMemoryFootprint(1);
   c.initializeCircularBuffer();
   long cap1 = c.getBufferTotalCapacity();

   c.setCircularBufferMemoryFootprint(2);
   c.initializeCircularBuffer();
   long cap2 = c.getBufferTotalCapacity();

   CHECK(cap2 > cap1);
}

// Overflow

TEST_CASE("Overflow with overwrite disabled", "[CircularBuffer]") {
   StubCamera cam;
   MockAdapterWithDevices adapter{{"cam", &cam}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("cam");
   c.setCircularBufferMemoryFootprint(1);
   c.startSequenceAcquisition(1'000'000, 0.0, true);

   long total = c.getBufferTotalCapacity();
   REQUIRE(total == 4);

   for (long i = 0; i < total; ++i) {
      CHECK(cam.InsertTestImage() == DEVICE_OK);
   }

   CHECK(cam.InsertTestImage() == DEVICE_BUFFER_OVERFLOW);
   CHECK(c.isBufferOverflowed() == true);
   CHECK(c.getRemainingImageCount() == total);
   c.stopSequenceAcquisition();
}

TEST_CASE("Overflow with overwrite enabled", "[CircularBuffer]") {
   StubCamera cam;
   MockAdapterWithDevices adapter{{"cam", &cam}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("cam");
   c.setCircularBufferMemoryFootprint(1);
   // stopOnOverflow=false enables overwrite mode
   c.startSequenceAcquisition(100, 0.0, false);

   long total = c.getBufferTotalCapacity();
   REQUIRE(total == 4);

   for (long i = 0; i < total; ++i) {
      CHECK(cam.InsertTestImage() == DEVICE_OK);
   }

   CHECK(cam.InsertTestImage() == DEVICE_OK);
   CHECK(c.isBufferOverflowed() == false);
   CHECK(c.getRemainingImageCount() == 1);
   c.stopSequenceAcquisition();
}

// Clear

TEST_CASE("clearCircularBuffer resets remaining count", "[CircularBuffer]") {
   StubCamera cam;
   MockAdapterWithDevices adapter{{"cam", &cam}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("cam");
   c.startSequenceAcquisition(1'000'000, 0.0, true);
   REQUIRE(cam.InsertTestImage() == DEVICE_OK);
   REQUIRE(cam.InsertTestImage() == DEVICE_OK);
   c.clearCircularBuffer();
   CHECK(c.getRemainingImageCount() == 0);
   c.stopSequenceAcquisition();
}

TEST_CASE("Overflow is sticky until buffer is cleared", "[CircularBuffer]") {
   StubCamera cam;
   MockAdapterWithDevices adapter{{"cam", &cam}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("cam");
   c.setCircularBufferMemoryFootprint(1);
   c.startSequenceAcquisition(1'000'000, 0.0, true);

   long total = c.getBufferTotalCapacity();
   REQUIRE(total == 4);
   for (long i = 0; i < total; ++i)
      REQUIRE(cam.InsertTestImage() == DEVICE_OK);
   REQUIRE(cam.InsertTestImage() == DEVICE_BUFFER_OVERFLOW);

   // Popping one image frees a slot, but insert should still fail until clear.
   REQUIRE(c.popNextImage() != nullptr);
   CHECK(cam.InsertTestImage() == DEVICE_BUFFER_OVERFLOW);

   c.clearCircularBuffer();
   CHECK(cam.InsertTestImage() == DEVICE_OK);
   c.stopSequenceAcquisition();
}

TEST_CASE("clearCircularBuffer resets overflow flag", "[CircularBuffer]") {
   StubCamera cam;
   MockAdapterWithDevices adapter{{"cam", &cam}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("cam");
   c.setCircularBufferMemoryFootprint(1);
   c.startSequenceAcquisition(1'000'000, 0.0, true);

   long total = c.getBufferTotalCapacity();
   for (long i = 0; i < total; ++i)
      REQUIRE(cam.InsertTestImage() == DEVICE_OK);
   cam.InsertTestImage();
   REQUIRE(c.isBufferOverflowed() == true);

   c.clearCircularBuffer();
   CHECK(c.isBufferOverflowed() == false);
   c.stopSequenceAcquisition();
}
