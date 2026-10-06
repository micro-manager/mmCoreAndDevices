#include <catch2/catch_all.hpp>

#include "MMCore.h"
#include "CameraImageMetadata.h"
#include "ImageMetadata.h"
#include "MMDeviceConstants.h"
#include "MockDeviceUtils.h"
#include "StubDevices.h"

#include <string>

TEST_CASE("All core metadata fields present for GRAY8 image") {
   StubCamera cam;
   MockAdapterWithDevices adapter{{"cam", &cam}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("cam");
   c.startSequenceAcquisition(10, 0.0, true);

   cam.InsertTestImage();

   Metadata md;
   c.getLastImageMD(md);
   c.stopSequenceAcquisition();

   CHECK(md.GetSingleTag(MM::g_Keyword_Metadata_CameraLabel).GetValue() == "cam");
   CHECK(md.GetSingleTag(MM::g_Keyword_Metadata_ImageNumber).GetValue() ==
         "0");
   CHECK(md.GetSingleTag(MM::g_Keyword_Metadata_Width).GetValue() == "512");
   CHECK(md.GetSingleTag(MM::g_Keyword_Metadata_Height).GetValue() == "512");
   CHECK(md.GetSingleTag(MM::g_Keyword_PixelType).GetValue() ==
         MM::g_Keyword_PixelType_GRAY8);

   auto elapsed =
       md.GetSingleTag(MM::g_Keyword_Elapsed_Time_ms).GetValue();
   CHECK(std::stod(elapsed) >= 0.0);

   auto timeInCore =
       md.GetSingleTag(MM::g_Keyword_Metadata_TimeInCore).GetValue();
   CHECK(!timeInCore.empty());

   CHECK(md.GetKeys().size() == 7);
}

TEST_CASE("PixelType metadata for all pixel formats") {
   StubCamera cam;
   unsigned bytesPerPixel;
   unsigned nComponents;
   const char* expectedPixelType;

   SECTION("GRAY8") {
      bytesPerPixel = 1;
      nComponents = 1;
      expectedPixelType = MM::g_Keyword_PixelType_GRAY8;
   }
   SECTION("GRAY16") {
      bytesPerPixel = 2;
      nComponents = 1;
      expectedPixelType = MM::g_Keyword_PixelType_GRAY16;
   }
   SECTION("GRAY32") {
      bytesPerPixel = 4;
      nComponents = 1;
      expectedPixelType = MM::g_Keyword_PixelType_GRAY32;
   }
   SECTION("RGB32") {
      bytesPerPixel = 4;
      nComponents = 4;
      expectedPixelType = MM::g_Keyword_PixelType_RGB32;
   }
   SECTION("RGB64") {
      bytesPerPixel = 8;
      nComponents = 4;
      expectedPixelType = MM::g_Keyword_PixelType_RGB64;
   }
   SECTION("Unknown") {
      bytesPerPixel = 3;
      nComponents = 1;
      expectedPixelType = MM::g_Keyword_PixelType_Unknown;
   }

   cam.bytesPerPixel = bytesPerPixel;
   cam.nComponents = nComponents;
   MockAdapterWithDevices adapter{{"cam", &cam}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("cam");
   c.startSequenceAcquisition(10, 0.0, true);

   cam.InsertTestImage();

   Metadata md;
   c.getLastImageMD(md);
   CHECK(md.GetSingleTag(MM::g_Keyword_PixelType).GetValue() ==
         expectedPixelType);
   c.stopSequenceAcquisition();
}

TEST_CASE("Camera label matches device label") {
   StubCamera cam;
   MockAdapterWithDevices adapter{{"myCam", &cam}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("myCam");
   c.startSequenceAcquisition(10, 0.0, true);

   cam.InsertTestImage();

   Metadata md;
   c.getLastImageMD(md);
   CHECK(md.GetSingleTag(MM::g_Keyword_Metadata_CameraLabel).GetValue() == "myCam");
   c.stopSequenceAcquisition();
}

TEST_CASE("Width and Height reflect camera dimensions") {
   StubCamera cam;
   cam.width = 256;
   cam.height = 128;
   MockAdapterWithDevices adapter{{"cam", &cam}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("cam");
   c.startSequenceAcquisition(10, 0.0, true);

   cam.InsertTestImage();

   Metadata md;
   c.getLastImageMD(md);
   CHECK(md.GetSingleTag(MM::g_Keyword_Metadata_Width).GetValue() == "256");
   CHECK(md.GetSingleTag(MM::g_Keyword_Metadata_Height).GetValue() == "128");
   c.stopSequenceAcquisition();
}

TEST_CASE("ImageNumber increments across inserted images") {
   StubCamera cam;
   MockAdapterWithDevices adapter{{"cam", &cam}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("cam");
   c.startSequenceAcquisition(10, 0.0, true);

   cam.InsertTestImage();
   cam.InsertTestImage();
   cam.InsertTestImage();

   Metadata md;
   c.popNextImageMD(md);
   CHECK(md.GetSingleTag(MM::g_Keyword_Metadata_ImageNumber).GetValue() ==
         "0");
   c.popNextImageMD(md);
   CHECK(md.GetSingleTag(MM::g_Keyword_Metadata_ImageNumber).GetValue() ==
         "1");
   c.popNextImageMD(md);
   CHECK(md.GetSingleTag(MM::g_Keyword_Metadata_ImageNumber).GetValue() ==
         "2");
   c.stopSequenceAcquisition();
}

TEST_CASE("Unconditionally-added fields overwrite camera-provided values") {
   StubCamera cam;
   MockAdapterWithDevices adapter{{"cam", &cam}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("cam");
   c.startSequenceAcquisition(10, 0.0, true);

   MM::CameraImageMetadata camMd;
   camMd.AddTag(MM::g_Keyword_Metadata_CameraLabel, "WRONG");
   camMd.AddTag(MM::g_Keyword_Metadata_Width, 9999);
   camMd.AddTag(MM::g_Keyword_Metadata_Height, 9999);
   camMd.AddTag(MM::g_Keyword_PixelType, "WRONG");
   camMd.AddTag(MM::g_Keyword_Metadata_ImageNumber, "999");
   camMd.AddTag(MM::g_Keyword_Metadata_TimeInCore, "wrong");
   cam.InsertTestImage(camMd);

   Metadata md;
   c.getLastImageMD(md);
   CHECK(md.GetSingleTag(MM::g_Keyword_Metadata_CameraLabel).GetValue() ==
         "cam");
   CHECK(md.GetSingleTag(MM::g_Keyword_Metadata_Width).GetValue() == "512");
   CHECK(md.GetSingleTag(MM::g_Keyword_Metadata_Height).GetValue() == "512");
   CHECK(md.GetSingleTag(MM::g_Keyword_PixelType).GetValue() ==
         MM::g_Keyword_PixelType_GRAY8);
   CHECK(md.GetSingleTag(MM::g_Keyword_Metadata_ImageNumber).GetValue() ==
         "0");
   CHECK(md.GetSingleTag(MM::g_Keyword_Metadata_TimeInCore).GetValue() !=
         "wrong");
   CHECK(md.GetKeys().size() == 7);
   c.stopSequenceAcquisition();
}

TEST_CASE("ElapsedTime-ms preserved when camera provides it") {
   StubCamera cam;
   MockAdapterWithDevices adapter{{"cam", &cam}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("cam");
   c.startSequenceAcquisition(10, 0.0, true);

   MM::CameraImageMetadata camMd;
   camMd.AddTag(MM::g_Keyword_Elapsed_Time_ms, "42.0");
   cam.InsertTestImage(camMd);

   Metadata md;
   c.getLastImageMD(md);
   CHECK(md.GetSingleTag(MM::g_Keyword_Elapsed_Time_ms).GetValue() == "42.0");
   c.stopSequenceAcquisition();
}

TEST_CASE("ImageNumber is tracked per camera across interleaved inserts") {
   StubCamera camA;
   StubCamera camB;
   MockAdapterWithDevices adapter{{"camA", &camA}, {"camB", &camB}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("camA");
   c.startSequenceAcquisition("camA", 10, 0.0, true);
   c.startSequenceAcquisition("camB", 10, 0.0, true);

   REQUIRE(camA.InsertTestImage() == DEVICE_OK);
   REQUIRE(camB.InsertTestImage() == DEVICE_OK);
   REQUIRE(camA.InsertTestImage() == DEVICE_OK);
   REQUIRE(camB.InsertTestImage() == DEVICE_OK);

   struct Expected { const char* label; const char* number; };
   const Expected expected[] = {
      {"camA", "0"}, {"camB", "0"}, {"camA", "1"}, {"camB", "1"},
   };
   for (const auto& e : expected) {
      Metadata md;
      c.popNextImageMD(md);
      CHECK(md.GetSingleTag(MM::g_Keyword_Metadata_CameraLabel).GetValue() ==
            e.label);
      CHECK(md.GetSingleTag(MM::g_Keyword_Metadata_ImageNumber).GetValue() ==
            e.number);
   }

   c.stopSequenceAcquisition("camA");
   c.stopSequenceAcquisition("camB");
}

TEST_CASE("ImageNumber resets after clearCircularBuffer") {
   StubCamera cam;
   MockAdapterWithDevices adapter{{"cam", &cam}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("cam");
   c.startSequenceAcquisition(10, 0.0, true);

   REQUIRE(cam.InsertTestImage() == DEVICE_OK);
   REQUIRE(cam.InsertTestImage() == DEVICE_OK);
   REQUIRE(cam.InsertTestImage() == DEVICE_OK);

   c.clearCircularBuffer();
   REQUIRE(cam.InsertTestImage() == DEVICE_OK);

   Metadata md;
   c.popNextImageMD(md);
   CHECK(md.GetSingleTag(MM::g_Keyword_Metadata_ImageNumber).GetValue() == "0");
   c.stopSequenceAcquisition();
}

TEST_CASE("ImageNumber resets after re-initializeCircularBuffer") {
   StubCamera cam;
   MockAdapterWithDevices adapter{{"cam", &cam}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("cam");
   c.startSequenceAcquisition(10, 0.0, true);

   REQUIRE(cam.InsertTestImage() == DEVICE_OK);
   REQUIRE(cam.InsertTestImage() == DEVICE_OK);
   c.stopSequenceAcquisition();

   c.startSequenceAcquisition(10, 0.0, true);
   REQUIRE(cam.InsertTestImage() == DEVICE_OK);

   Metadata md;
   c.popNextImageMD(md);
   CHECK(md.GetSingleTag(MM::g_Keyword_Metadata_ImageNumber).GetValue() == "0");
   c.stopSequenceAcquisition();
}

TEST_CASE("ImageNumber is monotonic across overwrite-on-overflow wrap") {
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
   for (long i = 0; i < total + 1; ++i)
      REQUIRE(cam.InsertTestImage() == DEVICE_OK);

   // The wrap-on-overflow path discards buffered images, so only the last
   // insert remains. Under the old code, the mid-acquisition wrap also reset
   // the per-camera ImageNumber counter, producing "0". Under the new code,
   // the counter is independent of buffer state, so the retained image
   // carries its original acquisition index.
   REQUIRE(c.getRemainingImageCount() == 1);
   Metadata md;
   c.popNextImageMD(md);
   CHECK(md.GetSingleTag(MM::g_Keyword_Metadata_ImageNumber).GetValue() ==
         std::to_string(total));
}

TEST_CASE("ImageNumbers are contiguous under stop-on-overflow") {
   StubCamera cam;
   MockAdapterWithDevices adapter{{"cam", &cam}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("cam");
   c.setCircularBufferMemoryFootprint(1);
   c.startSequenceAcquisition(100, 0.0, true);

   long total = c.getBufferTotalCapacity();
   REQUIRE(total == 4);

   for (long i = 0; i < total; ++i)
      REQUIRE(cam.InsertTestImage() == DEVICE_OK);
   REQUIRE(cam.InsertTestImage() == DEVICE_BUFFER_OVERFLOW);
   REQUIRE(cam.InsertTestImage() == DEVICE_BUFFER_OVERFLOW);

   for (long i = 0; i < total; ++i) {
      Metadata md;
      c.popNextImageMD(md);
      CHECK(md.GetSingleTag(MM::g_Keyword_Metadata_ImageNumber).GetValue() ==
            std::to_string(i));
   }
   c.stopSequenceAcquisition();
}
