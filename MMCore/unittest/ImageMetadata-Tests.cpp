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
   c.initializeCircularBuffer();

   cam.InsertTestImage();

   Metadata md;
   c.getLastImageMD(md);

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
   c.initializeCircularBuffer();

   cam.InsertTestImage();

   Metadata md;
   c.getLastImageMD(md);
   CHECK(md.GetSingleTag(MM::g_Keyword_PixelType).GetValue() ==
         expectedPixelType);
}

TEST_CASE("Camera label matches device label") {
   StubCamera cam;
   MockAdapterWithDevices adapter{{"myCam", &cam}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("myCam");
   c.initializeCircularBuffer();

   cam.InsertTestImage();

   Metadata md;
   c.getLastImageMD(md);
   CHECK(md.GetSingleTag(MM::g_Keyword_Metadata_CameraLabel).GetValue() == "myCam");
}

TEST_CASE("Width and Height reflect camera dimensions") {
   StubCamera cam;
   cam.width = 256;
   cam.height = 128;
   MockAdapterWithDevices adapter{{"cam", &cam}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("cam");
   c.initializeCircularBuffer();

   cam.InsertTestImage();

   Metadata md;
   c.getLastImageMD(md);
   CHECK(md.GetSingleTag(MM::g_Keyword_Metadata_Width).GetValue() == "256");
   CHECK(md.GetSingleTag(MM::g_Keyword_Metadata_Height).GetValue() == "128");
}

TEST_CASE("ImageNumber increments across inserted images") {
   StubCamera cam;
   MockAdapterWithDevices adapter{{"cam", &cam}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("cam");
   c.initializeCircularBuffer();

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
}

TEST_CASE("Unconditionally-added fields overwrite camera-provided values") {
   StubCamera cam;
   MockAdapterWithDevices adapter{{"cam", &cam}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("cam");
   c.initializeCircularBuffer();

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
}

TEST_CASE("ElapsedTime-ms preserved when camera provides it") {
   StubCamera cam;
   MockAdapterWithDevices adapter{{"cam", &cam}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("cam");
   c.initializeCircularBuffer();

   MM::CameraImageMetadata camMd;
   camMd.AddTag(MM::g_Keyword_Elapsed_Time_ms, "42.0");
   cam.InsertTestImage(camMd);

   Metadata md;
   c.getLastImageMD(md);
   CHECK(md.GetSingleTag(MM::g_Keyword_Elapsed_Time_ms).GetValue() == "42.0");
}

TEST_CASE("Custom device tag is transmitted") {
   StubCamera cam;
   MockAdapterWithDevices adapter{{"cam", &cam}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("cam");
   c.initializeCircularBuffer();

   cam.AddTag("MyCustomTag", "cam", "hello");
   cam.InsertTestImage();

   Metadata md;
   c.getLastImageMD(md);
   CHECK(md.GetSingleTag("cam-MyCustomTag").GetValue() == "hello");
   CHECK(md.GetKeys().size() == 8);
}

TEST_CASE("RemoveTag removes a previously added device tag") {
   StubCamera cam;
   MockAdapterWithDevices adapter{{"cam", &cam}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("cam");
   c.initializeCircularBuffer();

   cam.AddTag("MyCustomTag", "cam", "hello");
   cam.RemoveTag("cam-MyCustomTag");
   cam.InsertTestImage();

   Metadata md;
   c.getLastImageMD(md);
   CHECK(md.GetKeys().size() == 7);
}
