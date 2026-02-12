#include <catch2/catch_all.hpp>

#include "MMCore.h"
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

TEST_CASE("PixelType is GRAY16 for 2-byte images") {
   StubCamera cam;
   cam.bytesPerPixel = 2;
   cam.bitDepth = 16;
   MockAdapterWithDevices adapter{{"cam", &cam}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("cam");
   c.initializeCircularBuffer();

   cam.InsertTestImage();

   Metadata md;
   c.getLastImageMD(md);
   CHECK(md.GetSingleTag(MM::g_Keyword_PixelType).GetValue() ==
         MM::g_Keyword_PixelType_GRAY16);
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

   Metadata camMd;
   camMd.PutImageTag(MM::g_Keyword_Metadata_CameraLabel, "WRONG");
   camMd.PutImageTag(MM::g_Keyword_Metadata_Width, 9999);
   camMd.PutImageTag(MM::g_Keyword_Metadata_Height, 9999);
   camMd.PutImageTag(MM::g_Keyword_PixelType, "WRONG");
   camMd.PutImageTag(MM::g_Keyword_Metadata_ImageNumber, "999");
   camMd.PutImageTag(MM::g_Keyword_Metadata_TimeInCore, "wrong");
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

   Metadata camMd;
   camMd.PutImageTag(MM::g_Keyword_Elapsed_Time_ms, "42.0");
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
