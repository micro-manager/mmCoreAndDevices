#include <catch2/catch_all.hpp>

#include "MMCore.h"
#include "../../MMDevice/DeviceBase.h"
#include "MockDeviceUtils.h"

namespace {

struct MockCamera : public CCameraBase<MockCamera> {
   int Initialize() override { return DEVICE_OK; }
   int Shutdown() override { return DEVICE_OK; }
   bool Busy() override { return false; }
   void GetName(char* name) const override {
      snprintf(name, MM::MaxStrLength, "MockCamera");
   }

   int SnapImage() override { return DEVICE_ERR; }
   const unsigned char* GetImageBuffer() override { return nullptr; }
   long GetImageBufferSize() const override { return 0; }
   unsigned GetImageWidth() const override { return 0; }
   unsigned GetImageHeight() const override { return 0; }
   unsigned GetImageBytesPerPixel() const override { return 0; }
   unsigned GetBitDepth() const override { return 0; }
   int GetBinning() const override { return 1; }
   int SetBinning(int) override { return DEVICE_ERR; }
   void SetExposure(double) override { FAIL(); }
   double GetExposure() const override { return 0.0; }
   int SetROI(unsigned, unsigned, unsigned, unsigned) override { return DEVICE_ERR; }
   int GetROI(unsigned&, unsigned&, unsigned&, unsigned&) override { return DEVICE_ERR; }
   int ClearROI() override { return DEVICE_ERR; }
   int IsExposureSequenceable(bool&) const override { return DEVICE_ERR; }
};

TEST_CASE("Unload the current camera", "[regression]") {
   MockCamera cam;
   MockAdapterWithDevices adapter{{"cam", &cam}};
   CMMCore c;
   adapter.LoadIntoCore(c);

   c.setCameraDevice("cam");
   c.unloadDevice("cam");
}

TEST_CASE("Unload all devices with current camera set", "[regression]") {
   MockCamera cam;
   MockAdapterWithDevices adapter{{"cam", &cam}};
   CMMCore c;
   adapter.LoadIntoCore(c);

   c.setCameraDevice("cam");
   c.unloadAllDevices();
}

// TODO Similar tests should be added for every device role (but maybe once we
// have a way to avoid boilerplate when mocking devices)

}