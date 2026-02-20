// Concrete stub device classes with sensible defaults for use in MMCore unit
// tests. Each stub provides the minimum overrides needed to satisfy the pure
// virtual interface, with public fields for commonly configured values.

#pragma once

#include "CameraImageMetadata.h"
#include "DeviceBase.h"

#include <string>
#include <vector>

struct StubGeneric : CGenericBase<StubGeneric> {
   std::string name = "StubGeneric";

   int Initialize() override { return DEVICE_OK; }
   int Shutdown() override { return DEVICE_OK; }
   bool Busy() override { return false; }
   void GetName(char* buf) const override {
      CDeviceUtils::CopyLimitedString(buf, name.c_str());
   }
};

struct StubCamera : CCameraBase<StubCamera> {
   std::string name = "StubCamera";
   unsigned width = 512;
   unsigned height = 512;
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
   int StartSequenceAcquisition(long, double, bool) override {
      return DEVICE_OK;
   }
   int StartSequenceAcquisition(double) override { return DEVICE_OK; }
   int StopSequenceAcquisition() override { return DEVICE_OK; }
   bool IsCapturing() override { return false; }

   int InsertTestImage(
         const MM::CameraImageMetadata& md = MM::CameraImageMetadata{},
         const unsigned char* pixels = nullptr) {
      std::vector<unsigned char> defaultBuf;
      const unsigned char* buf = pixels;
      if (!buf) {
         defaultBuf.assign(
            static_cast<size_t>(width) * height * bytesPerPixel, 0);
         buf = defaultBuf.data();
      }
      return GetCoreCallback()->InsertImage(this, buf,
         width, height, bytesPerPixel, nComponents,
         md.Serialize());
   }

private:
   std::vector<unsigned char> imgBuf_;
};

struct StubStage : CStageBase<StubStage> {
   std::string name = "StubStage";
   double positionUm = 0.0;
   long positionSteps = 0;
   double lowerLimit = -10000.0;
   double upperLimit = 10000.0;

   int Initialize() override { return DEVICE_OK; }
   int Shutdown() override { return DEVICE_OK; }
   bool Busy() override { return false; }
   void GetName(char* buf) const override {
      CDeviceUtils::CopyLimitedString(buf, name.c_str());
   }

   int SetPositionUm(double pos) override {
      positionUm = pos;
      return DEVICE_OK;
   }
   int GetPositionUm(double& pos) override {
      pos = positionUm;
      return DEVICE_OK;
   }
   int SetPositionSteps(long steps) override {
      positionSteps = steps;
      return DEVICE_OK;
   }
   int GetPositionSteps(long& steps) override {
      steps = positionSteps;
      return DEVICE_OK;
   }
   int SetOrigin() override {
      positionUm = 0.0;
      positionSteps = 0;
      return DEVICE_OK;
   }
   int GetLimits(double& lower, double& upper) override {
      lower = lowerLimit;
      upper = upperLimit;
      return DEVICE_OK;
   }
   int IsStageSequenceable(bool& seq) const override {
      seq = false;
      return DEVICE_OK;
   }
   bool IsContinuousFocusDrive() const override { return false; }
};

struct StubXYStage : CXYStageBase<StubXYStage> {
   std::string name = "StubXYStage";
   long posXSteps = 0;
   long posYSteps = 0;
   double stepSizeX = 1.0;
   double stepSizeY = 1.0;

   int Initialize() override { return DEVICE_OK; }
   int Shutdown() override { return DEVICE_OK; }
   bool Busy() override { return false; }
   void GetName(char* buf) const override {
      CDeviceUtils::CopyLimitedString(buf, name.c_str());
   }

   int SetPositionSteps(long x, long y) override {
      posXSteps = x;
      posYSteps = y;
      return DEVICE_OK;
   }
   int GetPositionSteps(long& x, long& y) override {
      x = posXSteps;
      y = posYSteps;
      return DEVICE_OK;
   }
   double GetStepSizeXUm() override { return stepSizeX; }
   double GetStepSizeYUm() override { return stepSizeY; }
   int GetLimitsUm(double& xMin, double& xMax, double& yMin,
                    double& yMax) override {
      xMin = -100000.0; xMax = 100000.0;
      yMin = -100000.0; yMax = 100000.0;
      return DEVICE_OK;
   }
   int GetStepLimits(long& xMin, long& xMax, long& yMin,
                     long& yMax) override {
      xMin = -100000; xMax = 100000;
      yMin = -100000; yMax = 100000;
      return DEVICE_OK;
   }
   int Home() override { return DEVICE_OK; }
   int Stop() override { return DEVICE_OK; }
   int SetOrigin() override { return DEVICE_OK; }
   int IsXYStageSequenceable(bool& seq) const override {
      seq = false;
      return DEVICE_OK;
   }
};

struct StubStateDevice : CStateDeviceBase<StubStateDevice> {
   std::string name = "StubStateDevice";
   unsigned long numPositions = 10;

   int Initialize() override { return DEVICE_OK; }
   int Shutdown() override { return DEVICE_OK; }
   bool Busy() override { return false; }
   void GetName(char* buf) const override {
      CDeviceUtils::CopyLimitedString(buf, name.c_str());
   }

   unsigned long GetNumberOfPositions() const override {
      return numPositions;
   }
};

struct StubShutter : CShutterBase<StubShutter> {
   std::string name = "StubShutter";
   bool open = false;

   int Initialize() override { return DEVICE_OK; }
   int Shutdown() override { return DEVICE_OK; }
   bool Busy() override { return false; }
   void GetName(char* buf) const override {
      CDeviceUtils::CopyLimitedString(buf, name.c_str());
   }

   int SetOpen(bool o) override { open = o; return DEVICE_OK; }
   int GetOpen(bool& o) override { o = open; return DEVICE_OK; }
   int Fire(double) override { return DEVICE_OK; }
};

struct StubMagnifier : CMagnifierBase<StubMagnifier> {
   std::string name = "StubMagnifier";
   double magnification = 1.0;

   int Initialize() override { return DEVICE_OK; }
   int Shutdown() override { return DEVICE_OK; }
   bool Busy() override { return false; }
   void GetName(char* buf) const override {
      CDeviceUtils::CopyLimitedString(buf, name.c_str());
   }

   double GetMagnification() override { return magnification; }
};
