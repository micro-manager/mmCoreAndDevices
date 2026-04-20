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
   using CGenericBase::OnPropertiesChanged;

   int Initialize() override { return DEVICE_OK; }
   int Shutdown() override { return DEVICE_OK; }
   bool Busy() override { return false; }
   void GetName(char* buf) const override {
      CDeviceUtils::CopyLimitedString(buf, name.c_str());
   }
};

struct StubCamera : CCameraBase<StubCamera> {
   std::string name = "StubCamera";
   using CCameraBase::OnExposureChanged;
   unsigned width = 512;
   unsigned height = 512;
   unsigned bytesPerPixel = 1;
   unsigned nComponents = 1;
   unsigned bitDepth = 8;
   int binning = 1;
   double exposure = 10.0;
   bool capturing = false;

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
   bool IsCapturing() override { return capturing; }

   int PrepareForAcq() {
      return GetCoreCallback()->PrepareForAcq(this);
   }

   int AcqFinished(int statusCode = 0) {
      return GetCoreCallback()->AcqFinished(this, statusCode);
   }

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
   using CStageBase::OnStagePositionChanged;
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
   using CXYStageBase::OnXYStagePositionChanged;
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
   using CShutterBase::GetCoreCallback; // No OnShutterOpenChanged on device side
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
   using CMagnifierBase::OnMagnifierChanged;
   double magnification = 1.0;

   int Initialize() override { return DEVICE_OK; }
   int Shutdown() override { return DEVICE_OK; }
   bool Busy() override { return false; }
   void GetName(char* buf) const override {
      CDeviceUtils::CopyLimitedString(buf, name.c_str());
   }

   double GetMagnification() override { return magnification; }
};

struct StubAutoFocus : CAutoFocusBase<StubAutoFocus> {
   std::string name = "StubAutoFocus";
   bool continuousFocusing = false;
   double offset = 0.0;
   double lastScore = 0.0;
   double currentScore = 0.0;

   int Initialize() override { return DEVICE_OK; }
   int Shutdown() override { return DEVICE_OK; }
   bool Busy() override { return false; }
   void GetName(char* buf) const override {
      CDeviceUtils::CopyLimitedString(buf, name.c_str());
   }

   int SetContinuousFocusing(bool state) override {
      continuousFocusing = state;
      return DEVICE_OK;
   }
   int GetContinuousFocusing(bool& state) override {
      state = continuousFocusing;
      return DEVICE_OK;
   }
   bool IsContinuousFocusLocked() override { return false; }
   int FullFocus() override { return DEVICE_OK; }
   int IncrementalFocus() override { return DEVICE_OK; }
   int GetLastFocusScore(double& score) override {
      score = lastScore;
      return DEVICE_OK;
   }
   int GetCurrentFocusScore(double& score) override {
      score = currentScore;
      return DEVICE_OK;
   }
   int GetOffset(double& o) override { o = offset; return DEVICE_OK; }
   int SetOffset(double o) override { offset = o; return DEVICE_OK; }
};

struct StubImageProcessor : CImageProcessorBase<StubImageProcessor> {
   std::string name = "StubImageProcessor";

   int Initialize() override { return DEVICE_OK; }
   int Shutdown() override { return DEVICE_OK; }
   bool Busy() override { return false; }
   void GetName(char* buf) const override {
      CDeviceUtils::CopyLimitedString(buf, name.c_str());
   }

   int Process(unsigned char*, unsigned, unsigned, unsigned) override {
      return DEVICE_OK;
   }
};

struct StubSLM : CSLMBase<StubSLM> {
   std::string name = "StubSLM";
   using CSLMBase::OnSLMExposureChanged;
   unsigned width = 64;
   unsigned height = 64;
   unsigned nComponents = 1;
   unsigned bytesPerPixel = 1;
   double exposure = 0.0;

   int Initialize() override { return DEVICE_OK; }
   int Shutdown() override { return DEVICE_OK; }
   bool Busy() override { return false; }
   void GetName(char* buf) const override {
      CDeviceUtils::CopyLimitedString(buf, name.c_str());
   }

   int SetImage(unsigned char*) override { return DEVICE_OK; }
   int SetImage(unsigned int*) override { return DEVICE_OK; }
   int DisplayImage() override { return DEVICE_OK; }
   int SetPixelsTo(unsigned char) override { return DEVICE_OK; }
   int SetPixelsTo(unsigned char, unsigned char, unsigned char) override {
      return DEVICE_OK;
   }
   int SetExposure(double e) override { exposure = e; return DEVICE_OK; }
   double GetExposure() override { return exposure; }
   unsigned GetWidth() override { return width; }
   unsigned GetHeight() override { return height; }
   unsigned GetNumberOfComponents() override { return nComponents; }
   unsigned GetBytesPerPixel() override { return bytesPerPixel; }
   int IsSLMSequenceable(bool& seq) const override {
      seq = false;
      return DEVICE_OK;
   }
};

struct StubGalvo : CGalvoBase<StubGalvo> {
   std::string name = "StubGalvo";
   double posX = 0.0;
   double posY = 0.0;

   int Initialize() override { return DEVICE_OK; }
   int Shutdown() override { return DEVICE_OK; }
   bool Busy() override { return false; }
   void GetName(char* buf) const override {
      CDeviceUtils::CopyLimitedString(buf, name.c_str());
   }

   int PointAndFire(double, double, double) override { return DEVICE_OK; }
   int SetSpotInterval(double) override { return DEVICE_OK; }
   int SetPosition(double x, double y) override {
      posX = x; posY = y;
      return DEVICE_OK;
   }
   int GetPosition(double& x, double& y) override {
      x = posX; y = posY;
      return DEVICE_OK;
   }
   int SetIlluminationState(bool) override { return DEVICE_OK; }
   double GetXRange() override { return 100.0; }
   double GetYRange() override { return 100.0; }
   int AddPolygonVertex(int, double, double) override { return DEVICE_OK; }
   int DeletePolygons() override { return DEVICE_OK; }
   int RunSequence() override { return DEVICE_OK; }
   int LoadPolygons() override { return DEVICE_OK; }
   int SetPolygonRepetitions(int) override { return DEVICE_OK; }
   int RunPolygons() override { return DEVICE_OK; }
   int StopSequence() override { return DEVICE_OK; }
   int GetChannel(char* channelName) override {
      CDeviceUtils::CopyLimitedString(channelName, "");
      return DEVICE_OK;
   }
};
