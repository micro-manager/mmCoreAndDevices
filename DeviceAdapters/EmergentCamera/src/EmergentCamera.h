// Copyright (c) 2026, Dr. Volodymyr Cherkas
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "DeviceBase.h"
#include "DeviceThreads.h"
#include "ImgBuffer.h"

#include <EmergentCameraAPIs.h>
#include <emergentframe.h>
#include <gigevisiondeviceinfo.h>

#include <atomic>
#include <memory>
#include <string>
#include <utility>
#include <vector>

extern const char* g_EmergentCameraName;

class EmergentSequenceThread;

class EmergentCamera : public CCameraBase<EmergentCamera>
{
public:
   EmergentCamera();
   ~EmergentCamera();

   // MM::Device API
   int Initialize();
   int Shutdown();
   void GetName(char* name) const;
   bool Busy() { return false; }
   bool GetPropertyName(unsigned index, char* name) const override;

   // MM::Camera API
   int SnapImage();
   const unsigned char* GetImageBuffer();
   unsigned GetImageWidth() const { return width_; }
   unsigned GetImageHeight() const { return height_; }
   unsigned GetImageBytesPerPixel() const { return bytesPerPixel_; }
   unsigned GetBitDepth() const { return bitDepth_; }
   long GetImageBufferSize() const;
   unsigned GetNumberOfComponents() const { return 1; }

   double GetExposure() const;
   void SetExposure(double exposureMs);
   int GetBinning() const { return binning_; }
   int SetBinning(int binSize);

   int SetROI(unsigned x, unsigned y, unsigned xSize, unsigned ySize);
   int GetROI(unsigned& x, unsigned& y, unsigned& xSize, unsigned& ySize);
   int ClearROI();
   bool SupportsMultiROI() { return false; }
   bool IsMultiROISet() { return false; }
   int GetMultiROICount(unsigned& count);
   int SetMultiROI(const unsigned*, const unsigned*, const unsigned*, const unsigned*, unsigned);
   int GetMultiROI(unsigned*, unsigned*, unsigned*, unsigned*, unsigned*);

   int StartSequenceAcquisition(double intervalMs);
   int StartSequenceAcquisition(long numImages, double intervalMs, bool stopOnOverflow);
   int StopSequenceAcquisition();
   bool IsCapturing();

   int IsExposureSequenceable(bool& isSequenceable) const;
   int GetExposureSequenceMaxLength(long& nrEvents) const;
   int StartExposureSequence();
   int StopExposureSequence();
   int ClearExposureSequence();
   int AddToExposureSequence(double exposureTimeMs);
   int SendExposureSequence() const;

   // Property handlers
   int OnIPAddress(MM::PropertyBase* property, MM::ActionType action);
   int OnCameraSelection(MM::PropertyBase* property, MM::ActionType action);
   int OnConnectionSpeed(MM::PropertyBase* property, MM::ActionType action);
   int OnSettingsFile(MM::PropertyBase* property, MM::ActionType action);
   int OnRememberSettings(MM::PropertyBase* property, MM::ActionType action);
   int OnSettingsStatus(MM::PropertyBase* property, MM::ActionType action);
   int OnSaveSettings(MM::PropertyBase* property, MM::ActionType action);
   int OnExposure(MM::PropertyBase* property, MM::ActionType action);
   int OnFrameRateLimit(MM::PropertyBase* property, MM::ActionType action);
   int OnFrameRate(MM::PropertyBase* property, MM::ActionType action);
   int OnPixelType(MM::PropertyBase* property, MM::ActionType action);
   int OnGain(MM::PropertyBase* property, MM::ActionType action);
   int OnAutoGain(MM::PropertyBase* property, MM::ActionType action);
   int OnBinning(MM::PropertyBase* property, MM::ActionType action);
   int OnADC(MM::PropertyBase* property, MM::ActionType action);
   int OnRoiX(MM::PropertyBase* property, MM::ActionType action);
   int OnRoiY(MM::PropertyBase* property, MM::ActionType action);
   int OnRoiWidth(MM::PropertyBase* property, MM::ActionType action);
   int OnRoiHeight(MM::PropertyBase* property, MM::ActionType action);
   int OnRoiCenterX(MM::PropertyBase* property, MM::ActionType action);
   int OnRoiCenterY(MM::PropertyBase* property, MM::ActionType action);
   int OnUInt32Feature(MM::PropertyBase* property, MM::ActionType action,
                       long featureIndex);
   int OnBoolFeature(MM::PropertyBase* property, MM::ActionType action,
                     long featureIndex);
   int OnEnumFeature(MM::PropertyBase* property, MM::ActionType action,
                     long featureIndex);
   int OnTriggerMode(MM::PropertyBase* property, MM::ActionType action);
   int OnTriggerInput(MM::PropertyBase* property, MM::ActionType action);
   int OnTriggerInputEdge(MM::PropertyBase* property, MM::ActionType action);
   int OnTriggerEndMode(MM::PropertyBase* property, MM::ActionType action);
   int OnTriggerEndEdge(MM::PropertyBase* property, MM::ActionType action);
   int OnTriggerDelay(MM::PropertyBase* property, MM::ActionType action);
   int OnOutput0Source(MM::PropertyBase* property, MM::ActionType action);
   int OnOutput0Polarity(MM::PropertyBase* property, MM::ActionType action);
   int OnOutput1Source(MM::PropertyBase* property, MM::ActionType action);
   int OnOutput1Polarity(MM::PropertyBase* property, MM::ActionType action);
   int OnSensorTemperature(MM::PropertyBase* property, MM::ActionType action);
   int OnCameraTemperature(MM::PropertyBase* property, MM::ActionType action);
   int OnPhyTemperature(MM::PropertyBase* property, MM::ActionType action);
   int OnPtpOffset(MM::PropertyBase* property, MM::ActionType action);
   int OnCameraTimestamp(MM::PropertyBase* property, MM::ActionType action);

   // Called only by EmergentSequenceThread.
   int SequenceLoop(long numImages, double intervalMs, bool stopOnOverflow);
   void SequenceFinished(int status) throw();

private:
   friend class EmergentSequenceThread;

   enum ErrorCode
   {
      ERR_CAMERA_NOT_FOUND = 10001,
      ERR_ESDK_CALL = 10002,
      ERR_UNSUPPORTED_PIXEL_FORMAT = 10003,
      ERR_ACQUISITION_TIMEOUT = 10004,
      ERR_INVALID_ROI = 10005,
      ERR_FRAME_FORMAT = 10006,
      ERR_ACQUISITION_STOPPED = 10007,
      ERR_SETTINGS_FILE = 10008
   };

   int FindCameraByIp(GigEVisionDeviceInfo& result);
   void DiscoverCameras();
   int ConfigureCameraLocked();
   int ConfigureGpioDefaultsLocked();
   int SaveSettingsLocked();
   int RestoreSettingsLocked();
   static std::string DefaultSettingsFile();
   int CreateRuntimeProperties();
   int CreateAdditionalFeatureProperties();
   int ReadCameraIdentityLocked();
   int ReadPixelFormatsLocked();
   int RefreshGeometryLocked();
   void RefreshGeometryLimitsLocked();
   void PublishGeometryState();
   unsigned UsableUInt32Maximum(const char* sdkName, unsigned maximum) const;
   int ConfigureFullFrameLocked();
   int ConfigureRoiLocked(unsigned x, unsigned y, unsigned width, unsigned height);
   int SetPixelFormatLocked(const std::string& format);
   std::vector<std::string> ReadEnumValuesLocked(const char* name);
   int ConfigureTriggerModeLocked(const std::string& mode);
   int FireSoftwareTriggerLocked();
   int ApplyRoiLocked(unsigned x, unsigned y, unsigned width, unsigned height);
   int ApplyFrameRatePolicyLocked(bool allowWrite = true);
   int RefreshExposureLimitsLocked(bool relaxFrameRate);
   void PublishFrameRateState();
   int OnEnumProperty(MM::PropertyBase* property, MM::ActionType action,
                      const char* sdkName);
   int OnBoolPolarityProperty(MM::PropertyBase* property, MM::ActionType action,
                              const char* sdkName);
   int OnTemperatureProperty(MM::PropertyBase* property, MM::ActionType action,
                             const char* sdkName);

   int OpenStreamLocked();
   void CloseStreamLocked();
   int StartAcquisitionLocked();
   void StopAcquisitionLocked();
   void DrainCompletedFramesLocked();
   int AcquireOneFrameLocked(bool interruptible);
   int CopyFrameToImageLocked(const Emergent::CEmergentFrame& frame);
   int InsertCurrentImage(bool stopOnOverflow);

   int SdkError(const char* operation, EVT_ERROR error) const;
   static std::vector<std::string> SplitEnumValues(const char* values);
   static bool IsSupportedMonoFormat(const std::string& value);
   static PIXEL_FORMAT PixelFormatCode(const std::string& value);
   static unsigned PixelBytes(const std::string& value);
   static unsigned PixelBits(const std::string& value);
   static unsigned AlignDown(unsigned value, unsigned increment);
   static unsigned ClampAndAlignSize(unsigned requested, unsigned minimum,
                                     unsigned maximum, unsigned increment);

   mutable MMThreadLock cameraLock_;
   mutable MMThreadLock imageLock_;
   Emergent::CEmergentCamera camera_;
   GigEVisionDeviceInfo deviceInfo_{};
   std::vector<std::unique_ptr<Emergent::CEmergentFrame>> frameBuffers_;
   ImgBuffer image_;
   std::unique_ptr<EmergentSequenceThread> sequenceThread_;

   std::string ipAddress_;
   std::string cameraSelection_;
   std::string model_;
   std::string serial_;
   std::string firmware_;
   std::string pixelFormat_;
   std::vector<std::string> pixelFormats_;
   std::vector<std::pair<std::string, std::string>> discoveredCameras_;
   std::string triggerMode_;

   bool initialized_;
   bool propertiesCreated_;
   bool cameraOpen_;
   bool streamOpen_;
   bool acquisitionRunning_;
   bool softwareTriggerPending_;
   bool softwareTriggerPerFrame_;
   bool hardwareBinningSupported_;
   bool frameRateSupported_;
   bool frameRateLimitEnabled_;
   bool rememberSettings_;
   bool restoringSettings_;
   bool settingsFileHealthy_;
   std::string settingsFile_;
   std::string settingsStatus_;

   unsigned sensorWidth_;
   unsigned sensorHeight_;
   unsigned width_;
   unsigned height_;
   unsigned offsetX_;
   unsigned offsetY_;
   unsigned widthMin_;
   unsigned heightMin_;
   unsigned widthInc_;
   unsigned heightInc_;
   unsigned offsetXInc_;
   unsigned offsetYInc_;
   unsigned bytesPerPixel_;
   unsigned bitDepth_;
   int binning_;
   double exposureMs_;
   double gainDb_;
   double exposureMinimumMs_;
   double exposureMaximumMs_;
   unsigned requestedFrameRateFps_;
   unsigned frameRateFps_;
   unsigned frameRateMinimumFps_;
   unsigned frameRateMaximumFps_;
   unsigned sequenceBurstSize_;
   unsigned long long frameTimestampNs_;
   unsigned long long frameId_;
   MM::MMTime sequenceStartTime_;
};

class EmergentSequenceThread : public MMDeviceThreadBase
{
public:
   explicit EmergentSequenceThread(EmergentCamera* camera);

   void Start(long numImages, double intervalMs, bool stopOnOverflow);
   void Stop();
   bool IsStopped() const;

private:
   int svc() throw();

   EmergentCamera* camera_;
   std::atomic<bool> stop_;
   long numImages_;
   double intervalMs_;
   bool stopOnOverflow_;
};
