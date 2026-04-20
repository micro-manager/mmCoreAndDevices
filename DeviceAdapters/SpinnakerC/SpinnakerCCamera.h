// SpinnakerC device adapter
// Translated from the SpinnakerCamera device adapter (by Cairn), which used
// the Spinnaker C++ API; SpinnakerC uses the C API.

#pragma once

#include "DeviceBase.h"
#include "ImgBuffer.h"
#include "DeviceThreads.h"

#include <SpinnakerC.h>
#include <CameraDefsC.h>

#include <string>
#include <vector>

#define SPKRC_ERROR 10002

class SpinnakerCAcquisitionThread;

class SpinnakerCCamera : public CCameraBase<SpinnakerCCamera>
{
public:
   SpinnakerCCamera(const char* deviceName);
   ~SpinnakerCCamera();

   int Initialize();
   int Shutdown();
   void GetName(char* name) const;
   bool Busy() { return false; }

   int SnapImage();
   const unsigned char* GetImageBuffer();
   unsigned GetImageWidth() const;
   unsigned GetImageHeight() const;
   unsigned GetImageBytesPerPixel() const;
   unsigned GetNumberOfComponents() const;
   unsigned GetBitDepth() const;
   long GetImageBufferSize() const;
   double GetExposure() const;
   void SetExposure(double exp);
   int SetROI(unsigned x, unsigned y, unsigned xSize, unsigned ySize);
   int GetROI(unsigned& x, unsigned& y, unsigned& xSize, unsigned& ySize);
   int ClearROI();
   int GetBinning() const;
   int SetBinning(int binSize);
   int IsExposureSequenceable(bool& isSequenceable) const { isSequenceable = false; return DEVICE_OK; }

   int StartSequenceAcquisition(double interval);
   int StartSequenceAcquisition(long numImages, double interval_ms, bool stopOnOverflow);
   int StopSequenceAcquisition();
   bool IsCapturing();

   int MoveImageToCircularBuffer();

   int OnTemperature(MM::PropertyBase* pProp, MM::ActionType eAct);

   int OnPixelFormat(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnTestPattern(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnFrameRateEnabled(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnFrameRateAuto(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnExposureAuto(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnFrameRate(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnVideoMode(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnBinningInt(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnBinningModeEnum(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnADCBitDepth(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnReverseX(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnReverseY(MM::PropertyBase* pProp, MM::ActionType eAct);

   int OnGain(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnGainAuto(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnGammaEnabled(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnGamma(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnBlackLevel(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnBlackLevelAuto(MM::PropertyBase* pProp, MM::ActionType eAct);

   int OnTriggerSelector(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnTriggerMode(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnTriggerSource(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnTriggerActivation(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnTriggerOverlap(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnTriggerDelay(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnExposureMode(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnUserOutputSelector(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnUserOutputValue(MM::PropertyBase* pProp, MM::ActionType eAct);

   int OnLineSelector(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnLineMode(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnLineInverter(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnLineSource(MM::PropertyBase* pProp, MM::ActionType eAct);

private:
   int checkError(spinError err, const char* context);
   int allocateImageBuffer(std::size_t size, int64_t pixelFormatEnumValue);

   friend class SpinnakerCAcquisitionThread;

#pragma pack(push, 1)
   struct Unpack12Struct {
      uint8_t _2;
      uint8_t _1;
      uint8_t _0;
   };
#pragma pack(pop)

   // Node access helpers
   spinError getNodeHandle(const char* name, spinNodeHandle* hNode) const;
   bool isNodeReadable(spinNodeHandle hNode) const;
   bool isNodeWritable(spinNodeHandle hNode) const;
   spinError getEnumSymbolic(spinNodeHandle hNode, std::string& symbolic) const;
   spinError setEnumByName(spinNodeHandle hNode, const char* symbolic) const;
   spinError getEnumSymbolics(spinNodeHandle hNode, std::vector<std::string>& symbolics) const;
   spinError getEnumIntValue(spinNodeHandle hNode, int64_t& value) const;
   spinError getFloatValue(spinNodeHandle hNode, double& value) const;
   spinError setFloatValue(spinNodeHandle hNode, double value) const;
   spinError getIntValue(spinNodeHandle hNode, int64_t& value) const;
   spinError setIntValue(spinNodeHandle hNode, int64_t value) const;
   spinError getIntMin(spinNodeHandle hNode, int64_t& value) const;
   spinError getIntMax(spinNodeHandle hNode, int64_t& value) const;
   spinError getIntInc(spinNodeHandle hNode, int64_t& value) const;
   spinError getBoolValue(spinNodeHandle hNode, bool8_t& value) const;
   spinError setBoolValue(spinNodeHandle hNode, bool8_t value) const;
   spinError executeCommand(spinNodeHandle hNode) const;

   // Property creation/update helpers
   void CreatePropertyFromEnum(const char* nodeName, const char* mmPropName,
      int (SpinnakerCCamera::*fpt)(MM::PropertyBase* pProp, MM::ActionType eAct));
   void CreatePropertyFromFloat(const char* nodeName, const char* mmPropName,
      int (SpinnakerCCamera::*fpt)(MM::PropertyBase* pProp, MM::ActionType eAct));
   void CreatePropertyFromBool(const char* nodeName, const char* mmPropName,
      int (SpinnakerCCamera::*fpt)(MM::PropertyBase* pProp, MM::ActionType eAct));

   int OnEnumPropertyChanged(const char* nodeName,
      MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnFloatPropertyChanged(const char* nodeName,
      MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnBoolPropertyChanged(const char* nodeName,
      MM::PropertyBase* pProp, MM::ActionType eAct);

   void Unpack12Bit(uint16_t* dst, const uint8_t* packed,
      size_t width, size_t height, bool flip);
   void RGBtoBGRA(uint8_t* data, size_t imageBuffLength);

   int64_t getPixelFormatEnumValue() const;
   int64_t getPixelSizeEnumValue() const;

   std::string m_deviceName;
   std::string m_serialNumber;
   spinSystem m_system;
   spinCamera m_cam;
   spinImage m_imagePtr;
   spinNodeMapHandle m_nodeMap;
   unsigned char* m_imageBuff;

   SpinnakerCAcquisitionThread* m_aqThread;
   MMThreadLock m_pixelLock;
   bool m_stopOnOverflow;
};


class SpinnakerCAcquisitionThread : public MMDeviceThreadBase
{
public:
   SpinnakerCAcquisitionThread(SpinnakerCCamera* pCam);
   ~SpinnakerCAcquisitionThread();
   void Stop();
   void Start(long numImages, double intervalMs);
   bool IsStopped();
   void Suspend();
   bool IsSuspended();
   void Resume();
   void SetLength(long images) { m_numImages = images; }
   long GetLength() const { return m_numImages; }
   long GetImageCounter() { return m_imageCounter; }
   MM::MMTime GetStartTime() { return m_startTime; }
   MM::MMTime GetActualDuration() { return m_actualDuration; }
private:
   friend class SpinnakerCCamera;
   int svc(void) throw();
   long m_numImages;
   double m_intervalMs;
   long m_imageCounter;
   bool m_stop;
   bool m_suspend;
   SpinnakerCCamera* m_spkrCam;

   MM::MMTime m_startTime;
   MM::MMTime m_actualDuration;
   MM::MMTime m_lastFrameTime;
   MMThreadLock m_stopLock;
   MMThreadLock m_suspendLock;
};

