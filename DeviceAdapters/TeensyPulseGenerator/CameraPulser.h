#pragma once


#include "DeviceBase.h"
#include "MMDevice.h"
#include "ImgBuffer.h"
#include "TeensyCom.h"

#define ERR_INVALID_DEVICE_NAME            10001
#define ERR_NO_PHYSICAL_CAMERA             10010
#define ERR_NO_EQUAL_SIZE                  10011
#define ERR_FIRMWARE_VERSION_TOO_NEW       10012
#define ERR_FIRMWARE_VERSION_TOO_OLD       10013
#define MAX_NUMBER_PHYSICAL_CAMERAS       1


class CameraPulser : public CLegacyCameraBase<CameraPulser>
{
public:
   CameraPulser();
   ~CameraPulser();

   // MMDevice API
   // ------------
   int Initialize();
   int Shutdown();
  
   void GetName(char* name) const;      
   
   // MMCamera API
   // ------------

   int SnapImage();
   const unsigned char* GetImageBuffer();
   const unsigned char* GetImageBuffer(unsigned channelNr);
   unsigned GetImageWidth() const;
   unsigned GetImageHeight() const;
   unsigned GetImageBytesPerPixel() const;
   unsigned GetBitDepth() const;
   long GetImageBufferSize() const;
   double GetExposure() const;
   void SetExposure(double exp);
   int SetROI(unsigned x, unsigned y, unsigned xSize, unsigned ySize);
   int GetROI(unsigned& x, unsigned& y, unsigned& xSize, unsigned& ySize);
   int ClearROI();
   int PrepareSequenceAcqusition();
   int StartSequenceAcquisition(double interval);
   int StartSequenceAcquisition(long numImages, double interval_ms, bool stopOnOverflow);
   int StopSequenceAcquisition();
   int GetBinning() const;
   int SetBinning(int bS);
   int IsExposureSequenceable(bool& isSequenceable) const;
   unsigned  GetNumberOfComponents() const;
   unsigned  GetNumberOfChannels() const;
   int GetChannelName(unsigned channel, char* name);
   bool IsCapturing();

   // action interface
   // ---------------

   // property handlers
   int OnPort(MM::PropertyBase* pPropt, MM::ActionType eAct);
   int OnPhysicalCamera(MM::PropertyBase* pProp, MM::ActionType eAct, long i);
   int OnBinning(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnVersion(MM::PropertyBase* pProp, MM::ActionType pAct);
   int OnPulseDuration(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnIntervalBeyondExposure(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnWaitForInput(MM::PropertyBase* pProp, MM::ActionType eAct);

private:
   bool ImageSizesAreEqual();
   int Logical2Physical(int logical);

   std::vector<std::string> availableCameras_;
   std::vector<std::string> usedCameras_;
   std::string usedCamera_;
   double intervalBeyondExposure_; // Interval beyond exposure time in ms
   double pulseDuration_; // Pulse duration in milli-seconds
   bool waitForInput_; // Whether to wait sending the pulse for the input to go high
   bool initialized_;
   unsigned int nrCamerasInUse_;
   ImgBuffer img_;
   std::string port_;

   TeensyCom* teensyCom_;

   uint32_t version_;
   uint32_t nrPulses_; // nr Pulses we request
   uint32_t nrPulsesCounted_; // as returned by the Teensy
};
