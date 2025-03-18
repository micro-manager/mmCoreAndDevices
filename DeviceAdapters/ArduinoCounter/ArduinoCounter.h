 //////////////////////////////////////////////////////////////////////////////
// FILE:          ArduinoCounter.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Adapter for Arduino board with pulse counting firmware
//
// AUTHOR:        Nico Stuurman, nstuurman@altoslabs.com    2023/07/10
//
//

#ifndef _ArduinoCounter_H_
#define _ArduinoCounter_H_

#include "MMDevice.h"
#include "DeviceBase.h"
#include "ImgBuffer.h"
#include <string>
#include <map>

//////////////////////////////////////////////////////////////////////////////
// Error codes
//
#define ERR_UNKNOWN_POSITION 101
#define ERR_INITIALIZE_FAILED 102
#define ERR_WRITE_FAILED 103
#define ERR_CLOSE_FAILED 104
#define ERR_BOARD_NOT_FOUND 105
#define ERR_PORT_OPEN_FAILED 106
#define ERR_COMMUNICATION 107
#define ERR_NO_PORT_SET 108
#define ERR_VERSION_MISMATCH 109
#define ERR_INVALID_DEVICE_NAME            10001
#define ERR_NO_PHYSICAL_CAMERA             10010
#define ERR_NO_EQUAL_SIZE                  10011
#define ERR_FIRMWARE_VERSION_TOO_NEW       10012
#define ERR_FIRMWARE_VERSION_TOO_OLD       10013

//////////////////////////////////////////////////////////////////////////////
// Max number of physical cameras
//
#define MAX_NUMBER_PHYSICAL_CAMERAS       1



/**
 * CameraSnapThread: helper thread for MultiCamera
 */
class CameraSnapThread : public MMDeviceThreadBase
{
public:
   CameraSnapThread() :
      camera_(0),
      started_(false)
   {}

   ~CameraSnapThread() { if (started_) wait(); }

   void SetCamera(MM::Camera* camera) { camera_ = camera; }

   int svc() { camera_->SnapImage(); return 0; }

   void Start() { activate(); started_ = true; }

private:
   MM::Camera* camera_;
   bool started_;
};



/*
 * ArduinoCounter: 
 */
class ArduinoCounterCamera : public CLegacyCameraBase<ArduinoCounterCamera>
{
public:
   ArduinoCounterCamera();
   ~ArduinoCounterCamera();

   int Initialize();
   int Shutdown();

   void GetName(char* name) const;

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
   int OnPhysicalCamera(MM::PropertyBase* pProp, MM::ActionType eAct, long nr);
   int OnBinning(MM::PropertyBase* pProp, MM::ActionType eAct);

   // property handlers
   int OnPort(MM::PropertyBase* pPropt, MM::ActionType eAct);
   int OnLogic(MM::PropertyBase* pPropt, MM::ActionType eAct);
   int OnVersion(MM::PropertyBase* pPropt, MM::ActionType eAct);

private:
   int Logical2Physical(int logical);
   bool ImageSizesAreEqual();
   int startCommunication();
   int startCounting(int number);
   int stopCounting();

   std::vector<std::string> availableCameras_;
   std::vector<std::string> usedCameras_;
   std::vector<int> cameraWidths_;
   std::vector<int> cameraHeights_;
   unsigned int nrCamerasInUse_;
   bool initialized_;
   ImgBuffer img_;
   std::string port_;
   double version_;
   bool portAvailable_;
   bool invert_;
};


#endif //_ArduinoCounter_H_