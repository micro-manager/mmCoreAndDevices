///////////////////////////////////////////////////////////////////////////////
// FILE:          AndorSDK3Camera.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   The example implementation of the demo camera.
//                Simulates generic digital camera and associated automated
//                microscope devices and enables testing of the rest of the
//                system without the need to connect to the actual hardware. 
//                
// AUTHOR:        Nenad Amodaj, nenad@amodaj.com, 06/08/2005
//                
//                Karl Hoover (stuff such as programmable CCD size and transpose processor)
//
// COPYRIGHT:     University of California, San Francisco, 2006
//                100X Imaging Inc, 2008
//
// LICENSE:       This file is distributed under the BSD license.
//                License text is included with the source distribution.
//
//                This file is distributed in the hope that it will be useful,
//                but WITHOUT ANY WARRANTY; without even the implied warranty
//                of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
//
//                IN NO EVENT SHALL THE COPYRIGHT OWNER OR
//                CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
//                INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES.
//
// CVS:           $Id: AndorSDK3Camera.h 6793 2011-03-28 19:10:30Z karlh $
//

#ifndef _ANDORSDK3_H_
#define _ANDORSDK3_H_

#include "DeviceBase.h"
#include "ImgBuffer.h"
#include "DeviceThreads.h"
#include "atcore.h"
#include "IProperty.h"
#include <deque>

class MySequenceThread;
namespace andor {
   class IDevice;
   class IDeviceManager;
   class IEnum;
   class IBufferControl;
   class ICommand;
};

class TEnumProperty;
class TIntegerProperty;
class TFloatProperty;
class TExposureProperty;
class TFloatStringProperty;
class TBooleanProperty;
class TAOIProperty;
class TBooleanPropertyWithPoiseControl;
class SnapShotControl;
class TAndorEnumValueMapper;
class TTriggerRemapper;
class CEventsManager;
class ICallBackManager;

class SRRFControl;
class SRRFAndorSDK3Camera;


//////////////////////////////////////////////////////////////////////////////
// CAndorSDK3Camera class
//////////////////////////////////////////////////////////////////////////////


class CAndorSDK3Camera : public CCameraBase<CAndorSDK3Camera>  
{
public:
   CAndorSDK3Camera();
   ~CAndorSDK3Camera();
  
   int GetNumberOfDevicesPresent() { return number_of_devices_; };

   // MMDevice API
   // ------------
   int Initialize();
   int Shutdown();
  
   void GetName(char* name) const;      

   bool Busy() { return false; }
   
   andor::IDevice * GetCameraDevice() { return cameraDevice; };

   // MMCamera API
   // ------------
   int SnapImage();
   const unsigned char* GetImageBuffer();
   unsigned GetImageWidth() const;
   unsigned GetImageHeight() const;
   unsigned GetImageBytesPerPixel() const;
   unsigned GetBitDepth() const;
   long GetImageBufferSize() const;
   int ResizeImageBuffer();
   double GetExposure() const;
   void SetExposure(double exp);
   int SetROI(unsigned x, unsigned y, unsigned xSize, unsigned ySize); 
   int GetROI(unsigned& x, unsigned& y, unsigned& xSize, unsigned& ySize); 
   int ClearROI();
   int PrepareSequenceAcqusition() {return DEVICE_OK;}
   int StartSequenceAcquisition(double interval);
   int StartSequenceAcquisition(long numImages, double interval_ms, bool stopOnOverflow);
   int StopSequenceAcquisition();
   int InsertImage();
   int ThreadRun();
   bool IsCapturing();
   void OnThreadExiting() throw(); 
   //double GetNominalPixelSizeUm() const {return nominalPixelSizeUm_;}
   int GetBinning() const;
   int SetBinning(int bS);
   int IsExposureSequenceable(bool& isSequenceable) const {isSequenceable = false; return DEVICE_OK;}
   void RestartLiveAcquisition();

   // Used by SRRFControl via SRRFAndorSDK3Camera
   int AddProperty(const char* name, const char* value, MM::PropertyType eType, bool readOnly, MM::ActionFunctor* pAct);
   void ResizeSRRFImage(long radiality);
   
private:
   std::wstring currentSoftwareVersion_;
   std::wstring PerformReleaseVersionCheck();

   void InitialiseSDK3Defaults();
   void UnpackDataWithPadding(unsigned char* _pucSrcBuffer);
   bool InitialiseDeviceCircularBuffer(const unsigned numBuffers);
   bool CleanUpDeviceCircularBuffer();
   int  SetupCameraForSeqAcquisition(long numImages);
   int  CameraStart();
   int  checkForBufferOverflow();
   bool waitForData(unsigned char *& return_buffer, int & buffer_size, bool is_first_frame);
   const unsigned char* GetImageBufferSRRF();
   int InsertImageWithSRRF();
   int AcquireFrameInSequence(bool isFirstFrame);
   int AcquireSRRFImage(bool insertImage, long imageCounter);
   bool IsSRRFEnabled() const;
   int InsertMMImage(const ImgBuffer& image, const Metadata& md);
   std::wstring GetPreferredFeature(std::wstring Name, std::wstring FallbackName) const;
   void AddSimpleEnumProperty(std::wstring Name, std::string DisplayName="");
   void AddSimpleBoolProperty(std::wstring Name, std::string DisplayName="");
   void AddSimpleIntProperty(std::wstring Name, std::string DisplayName="");
   void AddSimpleFloatProperty(std::wstring Name, std::string DisplayName="");
   std::string ToNarrowString(std::wstring wstr) const;

   static const double nominalPixelSizeUm_;
   static const int CID_FPGA_TICKS = 1;

   ImgBuffer img_;
   bool busy_;
   bool initialized_;
   AT_64 sequenceStartTime_;
   AT_64 fpgaTSclockFrequency_;
   AT_64 timeStamp_;
   AT_64 startSRRFImageTime_;
   AT_64 startSRRFSequenceTime_;
   int number_of_devices_;
   int deviceInUseIndex_;
   bool keep_trying_;
   bool in_external_;
   unsigned int currentSeqExposure_;
   bool stopOnOverflow_;

   unsigned char** image_buffers_;
   unsigned int numImgBuffersAllocated_;

   bool b_cameraPresent_;

   bool GetCameraPresent() { return b_cameraPresent_; };
   void SetNumberOfDevicesPresent(int deviceCount) { number_of_devices_ = deviceCount; };

   AT_64 GetTimeStamp(unsigned char* pBuf);

   MMThreadLock* pDemoResourceLock_;
   MMThreadLock imgPixelsLock_;
   void TestResourceLocking(const bool);
   friend class MySequenceThread;
   friend class TAOIProperty;
   MySequenceThread * thd_;
   SnapShotControl* snapShotController_;

   // SRRF
   SRRFControl* SRRFControl_;
   SRRFAndorSDK3Camera* SRRFCamera_;
   ImgBuffer *SRRFImage_;

   // Properties for the property browser
   std::deque<IProperty*> simpleProperties;

   // Specialised properties
   TEnumProperty* binning_property;
   TEnumProperty* pixelEncoding_property;
   TAOIProperty* aoi_property;
   TEnumProperty* triggerMode_property;
   TExposureProperty* exposureTime_property;
   TFloatProperty* frameRate_property;
   TFloatStringProperty* frameRateLimits_property;
   

   // atcore++ objects
   andor::IDeviceManager* deviceManager;
   andor::IDevice* cameraDevice;
   andor::IBufferControl* bufferControl;
   andor::ICommand* startAcquisitionCommand;
   andor::ICommand* sendSoftwareTrigger;

   // Objects used by the properties
   andor::IEnum* triggerMode_Enum;
   TTriggerRemapper* triggerMode_remapper;
   TAndorEnumValueMapper* triggerMode_valueMapper;
   TAndorEnumValueMapper* gateMode_valueMapper;

   CEventsManager* eventsManager_;
   ICallBackManager * callbackManager_;
};

class MySequenceThread : public MMDeviceThreadBase
{
   friend class CAndorSDK3Camera;
   enum { default_numImages=1, default_intervalMS = 0 };
public:
   MySequenceThread(CAndorSDK3Camera* pCam);
   ~MySequenceThread();
   void Stop();
   void Start(long numImages, double intervalMs);
   bool IsStopped();
   void Suspend();
   bool IsSuspended();
   void Resume();
   double GetIntervalMs(){return intervalMs_;}                               
   void SetLength(long images) {numImages_ = images;}                        
   long GetLength() const {return numImages_;}
   long GetImageCounter(){return imageCounter_;}                             
   MM::MMTime GetStartTime(){return startTime_;}                             
   MM::MMTime GetActualDuration(){return actualDuration_;}

private:                                                                     
   int svc(void) throw();
   CAndorSDK3Camera* camera_;                                                     
   bool stop_;                                                               
   bool suspend_;                                                            
   long numImages_;                                                          
   long imageCounter_;                                                       
   double intervalMs_;                                                       
   MM::MMTime startTime_;                                                    
   MM::MMTime actualDuration_;                                               
   MMThreadLock stopLock_;                                                   
   MMThreadLock suspendLock_;                                                
};


#endif //_ANDORSDK3_H_
