///////////////////////////////////////////////////////////////////////////////
// FILE:          MMCamera.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Skeleton code for the micro-manager camera adapter. Use it as
//                starting point for writing custom device adapters
//                
// AUTHOR:        Nenad Amodaj, http://nenad.amodaj.com
//                
// COPYRIGHT:     University of California, San Francisco, 2011
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

#ifndef _MMCAMERA_H_
#define _MMCAMERA_H_

#include "DeviceBase.h"
#include "ImgBuffer.h"
#include "DeviceThreads.h"

#include "AtikCameras.h"

#include <fstream>
#include <ctime>
#include <string>

class SequenceThread;

class Atik : public CLegacyCameraBase<Atik>  
{
public:
   Atik();
   ~Atik();
  
   // MMDevice API
   // ------------
   int Initialize();
   int Shutdown();
  
   void GetName(char* name) const;      
   
   // MMCamera API
   // ------------
   int SnapImage();
   const unsigned char* GetImageBuffer();
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
   bool IsCapturing();
   int GetBinning() const;
   int SetBinning(int binSize);
   int IsExposureSequenceable(bool& seq) const {seq = false; return DEVICE_OK;}

   // action interface
   // ----------------
   int OnBinning(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnSensorTemp(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnCoolingEnable(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnCoolingTargetTemp(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnCoolingPower(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnPixelType(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnGain(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnOffset(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnDarkMode(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnTrigger(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnPreview(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnExposureMode(MM::PropertyBase* pProp, MM::ActionType eAct);

private:
   friend class SequenceThread;

   SequenceThread* thd_;
   int binning_;
   int bytesPerPixel_;
   int gain_;
   int offset_;
   double exposureMs_;
   bool initialized_;
   ImgBuffer img_;
   int roiX_, roiY_;
   std::string exposureMode_;

   ArtemisHandle handle;
   enum GO_Type {FX2, FX3};
   int width;
   int height;
   int setGOType;
   long coolingEnabled;
   long darkModeEnabled;
   long triggerEnabled;
   long previewEnabled;
   long coolingTargetTemp;
   long coolingPower;
   char modelName[128];
   bool enableLogging;

   bool hasPowerlvl;
   bool hasSetPoint;

   int ResizeImageBuffer();
   int InsertImage();

   int initialiseAtikCamera();

   template<typename ... Args>
   void log(const char* format, Args ... args) const;
};

class SequenceThread : public MMDeviceThreadBase
{
   public:
      SequenceThread(Atik* pCam);
      ~SequenceThread();
      void Stop();
      void Start(long numImages, double intervalMs);
      bool IsStopped();
      double GetIntervalMs(){return intervalMs_;}                               
      void SetLength(long images) {numImages_ = images;}                        
      long GetLength() const {return numImages_;}
      long GetImageCounter(){return imageCounter_;} 

   private:                                                                     
      int svc(void) throw();
      Atik* camera_;                                                     
      bool stop_;                                                               
      long numImages_;                                                          
      long imageCounter_;                                                       
      double intervalMs_;                                                       
}; 

#endif //_MMCAMERA_H_
