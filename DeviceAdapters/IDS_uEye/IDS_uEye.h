///////////////////////////////////////////////////////////////////////////////
// FILE:          IDS_uEye.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Driver for IDS uEye series of USB cameras
//                (also Thorlabs DCUxxxx USB, Edmund EO-xxxxM USB
//                 with IDS hardware)
//
//                based on IDS uEye SDK and Micromanager DemoCamera example
//                tested with SDK version 3.82, 4.02, 4.20 and 4.30
//                (3.82-specific functions are still present but not used)
//                
// AUTHOR:        Wenjamin Rosenfeld
//
// YEAR:          2012 - 2014
//                
// VERSION:       1.3
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
//LAST UPDATE:    09.04.2014 WR



#ifndef _IDS_uEYE_H_
#define _IDS_uEYE_H_

#include "DeviceBase.h"
#include "ImgBuffer.h"
#include "DeviceThreads.h"
#include <string>
#include <map>
#include <algorithm>

#ifdef _MSC_VER
// Disable Visual C++ warnings (caused by const fields in a struct)
#pragma warning(push)
#pragma warning(disable: 4510)
#pragma warning(disable: 4512)
#pragma warning(disable: 4610)
#endif // _MSC_VER

#ifdef __linux__
#include <ueye.h>
#else
#include <uEye.h>
#endif

#ifdef _MSC_VER
#pragma warning(pop)
#endif // _MSC_VER



using namespace std;


//////////////////////////////////////////////////////////////////////////////
// General definitions
//

#define EXPOSURE_MAX 10000               //maximal exposure (ms) to use, even if the camera reports higher values


//////////////////////////////////////////////////////////////////////////////
// Error codes
//
#define ERR_UNKNOWN_MODE         102

#define ERR_CAMERA_NOT_FOUND     103
#define ERR_MEM_ALLOC            105
#define ERR_ROI_INVALID          110


const char* Err_UNKNOWN_MODE =          "Unknown mode";
const char* Err_CAMERA_NOT_FOUND =      "uEye camera not found";
const char* Err_MEM_ALLOC =             "Could not allocate image memory";
const char* Err_ROI_INVALID =           "Invalid ROI size";




////////////////////////////////////////////////////////////////////////////
// Binning mode structure
//
struct binMode{
  string name;          //readable name
  int mode;             //specifies the mode in the language of SDK
  int binX;
  int binY;
};


//available bin modes from ueye.h
const int binModesH[]={IS_BINNING_DISABLE,
               IS_BINNING_2X_HORIZONTAL,
               IS_BINNING_3X_HORIZONTAL,
               IS_BINNING_4X_HORIZONTAL,
               IS_BINNING_5X_HORIZONTAL,
               IS_BINNING_6X_HORIZONTAL,
               IS_BINNING_8X_HORIZONTAL,
               IS_BINNING_16X_HORIZONTAL};

const int binModesV[]={IS_BINNING_DISABLE,
               IS_BINNING_2X_VERTICAL,
               IS_BINNING_3X_VERTICAL,
               IS_BINNING_4X_VERTICAL,
               IS_BINNING_5X_VERTICAL,
               IS_BINNING_6X_VERTICAL,
               IS_BINNING_8X_VERTICAL,
               IS_BINNING_16X_VERTICAL};


const int binModesHNum=sizeof(binModesH)/sizeof(binModesH[0]);
const int binModesVNum=sizeof(binModesV)/sizeof(binModesV[0]);

binMode binModes[binModesHNum][binModesVNum];


/* convert color mode code to user readable string (as defined in uEye.h) */
string colorModeToString(int colorMode){
  
  string outString="";

  switch(colorMode){
     
  case(IS_CM_SENSOR_RAW8):      outString="SENSOR_RAW8"; break;         //Raw sensor data, occupies 8 bits
  case(IS_CM_SENSOR_RAW10):     outString="SENSOR_RAW10"; break;        //Raw sensor data, occupies 16 bits 
  case(IS_CM_SENSOR_RAW12):     outString="SENSOR_RAW12"; break;        //Raw sensor data, occupies 16 bits
  case(IS_CM_SENSOR_RAW16):     outString="SENSOR_RAW16"; break;        //Raw sensor data, occupies 16 bits

  case(IS_CM_MONO8):            outString="MONO8"; break;               //Mono, occupies 8 bit
  case(IS_CM_MONO10):           outString="MONO10"; break;              //Mono, occupies 16 bits
  case(IS_CM_MONO12):           outString="MONO12"; break;              //Mono, occupies 16 bits
  case(IS_CM_MONO16):           outString="MONO16"; break;              //Mono, occupies 16 bits

  case(IS_CM_BGR5_PACKED):      outString="BGR5_PACKED"; break;         //BGR (5 5 5 1), 1 bit not used, occupies 16 bits
  case(IS_CM_BGR565_PACKED):    outString="BGR565_PACKED"; break;       //BGR (5 6 5), occupies 16 bits

  case(IS_CM_RGB8_PACKED):      outString="RGB8_PACKED"; break;         //BGR and RGB (8 8 8), occupies 24 bits
  case(IS_CM_BGR8_PACKED):      outString="BGR8_PACKED"; break;

  case(IS_CM_RGBA8_PACKED):     outString="RGBA8_PACKED"; break;        //BGRA and RGBA (8 8 8 8), alpha not used, occupies 32 bits
  case(IS_CM_BGRA8_PACKED):     outString="BGRA8_PACKED"; break;

  case(IS_CM_RGBY8_PACKED):     outString="RGBY8_PACKED"; break;        //BGRY and RGBY (8 8 8 8), occupies 32 bits
  case(IS_CM_BGRY8_PACKED):     outString="BGRY8_PACKED"; break;

  case(IS_CM_RGB10_PACKED):     outString="RGB10_PACKED"; break;        //BGR and RGB (10 10 10 2), 2 bits not used, occupies 32 bits, debayering is done from 12 bit raw
  case(IS_CM_BGR10_PACKED):     outString="BGR10_PACKED"; break;

  case(IS_CM_RGB10_UNPACKED):   outString="RGB10_UNPACKED"; break;      //BGR and RGB (10(16) 10(16) 10(16)), 6 MSB bits not used respectively, occupies 48 bits
  case(IS_CM_BGR10_UNPACKED):   outString="BGR10_UNPACKED"; break;

  case(IS_CM_RGB12_UNPACKED):   outString="RGB12_UNPACKED"; break;      //BGR and RGB (12(16) 12(16) 12(16)), 4 MSB bits not used respectively, occupies 48 bits
  case(IS_CM_BGR12_UNPACKED):   outString="BGR12_UNPACKED"; break;

  case(IS_CM_RGBA12_UNPACKED):  outString="RGBA12_UNPACKED"; break;     //BGRA and RGBA (12(16) 12(16) 12(16) 16), 4 MSB bits not used respectively, alpha not used, occupies 64 bits
  case(IS_CM_BGRA12_UNPACKED):  outString="BGRA12_UNPACKED"; break;

  case(IS_CM_JPEG): outString="JPEG";

  case(IS_CM_UYVY_PACKED):      outString="UYVY_PACKED"; break;         //YUV422 (8 8), occupies 16 bits
  case(IS_CM_UYVY_MONO_PACKED): outString="UYVY_MONO_PACKED"; break;
  case(IS_CM_UYVY_BAYER_PACKED):outString="UYVY_BAYER_PACKED"; break;

  case(IS_CM_CBYCRY_PACKED):    outString="CBYCRY_PACKE"; break;        //YCbCr422 (8 8), occupies 16 bits

  case(IS_CM_RGB8_PLANAR):      outString="RGB8_PLANAR"; break;         //RGB planar (8 8 8), occupies 24 bits


  case(IS_CM_ALL_POSSIBLE):     outString="ALL_POSSIBLE"; break;


  default: outString+="unknown";

  }
  
  return(outString);
}


/* convert trigger mode code to user readable string (as defined in uEye.h) */
string triggerModeToString(int triggerMode){
  
  string outString="";

  switch(triggerMode){
    
  case(IS_SET_TRIGGER_OFF):             outString="TRIGGER_OFF"; break;
  case(IS_SET_TRIGGER_HI_LO):           outString="TRIGGER_HI_LO"; break;
  case(IS_SET_TRIGGER_LO_HI):           outString="TRIGGER_LO_HI"; break;
  case(IS_SET_TRIGGER_SOFTWARE):        outString="TRIGGER_SOFTWARE"; break;   

  default: outString+="unknown";

  }

  return(outString);
}


//////////////////////////////////////////////////////////////////////////////
// CIDS_uEye class
//
//////////////////////////////////////////////////////////////////////////////

class MySequenceThread;

class CIDS_uEye : public CCameraBase<CIDS_uEye>  
{
  
 public:

  CIDS_uEye();                                           //constructor
  ~CIDS_uEye();                                          //destructor
  

  // uEye interface
  // ------------
  HIDS hCam;                                            //handle for the camera
  char* pcImgMem;                                       //image memory
  int memPid;                                           //ID for image memory


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
  
  int PrepareSequenceAcquisition()
  {
    return DEVICE_OK;
  }

  int StartSequenceAcquisition(double interval);
  int StartSequenceAcquisition(long numImages, double interval_ms, bool stopOnOverflow);
  int StopSequenceAcquisition();
  int InsertImage();
  int ThreadRun();
  bool IsCapturing();
  void OnThreadExiting() throw(); 

  double GetNominalPixelSizeUm() const 
  {return nominalPixelSizeUm_;}

  double GetPixelSizeUm() const 
  {return nominalPixelSizeUm_ * GetBinning();}

  int GetBinning() const;
  int SetBinning(int bS);
  int IsExposureSequenceable(bool& isSequenceable) const {isSequenceable = false; return DEVICE_OK;}

  unsigned  GetNumberOfComponents() const
  { return nComponents_;};


  //convenience functions
  void GetPixelClockRange(HIDS hCam, UINT* pixClkMin, UINT* pixClkMax);
  void SetPixelClock(UINT pixClk);
  void GetFramerateRange(HIDS hCam, double* fpsMin, double* fpsMax);
  void GetExposureRange(HIDS hCam, double* expMin, double* expMax, double* expIncrement);


  // action interface
  // ----------------

  // handlers for properties
  int OnBinning(MM::PropertyBase* pProp, MM::ActionType eAct);
  int OnPixelClock(MM::PropertyBase* pProp, MM::ActionType eAct);
  int OnFramerate(MM::PropertyBase* pProp, MM::ActionType eAct);
  int OnExposure(MM::PropertyBase* pProp, MM::ActionType eAct);
  int OnPixelType(MM::PropertyBase* pProp, MM::ActionType eAct);
  int OnReadoutTime(MM::PropertyBase* pProp, MM::ActionType eAct);
  int OnErrorSimulation(MM::PropertyBase* , MM::ActionType );
  int OnCameraCCDXSize(MM::PropertyBase* , MM::ActionType );
  int OnCameraCCDYSize(MM::PropertyBase* , MM::ActionType );
  int OnTrigger(MM::PropertyBase* pProp, MM::ActionType eAct);
  int OnTriggerDevice(MM::PropertyBase* pProp, MM::ActionType eAct);
  int OnDropPixels(MM::PropertyBase* pProp, MM::ActionType eAct);
  int OnFractionOfPixelsToDropOrSaturate(MM::PropertyBase* pProp, MM::ActionType eAct);
  int OnGainMaster(MM::PropertyBase* pProp, MM::ActionType eAct);
  int OnFlashMode(MM::PropertyBase* pProp, MM::ActionType eAct);

 private:

  double nominalPixelSizeUm_;                                   //pixel pitch

  int bitDepthADC_;
  int bitDepthReal_;
  
  double dPhase_;
  ImgBuffer img_;
  UEYEIMAGEINFO imgInfo_;

  bool busy_;
  bool stopOnOverflow_;
  bool initialized_;
  double readoutUs_;
  MM::MMTime readoutStartTime_;
  int bitDepth_;

  unsigned roiX_;                                               //x-position of the ROI corner
  unsigned roiY_;                                               //y-position of the ROI corner
  unsigned roiXSize_;                                           //width of the ROI, screen pixel (incl. binning)
  unsigned roiYSize_;                                           //height of the ROI, screen pixel (incl. binning)
  unsigned roiXSizeReal_;                                       //width of the ROI, CCD pixel
  unsigned roiYSizeReal_;                                       //height of the ROI, CCD pixel
  
  long cameraCCDXSize_;                                         //sensor width in pixels
  long cameraCCDYSize_;                                         //sensor height in pixels


  MM::MMTime sequenceStartTime_;
  long imageCounter_;
  double ccdT_;
  std::string triggerDevice_;

  UINT pixelClkMin_;                                            //minimal pixel clock (MHz)
  UINT pixelClkMax_;                                            //maximal pixel clock (MHz)
  UINT pixelClkCur_;                                            //current (real) pixel clock
  UINT pixelClkDef_;                                            //default pixel clock 

  double framerateMin_;                                         //minimal framerate
  double framerateMax_;                                         //maximal framerate
  double framerateCur_;                                         //current (real) framerate

  double exposureMin_;                                          //minimal allowed exposure
  double exposureMax_;                                          //maximal allowed exposure
  double exposureIncrement_;
  double exposureSet_;                                          //set (desired) exposure value
  double exposureCur_;                                          //current (real) exposure value

  int colorModeDef_;                                            //default color mode

  long gainMaster_;                                             //master gain
  long gainRed_;
  long gainGreen_;
  long gainBlue_;

  long binX_;                                                   //horizontal binning factor
  long binY_;                                                   //vertical binning factor
  string binModeName_;                                          //name of the current binning mode

  

  bool dropPixels_;
  bool saturatePixels_;
  double fractionOfPixelsToDropOrSaturate_;
  
  double testProperty_[10];
  MMThreadLock* pDemoResourceLock_;
  MMThreadLock imgPixelsLock_;
  int nComponents_;
  friend class MySequenceThread;
  MySequenceThread * thd_;

  
  int SetAllowedBinning();
  void TestResourceLocking(const bool);
  void ClearImageBuffer(ImgBuffer& img);
  int ResizeImageBuffer();

  int SetImageMemory();

  int setSensorPixelParameters(WORD sensorID);
  



};


class MySequenceThread : public MMDeviceThreadBase
{

  friend class CIDS_uEye;
  enum { default_numImages=1, default_intervalMS = 100 };

 public:
  MySequenceThread(CIDS_uEye* pCam);
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
  CIDS_uEye* camera_;                                                     
  bool stop_;                                                               
  bool suspend_;                                                            
  long numImages_;                                                          
  long imageCounter_;                                                       
  double intervalMs_;                                                       
  MM::MMTime startTime_;                                                    
  MM::MMTime actualDuration_;                                               
  MM::MMTime lastFrameTime_;                                                
  MMThreadLock stopLock_;                                                   
  MMThreadLock suspendLock_;
}; 




#endif //_IDS_UEYE_H_
