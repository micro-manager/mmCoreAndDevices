///////////////////////////////////////////////////////////////////////////////
// FILE:          MMTUCam.h
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
//                Karl Hoover (stuff such as programmable CCD size  & the various image processors)
//                Arther Edelstein ( equipment error simulation)
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

#ifndef _MMTUCAM_H_
#define _MMTUCAM_H_

#include <windows.h>
#include "./sdk3/inc/TucApi.h"
#include "./sdk3/inc/TucVendorApi.h"
#include "TuCamTypes.h"
#include "DeviceBase.h"
#include "ImgBuffer.h"
#include "DeviceThreads.h"
#include "DecoderCallbackRaw.h"
#include <string>
#include <map>
#include <vector>
#include <algorithm>
#include <queue>
#include <memory>
#include <mutex>
#include <atomic>
#include <thread>
#include <condition_variable>

class TuEnumProperty;
class TuIntegerProperty;
class TuFloatProperty;
class TuBooleanProperty;
class TuCommandProperty;
class TuCstringProperty;

typedef std::vector<TuEnumProperty *>    VECTOR_ENUM;
typedef std::vector<TuIntegerProperty *> VECTOR_INTEGER;
typedef std::vector<TuFloatProperty *>   VECTOR_FLOAT;
typedef std::vector<TuBooleanProperty *> VECTOR_BOOLEAN;
typedef std::vector<TuCommandProperty *> VECTOR_COMMAND;
typedef std::vector<TuCstringProperty *> VECTOR_STRING;
//////////////////////////////////////////////////////////////////////////////
// Error codes
//
#define ERR_UNKNOWN_MODE         102
#define ERR_UNKNOWN_POSITION     103
#define ERR_IN_SEQUENCE          104
#define ERR_SEQUENCE_INACTIVE    105
#define ERR_STAGE_MOVING         106
#define HUB_NOT_AVAILABLE        107

///const char* NoHubError = "Parent Hub not defined.";

// Defines which segments in a seven-segment display are lit up for each of
// the numbers 0-9. Segments are:
//
//  0       1
// 1 2     2 4
//  3       8
// 4 5    16 32
//  6      64
const int SEVEN_SEGMENT_RULES[] = {1+2+4+16+32+64, 4+32, 1+4+8+16+64,
      1+4+8+32+64, 2+4+8+32, 1+2+8+32+64, 2+8+16+32+64, 1+4+32,
      1+2+4+8+16+32+64, 1+2+4+8+32+64};
// Indicates if the segment is horizontal or vertical.
const int SEVEN_SEGMENT_HORIZONTALITY[] = {1, 0, 0, 1, 0, 0, 1};
// X offset for this segment.
const int SEVEN_SEGMENT_X_OFFSET[] = {0, 0, 1, 0, 0, 1, 0};
// Y offset for this segment.
const int SEVEN_SEGMENT_Y_OFFSET[] = {0, 0, 0, 1, 1, 1, 2};

////////////////////////
// DemoHub
//////////////////////
#define TUDBG_PRINTF(format,...)	{char dbgMsg[1024] = {0}; sprintf_s(dbgMsg, "" format"", ##__VA_ARGS__); OutputDebugString(dbgMsg);}

//#define ENABLE_FPS_STATISTICS // 帧率统计功能开关

///////////////////////
// CameraPID
///////////////////////
#define DHYANA_400D_X45     0x6404
#define DHYANA_D95_X45      0x6405
#define DHYANA_400DC_X45    0x6804
#define DHYANA_400DC_X100   0xEC03
#define DHYANA_400D_X100    0xE404
#define DHYANA_400BSIV1     0xE405
#define DHYANA_D95_X100     0xE406
#define DHYANA_400BSIV2     0xE408
#define PID_FL_20BW         0xE40D
#define DHYANA_D95_V2       0xE40F
#define DHYANA_401D         0xE005
#define ARIES_6510          0xE429
#define ARIES_6506          0xE42E
#define ARIES_6504          0xE440
#define ARIES_6504LT        0xE00E
#define LEO_3243            0xE438
#define LEO_3243_PRO        0xE42D
#define LEO_5514            0xE430
#define LIBRA_3412M         0xC401
#define LIBRA_3412C         0xCC01
#define LIBRA_3405M         0xC402
#define LIBRA_3412UC        0xC801
#define LIBRA_3412UM        0xC009
#define LIBRA_3405UC        0xC802
#define LIBRA_3405UM        0xC00A

#define GENICAM_PID_ADDRESS 0x80000

///////////////////////
// ImgMode
///////////////////////
#define MODE_12BIT    0x00
#define MODE_CMS      0x01
#define MODE_11BIT    0x02
#define MODE_GLRESET  0x03

class DemoHub : public HubBase<DemoHub>
{
public:
   DemoHub() :
      initialized_(false),
      busy_(false)
   {}
   ~DemoHub() {}

   // Device API
   // ---------
   int Initialize();
   int Shutdown() {return DEVICE_OK;};
   void GetName(char* pName) const; 
   bool Busy() { return busy_;} ;

   // HUB api
   int DetectInstalledDevices();

private:
   void GetPeripheralInventory();

   std::vector<std::string> peripherals_;
   bool initialized_;
   bool busy_;
};


//////////////////////////////////////////////////////////////////////////////
// CMMTUCam class
// Simulation of the Camera device
//////////////////////////////////////////////////////////////////////////////

class CTUCamThread;

class CMMTUCam : public CCameraBase<CMMTUCam>  
{
public:
    CMMTUCam();
    ~CMMTUCam();
  
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

    int PrepareSequenceAcqusition() { return DEVICE_OK; }
    int StartSequenceAcquisition(double interval);
    int StartSequenceAcquisition(long numImages, double interval_ms, bool stopOnOverflow);
    int StopSequenceAcquisition();
    int InsertImage();
    int RunSequenceOnThread(MM::MMTime startTime);
    bool IsCapturing();
    void OnThreadExiting() throw();
    double GetNominalPixelSizeUm() const {return nominalPixelSizeUm_;}
    // GetPixelSizeUm() is final in MMCore 74, concrete cameras must not override it
    bool Busy() override { return false; }
    int GetBinning() const;
    int SetBinning(int bS);

    int IsExposureSequenceable(bool& isSequenceable) const;
    int GetExposureSequenceMaxLength(long& nrEvents) const;
    int StartExposureSequence();
    int StopExposureSequence();
    int ClearExposureSequence();
    int AddToExposureSequence(double exposureTime_ms);
    int SendExposureSequence() const;

    unsigned  GetNumberOfComponents() const { return nComponents_;};

    // action interface
    // ----------------
    // floating point read-only properties for testing
	int OnPixelType(MM::PropertyBase* pProp, MM::ActionType eAct); 
    int OnReadoutTime(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnScanMode(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnErrorSimulation(MM::PropertyBase* , MM::ActionType );
    int OnCameraCCDXSize(MM::PropertyBase* , MM::ActionType );
    int OnCameraCCDYSize(MM::PropertyBase* , MM::ActionType );
    int OnTriggerDevice(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnDropPixels(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnFastImage(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnSaturatePixels(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnFractionOfPixelsToDropOrSaturate(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnShouldRotateImages(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnShouldDisplayImageNumber(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnStripeWidth(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnCCDTemp(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnIsSequenceable(MM::PropertyBase* pProp, MM::ActionType eAct);

	bool OnCapLiving() { return m_bLiving; }
    void OnCaptureStart();
	void OnCaptureStop();
	void UpdateLevelsRange();

	void PropertyReadOnly(MM::PropertyBase* pProp, const char* featureName, TUC_FEATURE_ACCESS access);
	int PropertyChanged(const char* propName, const char* propValue);

    int UpdateImageBuffer();

    TUC_HDEVICE GetDeviceHandle() const { return m_hDevice; }

private:

    void TestResourceLocking(const bool);
    void GenerateEmptyImage(ImgBuffer& img);
    void GenerateSyntheticImage(ImgBuffer& img, double exp);
    int ResizeImageBuffer();
	
    static const double nominalPixelSizeUm_;

    double exposureMaximum_;
    double exposureMinimum_;
    double dPhase_;
    ImgBuffer img_;
    std::unique_ptr<UCHAR[]> decodedImg_;
    bool busy_;
    bool stopOnOverFlow_;
    bool initialized_;
    double readoutUs_;
    MM::MMTime readoutStartTime_;
    long scanMode_;
    int bitDepth_;
    unsigned roiX_;
    unsigned roiY_;
    MM::MMTime sequenceStartTime_;
    bool isSequenceable_;
    long sequenceMaxLength_;
    bool sequenceRunning_;
    unsigned long sequenceIndex_;
    double GetSequenceExposure();
    std::vector<double> exposureSequence_;
    long imageCounter_;
    long binSize_;
    long cameraCCDXSize_;
    long cameraCCDYSize_;
    double ccdT_;
    std::string triggerDevice_;

    bool stopOnOverflow_;

    bool dropPixels_;
    bool fastImage_;
    bool saturatePixels_;
    double fractionOfPixelsToDropOrSaturate_;
    bool shouldRotateImages_;
    bool shouldDisplayImageNumber_;
    double stripeWidth_;

    double testProperty_[10];
    MMThreadLock imgPixelsLock_;
    int nComponents_;
    bool returnToSoftwareTriggers_;

    friend class CTUCamThread;
    CTUCamThread * thd_;

    std::mutex queueLock_;
    std::condition_variable queueCV_;
    std::queue<TUC_HBUFFER> queueBuffers_;
    bool callbackFlag_ = false;
    const int cycBufSize_ = 32;
    DecoderCallbackRaw decoder_;

	std::mutex frameCopyLock_;
    std::mutex frameQueueLock_;
    std::condition_variable frameQueueCV_;
    std::queue<TuCamImageFrame> frameQueue_;
    std::atomic<bool> frameProcessingThreadRunning_;
    std::thread frameProcessingThread_;

    // Callback path: N workers Convert → host pool → Queue (WF-C-02 style).
    // convertLock_ serializes Convert on one device; memcpy/Queue overlap across workers.
    static const int kConvertWorkerCount = 2;
    std::atomic<bool> convertWorkersRunning_;
    std::vector<std::thread> convertWorkers_;
    std::mutex convertLock_;
    std::mutex convertWaitLock_;
    std::condition_variable convertWaitCV_;
    size_t convertReady_;
    std::atomic<bool> convertFailed_;    
    
#ifdef ENABLE_FPS_STATISTICS
    // 统计变量：用于统计每秒执行次数
    std::atomic<unsigned long long> callbackCount_;           // OnCallbackBuffer调用次数
    std::atomic<unsigned long long> processFrameCount_;      // ProcessFrameAndInsertImage调用次数
    std::atomic<unsigned long long> waitForFrameCount_;      // WaitForFrame调用次数
    MM::MMTime statsStartTime_;                              // 统计开始时间
    MM::MMTime lastStatsOutputTime_;                         // 上次输出统计信息的时间
    std::mutex statsLock_;                                   // 统计锁
#endif

private:
	void TestImage(ImgBuffer& img, double exp);

	static void TUC_CALLBACK OnCallbackBuffer(TUC_HDEVICE hDevice, TUC_HBUFFER hBuffer, void *userContext);
    void ResetCallback();
    int QueueAllAnnouncedBuffers();
    void RequeueBuffer(TUC_HBUFFER hBuffer);
    // Keep most SDK buffers in INPUT. Ring only holds a few handles.
    size_t HandleRingCapacity() const;
    bool HostPoolHasFreeSlot();
    uint8_t* AcquireHostFrame(uint32_t bytes);
    void ReleaseHostFrame(uint8_t* buffer);
    void FreeHostFramePool();
    void FreeHostFramePoolLocked();
    int StartCaptureWithCallback();

    int InitTUCamApi();
    int UninitTUCamApi();

    int AllocBuffer();
    int ReleaseBuffer();

    int StopCapture();
    int StartCapture();
    int WaitForFrame(ImgBuffer& img);
    int WaitForFrameInSnap();
    void CopyImg(const TuCamImageFrame& frame);
    void FillFrameFromImageInfo(const TUC_IMAGE_INFO& info, TuCamImageFrame& frame);
    void UpdateFrameGeometryFromFeatures();
    bool TrySetFeatureBool(const char* name, bool value);
    bool TrySetFeatureFloat(const char* name, double value);
    bool TryGetFeatureInt64(const char* name, int64_t* value) const;
    bool TrySetFeatureInt64(const char* name, int64_t value);
    bool TryGetFeatureFloat(const char* name, double* value) const;

    void StartFrameProcessingThread();
    void StopFrameProcessingThread();
    void FrameProcessingThreadFunc();
    void StartConvertWorkers();
    void StopConvertWorkers();
    void ConvertWorkerFunc();
    int ProcessFrameAndInsertImage(TuCamImageFrame frame);
    void ReleaseOwnedFrame(TuCamImageFrame& frame);

    bool SaveRaw(char *pfileName, unsigned char *pData, unsigned long ulSize);
    int GetPID();

	static int   	s_nNumCam;
	static int		s_nCntCam;

	int             m_nPID;
	int             m_nBCD;
    int             m_nIdxGain;
    int             m_nMaxHeight;
    char            m_szImgPath[MAX_PATH];
    float           m_fCurTemp;
    float           m_fValTemp;
    int             m_nMidTemp;
    bool            m_bROI;
    bool            m_bSaving;
    bool            m_bTemping;
    bool            m_bLiving;
	HANDLE          m_hThdWaitEvt;
    HANDLE          m_hThdTempEvt;

    TUC_HDEVICE      m_hDevice;
    TUC_HBUFFERS     m_hBuffers;
    TuCamImageFrame  m_frame;
    uint32_t         m_bufferCount;
    bool             converterCreated_;
    std::mutex       hostPoolLock_;
    std::vector<uint8_t*> hostFramePool_;
    std::vector<uint8_t*> hostFrameFree_;
    uint32_t         hostFrameBytes_;

	VECTOR_ENUM      m_vecEnum;
    VECTOR_INTEGER   m_vecInteger;
    VECTOR_FLOAT     m_vecFloat;
    VECTOR_BOOLEAN   m_vecBoolean;
    VECTOR_COMMAND   m_vecCommand;
	VECTOR_STRING    m_vecString; 

    int m_pid;
};

class CTUCamThread : public MMDeviceThreadBase
{
   friend class CMMTUCam;
   enum { default_numImages=1, default_intervalMS = 100 };
   public:
      CTUCamThread(CMMTUCam* pCam);
      ~CTUCamThread();
      void Stop();
      void Start(long numImages, double intervalMs);
      bool IsStopped();
      void Suspend();
      bool IsSuspended();
      void Resume();
      bool HasTakenSpecifiedPhotos();
      double GetIntervalMs(){return intervalMs_;}                               
      void SetLength(long images) {numImages_ = images;}                        
      long GetLength() const {return numImages_;}
      long GetImageCounter(){return imageCounter_;}                             
      MM::MMTime GetStartTime(){return startTime_;}                             
      MM::MMTime GetActualDuration(){return actualDuration_;}
   private:                                                                     
      int svc(void) throw();
      double intervalMs_;                                                       
      long numImages_;                                                          
      long imageCounter_;                                                       
      std::atomic<bool> stop_;
      std::atomic<bool> suspend_;
      CMMTUCam* camera_;                                                     
      MM::MMTime startTime_;                                                    
      MM::MMTime actualDuration_;                                               
      MM::MMTime lastFrameTime_;                                                
      MMThreadLock stopLock_;                                                   
      MMThreadLock suspendLock_;                                                
}; 


#endif //_MMTUCAM_H_