///////////////////////////////////////////////////////////////////////////////
// FILE:          MMSCCam.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// COPYRIGHT:     HF Agile Device Co., Ltd., 2024
//
// LICENSE:       This file is distributed under the LGPL license.
//                License text is included with the source distribution.

#pragma once

#include "DeviceBase.h"
#include "ImgBuffer.h"
#include "DeviceThreads.h"
#include "SCApi.h"
#include <string>
#include <map>
#include <algorithm>

enum ReadoutMode {
    bit11_HS_Low = 0,
    bit11_HS_High,
    bit11_HDR_Low,
    bit11_HDR_High,
    bit12_HDR_Low,
    bit12_HDR_High,
    bit12_CMS,
    bit16_From11,
    bit16_From12
};

class SCCamera : public CCameraBase<SCCamera>  
{
public:
    SCCamera();
    ~SCCamera();
  
    // MMDevice API
    // ------------
    int Initialize();
    int Shutdown();

    void GetName(char* name) const;      

    bool Busy() { return false; }

    // MMCamera API
    // ------------
    int SnapImage();
    const unsigned char* GetImageBuffer();
    unsigned GetImageWidth() const;
    unsigned GetImageHeight() const ;
    unsigned GetImageBytesPerPixel() const;
    unsigned GetBitDepth() const;
    long GetImageBufferSize() const;
    double GetExposure() const;
    void SetExposure(double exp);
    int SetROI(unsigned x, unsigned y, unsigned xSize, unsigned ySize) ; 
    int GetROI(unsigned& x, unsigned& y, unsigned& xSize, unsigned& ySize); 
    int ClearROI();
    virtual int GetProperty(const char* name, char* value) const;
    virtual int SetProperty(const char* name, const char* value);

    int StartSequenceAcquisition(double interval);
    int StartSequenceAcquisition(long numImages, double interval_ms, bool stopOnOverflow);
    int StopSequenceAcquisition();
    int InsertImage();
    int RunSequenceOnThread(MM::MMTime startTime);
    bool IsCapturing();
    void OnThreadExiting() throw(); 
    int GetBinning() const;
    int SetBinning(int bS);

    int IsExposureSequenceable(bool& isSequenceable) const;
    int GetExposureSequenceMaxLength(long& length) const;
    int StartExposureSequence();
    int StopExposureSequence();
    int ClearExposureSequence();
    int AddToExposureSequence(double exposureTime_ms);
    int SendExposureSequence() const;

	//double LineIntervalTime(int nLineDelayTm);
	//int LineIntervalCal(int nVal, bool bExpChange = true);

    unsigned  GetNumberOfComponents() const;
    //bool Busy() { return true; }
    virtual int ThreadRun(void);

    
    // action interface
    // ----------------
    // floating point read-only properties for testing
    int OnTestProperty(MM::PropertyBase* pProp, MM::ActionType eAct, long){ return 0; };

    int OnBinning(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnBinningSum(MM::PropertyBase* pProp, MM::ActionType eAct){ return 0; };
    int OnExposure(MM::PropertyBase* pProp, MM::ActionType eAct){ return 0; };
	int OnFrameRate(MM::PropertyBase* pProp, MM::ActionType eAct){ return 0; };
    int OnImageMode(MM::PropertyBase* pProp, MM::ActionType eAct){ return 0; };
    int OnPixelType(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnIsSequenceable(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnReadOutMode(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnTriggerIn(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnTriggerSelector(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnTriggerOut(MM::PropertyBase* pProp, MM::ActionType eAct);

    MM::MMTime GetCurrentMMTime()
	{
		return CCameraBase<SCCamera>::GetCurrentMMTime();
	};

private:
    int32_t insertCount_ = 0;

    bool initialized_ = false;
    SC_DEV_HANDLE devHandle_ = nullptr;
    bool isAcqusition_ = false;

    unsigned  xoffset_ = 0;
    unsigned  yoffset_ = 0;
    unsigned  depth_ = 0;
    ImgBuffer image_;
    uint8_t* rawBuffer_ = nullptr;
    SC_Frame  *recvImage_ = nullptr;

    friend class CameraRecvThread;
    CameraRecvThread* thd_ = nullptr;
    MMThreadLock imgPixelsLock_;
    bool stopOnOverflow_;

    bool isSequenceable_ = false;
    int32_t sequenceMaxLength_ = 100;
    bool sequenceRunning_ = false;
    int32_t sequenceIndex_ = 0;
    std::vector<double> exposureSequence_;

    std::vector<std::string>    triggerOutTypes_;
    std::vector<std::string>    triggerActivations_;
    std::vector<std::string>    readOutModes_;

private:
       int ResizeImageBuffer(
                           int imageSizeW, 
                           int imageSizeH, 
                           int byteDepth, 
                           int binSize = 1);

       void getActualRoi(
                        unsigned& actualX, 
                        unsigned& actualY, 
                        unsigned& actualXSize, 
                        unsigned& actualYSize);

       int getDepth(const char* pixelType, int &dyteDepth);

       // acquisition
       int getNextFrame();
       double GetSequenceExposure();
       int convertPixelFormatToDepth(SC_EPixelType pixel);
       int convertBinningModeToBinningValue(int binngingMode) const;
       int convertBinningValueToBinningMode(int binngingValue) const;

       std::string convertReadOutModeStr(const char* name, uint64_t vv) const;
       uint32_t convertReadOutModeEnum(const char* name, const char* vvStr) const;
       uint32_t getReadOutModeEnum(const char*vvStr) const;

       std::string convertTriggerTypeToValue(int32_t type);
       int32_t convertTriggerValueToType(const char* value);
       std::string convertTriggerOutSelectorToVal(int32_t selector);
       int32_t convertTriggerOutSeletorToEnum(const char* selector);
       int32_t convertTriggerOutTypeToEnum(const char* type);

       
       int32_t createTriggerProperty();
       int32_t createTriggerOutProperty();
                                        //const char* ifname,
                                        //const char* triggerOutTypeName,
                                        //const char* triggerOutActName,
                                        //const char* triggerOutDelayName,
                                        //const char* triggerOutPulseWidthName);

       int32_t updateTriggerInActivation(const char *name, const std::vector<std::string> &AllowPropertiesValue);
};


class CameraRecvThread : public MMDeviceThreadBase
{
    friend class SCCamera;
   enum { default_numImages=1, default_intervalMS = 100 };
public:
	CameraRecvThread(SCCamera* pCam);

	~CameraRecvThread();

	void Stop();

	void Start(long numImages, double intervalMs);
    bool IsStopped();
    void Suspend();
    bool IsSuspended();
    void Resume();
    double GetIntervalMs();
    void SetLength(long images);
    long GetLength() const;

    long GetImageCounter();
    MM::MMTime GetStartTime();
    MM::MMTime GetActualDuration();

private:
    int svc(void) throw();

protected:
   SCCamera* camera_;
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
