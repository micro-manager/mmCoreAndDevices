///////////////////////////////////////////////////////////////////////////////
// FILE:          POACamera.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   This is a device adapter of Micro-Manager for Player One cameras.
//				  This is modified based on DemoCamera project.
//                
// AUTHOR:        Lei Zhang, lei.zhang@player-one-astronomy.com, Feb 2024
//                
// COPYRIGHT:     Player One Astronomy, SUZHOU, 2024
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

#pragma once

#include "DeviceBase.h"
#include "ImgBuffer.h"
#include "DeviceThreads.h"
#include "PlayerOneCamera.h"
#include "PlayerOnePW.h"

#include <string>
#include <map>
#include <algorithm>
#include <stdint.h>
#include <future>

//////////////////////////////////////////////////////////////////////////////
// Error codes
//
#define ERR_UNKNOWN_MODE         102
#define ERR_UNKNOWN_POSITION     103
#define ERR_IN_SEQUENCE          104
#define ERR_SEQUENCE_INACTIVE    105
#define ERR_STAGE_MOVING         106
#define HUB_NOT_AVAILABLE        107

const char* NoHubError = "Parent Hub not defined.";

////////////////////////
// DemoHub
//////////////////////

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
    int Shutdown() { return DEVICE_OK; };
    void GetName(char* pName) const;
    bool Busy() { return busy_; };

    // HUB api
    int DetectInstalledDevices();

private:
    void GetPeripheralInventory();

    std::vector<std::string> peripherals_;
    bool initialized_;
    bool busy_;
};

//////////////////////////////////////////////////////////////////////////////
// POACamera class
//////////////////////////////////////////////////////////////////////////////

class MySequenceThread;

class POACamera : public CLegacyCameraBase<POACamera>
{
public:
    POACamera();
    ~POACamera();

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
    double GetNominalPixelSizeUm() const { return nominalPixelSizeUm_; }
    double GetPixelSizeUm() const { return nominalPixelSizeUm_ * GetBinning(); }
    int GetBinning() const;
    int SetBinning(int bS);

    bool SupportsMultiROI();

    int PrepareSequenceAcqusition();
    int StartSequenceAcquisition(double interval);
    int StartSequenceAcquisition(long numImages, double interval_ms, bool stopOnOverflow);
    int StopSequenceAcquisition(); 
    bool IsCapturing();

    int InsertImage();
    int RunSequenceOnThread(MM::MMTime startTime);
    void OnThreadExiting() throw();

    int IsExposureSequenceable(bool& isSequenceable) const;
    int GetExposureSequenceMaxLength(long& nrEvents) const;
    int StartExposureSequence();
    int StopExposureSequence();
    int ClearExposureSequence();
    int AddToExposureSequence(double exposureTime_ms);
    int SendExposureSequence() const;
   
    unsigned  GetNumberOfComponents() const { return nComponents_; };

    // action interface
    // ----------------
    int OnSelectCamIndex(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnMaxExposure(MM::PropertyBase* pProp, MM::ActionType eAct);

    int OnAsyncFollower(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnAsyncLeader(MM::PropertyBase* pProp, MM::ActionType eAct);
    void SlowPropUpdate(std::string leaderValue);

    int OnExposure(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnExpAuto(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnGain(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnGainAuto(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnAutoExpBrightness(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnBinning(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnPixelType(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnOffset(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnGamma(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnUSBBandwidthLimit(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnFrameRateLimit(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnFlip(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnCCDTemp(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnHardBin(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnPixelBinSum(MM::PropertyBase* pProp, MM::ActionType eAct);

    int OnWB_Red(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnWB_Green(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnWB_Blue(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnAutoWB(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnMonoBin(MM::PropertyBase* pProp, MM::ActionType eAct);

    int OnTargetTEMP(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnCoolerOn(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnCoolerPower(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnFanPower(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnHeaterPower(MM::PropertyBase* pProp, MM::ActionType eAct);

    int OnGainPreset(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnOffsetPreset(MM::PropertyBase* pProp, MM::ActionType eAct);

    int OnIsSequenceable(MM::PropertyBase* pProp, MM::ActionType eAct);

    long GetCCDXSize() { return cameraCCDXSize_; }
    long GetCCDYSize() { return cameraCCDYSize_; }


private:
    void TestResourceLocking(const bool);
    void GenerateEmptyImage(ImgBuffer& img);
    bool GenerateColorTestPattern(ImgBuffer& img);
    int ResizeImageBuffer();
    const char* ImgFmtToString(const POAImgFormat &imgFmt);
    // Refresh the current ROI parameters
    void RefreshCurrROIParas();
    void ManageRGB24Memory();
    void ResetGammaTable();
    void BGR888ToRGB32(unsigned char* pBGR888Buf, unsigned char* pRGB32Buf, int bgr888Len, bool isEnableGamma = false);

    bool initialized_;
    unsigned char* pRGB24;
    std::size_t RGB24BufSize_;

    std::vector<std::string> connectCamerasName_;
    std::string selectCamName_;
    int selectCamIndex_;
    POACameraProperties camProp_;
    int roiX_;
    int roiY_;
    int cameraCCDXSize_;
    int cameraCCDYSize_;
    int binSize_;
    POAImgFormat imgFmt_;
    int bitDepth_;
    volatile bool m_bIsToStopExposure = false;
    double gammaValue_;
    unsigned char *p8bitGammaTable;
    unsigned short *p16bitGammaTable;
    double nominalPixelSizeUm_;
    double exposureMaximum_;
    ImgBuffer img_;
    double ccdT_;

    int gainHighestDR_, HCGain_, unityGain_, gainLowestRN_;
    int offsetHighestDR_, offsetHCGain_, offsetUnityGain_, offsetLowestRN_;

    double readoutUs_;
    MM::MMTime readoutStartTime_;
    
    MM::MMTime sequenceStartTime_;
    bool isSequenceable_;
    long sequenceMaxLength_;
    bool sequenceRunning_;
    unsigned long sequenceIndex_;
    double GetSequenceExposure();
    std::vector<double> exposureSequence_;
    long imageCounter_;
    
    bool stopOnOverflow_;

    bool supportsMultiROI_;

    std::string asyncLeader_;
    std::string asyncFollower_;
    MMThreadLock imgPixelsLock_;
    MMThreadLock asyncFollowerLock_;
    friend class MySequenceThread;
    int nComponents_;
    MySequenceThread* thd_;
    std::future<void> fut_;
};

class MySequenceThread : public MMDeviceThreadBase
{
    friend class POACamera;
    enum { default_numImages = 1, default_intervalMS = 100 };
public:
    MySequenceThread(POACamera* pCam);
    ~MySequenceThread();
    void Stop();
    void Start(long numImages, double intervalMs);
    bool IsStopped();
    void Suspend();
    bool IsSuspended();
    void Resume();
    double GetIntervalMs() { return intervalMs_; }
    void SetLength(long images) { numImages_ = images; }
    long GetLength() const { return numImages_; }
    long GetImageCounter() { return imageCounter_; }
    MM::MMTime GetStartTime() { return startTime_; }
    MM::MMTime GetActualDuration() { return actualDuration_; }
private:
    int svc(void) throw();
    double intervalMs_;
    long numImages_;
    long imageCounter_;
    bool stop_;
    bool suspend_;
    POACamera* camera_;
    MM::MMTime startTime_;
    MM::MMTime actualDuration_;
    MM::MMTime lastFrameTime_;
    MMThreadLock stopLock_;
    MMThreadLock suspendLock_;
};

//////////////////////////////////////////////////////////////////////////////
// POAFilterWheel class
//////////////////////////////////////////////////////////////////////////////

class POAFilterWheel : public CStateDeviceBase<POAFilterWheel>
{
public:
    POAFilterWheel();
    ~POAFilterWheel();

    // MMDevice API
    // ------------
    int Initialize();
    int Shutdown();

    void GetName(char* pszName) const;
    bool Busy();
    unsigned long GetNumberOfPositions()const;

    // action interface
    // ----------------
    int OnState(MM::PropertyBase* pProp, MM::ActionType eAct);

    int OnSelectPWIndex(MM::PropertyBase* pProp, MM::ActionType eAct);

private:
    //long numPos_;
    bool initialized_;
    MM::MMTime changedTime_;
    int position_;

    std::vector<std::string> connectPWsName_;
    std::string selectPWName_;
    int selectPWIndex_;
    PWProperties PWProp_;
    bool isBusyWait;
};

