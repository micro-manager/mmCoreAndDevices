///////////////////////////////////////////////////////////////////////////////
// FILE:          IDSPeak.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Driver for IDS peak series of USB cameras
//
//                Based on IDS peak SDK and Micro-manager DemoCamera example
//                tested with SDK version 2.5
//                Requires Micro-manager Device API 71 or higher!
//                
// AUTHOR:        Lars Kool, Institut Pierre-Gilles de Gennes
//
// YEAR:          2023
//                
// VERSION:       1.1.1
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
//LAST UPDATE:    03.12.2024 LK

#ifndef _IDSPeak_H_
#define _IDSPeak_H_

#include "DeviceBase.h"
#include "ImgBuffer.h"
#include "DeviceThreads.h"
#include <string>
#include <map>
#include <algorithm>
#include <stdint.h>
#include <future>

#include <ids_peak_comfort_c/ids_peak_comfort_c.h>

using namespace std;

#define EXPOSURE_MAX 1000000

////////////////////////////////////////
// Error codes
////////////////////////////////////////
#define ERR_LIBRARY_NOT_INIT     101
#define ERR_UNKNOWN_MODE         102
#define ERR_UNKNOWN_POSITION     103
#define ERR_IN_SEQUENCE          104
#define ERR_SEQUENCE_INACTIVE    105
#define ERR_STAGE_MOVING         106
#define HUB_NOT_AVAILABLE        107
#define ERR_MEM_ALLOC            108
#define ERR_ROI_INVALID          109
#define ERR_CAMERA_NOT_FOUND     110
#define ERR_DEVICE_NOT_AVAILABLE 111
#define ERR_NO_READ_ACCESS       112
#define ERR_ACQ_START            113
#define ERR_ACQ_FRAME            114
#define ERR_ACQ_RELEASE          115
#define ERR_ACQ_TIMEOUT          116
#define ERR_NO_WRITE_ACCESS      117

const char* NoHubError = "Parent Hub not defined.";

//////////////////////////////////////////////////////////////////////////////
// IDSPeakHub class
// Hub for IDS Peak cameras
//////////////////////////////////////////////////////////////////////////////

class IDSPeakHub : public HubBase<IDSPeakHub>
{
public:
    IDSPeakHub();
    ~IDSPeakHub();

    // Device API
    int Initialize();
    int Shutdown();
    void GetName(char* pName) const;
    bool Busy();

    // HUB api
    int DetectInstalledDevices();

    // Hub parameters
    peak_status status = PEAK_STATUS_SUCCESS;
    size_t nCameras_ = 0;
    vector<peak_camera_handle> hCams;
    int testVal;

    // Hub action handlers

    // Hub members

private:
    bool initialized_;
    bool busy_;
    unsigned char errorCode_;
    string unit_;
};

//////////////////////////////////////////////////////////////////////////////
// CIDSPeak class
//////////////////////////////////////////////////////////////////////////////

class MySequenceThread;

class CIDSPeak : public CLegacyCameraBase<CIDSPeak>
{
public:
    CIDSPeak(int deviceIdx);
    ~CIDSPeak();

    // MMDevice API
    // ------------
    int Initialize();
    int Shutdown();

    peak_camera_handle hCam = PEAK_INVALID_HANDLE;
    peak_status status = PEAK_STATUS_SUCCESS;
    void GetName(char* name) const;
    int deviceIdx_;

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
    bool SupportsMultiROI();
    bool IsMultiROISet();
    int GetMultiROICount(unsigned& count);
    int SetMultiROI(const unsigned* xs, const unsigned* ys,
        const unsigned* widths, const unsigned* heights,
        unsigned numROIs);
    int GetMultiROI(unsigned* xs, unsigned* ys, unsigned* widths,
        unsigned* heights, unsigned* length);
    int PrepareSequenceAcqusition() { return DEVICE_OK; }
    int StartSequenceAcquisition(double interval);
    int StartSequenceAcquisition(long numImages, double interval_ms, bool stopOnOverflow);
    int StopSequenceAcquisition();
    int InsertImage();
    int RunSequenceOnThread();
    bool IsCapturing();
    void OnThreadExiting() throw();
    double GetNominalPixelSizeUm() const { return nominalPixelSizeUm_; }
    double GetPixelSizeUm() const { return nominalPixelSizeUm_ * GetBinning(); }
    int GetBinning() const;
    int SetBinning(int bS);

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
    int OnChangeCamera(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnModelName(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnSerialNumber(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnMaxExposure(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnBinning(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnPixelType(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnFrameRate(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnReadoutTime(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnCameraCCDXSize(MM::PropertyBase*, MM::ActionType);
    int OnCameraCCDYSize(MM::PropertyBase*, MM::ActionType);
    int OnTriggerDevice(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnSupportsMultiROI(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnMultiROIFillValue(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnCCDTemp(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnIsSequenceable(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnAutoWhiteBalance(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnGainMaster(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnGainRed(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnGainGreen(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnGainBlue(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnPeakPixelFormat(MM::PropertyBase* pProp, MM::ActionType eAct);

    long GetCCDXSize() { return cameraCCDXSize_; }
    long GetCCDYSize() { return cameraCCDYSize_; }

    // My methods
    int cleanExit();
    peak_bool checkForSuccess(peak_status status, peak_bool continueExecution);
    int getSensorInfo();
    peak_status getGFAString(const char* featureName, char* stringValue);
    peak_status getGFAInt(const char* featureName, int64_t* intValue);
    peak_status getGFAfloat(const char* featureName, double* floatValue);
    peak_status getTemperature(double* sensorTemp);
    void initializeAutoWBConversion();
    int transferBuffer(peak_frame_handle hFrame, ImgBuffer& img);
    int updateAutoWhiteBalance();
    int framerateSet(double framerate);
    int cameraChanged();
    bool isColorCamera();


private:
    int SetAllowedBinning();
    void GenerateEmptyImage(ImgBuffer& img);
    int ResizeImageBuffer();

    static const double nominalPixelSizeUm_;

    string modelName_;
    string serialNum_;
    size_t nCameras_;
    double exposureMin_;
    double exposureMax_;
    double exposureInc_;
    double exposureCur_;
    double framerateCur_;
    double framerateMax_;
    double framerateMin_;
    double framerateInc_;
    ImgBuffer img_;
    bool stopOnOverFlow_;
    bool initialized_;
    double readoutUs_;
    MM::MMTime readoutStartTime_;
    string pixelType_;
    int bitDepth_;
    int significantBitDepth_;
    int nComponents_;
    unsigned roiX_;
    unsigned roiY_;
    unsigned roiInc_;
    unsigned roiMinSizeX_;
    unsigned roiMinSizeY_;
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

    peak_auto_feature_mode peakAutoWhiteBalance_;
    map<int, string> peakAutoToString;
    map<string, int> stringToPeakAuto;
    double gainMaster_;
    double gainRed_;
    double gainGreen_;
    double gainBlue_;
    double gainMin_;
    double gainMax_;
    double gainInc_;

    std::vector<std::string> availablePixelFormats;
    peak_pixel_format currPixelFormat;


    bool stopOnOverflow_;

    bool supportsMultiROI_;
    int multiROIFillValue_;
    std::vector<unsigned> multiROIXs_;
    std::vector<unsigned> multiROIYs_;
    std::vector<unsigned> multiROIWidths_;
    std::vector<unsigned> multiROIHeights_;

    MMThreadLock imgPixelsLock_;
    friend class MySequenceThread;

    MySequenceThread* thd_;
    std::future<void> fut_;
};

class MySequenceThread : public MMDeviceThreadBase
{
    friend class CIDSPeak;
    enum { default_numImages = 1, default_intervalMS = 100 };
public:
    MySequenceThread(CIDSPeak* pCam);
    ~MySequenceThread();
    void Stop();
    void Start(long numImages, double intervalMs);
    bool IsStopped();
    void Suspend();
    bool IsSuspended();
    void Resume();
    double GetIntervalMs() { return intervalMs_; }
    //void SetIntervalMs(double intervalms);
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
    CIDSPeak* camera_;
    MM::MMTime startTime_;
    MM::MMTime actualDuration_;
    MM::MMTime lastFrameTime_;
    MMThreadLock stopLock_;
    MMThreadLock suspendLock_;
};

#endif //_IDSPeak_H_