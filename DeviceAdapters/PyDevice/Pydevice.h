///////////////////////////////////////////////////////////////////////////////
// FILE:          Pydevice.h
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
// COPYRIGHT:     University of California, San Francisco, 2006-2015
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

#ifndef _Pydevice_H_
#define _Pydevice_H_

#include "DeviceBase.h"
#include "ImgBuffer.h"
#include "DeviceThreads.h"
#include <string>
#include <map>
#include <algorithm>
#include <stdint.h>
#include <future>

#ifdef _DEBUG
#undef _DEBUG
#include <Python.h> // if you get a compiler error here, try building again and see if magic happens
#define _DEBUG
#else
#include <Python.h>
#endif


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

// Defines which segments in a seven-segment display are lit up for each of
// the numbers 0-9. Segments are:
//
//  0       1
// 1 2     2 4
//  3       8
// 4 5    16 32
//  6      64
const int SEVEN_SEGMENT_RULES[] = { 1 + 2 + 4 + 16 + 32 + 64, 4 + 32, 1 + 4 + 8 + 16 + 64,
      1 + 4 + 8 + 32 + 64, 2 + 4 + 8 + 32, 1 + 2 + 8 + 32 + 64, 2 + 8 + 16 + 32 + 64, 1 + 4 + 32,
      1 + 2 + 4 + 8 + 16 + 32 + 64, 1 + 2 + 4 + 8 + 32 + 64 };
// Indicates if the segment is horizontal or vertical.
const int SEVEN_SEGMENT_HORIZONTALITY[] = { 1, 0, 0, 1, 0, 0, 1 };
// X offset for this segment.
const int SEVEN_SEGMENT_X_OFFSET[] = { 0, 0, 1, 0, 0, 1, 0 };
// Y offset for this segment.
const int SEVEN_SEGMENT_Y_OFFSET[] = { 0, 0, 0, 1, 1, 1, 2 };

class ImgManipulator
{
public:
    virtual int ChangePixels(ImgBuffer& img) = 0;
};

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


    std::vector<std::string> peripherals_;
    bool initialized_;
    bool busy_;
};


//////////////////////////////////////////////////////////////////////////////
// CPydevice class
// Simulation of the Camera device
//////////////////////////////////////////////////////////////////////////////

class MySequenceThread;

class CPydevice : public CCameraBase<CPydevice>
{
public:
    CPydevice();
    ~CPydevice();

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
    // Added by Jeroen:
    int OnZoomFactor(MM::PropertyBase* pProp, MM::ActionType);
    int OnDelay(MM::PropertyBase* pProp, MM::ActionType);
    int OnDwelltime(MM::PropertyBase* pProp, MM::ActionType);
    int OnScanpadding(MM::PropertyBase* pProp, MM::ActionType);
    int OnInputMin(MM::PropertyBase* pProp, MM::ActionType);
    int OnInputMax(MM::PropertyBase* pProp, MM::ActionType);

    int OnBinning(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnPixelType(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnBitDepth(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnScanMode(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnScanXSteps(MM::PropertyBase*, MM::ActionType);
    int OnScanYSteps(MM::PropertyBase*, MM::ActionType);
    int OnTriggerDevice(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnDACPortOutx(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnDACPortOuty(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnDACPortIn(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnIsSequenceable(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnMode(MM::PropertyBase* pProp, MM::ActionType eAct);

    long GetScanXSteps() { return ScanXSteps_; }
    long GetScanYSteps() { return ScanYSteps_; }
    double GetZoomFactor() { return zoomFactor_; }
    long GetDelay() { return delay_; }
    double GetDwelltime() { return dwelltime_; }
    double GetScanpadding() { return scanpadding_; }

private:
    int SetAllowedBinning();
    void GenerateEmptyImage(ImgBuffer& img);
    int GeneratePythonImage(ImgBuffer& img);
    int ResizeImageBuffer();
    
    static const double nominalPixelSizeUm_;

    double exposureMaximum_;
    double dPhase_;
    ImgBuffer img_;
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
    long ScanXSteps_;
    long ScanYSteps_;
    double zoomFactor_;
    double inputmin_;
    double inputmax_;
    long delay_;
    double dwelltime_;
    double scanpadding_;
    std::string triggerDevice_;
    std::string dacportoutx_;
    std::string dacportouty_;
    std::string dacportin_;

    bool stopOnOverflow_;

    bool fastImage_;
    int multiROIFillValue_;
    std::vector<unsigned> multiROIXs_;
    std::vector<unsigned> multiROIYs_;
    std::vector<unsigned> multiROIWidths_;
    std::vector<unsigned> multiROIHeights_;

    MMThreadLock imgPixelsLock_;
    friend class MySequenceThread;
    int nComponents_;
    MySequenceThread* thd_;
    std::future<void> fut_;
    int mode_;
    ImgManipulator* imgManpl_;
    double pcf_;
};

class MySequenceThread : public MMDeviceThreadBase
{
    friend class CPydevice;
    enum { default_numImages = 1, default_intervalMS = 100 };
public:
    MySequenceThread(CPydevice* pCam);
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
    CPydevice* camera_;
    MM::MMTime startTime_;
    MM::MMTime actualDuration_;
    MM::MMTime lastFrameTime_;
    MMThreadLock stopLock_;
    MMThreadLock suspendLock_;
};




#endif //_Pydevice_H_
