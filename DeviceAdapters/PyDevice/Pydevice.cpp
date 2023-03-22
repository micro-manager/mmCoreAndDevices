///////////////////////////////////////////////////////////////////////////////
// FILE:          Pydevice.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   The implementation of the Python camera. Adapted from the Democamera in the 
//                Micromanager repository.
//                
// AUTHOR:        Jeroen Doornbos
//
// COPYRIGHT:     
// LICENSE:       ?

#include "Pydevice.h"
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <cstdio>
#include <string>
#include <math.h>
#include "ModuleInterface.h"
#include <sstream>
#include <algorithm>
#include "WriteCompactTiffRGB.h"
#include <iostream>
#include <future>
#include <algorithm>
#include <iomanip>
#include <sstream>


using namespace std;
const double CPydevice::nominalPixelSizeUm_ = 1.0;
double g_IntensityFactor_ = 1.0;

// External names used used by the rest of the system
// to load particular device from the "Pydevice.dll" library
const char* g_CameraDeviceName = "PyCam";

// constants for naming pixel types (allowed values of the "PixelType" property)
const char* g_PixelType_8bit = "8bit";
const char* g_PixelType_16bit = "16bit";
const char* g_PixelType_32bitRGB = "32bitRGB";
const char* g_PixelType_64bitRGB = "64bitRGB";
const char* g_PixelType_32bit = "32bit";  // floating point greyscale

// constants for naming camera modes
const char* g_Bidirectional = "Bidirectional";
const char* g_Unidirectional = "Unidirectional";


enum { MODE_BIDIRECTIONAL, MODE_UNIDIRECTIONAL};

///////////////////////////////////////////////////////////////////////////////
// Exported MMDevice API
///////////////////////////////////////////////////////////////////////////////

MODULE_API void InitializeModuleData()
{
    RegisterDevice(g_CameraDeviceName, MM::CameraDevice, "Python camera");
}

MODULE_API MM::Device* CreateDevice(const char* deviceName)
{
    if (deviceName == 0)
        return 0;

    // decide which device class to create based on the deviceName parameter
    if (strcmp(deviceName, g_CameraDeviceName) == 0)
    {
        // create camera
        return new CPydevice();
    }

    // ...supplied name not recognized
    return 0;
}

MODULE_API void DeleteDevice(MM::Device* pDevice)
{
    delete pDevice;
}

///////////////////////////////////////////////////////////////////////////////
// CPydevice implementation
// ~~~~~~~~~~~~~~~~~~~~~~~~~~

/**
* CPydevice constructor.
* Setup default all variables and create device properties required to exist
* before intialization. In this case, no such properties were required. All
* properties will be created in the Initialize() method.
*
* As a general guideline Micro-Manager devices do not access hardware in the
* the constructor. We should do as little as possible in the constructor and
* perform most of the initialization in the Initialize() method.
*/
CPydevice::CPydevice() :
    CCameraBase<CPydevice>(),
    exposureMaximum_(0),
    dPhase_(0),
    initialized_(false),
    readoutUs_(0.0),
    scanMode_(1),
    bitDepth_(8),
    roiX_(0),
    roiY_(0),
    sequenceStartTime_(0),
    isSequenceable_(true),
    dacportoutx_("Dev4/ao2"),
    dacportouty_("Dev4/ao3"),
    dacportin_("Dev4/ai24"),
    sequenceMaxLength_(100),
    sequenceRunning_(false),
    sequenceIndex_(0),
    binSize_(1),
    zoomFactor_(2.0),
    delay_(0),
    dwelltime_(4),
    scanpadding_(1),
    ScanXSteps_(512),
    ScanYSteps_(512),
    inputmin_(-2.0),
    inputmax_(2.0),
    triggerDevice_(""),
    fastImage_(false),
    nComponents_(1),
    mode_(MODE_BIDIRECTIONAL),
    imgManpl_(0)
{
    // call the base class method to set-up default error codes/messages
    InitializeDefaultErrorMessages();
    readoutStartTime_ = GetCurrentMMTime();
    thd_ = new MySequenceThread(this);

    // parent ID display
    CreateHubIDProperty();

    CreateStringProperty("DAC port analog out x", dacportoutx_.c_str(), false,
        new CPropertyAction(this, &CPydevice::OnDACPortOutx),
        true);

    CreateStringProperty("DAC port analog out y", dacportouty_.c_str(), false,
        new CPropertyAction(this, &CPydevice::OnDACPortOuty),
        true);

    CreateStringProperty("DAC port analog in", dacportin_.c_str(), false,
        new CPropertyAction(this, &CPydevice::OnDACPortIn),
        true);
}

/**
* CPydevice destructor.
* If this device used as intended within the Micro-Manager system,
* Shutdown() will be always called before the destructor. But in any case
* we need to make sure that all resources are properly released even if
* Shutdown() was not called.
*/
CPydevice::~CPydevice()
{
    StopSequenceAcquisition();
    delete thd_;
}

/**
* Obtains device name.
* Required by the MM::Device API.
*/
void CPydevice::GetName(char* name) const
{
    // Return the name used to referr to this device adapte
    CDeviceUtils::CopyLimitedString(name, g_CameraDeviceName);
}

/**
* Intializes the hardware.
* Required by the MM::Device API.
* Typically we access and initialize hardware at this point.
* Device properties are typically created here as well, except
* the ones we need to use for defining initialization parameters.
* Such pre-initialization properties are created in the constructor.
* (This device does not have any pre-initialization properties)
*/
int CPydevice::Initialize()
{
    if (initialized_)
        return DEVICE_OK;

    DemoHub* pHub = static_cast<DemoHub*>(GetParentHub());
    if (pHub)
    {
        char hubLabel[MM::MaxStrLength];
        pHub->GetLabel(hubLabel);
        SetParentID(hubLabel); // for backward comp.
    }
    else
        LogMessage(NoHubError);

    // set property list
    // -----------------
    // Python initialisation

    Py_Initialize();
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append(\".\")");
    // add paths in which the function is set
    PyRun_SimpleString("sys.path.append(\"C:/Users/Jeroen Doornbos/Documents/wfs_current/micromanager/pybridgeattempt\")");


    // Name
    int nRet = CreateStringProperty(MM::g_Keyword_Name, g_CameraDeviceName, true);
    if (DEVICE_OK != nRet)
        return nRet;

    // Description
    nRet = CreateStringProperty(MM::g_Keyword_Description, "Python camera Device Adapter", true);
    if (DEVICE_OK != nRet)
        return nRet;

    // CameraName
    nRet = CreateStringProperty(MM::g_Keyword_CameraName, "Pydevice-ConfocalScan", true);
    assert(nRet == DEVICE_OK);

    // CameraID
    nRet = CreateStringProperty(MM::g_Keyword_CameraID, "V1.0", true);
    assert(nRet == DEVICE_OK);

    // binning
    CPropertyAction* pAct = new CPropertyAction(this, &CPydevice::OnBinning);
    nRet = CreateIntegerProperty(MM::g_Keyword_Binning, 1, false, pAct);
    assert(nRet == DEVICE_OK);

    nRet = SetAllowedBinning();
    if (nRet != DEVICE_OK)
        return nRet;

    // pixel type
    pAct = new CPropertyAction(this, &CPydevice::OnPixelType);
    nRet = CreateStringProperty(MM::g_Keyword_PixelType, g_PixelType_8bit, false, pAct);
    assert(nRet == DEVICE_OK);

    vector<string> pixelTypeValues;
    pixelTypeValues.push_back(g_PixelType_8bit);
    pixelTypeValues.push_back(g_PixelType_16bit);
    pixelTypeValues.push_back(g_PixelType_32bitRGB);
    pixelTypeValues.push_back(g_PixelType_64bitRGB);
    pixelTypeValues.push_back(::g_PixelType_32bit);

    nRet = SetAllowedValues(MM::g_Keyword_PixelType, pixelTypeValues);
    if (nRet != DEVICE_OK)
        return nRet;

    // Bit depth
    pAct = new CPropertyAction(this, &CPydevice::OnBitDepth);
    nRet = CreateIntegerProperty("BitDepth", 8, false, pAct);
    assert(nRet == DEVICE_OK);

    vector<string> bitDepths;
    bitDepths.push_back("8");
    bitDepths.push_back("10");
    bitDepths.push_back("12");
    bitDepths.push_back("14");
    bitDepths.push_back("16");
    bitDepths.push_back("32");
    nRet = SetAllowedValues("BitDepth", bitDepths);
    if (nRet != DEVICE_OK)
        return nRet;

    // exposure
    nRet = CreateFloatProperty(MM::g_Keyword_Exposure, 0, false);
    assert(nRet == DEVICE_OK);
    SetPropertyLimits(MM::g_Keyword_Exposure, 0.0, exposureMaximum_);

    pAct = new CPropertyAction(this, &CPydevice::OnZoomFactor);
    CreateFloatProperty("ZoomFactor", 1.0, false, pAct);
    
    pAct = new CPropertyAction(this, &CPydevice::OnDelay);
    CreateIntegerProperty("Delay", 0, false, pAct);

    pAct = new CPropertyAction(this, &CPydevice::OnDwelltime);
    CreateFloatProperty("Dwelltime", 4, false, pAct);

    pAct = new CPropertyAction(this, &CPydevice::OnScanpadding);
    CreateFloatProperty("Scanpadding", 1.0, false, pAct);

     // scan mode
    pAct = new CPropertyAction(this, &CPydevice::OnScanMode);
    nRet = CreateIntegerProperty("ScanMode", 1, false, pAct);
    assert(nRet == DEVICE_OK);
    AddAllowedValue("ScanMode", "1");
    AddAllowedValue("ScanMode", "2");


    // camera gain
    nRet = CreateIntegerProperty(MM::g_Keyword_Gain, 0, false);
    assert(nRet == DEVICE_OK);
    SetPropertyLimits(MM::g_Keyword_Gain, -5, 8);

    // CCD size of the camera we are modeling
    pAct = new CPropertyAction(this, &CPydevice::OnScanXSteps);
    CreateIntegerProperty("ScanXSteps", 512, false, pAct);
    pAct = new CPropertyAction(this, &CPydevice::OnScanYSteps);
    CreateIntegerProperty("ScanYSteps", 512, false, pAct);

    pAct = new CPropertyAction(this, &CPydevice::OnInputMin);
    CreateFloatProperty("InputMin", -2.0, false, pAct);
    pAct = new CPropertyAction(this, &CPydevice::OnInputMax);
    CreateFloatProperty("InputMax", 2.0, false, pAct);




    // Trigger device
    pAct = new CPropertyAction(this, &CPydevice::OnTriggerDevice);
    CreateStringProperty("TriggerDevice", "", false, pAct);

    // Camera mode: 
    pAct = new CPropertyAction(this, &CPydevice::OnMode);
    std::string propName = "Mode";
    CreateStringProperty(propName.c_str(), g_Bidirectional, false, pAct);
    AddAllowedValue(propName.c_str(), g_Bidirectional);
    AddAllowedValue(propName.c_str(), g_Unidirectional);


    // synchronize all properties
    // --------------------------
    nRet = UpdateStatus();
    if (nRet != DEVICE_OK)
        return nRet;


    // setup the buffer
    // ----------------
    nRet = ResizeImageBuffer();
    if (nRet != DEVICE_OK)
        return nRet;

#ifdef TESTRESOURCELOCKING
    TestResourceLocking(true);
    LogMessage("TestResourceLocking OK", true);
#endif


    initialized_ = true;




    // initialize image buffer
    GenerateEmptyImage(img_);
    return DEVICE_OK;


}

/**
* Shuts down (unloads) the device.
* Required by the MM::Device API.
* Ideally this method will completely unload the device and release all resources.
* Shutdown() may be called multiple times in a row.
* After Shutdown() we should be allowed to call Initialize() again to load the device
* without causing problems.
*/
int CPydevice::Shutdown()
{
    initialized_ = false;
    // destroy the python interpreter <-- JEROEN
    Py_Finalize();
    return DEVICE_OK;
}

/**
* Performs exposure and grabs a single image.
* This function should block during the actual exposure and return immediately afterwards
* (i.e., before readout).  This behavior is needed for proper synchronization with the shutter.
* Required by the MM::Camera API.
*/
int CPydevice::SnapImage()
{
    static int callCounter = 0;
    ++callCounter;

    MM::MMTime startTime = GetCurrentMMTime();

    // Jeroen: New image taking implementation:
    GeneratePythonImage(img_);



    MM::MMTime s0(0, 0);
    
    readoutStartTime_ = GetCurrentMMTime();

    return DEVICE_OK;
}


/**
* Returns pixel data.
* Required by the MM::Camera API.
* The calling program will assume the size of the buffer based on the values
* obtained from GetImageBufferSize(), which in turn should be consistent with
* values returned by GetImageWidth(), GetImageHight() and GetImageBytesPerPixel().
* The calling program allso assumes that camera never changes the size of
* the pixel buffer on its own. In other words, the buffer can change only if
* appropriate properties are set (such as binning, pixel type, etc.)
*/
const unsigned char* CPydevice::GetImageBuffer()
{
    MMThreadGuard g(imgPixelsLock_);


    
    //GenerateEmptyImage(img_);

    MM::MMTime readoutTime(readoutUs_);

    while (readoutTime > (GetCurrentMMTime() - readoutStartTime_)) {}

    unsigned char* pB = (unsigned char*)(img_.GetPixels());
    return pB;
}

/**
* Returns image buffer X-size in pixels.
* Required by the MM::Camera API.
*/
unsigned CPydevice::GetImageWidth() const
{
    return img_.Width();
}

/**
* Returns image buffer Y-size in pixels.
* Required by the MM::Camera API.
*/
unsigned CPydevice::GetImageHeight() const
{
    return img_.Height();
}

/**
* Returns image buffer pixel depth in bytes.
* Required by the MM::Camera API.
*/
unsigned CPydevice::GetImageBytesPerPixel() const
{
    return img_.Depth();
}

/**
* Returns the bit depth (dynamic range) of the pixel.
* This does not affect the buffer size, it just gives the client application
* a guideline on how to interpret pixel values.
* Required by the MM::Camera API.
*/
unsigned CPydevice::GetBitDepth() const
{
    return bitDepth_;
}

/**
* Returns the size in bytes of the image buffer.
* Required by the MM::Camera API.
*/
long CPydevice::GetImageBufferSize() const
{
    return img_.Width() * img_.Height() * GetImageBytesPerPixel();
}

/**
* Sets the camera Region Of Interest.
* Required by the MM::Camera API.
* This command will change the dimensions of the image.
* Depending on the hardware capabilities the camera may not be able to configure the
* exact dimensions requested - but should try do as close as possible.
* If the hardware does not have this capability the software should simulate the ROI by
* appropriately cropping each frame.
* This demo implementation ignores the position coordinates and just crops the buffer.
* If multiple ROIs are currently set, then this method clears them in favor of
* the new ROI.
* @param x - top-left corner coordinate
* @param y - top-left corner coordinate
* @param xSize - width
* @param ySize - height
*/
int CPydevice::SetROI(unsigned x, unsigned y, unsigned xSize, unsigned ySize)
{
    multiROIXs_.clear();
    multiROIYs_.clear();
    multiROIWidths_.clear();
    multiROIHeights_.clear();
    if (xSize == 0 && ySize == 0)
    {
        // effectively clear ROI
        ResizeImageBuffer();
        roiX_ = 0;
        roiY_ = 0;
    }
    else
    {
        // apply ROI
        img_.Resize(xSize, ySize);
        roiX_ = x;
        roiY_ = y;
    }
    return DEVICE_OK;
}

/**
* Returns the actual dimensions of the current ROI.
* If multiple ROIs are set, then the returned ROI should encompass all of them.
* Required by the MM::Camera API.
*/
int CPydevice::GetROI(unsigned& x, unsigned& y, unsigned& xSize, unsigned& ySize)
{
    x = roiX_;
    y = roiY_;

    xSize = img_.Width();
    ySize = img_.Height();

    return DEVICE_OK;
}

/**
* Resets the Region of Interest to full frame.
* Required by the MM::Camera API.
*/
int CPydevice::ClearROI()
{
    ResizeImageBuffer();
    roiX_ = 0;
    roiY_ = 0;
    multiROIXs_.clear();
    multiROIYs_.clear();
    multiROIWidths_.clear();
    multiROIHeights_.clear();
    return DEVICE_OK;
}

/**
* Returns the current exposure setting in milliseconds.
* Required by the MM::Camera API.
*/
double CPydevice::GetExposure() const
{
    char buf[MM::MaxStrLength];
    int ret = GetProperty(MM::g_Keyword_Exposure, buf);
    if (ret != DEVICE_OK)
        return 0.0;
    return atof(buf);
}

/**
 * Returns the current exposure from a sequence and increases the sequence counter
 * Used for exposure sequences
 */
double CPydevice::GetSequenceExposure()
{
    if (exposureSequence_.size() == 0)
        return this->GetExposure();

    double exposure = exposureSequence_[sequenceIndex_];

    sequenceIndex_++;
    if (sequenceIndex_ >= exposureSequence_.size())
        sequenceIndex_ = 0;

    return exposure;
}

/**
* Sets exposure in milliseconds.
* Required by the MM::Camera API.
*/
void CPydevice::SetExposure(double exp)
{
    SetProperty(MM::g_Keyword_Exposure, CDeviceUtils::ConvertToString(exp));
    GetCoreCallback()->OnExposureChanged(this, exp);;
}

/**
* Returns the current binning factor.
* Required by the MM::Camera API.
*/
int CPydevice::GetBinning() const
{
    char buf[MM::MaxStrLength];
    int ret = GetProperty(MM::g_Keyword_Binning, buf);
    if (ret != DEVICE_OK)
        return 1;
    return atoi(buf);
}

/**
* Sets binning factor.
* Required by the MM::Camera API.
*/
int CPydevice::SetBinning(int binF)
{
    return SetProperty(MM::g_Keyword_Binning, CDeviceUtils::ConvertToString(binF));
}

int CPydevice::IsExposureSequenceable(bool& isSequenceable) const
{
    isSequenceable = isSequenceable_;
    return DEVICE_OK;
}

int CPydevice::GetExposureSequenceMaxLength(long& nrEvents) const
{
    if (!isSequenceable_) {
        return DEVICE_UNSUPPORTED_COMMAND;
    }

    nrEvents = sequenceMaxLength_;
    return DEVICE_OK;
}

int CPydevice::StartExposureSequence()
{
    if (!isSequenceable_) {
        return DEVICE_UNSUPPORTED_COMMAND;
    }

    // may need thread lock
    sequenceRunning_ = true;
    return DEVICE_OK;
}

int CPydevice::StopExposureSequence()
{
    if (!isSequenceable_) {
        return DEVICE_UNSUPPORTED_COMMAND;
    }

    // may need thread lock
    sequenceRunning_ = false;
    sequenceIndex_ = 0;
    return DEVICE_OK;
}

/**
 * Clears the list of exposures used in sequences
 */
int CPydevice::ClearExposureSequence()
{
    if (!isSequenceable_) {
        return DEVICE_UNSUPPORTED_COMMAND;
    }

    exposureSequence_.clear();
    return DEVICE_OK;
}

/**
 * Adds an exposure to a list of exposures used in sequences
 */
int CPydevice::AddToExposureSequence(double exposureTime_ms)
{
    if (!isSequenceable_) {
        return DEVICE_UNSUPPORTED_COMMAND;
    }

    exposureSequence_.push_back(exposureTime_ms);
    return DEVICE_OK;
}

int CPydevice::SendExposureSequence() const {
    if (!isSequenceable_) {
        return DEVICE_UNSUPPORTED_COMMAND;
    }

    return DEVICE_OK;
}

int CPydevice::SetAllowedBinning()
{ 
    vector<string> binValues;
    binValues.push_back("1");
    binValues.push_back("2");
    if (scanMode_ < 3)
        binValues.push_back("4");
    if (scanMode_ < 2)
        binValues.push_back("8");
    if (binSize_ == 8 && scanMode_ == 3) {
        SetProperty(MM::g_Keyword_Binning, "2");
    }
    else if (binSize_ == 8 && scanMode_ == 2) {
        SetProperty(MM::g_Keyword_Binning, "4");
    }
    else if (binSize_ == 4 && scanMode_ == 3) {
        SetProperty(MM::g_Keyword_Binning, "2");
    }

    LogMessage("Setting Allowed Binning settings", true);
    return SetAllowedValues(MM::g_Keyword_Binning, binValues);
}


/**
 * Required by the MM::Camera API
 * Please implement this yourself and do not rely on the base class implementation
 * The Base class implementation is deprecated and will be removed shortly
 */
int CPydevice::StartSequenceAcquisition(double interval)
{
    return StartSequenceAcquisition(LONG_MAX, interval, false);
}

/**
* Stop and wait for the Sequence thread finished
*/
int CPydevice::StopSequenceAcquisition()
{
    if (!thd_->IsStopped()) {
        thd_->Stop();
        thd_->wait();
    }

    return DEVICE_OK;
}

/**
* Simple implementation of Sequence Acquisition
* A sequence acquisition should run on its own thread and transport new images
* coming of the camera into the MMCore circular buffer.
*/
int CPydevice::StartSequenceAcquisition(long numImages, double interval_ms, bool stopOnOverflow)
{
    if (IsCapturing())
        return DEVICE_CAMERA_BUSY_ACQUIRING;

    int ret = GetCoreCallback()->PrepareForAcq(this);
    if (ret != DEVICE_OK)
        return ret;
    sequenceStartTime_ = GetCurrentMMTime();
    imageCounter_ = 0;
    thd_->Start(numImages, interval_ms);
    stopOnOverflow_ = stopOnOverflow;
    return DEVICE_OK;
}

/*
 * Inserts Image and MetaData into MMCore circular Buffer
 */

int CPydevice::InsertImage()
{
    MM::MMTime timeStamp = this->GetCurrentMMTime();
    char label[MM::MaxStrLength];
    this->GetLabel(label);

    // Important:  metadata about the image are generated here:
    Metadata md;
    md.put("Camera", label);
    md.put(MM::g_Keyword_Elapsed_Time_ms, CDeviceUtils::ConvertToString((timeStamp - sequenceStartTime_).getMsec()));
    md.put(MM::g_Keyword_Metadata_ROI_X, CDeviceUtils::ConvertToString((long)roiX_));
    md.put(MM::g_Keyword_Metadata_ROI_Y, CDeviceUtils::ConvertToString((long)roiY_));

    imageCounter_++;

    char buf[MM::MaxStrLength];
    GetProperty(MM::g_Keyword_Binning, buf);
    md.put(MM::g_Keyword_Binning, buf);

    MMThreadGuard g(imgPixelsLock_);

    const unsigned char* pI;
    pI = GetImageBuffer();


    unsigned int w = GetImageWidth();
    unsigned int h = GetImageHeight();
    unsigned int b = GetImageBytesPerPixel();

    int ret = GetCoreCallback()->InsertImage(this, pI, w, h, b, nComponents_, md.Serialize().c_str());
    if (!stopOnOverflow_ && ret == DEVICE_BUFFER_OVERFLOW)
    {
        // do not stop on overflow - just reset the buffer
        GetCoreCallback()->ClearImageBuffer(this);
        // don't process this same image again...
        return GetCoreCallback()->InsertImage(this, pI, w, h, b, nComponents_, md.Serialize().c_str(), false);
    }
    else
    {
        return ret;
    }
}

/*
 * Do actual capturing
 * Called from inside the thread
 */
int CPydevice::RunSequenceOnThread()
{
    int ret = DEVICE_ERR;
    MM::MMTime startTime = GetCurrentMMTime();

    // Trigger
    if (triggerDevice_.length() > 0) {
        MM::Device* triggerDev = GetDevice(triggerDevice_.c_str());
        if (triggerDev != 0) {
            LogMessage("trigger requested");
            triggerDev->SetProperty("Trigger", "+");
        }
    }
    
    GeneratePythonImage(img_);


    ret = InsertImage();

    if (ret != DEVICE_OK)
    {
        return ret;
    }
    return ret;
};

bool CPydevice::IsCapturing() {
    return !thd_->IsStopped();
}

/*
 * called from the thread function before exit
 */
void CPydevice::OnThreadExiting() throw()
{
    try
    {
        LogMessage(g_Msg_SEQUENCE_ACQUISITION_THREAD_EXITING);
        GetCoreCallback() ? GetCoreCallback()->AcqFinished(this, 0) : DEVICE_OK;
    }
    catch (...)
    {
        LogMessage(g_Msg_EXCEPTION_IN_ON_THREAD_EXITING, false);
    }
}


MySequenceThread::MySequenceThread(CPydevice* pCam)
    :intervalMs_(default_intervalMS)
    , numImages_(default_numImages)
    , imageCounter_(0)
    , stop_(true)
    , suspend_(false)
    , camera_(pCam)
    , startTime_(0)
    , actualDuration_(0)
    , lastFrameTime_(0)
{};

MySequenceThread::~MySequenceThread() {};

void MySequenceThread::Stop() {
    MMThreadGuard g(this->stopLock_);
    stop_ = true;
}

void MySequenceThread::Start(long numImages, double intervalMs)
{
    MMThreadGuard g1(this->stopLock_);
    MMThreadGuard g2(this->suspendLock_);
    numImages_ = numImages;
    intervalMs_ = intervalMs;
    imageCounter_ = 0;
    stop_ = false;
    suspend_ = false;
    activate();
    actualDuration_ = MM::MMTime{};
    startTime_ = camera_->GetCurrentMMTime();
    lastFrameTime_ = MM::MMTime{};
}

bool MySequenceThread::IsStopped() {
    MMThreadGuard g(this->stopLock_);
    return stop_;
}

void MySequenceThread::Suspend() {
    MMThreadGuard g(this->suspendLock_);
    suspend_ = true;
}

bool MySequenceThread::IsSuspended() {
    MMThreadGuard g(this->suspendLock_);
    return suspend_;
}

void MySequenceThread::Resume() {
    MMThreadGuard g(this->suspendLock_);
    suspend_ = false;
}

int MySequenceThread::svc(void) throw()
{
    int ret = DEVICE_ERR;
    try
    {
        do
        {
            ret = camera_->RunSequenceOnThread();
        } while (DEVICE_OK == ret && !IsStopped() && imageCounter_++ < numImages_ - 1);
        if (IsStopped())
            camera_->LogMessage("SeqAcquisition interrupted by the user\n");
    }
    catch (...) {
        camera_->LogMessage(g_Msg_EXCEPTION_IN_THREAD, false);
    }
    stop_ = true;
    actualDuration_ = camera_->GetCurrentMMTime() - startTime_;
    camera_->OnThreadExiting();
    return ret;
}


///////////////////////////////////////////////////////////////////////////////
// CPydevice Action handlers
///////////////////////////////////////////////////////////////////////////////


/*
* this Read Only property will update whenever any property is modified
*/

/**
* Handles "Binning" property.
*/
int CPydevice::OnBinning(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    int ret = DEVICE_ERR;
    switch (eAct)
    {
    case MM::AfterSet:
    {
        if (IsCapturing())
            return DEVICE_CAMERA_BUSY_ACQUIRING;

        // the user just set the new value for the property, so we have to
        // apply this value to the 'hardware'.
        long binFactor;
        pProp->Get(binFactor);
        if (binFactor > 0 && binFactor < 10)
        {
            // calculate ROI using the previous bin settings
            double factor = (double)binFactor / (double)binSize_;
            roiX_ = (unsigned int)(roiX_ / factor);
            roiY_ = (unsigned int)(roiY_ / factor);
            for (unsigned int i = 0; i < multiROIXs_.size(); ++i)
            {
                multiROIXs_[i] = (unsigned int)(multiROIXs_[i] / factor);
                multiROIYs_[i] = (unsigned int)(multiROIYs_[i] / factor);
                multiROIWidths_[i] = (unsigned int)(multiROIWidths_[i] / factor);
                multiROIHeights_[i] = (unsigned int)(multiROIHeights_[i] / factor);
            }
            img_.Resize((unsigned int)(img_.Width() / factor),
                (unsigned int)(img_.Height() / factor));
            binSize_ = binFactor;
            std::ostringstream os;
            os << binSize_;
            OnPropertyChanged("Binning", os.str().c_str());
            ret = DEVICE_OK;
        }
    }break;
    case MM::BeforeGet:
    {
        ret = DEVICE_OK;
        pProp->Set(binSize_);
    }break;
    default:
        break;
    }
    return ret;
}

/**
* Handles "PixelType" property.
*/
int CPydevice::OnPixelType(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    int ret = DEVICE_ERR;
    switch (eAct)
    {
    case MM::AfterSet:
    {
        if (IsCapturing())
            return DEVICE_CAMERA_BUSY_ACQUIRING;

        string pixelType;
        pProp->Get(pixelType);

        if (pixelType.compare(g_PixelType_8bit) == 0)
        {
            nComponents_ = 1;
            img_.Resize(img_.Width(), img_.Height(), 1);
            bitDepth_ = 8;
            ret = DEVICE_OK;
        }
        else if (pixelType.compare(g_PixelType_16bit) == 0)
        {
            nComponents_ = 1;
            img_.Resize(img_.Width(), img_.Height(), 2);
            bitDepth_ = 16;
            ret = DEVICE_OK;
        }
        else if (pixelType.compare(g_PixelType_32bitRGB) == 0)
        {
            nComponents_ = 4;
            img_.Resize(img_.Width(), img_.Height(), 4);
            bitDepth_ = 8;
            ret = DEVICE_OK;
        }
        else if (pixelType.compare(g_PixelType_64bitRGB) == 0)
        {
            nComponents_ = 4;
            img_.Resize(img_.Width(), img_.Height(), 8);
            bitDepth_ = 16;
            ret = DEVICE_OK;
        }
        else if (pixelType.compare(g_PixelType_32bit) == 0)
        {
            nComponents_ = 1;
            img_.Resize(img_.Width(), img_.Height(), 4);
            bitDepth_ = 32;
            ret = DEVICE_OK;
        }
        else
        {
            // on error switch to default pixel type
            nComponents_ = 1;
            img_.Resize(img_.Width(), img_.Height(), 1);
            pProp->Set(g_PixelType_8bit);
            bitDepth_ = 8;
            ret = ERR_UNKNOWN_MODE;
        }
    }
    break;
    case MM::BeforeGet:
    {
        long bytesPerPixel = GetImageBytesPerPixel();
        if (bytesPerPixel == 1)
        {
            pProp->Set(g_PixelType_8bit);
        }
        else if (bytesPerPixel == 2)
        {
            pProp->Set(g_PixelType_16bit);
        }
        else if (bytesPerPixel == 4)
        {
            if (nComponents_ == 4)
            {
                pProp->Set(g_PixelType_32bitRGB);
            }
            else if (nComponents_ == 1)
            {
                pProp->Set(::g_PixelType_32bit);
            }
        }
        else if (bytesPerPixel == 8)
        {
            pProp->Set(g_PixelType_64bitRGB);
        }
        else
        {
            pProp->Set(g_PixelType_8bit);
        }
        ret = DEVICE_OK;
    } break;
    default:
        break;
    }
    return ret;
}

/**
* Handles "BitDepth" property.
*/
int CPydevice::OnBitDepth(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    int ret = DEVICE_ERR;
    switch (eAct)
    {
    case MM::AfterSet:
    {
        if (IsCapturing())
            return DEVICE_CAMERA_BUSY_ACQUIRING;

        long bitDepth;
        pProp->Get(bitDepth);

        unsigned int bytesPerComponent;

        switch (bitDepth) {
        case 8:
            bytesPerComponent = 1;
            bitDepth_ = 8;
            ret = DEVICE_OK;
            break;
        case 10:
            bytesPerComponent = 2;
            bitDepth_ = 10;
            ret = DEVICE_OK;
            break;
        case 12:
            bytesPerComponent = 2;
            bitDepth_ = 12;
            ret = DEVICE_OK;
            break;
        case 14:
            bytesPerComponent = 2;
            bitDepth_ = 14;
            ret = DEVICE_OK;
            break;
        case 16:
            bytesPerComponent = 2;
            bitDepth_ = 16;
            ret = DEVICE_OK;
            break;
        case 32:
            bytesPerComponent = 4;
            bitDepth_ = 32;
            ret = DEVICE_OK;
            break;
        default:
            // on error switch to default pixel type
            bytesPerComponent = 1;

            pProp->Set((long)8);
            bitDepth_ = 8;
            ret = ERR_UNKNOWN_MODE;
            break;
        }
        char buf[MM::MaxStrLength];
        GetProperty(MM::g_Keyword_PixelType, buf);
        std::string pixelType(buf);
        unsigned int bytesPerPixel = 1;


        // automagickally change pixel type when bit depth exceeds possible value
        if (pixelType.compare(g_PixelType_8bit) == 0)
        {
            if (2 == bytesPerComponent)
            {
                SetProperty(MM::g_Keyword_PixelType, g_PixelType_16bit);
                bytesPerPixel = 2;
            }
            else if (4 == bytesPerComponent)
            {
                SetProperty(MM::g_Keyword_PixelType, g_PixelType_32bit);
                bytesPerPixel = 4;

            }
            else
            {
                bytesPerPixel = 1;
            }
        }
        else if (pixelType.compare(g_PixelType_16bit) == 0)
        {
            bytesPerPixel = 2;
        }
        else if (pixelType.compare(g_PixelType_32bitRGB) == 0)
        {
            bytesPerPixel = 4;
        }
        else if (pixelType.compare(g_PixelType_32bit) == 0)
        {
            bytesPerPixel = 4;
        }
        else if (pixelType.compare(g_PixelType_64bitRGB) == 0)
        {
            bytesPerPixel = 8;
        }
        img_.Resize(img_.Width(), img_.Height(), bytesPerPixel);

    } break;
    case MM::BeforeGet:
    {
        pProp->Set((long)bitDepth_);
        ret = DEVICE_OK;
    } break;
    default:
        break;
    }
    return ret;
}
/**
* Handles "ReadoutTime" property.
*/

/*
* Handles "ScanMode" property.
* Changes allowed Binning values to test whether the UI updates properly
*/
int CPydevice::OnScanMode(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::AfterSet) {
        pProp->Get(scanMode_);
        SetAllowedBinning();
        if (initialized_) {
            int ret = OnPropertiesChanged();
            if (ret != DEVICE_OK)
                return ret;
        }
    }
    else if (eAct == MM::BeforeGet) {
        LogMessage("Reading property ScanMode", true);
        pProp->Set(scanMode_);
    }
    return DEVICE_OK;
}




int CPydevice::OnScanXSteps(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        pProp->Set(ScanXSteps_);
    }
    else if (eAct == MM::AfterSet)
    {
        long value;
        pProp->Get(value);
        if ((value < 16) || (33000 < value))
            return DEVICE_ERR;  // invalid image size
        if (value != ScanXSteps_)
        {
            ScanXSteps_ = value;
            img_.Resize(ScanXSteps_ / binSize_, ScanYSteps_ / binSize_);
        }
    }
    return DEVICE_OK;

}

int CPydevice::OnZoomFactor(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        pProp->Set(zoomFactor_);
    }
    else if (eAct == MM::AfterSet)
    {
        pProp->Get(zoomFactor_);
    }
    return DEVICE_OK;
}

int CPydevice::OnDelay(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        pProp->Set(delay_);
    }
    else if (eAct == MM::AfterSet)
    {
        pProp->Get(delay_);
    }
    return DEVICE_OK;
}

int CPydevice::OnDwelltime(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        pProp->Set(dwelltime_);
    }
    else if (eAct == MM::AfterSet)
    {
        pProp->Get(dwelltime_);
    }
    return DEVICE_OK;
}

int CPydevice::OnScanpadding(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        pProp->Set(scanpadding_);
    }
    else if (eAct == MM::AfterSet)
    {
        pProp->Get(scanpadding_);
    }
    return DEVICE_OK;
}

int CPydevice::OnInputMin(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        pProp->Set(inputmin_);
    }
    else if (eAct == MM::AfterSet)
    {
        pProp->Get(inputmin_);
    }
    return DEVICE_OK;
}


int CPydevice::OnInputMax(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        pProp->Set(inputmax_);
    }
    else if (eAct == MM::AfterSet)
    {
        pProp->Get(inputmax_);
    }
    return DEVICE_OK;
}


int CPydevice::OnScanYSteps(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        pProp->Set(ScanYSteps_);
    }
    else if (eAct == MM::AfterSet)
    {
        long value;
        pProp->Get(value);
        if ((value < 16) || (33000 < value))
            return DEVICE_ERR;  // invalid image size
        if (value != ScanYSteps_)
        {
            ScanYSteps_ = value;
            img_.Resize(ScanXSteps_ / binSize_, ScanYSteps_ / binSize_);
        }
    }
    return DEVICE_OK;

}

int CPydevice::OnTriggerDevice(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        pProp->Set(triggerDevice_.c_str());
    }
    else if (eAct == MM::AfterSet)
    {
        pProp->Get(triggerDevice_);
    }
    return DEVICE_OK;
}


int CPydevice::OnDACPortOutx(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        pProp->Set(dacportoutx_.c_str());
    }
    else if (eAct == MM::AfterSet)
    {
        pProp->Get(dacportoutx_);
    }
    return DEVICE_OK;
}


int CPydevice::OnDACPortOuty(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        pProp->Set(dacportouty_.c_str());
    }
    else if (eAct == MM::AfterSet)
    {
        pProp->Get(dacportouty_);
    }
    return DEVICE_OK;
}

int CPydevice::OnDACPortIn(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        pProp->Set(dacportin_.c_str());
    }
    else if (eAct == MM::AfterSet)
    {
        pProp->Get(dacportin_);
    }
    return DEVICE_OK;
}


int CPydevice::OnIsSequenceable(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    std::string val = "Yes";
    if (eAct == MM::BeforeGet)
    {
        if (!isSequenceable_)
        {
            val = "No";
        }
        pProp->Set(val.c_str());
    }
    else if (eAct == MM::AfterSet)
    {
        isSequenceable_ = false;
        pProp->Get(val);
        if (val == "Yes")
        {
            isSequenceable_ = true;
        }
    }

    return DEVICE_OK;
}


int CPydevice::OnMode(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    std::string val;
    if (eAct == MM::BeforeGet)
    {
        switch (mode_)
        {
        case MODE_BIDIRECTIONAL:
            val = g_Bidirectional;
            break;
        case MODE_UNIDIRECTIONAL:
            val = g_Unidirectional;
            break;
        default:
            val = g_Bidirectional;
            break;
        }
        pProp->Set(val.c_str());
    }
    else if (eAct == MM::AfterSet)
    {
        pProp->Get(val);
        if (val == g_Unidirectional)
        {
            mode_ = MODE_UNIDIRECTIONAL;
        }
        else
        {
            mode_ = MODE_BIDIRECTIONAL;
        }
    }
    return DEVICE_OK;
}


///////////////////////////////////////////////////////////////////////////////
// Private CPydevice methods
///////////////////////////////////////////////////////////////////////////////

/**
* Sync internal image buffer size to the chosen property values.
*/
int CPydevice::ResizeImageBuffer()
{
    char buf[MM::MaxStrLength];
    //int ret = GetProperty(MM::g_Keyword_Binning, buf);
    //if (ret != DEVICE_OK)
    //   return ret;
    //binSize_ = atol(buf);

    int ret = GetProperty(MM::g_Keyword_PixelType, buf);
    if (ret != DEVICE_OK)
        return ret;

    std::string pixelType(buf);
    int byteDepth = 0;

    if (pixelType.compare(g_PixelType_8bit) == 0)
    {
        byteDepth = 1;
    }
    else if (pixelType.compare(g_PixelType_16bit) == 0)
    {
        byteDepth = 2;
    }
    else if (pixelType.compare(g_PixelType_32bitRGB) == 0)
    {
        byteDepth = 4;
    }
    else if (pixelType.compare(g_PixelType_32bit) == 0)
    {
        byteDepth = 4;
    }
    else if (pixelType.compare(g_PixelType_64bitRGB) == 0)
    {
        byteDepth = 8;
    }

    img_.Resize(ScanXSteps_ / binSize_, ScanYSteps_ / binSize_, byteDepth);
    return DEVICE_OK;
}

void CPydevice::GenerateEmptyImage(ImgBuffer& img)
{
    MMThreadGuard g(imgPixelsLock_);
    if (img.Height() == 0 || img.Width() == 0 || img.Depth() == 0)
        return;
    unsigned char* pBuf = const_cast<unsigned char*>(img.GetPixels());
    memset(pBuf, 0, img.Height() * img.Width() * img.Depth());
}


int CPydevice::GeneratePythonImage(ImgBuffer& img)
{

    MMThreadGuard g(imgPixelsLock_);

    //std::string pixelType;
    char buf[MM::MaxStrLength];
    GetProperty(MM::g_Keyword_PixelType, buf);
    std::string pixelType(buf);

    if (img.Height() == 0 || img.Width() == 0 || img.Depth() == 0)
        return 0;

    unsigned imgWidth = img.Width();

    // Define all neccesary for bridge:
    auto modname = PyUnicode_FromString("Pyscanner");
    auto funcname = "cpp_single_capture";

    // define all the input variable seperately


    auto input = PyUnicode_FromString(("['" + dacportin_ + "']").c_str());
    auto output = PyUnicode_FromString(("['" + dacportoutx_ + "', '" + dacportouty_ + "']").c_str());

    // read the image settings
    auto height = std::to_string(ScanXSteps_);
    auto width = std::to_string(ScanYSteps_);

    auto resolution = PyUnicode_FromString(("[" + height + ", " + width + "]").c_str());

    std::string zoomFactorString = std::to_string(zoomFactor_);
    auto zoom = PyUnicode_FromString(("[" + zoomFactorString + "]").c_str());


    std::string delayString = std::to_string(delay_);
    auto delay = PyUnicode_FromString(("[" + delayString + "]").c_str());




    auto dwelltimefactor = dwelltime_ / 1000000;
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(12) << dwelltimefactor;
    auto dwelltime = PyUnicode_FromString(("[" + stream.str() + "]").c_str());

    std::string scanpaddingString = std::to_string(scanpadding_);
    auto scanpadding = PyUnicode_FromString(("[" + scanpaddingString + "]").c_str());


    std::string inputminString = std::to_string(inputmin_);
    std::string inputmaxString = std::to_string(inputmax_);

    auto input_range = PyUnicode_FromString(("[" + inputminString + ", " + inputmaxString + "]").c_str());

    auto module = PyImport_Import(modname);

    auto func = PyObject_GetAttrString(module, funcname);
    auto args = PyTuple_New(8);

    PyTuple_SetItem(args, 0, input);
    PyTuple_SetItem(args, 1, output);
    PyTuple_SetItem(args, 2, resolution);
    PyTuple_SetItem(args, 3, zoom);
    PyTuple_SetItem(args, 4, delay);
    PyTuple_SetItem(args, 5, dwelltime);
    PyTuple_SetItem(args, 6, scanpadding);
    PyTuple_SetItem(args, 7, input_range);

    auto returnvalue = PyObject_CallObject(func, args);
    Py_DECREF(args);

    import_array();

    PyObject* numpy = PyImport_ImportModule("numpy");

    // Convert the returnvalue to a numpy array
    PyObject* array = PyArray_FROM_OTF(returnvalue, NPY_DOUBLE, NPY_IN_ARRAY);

    npy_intp* dimensions = PyArray_DIMS(array);

    // Get a pointer to the data of the numpy array
    double* data = (double*)PyArray_DATA(array);
    long maxValue = (1L << bitDepth_) - 1;
    unsigned j, k;
    if (pixelType.compare(g_PixelType_8bit) == 0)
    {
        unsigned char* pBuf = const_cast<unsigned char*>(img.GetPixels());
        for (int i = 0; i < img.Height() * img.Width() * img.Depth(); i++)
        {
            pBuf[i] = (unsigned char)(data[i] * maxValue);
        }
        Py_DECREF(array);
        Py_DECREF(func);
        Py_DECREF(module);
    }
    else if (pixelType.compare(g_PixelType_16bit) == 0) // this is what we do with 16 bit images
    {
        unsigned short* pBuf = (unsigned short*) const_cast<unsigned char*>(img.GetPixels());

        double min_val = inputmin_;
        double max_val = inputmax_;


        // Scale each input value to the range of 16-bit unsigned integers
        for (int i = 0; i < imgWidth * img.Height(); i++) {
            double scaled_val = (data[i] - min_val) / (max_val - min_val) * 65535.0;
            pBuf[i] = (unsigned short)scaled_val;
        }
//        for (j = 0; j < img.Height(); j++)
//        {
//            for (k = 0; k < imgWidth; k++)
//            {
//                long lIndex = imgWidth * j + k;
//                double val = data[imgWidth * j + k];
//                val = val * maxValue;
//                *(pBuf + lIndex) = (unsigned short)val;
//            }
//        }
        // Decrement the reference count of the numpy array
        Py_DECREF(array);
        Py_DECREF(func);
        Py_DECREF(module);
    }

    
    return 0;
}