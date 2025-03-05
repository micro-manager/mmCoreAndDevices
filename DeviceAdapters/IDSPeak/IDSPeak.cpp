///////////////////////////////////////////////////////////////////////////////
// FILE:          IDSPeak.cpp
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

#include "IDSPeak.h"
#include <cstdio>
#include <string>
#include <math.h>
#include "ModuleInterface.h"
#include <sstream>
#include <algorithm>
#include <iostream>
#include <future>
#include <unordered_map>

#include <ids_peak_comfort_c/ids_peak_comfort_c.h>

using namespace std;
const double CIDSPeak::nominalPixelSizeUm_ = 1.0;
double g_IntensityFactor_ = 1.0;
const char* g_PixelType_8bit = "8bit";
const char* g_PixelType_16bit = "16bit";
const char* g_PixelType_32bitRGBA = "32bit RGBA";

// External names used used by the rest of the system
// to load particular device from the "IDSPeak.dll" library
const char* g_CameraDeviceName = "IDSCam";
const char* g_IDSPeakHubName = "IDS Peak Hub";
const char* g_keyword_Peak_PixelFormat = "IDS Pixel Format";


string pixelFormatToString(peak_pixel_format pixelFormat) {
    switch (pixelFormat) {
    case PEAK_PIXEL_FORMAT_BAYER_GR8:
        return "BAYER_GR8";
    case PEAK_PIXEL_FORMAT_BAYER_GR10:
        return "BAYER_GR10";
    case PEAK_PIXEL_FORMAT_BAYER_GR12:
        return "BAYER_GR12";
    case PEAK_PIXEL_FORMAT_BAYER_RG8:
        return "BAYER_RG8";
    case PEAK_PIXEL_FORMAT_BAYER_RG10:
        return "BAYER_RG10";
    case PEAK_PIXEL_FORMAT_BAYER_RG12:
        return "BAYER_RG12";
    case PEAK_PIXEL_FORMAT_BAYER_GB8:
        return "BAYER_GB8";
    case PEAK_PIXEL_FORMAT_BAYER_GB10:
        return "BAYER_GB10";
    case PEAK_PIXEL_FORMAT_BAYER_GB12:
        return "BAYER_GB12";
    case PEAK_PIXEL_FORMAT_BAYER_BG8:
        return "BAYER_BG8";
    case PEAK_PIXEL_FORMAT_BAYER_BG10:
        return "BAYER_BG10";
    case PEAK_PIXEL_FORMAT_BAYER_BG12:
        return "BAYER_BG12";
    case PEAK_PIXEL_FORMAT_MONO8:
        return "MONO8";
    case PEAK_PIXEL_FORMAT_MONO10:
        return "MONO10";
    case PEAK_PIXEL_FORMAT_MONO12:
        return "MONO12";
    case PEAK_PIXEL_FORMAT_RGB8:
        return "RGB8";
    case PEAK_PIXEL_FORMAT_RGB10:
        return "RGB10";
    case PEAK_PIXEL_FORMAT_RGB12:
        return "RGB12";
    case PEAK_PIXEL_FORMAT_BGR8:
        return "BGR8";
    case PEAK_PIXEL_FORMAT_BGR10:
        return "BGR10";
    case PEAK_PIXEL_FORMAT_BGR12:
        return "BGR12";
    case PEAK_PIXEL_FORMAT_RGBA8:
        return "RGBA8";
    case PEAK_PIXEL_FORMAT_RGBA10:
        return "RGBA10";
    case PEAK_PIXEL_FORMAT_RGBA12:
        return "RGBA12";
    case PEAK_PIXEL_FORMAT_BGRA8:
        return "BGRA8";
    case PEAK_PIXEL_FORMAT_BGRA10:
        return "BGRA10";
    case PEAK_PIXEL_FORMAT_BGRA12:
        return "BGRA12";
    case PEAK_PIXEL_FORMAT_BAYER_GR10P:
        return "BAYER_GR10P";
    case PEAK_PIXEL_FORMAT_BAYER_GR12P:
        return "BAYER_GR12P";
    case PEAK_PIXEL_FORMAT_BAYER_RG10P:
        return "BAYER_RG10P";
    case PEAK_PIXEL_FORMAT_BAYER_RG12P:
        return "BAYER_RG12P";
    case PEAK_PIXEL_FORMAT_BAYER_GB10P:
        return "BAYER_GB10P";
    case PEAK_PIXEL_FORMAT_BAYER_GB12P:
        return "BAYER_GB12P";
    case PEAK_PIXEL_FORMAT_BAYER_BG10P:
        return "BAYER_BG10P";
    case PEAK_PIXEL_FORMAT_BAYER_BG12P:
        return "BAYER_BG12P";
    case PEAK_PIXEL_FORMAT_MONO10P:
        return "MONO10P";
    case PEAK_PIXEL_FORMAT_MONO12P:
        return "MONO12P";
    case PEAK_PIXEL_FORMAT_RGB10P32:
        return "RGB10P32";
    case PEAK_PIXEL_FORMAT_BGR10P32:
        return "BGR10P32";
    case PEAK_PIXEL_FORMAT_BAYER_GR10G40_IDS:
        return "BAYER_GR10G40_IDS";
    case PEAK_PIXEL_FORMAT_BAYER_RG10G40_IDS:
        return "BAYER_RG10G40_IDS";
    case PEAK_PIXEL_FORMAT_BAYER_GB10G40_IDS:
        return "BAYER_GB10G40_IDS";
    case PEAK_PIXEL_FORMAT_BAYER_BG10G40_IDS:
        return "BAYER_BG10G40_IDS";
    case PEAK_PIXEL_FORMAT_BAYER_GR12G24_IDS:
        return "BAYER_GR12G24_IDS";
    case PEAK_PIXEL_FORMAT_BAYER_RG12G24_IDS:
        return "BAYER_RG12G24_IDS";
    case PEAK_PIXEL_FORMAT_BAYER_GB12G24_IDS:
        return "BAYER_GB12G24_IDS";
    case PEAK_PIXEL_FORMAT_BAYER_BG12G24_IDS:
        return "BAYER_BG12G24_IDS";
    case PEAK_PIXEL_FORMAT_MONO10G40_IDS:
        return "MONO10G40_IDS";
    case PEAK_PIXEL_FORMAT_MONO12G24_IDS:
        return "MONO12G24_IDS";

    default:
        return "";
    }
}
unordered_map<string, peak_pixel_format> stringPixelFormat = {
    { "BAYER_GR8", PEAK_PIXEL_FORMAT_BAYER_GR8 },
    { "BAYER_GR10", PEAK_PIXEL_FORMAT_BAYER_GR10 },
    { "BAYER_GR12", PEAK_PIXEL_FORMAT_BAYER_GR12 },
    { "BAYER_RG8", PEAK_PIXEL_FORMAT_BAYER_RG8 },
    { "BAYER_RG10", PEAK_PIXEL_FORMAT_BAYER_RG10 },
    { "BAYER_RG12", PEAK_PIXEL_FORMAT_BAYER_RG12 },
    { "BAYER_GB8", PEAK_PIXEL_FORMAT_BAYER_GB8 },
    { "BAYER_GB10", PEAK_PIXEL_FORMAT_BAYER_GB10 },
    { "BAYER_GB12", PEAK_PIXEL_FORMAT_BAYER_GB12 },
    { "BAYER_BG8", PEAK_PIXEL_FORMAT_BAYER_BG8 },
    { "BAYER_BG10", PEAK_PIXEL_FORMAT_BAYER_BG10 },
    { "BAYER_BG12", PEAK_PIXEL_FORMAT_BAYER_BG12 },
    { "MONO8", PEAK_PIXEL_FORMAT_MONO8 },
    { "MONO10", PEAK_PIXEL_FORMAT_MONO10 },
    { "MONO12", PEAK_PIXEL_FORMAT_MONO12 },
    { "RGB8", PEAK_PIXEL_FORMAT_RGB8 },
    { "RGB10", PEAK_PIXEL_FORMAT_RGB10 },
    { "RGB12", PEAK_PIXEL_FORMAT_RGB12 },
    { "BGR8", PEAK_PIXEL_FORMAT_BGR8 },
    { "BGR10", PEAK_PIXEL_FORMAT_BGR10 },
    { "BGR12", PEAK_PIXEL_FORMAT_BGR12 },
    { "RGBA8", PEAK_PIXEL_FORMAT_RGBA8 },
    { "RGBA10", PEAK_PIXEL_FORMAT_RGBA10 },
    { "RGBA12", PEAK_PIXEL_FORMAT_RGBA12 },
    { "BGRA8", PEAK_PIXEL_FORMAT_BGRA8 },
    { "BGRA10", PEAK_PIXEL_FORMAT_BGRA10 },
    { "BGRA12", PEAK_PIXEL_FORMAT_BGRA12 },
    { "BAYER_GR10P", PEAK_PIXEL_FORMAT_BAYER_GR10P },
    { "BAYER_GR12P", PEAK_PIXEL_FORMAT_BAYER_GR12P },
    { "BAYER_RG10P", PEAK_PIXEL_FORMAT_BAYER_RG10P },
    { "BAYER_RG12P", PEAK_PIXEL_FORMAT_BAYER_RG12P },
    { "BAYER_GB10P", PEAK_PIXEL_FORMAT_BAYER_GB10P },
    { "BAYER_GB12P", PEAK_PIXEL_FORMAT_BAYER_GB12P },
    { "BAYER_BG10P", PEAK_PIXEL_FORMAT_BAYER_BG10P },
    { "BAYER_BG12P", PEAK_PIXEL_FORMAT_BAYER_BG12P },
    { "MONO10P", PEAK_PIXEL_FORMAT_MONO10P },
    { "MONO12P", PEAK_PIXEL_FORMAT_MONO12P },
    { "RGB10P32", PEAK_PIXEL_FORMAT_RGB10P32 },
    { "BGR10P32", PEAK_PIXEL_FORMAT_BGR10P32 },
    { "BAYER_GR10G40_IDS", PEAK_PIXEL_FORMAT_BAYER_GR10G40_IDS },
    { "BAYER_RG10G40_IDS", PEAK_PIXEL_FORMAT_BAYER_RG10G40_IDS },
    { "BAYER_GB10G40_IDS", PEAK_PIXEL_FORMAT_BAYER_GB10G40_IDS },
    { "BAYER_BG10G40_IDS", PEAK_PIXEL_FORMAT_BAYER_BG10G40_IDS },
    { "BAYER_GR12G24_IDS", PEAK_PIXEL_FORMAT_BAYER_GR12G24_IDS },
    { "BAYER_RG12G24_IDS", PEAK_PIXEL_FORMAT_BAYER_RG12G24_IDS },
    { "BAYER_GB12G24_IDS", PEAK_PIXEL_FORMAT_BAYER_GB12G24_IDS },
    { "BAYER_BG12G24_IDS", PEAK_PIXEL_FORMAT_BAYER_BG12G24_IDS },
    { "MONO10G40_IDS", PEAK_PIXEL_FORMAT_MONO10G40_IDS },
    { "MONO12G24_IDS", PEAK_PIXEL_FORMAT_MONO12G24_IDS },


};
peak_pixel_format stringToPixelFormat(string PixelFormat) {
    return stringPixelFormat[PixelFormat];
}

///////////////////////////////////////////////////////////////////////////////
//  MMDevice API
///////////////////////////////////////////////////////////////////////////////

MODULE_API void InitializeModuleData()
{
    RegisterDevice(g_IDSPeakHubName, MM::HubDevice, "Hub for IDS cameras");
    RegisterDevice(g_CameraDeviceName, MM::CameraDevice, "Device adapter for IDS peak cameras");
}

MODULE_API MM::Device* CreateDevice(const char* deviceName)
{
    cout << deviceName << endl;
    if (!deviceName)
    {
        return 0; // Trying to create nothing, return nothing
    }
    if (strcmp(deviceName, g_IDSPeakHubName) == 0)
    {
        return new IDSPeakHub(); // Create Hub
    }
    if (strncmp(deviceName, g_CameraDeviceName, strlen(g_CameraDeviceName)) == 0)
    {
        string name_s = deviceName;
        string substr = name_s.substr(strlen(g_CameraDeviceName), strlen(deviceName));
        int deviceIdx = stoi(substr);
        //if (!deviceIdx) { return 0; } // If somehow the channel index is incorrect, return nothing
        return new CIDSPeak(deviceIdx); // Create channel
    }
    return 0; // If an unexpected name is provided, return nothing
}

MODULE_API void DeleteDevice(MM::Device* device)
{
    delete device;
}

//////////////////////////////////////////////////////////////////////////////
// IDSPeakHub class
// Hub for IDS Peak cameras
//////////////////////////////////////////////////////////////////////////////

IDSPeakHub::IDSPeakHub() :
    initialized_(false),
    busy_(false),
    nCameras_(0),
    errorCode_(0),
    testVal(10)
{
    CreateIntegerProperty("Test", testVal, false);
}

IDSPeakHub::~IDSPeakHub() {
    Shutdown();
}

/**
* Obtains device name.
* Required by the MM::Device API.
*/
void IDSPeakHub::GetName(char* name) const
{
    // Return the name used to refer to this device adapter
    string deviceName = g_IDSPeakHubName;
    CDeviceUtils::CopyLimitedString(name, deviceName.c_str());
}

int IDSPeakHub::Initialize()
{
    if (initialized_)
        return DEVICE_OK;

    // Initalize peak status
    status = PEAK_STATUS_SUCCESS;

    // Initialize peak library
    status = peak_Library_Init();
    if (status != PEAK_STATUS_SUCCESS) { return ERR_LIBRARY_NOT_INIT; }

    // Name
    int ret = CreateStringProperty(MM::g_Keyword_Name, g_IDSPeakHubName, true);
    if (DEVICE_OK != ret)
        return ret;

    // Description
    ret = CreateStringProperty(MM::g_Keyword_Description, "Hub for IDS Peak cameras", true);
    if (DEVICE_OK != ret)
        return ret;

    // Detect number of cameras
    status = peak_CameraList_Update(NULL);
    if (status != PEAK_STATUS_SUCCESS) { return ERR_CAMERA_NOT_FOUND; }

    status = peak_CameraList_Get(NULL, &nCameras_); // get length of camera list
    if (status != PEAK_STATUS_SUCCESS) { return ERR_CAMERA_NOT_FOUND; } // exit program if no camera was found

    initialized_ = true;
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
int IDSPeakHub::Shutdown()
{
    peak_Library_Exit();
    initialized_ = false;
    return DEVICE_OK;
}

int IDSPeakHub::DetectInstalledDevices()
{
    ClearInstalledDevices();
    for (int i = 0; i < nCameras_; i++)
    {
        MM::Device* device = new CIDSPeak(i);
        if (device)
        {
            AddInstalledDevice(device);
        }
    }
    return DEVICE_OK;
}

bool IDSPeakHub::Busy()
{
    return false;
}

///////////////////////////////////////////////////////////////////////////////
// CIDSPeak implementation
// ~~~~~~~~~~~~~~~~~~~~~~~~~~

/**
* CIDSPeak constructor.
* Setup default all variables and create device properties required to exist
* before intialization. In this case, no such properties were required. All
* properties will be created in the Initialize() method.
*
* As a general guideline Micro-Manager devices do not access hardware in the
* the constructor. We should do as little as possible in the constructor and
* perform most of the initialization in the Initialize() method.
*/
CIDSPeak::CIDSPeak(int idx) :
    CLegacyCameraBase<CIDSPeak>(),
    initialized_(false),
    readoutUs_(0.0),
    bitDepth_(8),
    roiX_(0),
    roiY_(0),
    roiMinSizeX_(0),
    roiMinSizeY_(0),
    roiInc_(1),
    sequenceStartTime_(0),
    isSequenceable_(false),
    sequenceMaxLength_(100),
    sequenceRunning_(false),
    sequenceIndex_(0),
    binSize_(1),
    cameraCCDXSize_(512),
    cameraCCDYSize_(512),
    ccdT_(0.0),
    triggerDevice_(""),
    stopOnOverflow_(false),
    supportsMultiROI_(false),
    multiROIFillValue_(0),
    nComponents_(1),
    exposureMax_(10000.0),
    exposureMin_(0.0),
    exposureInc_(1.0),
    exposureCur_(10.0),
    framerateCur_(10),
    framerateMax_(1000),
    framerateMin_(0.1),
    framerateInc_(0.1),
    imageCounter_(0),
    gainMaster_(1.0),
    gainRed_(1.0),
    gainGreen_(1.0),
    gainBlue_(1.0)
{
    // call the base class method to set-up default error codes/messages
    InitializeDefaultErrorMessages();
    readoutStartTime_ = GetCurrentMMTime();
    thd_ = new MySequenceThread(this);
    deviceIdx_ = idx;

    // parent ID display
    CreateHubIDProperty();
}

/**
* CIDSPeak destructor.
* If this device used as intended within the Micro-Manager system,
* Shutdown() will be always called before the destructor. But in any case
* we need to make sure that all resources are properly released even if
* Shutdown() was not called.
*/
CIDSPeak::~CIDSPeak()
{
    Shutdown();
    StopSequenceAcquisition();
    delete thd_;
}

/**
* Obtains device name.
* Required by the MM::Device API.
*/
void CIDSPeak::GetName(char* name) const
{
    // Return the name used to referr to this device adapte
    string deviceName = g_CameraDeviceName;
    deviceName.append(std::to_string(deviceIdx_));
    CDeviceUtils::CopyLimitedString(name, deviceName.c_str());
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
int CIDSPeak::Initialize()
{
    if (initialized_)
        return DEVICE_OK;

    // Get handle to Hub
    IDSPeakHub* pHub = static_cast<IDSPeakHub*>(GetParentHub());
    if (pHub)
    {
        char hubLabel[MM::MaxStrLength];
        pHub->GetLabel(hubLabel);
        SetParentID(hubLabel);
    }
    else
        LogMessage(NoHubError);

    // Initalize peak status
    status = PEAK_STATUS_SUCCESS;

    // Retrieve Cam handle
    size_t nCameras = 0;
    status = peak_CameraList_Update(NULL);
    if (status != PEAK_STATUS_SUCCESS) { return ERR_CAMERA_NOT_FOUND; }

    status = peak_CameraList_Get(NULL, &nCameras); // get length of camera list
    if (status != PEAK_STATUS_SUCCESS) { return ERR_CAMERA_NOT_FOUND; } // exit program if no camera was found

    peak_camera_descriptor* cameraList = (peak_camera_descriptor*)calloc(
        nCameras, sizeof(peak_camera_descriptor)); // allocate memory for the camera list
    status = peak_CameraList_Get(cameraList, &nCameras); // get the camera list
    if (status != PEAK_STATUS_SUCCESS) { return ERR_CAMERA_NOT_FOUND; } // exit program if no camera was found

    status = peak_Camera_Open(cameraList[deviceIdx_].cameraID, &hCam); // Get the camera handle
    if (status != PEAK_STATUS_SUCCESS) { return ERR_CAMERA_NOT_FOUND; } // exit if this is unsuccessful
    free(cameraList); // free the camera list, not needed any longer

    // check which camera was actually opened
    peak_camera_descriptor cameraInfo;
    status = peak_Camera_GetDescriptor(peak_Camera_ID_FromHandle(hCam), &cameraInfo);
    if (status != PEAK_STATUS_SUCCESS) { return ERR_CAMERA_NOT_FOUND; }

    // set property list
    // -----------------

    // Name
    int nRet = CreateStringProperty(MM::g_Keyword_Name, g_CameraDeviceName, true);
    assert(nRet == DEVICE_OK);

    // Description
    nRet = CreateStringProperty(MM::g_Keyword_Description, "IDS Peak Camera Adapter", true);
    assert(nRet == DEVICE_OK);

    // CameraName
    modelName_ = cameraInfo.modelName;
    nRet = CreateStringProperty(MM::g_Keyword_CameraName, modelName_.c_str(), true);
    assert(nRet == DEVICE_OK);

    // SerialNumber
    serialNum_ = cameraInfo.serialNumber;
    nRet = CreateStringProperty("Serial Number", serialNum_.c_str(), true);
    assert(nRet == DEVICE_OK);

    // Binning
    CPropertyAction* pAct = new CPropertyAction(this, &CIDSPeak::OnBinning);
    nRet = CreateIntegerProperty(MM::g_Keyword_Binning, 0, false, pAct);
    assert(nRet == DEVICE_OK);
    nRet = SetAllowedBinning();
    if (nRet != DEVICE_OK) { return nRet; }

    // MM Pixel type
    vector<string> pixelTypeValues;
    pixelTypeValues.push_back(g_PixelType_8bit);
    pixelTypeValues.push_back(g_PixelType_16bit);
    if (isColorCamera()) { pixelTypeValues.push_back(g_PixelType_32bitRGBA); }

    pAct = new CPropertyAction(this, &CIDSPeak::OnPixelType);
    nRet = CreateStringProperty(MM::g_Keyword_PixelType, g_PixelType_8bit, false, pAct);
    nRet = SetAllowedValues(MM::g_Keyword_PixelType, pixelTypeValues);
    pixelType_ = g_PixelType_8bit;
    if (nRet != DEVICE_OK)
        return nRet;

    // Pixel formats (IDS side)
    size_t nPixelFormats = 0;
    status = peak_PixelFormat_GetList(hCam, NULL, &nPixelFormats);
    peak_pixel_format* pixelFormats = (peak_pixel_format*)calloc(nPixelFormats, sizeof(peak_pixel_format));

    status = peak_PixelFormat_GetList(hCam, pixelFormats, &nPixelFormats);
    for (size_t i = 0; i < nPixelFormats; i++) {
        availablePixelFormats.push_back(pixelFormatToString(pixelFormats[i]));
    }
    status = peak_PixelFormat_Get(hCam, &currPixelFormat);
    pAct = new CPropertyAction(this, &CIDSPeak::OnPeakPixelFormat);
    nRet = CreateStringProperty(g_keyword_Peak_PixelFormat, pixelFormatToString(currPixelFormat).c_str(), false, pAct);
    nRet = ClearAllowedValues(g_keyword_Peak_PixelFormat);
    nRet = SetAllowedValues(g_keyword_Peak_PixelFormat, availablePixelFormats);

    // Exposure time
    nRet = CreateFloatProperty(MM::g_Keyword_Exposure, exposureCur_, false);
    assert(nRet == DEVICE_OK);
    status = peak_ExposureTime_GetRange(hCam, &exposureMin_, &exposureMax_, &exposureInc_);
    if (status != PEAK_STATUS_SUCCESS) { return ERR_DEVICE_NOT_AVAILABLE; }
    exposureMin_ /= 1000;
    exposureMax_ /= 1000;
    exposureInc_ /= 1000;
    nRet = SetPropertyLimits(MM::g_Keyword_Exposure, exposureMin_, exposureMax_);
    assert(nRet == DEVICE_OK);
    status = peak_ExposureTime_Get(hCam, &exposureCur_);
    exposureCur_ /= 1000;

    // Frame rate
    pAct = new CPropertyAction(this, &CIDSPeak::OnFrameRate);
    nRet = CreateFloatProperty("MDA framerate", 1, false, pAct);
    assert(nRet == DEVICE_OK);
    status = peak_FrameRate_GetRange(hCam, &framerateMin_, &framerateMax_, &framerateInc_);
    nRet = SetPropertyLimits("MDA framerate", framerateMin_, framerateMax_);
    assert(nRet == DEVICE_OK);
    status = peak_FrameRate_Get(hCam, &framerateCur_);

    // Auto white balance
    initializeAutoWBConversion();
    status = peak_AutoWhiteBalance_Mode_Get(hCam, &peakAutoWhiteBalance_);
    pAct = new CPropertyAction(this, &CIDSPeak::OnAutoWhiteBalance);
    nRet = CreateStringProperty("Auto white balance", "Off", false, pAct);
    assert(nRet == DEVICE_OK);

    vector<string> autoWhiteBalanceValues;
    autoWhiteBalanceValues.push_back("Off");
    autoWhiteBalanceValues.push_back("Once");
    autoWhiteBalanceValues.push_back("Continuous");

    nRet = SetAllowedValues("Auto white balance", autoWhiteBalanceValues);
    if (nRet != DEVICE_OK)
        return nRet;

    // Gain master
    status = peak_Gain_GetRange(hCam, PEAK_GAIN_TYPE_DIGITAL, PEAK_GAIN_CHANNEL_MASTER, &gainMin_, &gainMax_, &gainInc_);
    status = peak_Gain_Get(hCam, PEAK_GAIN_TYPE_DIGITAL, PEAK_GAIN_CHANNEL_MASTER, &gainMaster_);
    pAct = new CPropertyAction(this, &CIDSPeak::OnGainMaster);
    nRet = CreateFloatProperty("Gain Master", 1.0, false, pAct);
    assert(nRet == DEVICE_OK);
    nRet = SetPropertyLimits("Gain Master", gainMin_, gainMax_);
    if (nRet != DEVICE_OK)
        return nRet;

    // Gain Red (should be set after gain master)
    status = peak_Gain_Get(hCam, PEAK_GAIN_TYPE_DIGITAL, PEAK_GAIN_CHANNEL_RED, &gainRed_);
    pAct = new CPropertyAction(this, &CIDSPeak::OnGainRed);
    nRet = CreateFloatProperty("Gain Red", gainRed_, false, pAct);
    assert(nRet == DEVICE_OK);
    nRet = SetPropertyLimits("Gain Red", gainMin_, gainMax_);
    if (nRet != DEVICE_OK)
        return nRet;

    // Gain Green (should be set after gain master)
    status = peak_Gain_Get(hCam, PEAK_GAIN_TYPE_DIGITAL, PEAK_GAIN_CHANNEL_GREEN, &gainGreen_);
    pAct = new CPropertyAction(this, &CIDSPeak::OnGainGreen);
    nRet = CreateFloatProperty("Gain Green", gainGreen_, false, pAct);
    assert(nRet == DEVICE_OK);
    nRet = SetPropertyLimits("Gain Green", gainMin_, gainMax_);
    if (nRet != DEVICE_OK)
        return nRet;

    //Gain Blue (should be called after gain master)
    status = peak_Gain_Get(hCam, PEAK_GAIN_TYPE_DIGITAL, PEAK_GAIN_CHANNEL_BLUE, &gainBlue_);
    pAct = new CPropertyAction(this, &CIDSPeak::OnGainBlue);
    nRet = CreateFloatProperty("Gain Blue", gainBlue_, false, pAct);
    assert(nRet == DEVICE_OK);
    nRet = SetPropertyLimits("Gain Blue", gainMin_, gainMax_);
    if (nRet != DEVICE_OK)
        return nRet;

    // camera temperature ReadOnly, and request camera temperature
    pAct = new CPropertyAction(this, &CIDSPeak::OnCCDTemp);
    nRet = CreateFloatProperty("CCDTemperature", 0, true, pAct);
    assert(nRet == DEVICE_OK);

    // readout time
    pAct = new CPropertyAction(this, &CIDSPeak::OnReadoutTime);
    nRet = CreateFloatProperty(MM::g_Keyword_ReadoutTime, 0, false, pAct);
    assert(nRet == DEVICE_OK);

    // CCD size of the camera we are modeling
    // getSensorInfo needs to be called before the CreateIntegerProperty
    // calls, othewise the default (512) values will be displayed.
    nRet = getSensorInfo();
    pAct = new CPropertyAction(this, &CIDSPeak::OnCameraCCDXSize);
    nRet = CreateIntegerProperty("OnCameraCCDXSize", 512, true, pAct);
    assert(nRet == DEVICE_OK);
    pAct = new CPropertyAction(this, &CIDSPeak::OnCameraCCDYSize);
    nRet = CreateIntegerProperty("OnCameraCCDYSize", 512, true, pAct);
    assert(nRet == DEVICE_OK);

    // Trigger device
    pAct = new CPropertyAction(this, &CIDSPeak::OnTriggerDevice);
    nRet = CreateStringProperty("TriggerDevice", "", false, pAct);
    assert(nRet == DEVICE_OK);

    pAct = new CPropertyAction(this, &CIDSPeak::OnSupportsMultiROI);
    nRet = CreateIntegerProperty("AllowMultiROI", 0, false, pAct);
    assert(nRet == DEVICE_OK);
    nRet = AddAllowedValue("AllowMultiROI", "0");
    assert(nRet == DEVICE_OK);
    nRet = AddAllowedValue("AllowMultiROI", "1");
    assert(nRet == DEVICE_OK);

    pAct = new CPropertyAction(this, &CIDSPeak::OnMultiROIFillValue);
    nRet = CreateIntegerProperty("MultiROIFillValue", 0, false, pAct);
    assert(nRet == DEVICE_OK);
    nRet = SetPropertyLimits("MultiROIFillValue", 0, 65536);
    assert(nRet == DEVICE_OK);

    // Whether or not to use exposure time sequencing
    pAct = new CPropertyAction(this, &CIDSPeak::OnIsSequenceable);
    std::string propName = "UseExposureSequences";
    nRet = CreateStringProperty(propName.c_str(), "No", false, pAct);
    assert(nRet == DEVICE_OK);
    nRet = AddAllowedValue(propName.c_str(), "Yes");
    assert(nRet == DEVICE_OK);
    nRet = AddAllowedValue(propName.c_str(), "No");
    assert(nRet == DEVICE_OK);

    // initialize image buffer
    GenerateEmptyImage(img_);

    // Obtain ROI properties
    // The SetROI function used the CCD size, so this function should
    // always be put after the getSensorInfo call
    // It is assumed that the maximum ROI size is the size of the CCD
    // and that the increment in X and Y are identical
    // Also the ROI needs to be initialized beforehand, hence it should be 
    // placed after GenerateEmptyImage()
    peak_size roi_size_min;
    peak_size roi_size_max;
    peak_size roi_size_inc;
    peak_roi roi;
    status = peak_ROI_Size_GetRange(hCam, &roi_size_min, &roi_size_max, &roi_size_inc);
    if (status != PEAK_STATUS_SUCCESS) { return DEVICE_ERR; }
    roiMinSizeX_ = roi_size_min.width;
    roiMinSizeY_ = roi_size_min.height;
    roiInc_ = roi_size_inc.height;
    status = peak_ROI_Get(hCam, &roi);
    SetROI(roi.offset.x, roi.offset.y, roi.size.width, roi.size.height);
    img_.Resize(roi.size.width, roi.size.height, nComponents_ * (bitDepth_ / 8));

    if (nRet != DEVICE_OK) { return nRet; }

    // synchronize all properties
    // --------------------------
    nRet = UpdateStatus();
    if (nRet != DEVICE_OK)
        return nRet;

    initialized_ = true;
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
int CIDSPeak::Shutdown()
{
    if (initialized_ == false) { return DEVICE_OK; }
    cleanExit();
    initialized_ = false;
    return DEVICE_OK;
}

/**
* Performs exposure and grabs a single image.
* This function should block during the actual exposure and return immediately afterwards
* (i.e., before readout).  This behavior is needed for proper synchronization with the shutter.
* Required by the MM::Camera API.
*/
int CIDSPeak::SnapImage()
{
    int nRet = DEVICE_OK;
    static int callCounter = 0;
    ++callCounter;

    MM::MMTime startTime = GetCurrentMMTime();

    unsigned int framesToAcquire = 1;
    unsigned int pendingFrames = framesToAcquire;
    unsigned int timeoutCount = 0;

    // Make SnapImage responsive, even if low framerate has been set
    double framerateTemp = framerateCur_;
    nRet = framerateSet(1000 / exposureCur_);

    uint32_t three_frame_times_timeout_ms = (uint32_t)((3000.0 / framerateCur_) + 0.5);

    status = peak_Acquisition_Start(hCam, framesToAcquire);
    if (status != PEAK_STATUS_SUCCESS) { return ERR_ACQ_START; }

    while (pendingFrames > 0)
    {
        peak_frame_handle hFrame;
        status = peak_Acquisition_WaitForFrame(hCam, three_frame_times_timeout_ms, &hFrame);
        if (status == PEAK_STATUS_TIMEOUT)
        {
            timeoutCount++;
            if (timeoutCount > 99) { return ERR_ACQ_TIMEOUT; }
            else { continue; }
        }
        else if (status == PEAK_STATUS_ABORTED) { break; }
        else if (status != PEAK_STATUS_SUCCESS) { return ERR_ACQ_FRAME; }

        // At this point we successfully got a frame handle. We can deal with the info now!
        nRet = transferBuffer(hFrame, img_);

        // Now we have transfered all information, we can release the frame.
        status = peak_Frame_Release(hCam, hFrame);
        if (PEAK_ERROR(status)) { return ERR_ACQ_RELEASE; }
        pendingFrames--;
    }
    readoutStartTime_ = GetCurrentMMTime();
    // Revert framerate back to framerate before snapshot
    nRet = framerateSet(framerateTemp);
    return nRet;
}


/**
* Returns pixel data.
* Required by the MM::Camera API.
* The calling program will assume the size of the buffer based on the values
* obtained from GetImageBufferSize(), which in turn should be consistent with
* values returned by GetImageWidth(), GetImageHight() and GetImageBytesPerPixel().
* The calling program also assumes that camera never changes the size of
* the pixel buffer on its own. In other words, the buffer can change only if
* appropriate properties are set (such as binning, pixel type, etc.)
*/
const unsigned char* CIDSPeak::GetImageBuffer()
{
    MMThreadGuard g(imgPixelsLock_);
    MM::MMTime readoutTime(readoutUs_);
    while (readoutTime > (GetCurrentMMTime() - readoutStartTime_)) {}
    unsigned char* pB = (unsigned char*)(img_.GetPixels());
    return pB;
}

/**
* Returns image buffer X-size in pixels.
* Required by the MM::Camera API.
*/
unsigned CIDSPeak::GetImageWidth() const
{
    return img_.Width();
}

/**
* Returns image buffer Y-size in pixels.
* Required by the MM::Camera API.
*/
unsigned CIDSPeak::GetImageHeight() const
{
    return img_.Height();
}

/**
* Returns image buffer pixel depth in bytes.
* Required by the MM::Camera API.
*/
unsigned CIDSPeak::GetImageBytesPerPixel() const
{
    return img_.Depth();
}

/**
* Returns the bit depth (dynamic range) of the pixel.
* This does not affect the buffer size, it just gives the client application
* a guideline on how to interpret pixel values.
* Required by the MM::Camera API.
*/
unsigned CIDSPeak::GetBitDepth() const
{
    return bitDepth_;
}

/**
* Returns the size in bytes of the image buffer.
* Required by the MM::Camera API.
*/
long CIDSPeak::GetImageBufferSize() const
{
    return img_.Width() * img_.Height() * GetImageBytesPerPixel();
}

/**
* Sets the camera Region Of Interest.
* Required by the MM::Camera API.
* This command will change the dimensions of the image.
* Depending on the hardware capabilities the camera may not be able to configure the
* exact dimensions requested - but should try do as close as possible.
* If both xSize and ySize are set to 0, the ROI is set to the entire CCD
* @param x - top-left corner coordinate
* @param y - top-left corner coordinate
* @param xSize - width
* @param ySize - height
*/
int CIDSPeak::SetROI(unsigned x, unsigned y, unsigned xSize, unsigned ySize)
{
    int nRet = DEVICE_OK;
    if (peak_ROI_GetAccessStatus(hCam) == PEAK_ACCESS_READWRITE)
    {
        multiROIXs_.clear();
        multiROIYs_.clear();
        multiROIWidths_.clear();
        multiROIHeights_.clear();
        if (xSize == 0 && ySize == 0)
        {
            // effectively clear ROI
            nRet = ResizeImageBuffer();
            if (nRet != DEVICE_OK) { return nRet; }
            roiX_ = 0;
            roiY_ = 0;
            xSize = cameraCCDXSize_;
            ySize = cameraCCDYSize_;
        }
        else
        {
            // If ROI is smaller than the minimum required size, set size to minimum
            if (xSize < roiMinSizeX_) { xSize = roiMinSizeX_; }
            if (ySize < roiMinSizeY_) { ySize = roiMinSizeY_; }
            // If ROI is larger than the CCD, set size to CCD size
            if (xSize > (unsigned int)cameraCCDXSize_) { xSize = cameraCCDXSize_; }
            if (ySize > (unsigned int)cameraCCDYSize_) { ySize = cameraCCDYSize_; }
            // If ROI is not multiple of increment, reduce ROI such that it is
            xSize -= xSize % roiInc_;
            ySize -= ySize % roiInc_;
            // Check if ROI goes out of bounds, if so, push it in
            if (x + xSize > (unsigned int)cameraCCDXSize_) { x = cameraCCDXSize_ - xSize; }
            if (y + ySize > (unsigned int)cameraCCDYSize_) { y = cameraCCDYSize_ - ySize; }
            // apply ROI
            img_.Resize(xSize, ySize);
            roiX_ = x;
            roiY_ = y;
        }
        // Actually push the ROI settings to the camera
        peak_roi roi;
        roi.offset.x = roiX_;
        roi.offset.y = roiY_;
        roi.size.width = xSize;
        roi.size.height = ySize;
        status = peak_ROI_Set(hCam, roi);
    }
    else { return DEVICE_CAN_NOT_SET_PROPERTY; }
    return nRet;
}

/**
* Returns the actual dimensions of the current ROI.
* If multiple ROIs are set, then the returned ROI should encompass all of them.
* Required by the MM::Camera API.
*/
int CIDSPeak::GetROI(unsigned& x, unsigned& y, unsigned& xSize, unsigned& ySize)
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
int CIDSPeak::ClearROI()
{
    // Passing all zeros to SetROI sets the ROI to the full frame
    int nRet = SetROI(0, 0, 0, 0);
    return nRet;
}

/**
 * Queries if the camera supports multiple simultaneous ROIs.
 * Optional method in the MM::Camera API; by default cameras do not support
 * multiple ROIs.
 */
bool CIDSPeak::SupportsMultiROI()
{
    return supportsMultiROI_;
}

/**
 * Queries if multiple ROIs have been set (via the SetMultiROI method). Must
 * return true even if only one ROI was set via that method, but must return
 * false if an ROI was set via SetROI() or if ROIs have been cleared.
 * Optional method in the MM::Camera API; by default cameras do not support
 * multiple ROIs, so this method returns false.
 */
bool CIDSPeak::IsMultiROISet()
{
    return multiROIXs_.size() > 0;
}

/**
 * Queries for the current set number of ROIs. Must return zero if multiple
 * ROIs are not set (including if an ROI has been set via SetROI).
 * Optional method in the MM::Camera API; by default cameras do not support
 * multiple ROIs.
 */
int CIDSPeak::GetMultiROICount(unsigned int& count)
{
    count = (unsigned int)multiROIXs_.size();
    return DEVICE_OK;
}

/**
 * Set multiple ROIs. Replaces any existing ROI settings including ROIs set
 * via SetROI.
 * Optional method in the MM::Camera API; by default cameras do not support
 * multiple ROIs.
 * @param xs Array of X indices of upper-left corner of the ROIs.
 * @param ys Array of Y indices of upper-left corner of the ROIs.
 * @param widths Widths of the ROIs, in pixels.
 * @param heights Heights of the ROIs, in pixels.
 * @param numROIs Length of the arrays.
 */
int CIDSPeak::SetMultiROI(const unsigned int* xs, const unsigned int* ys,
    const unsigned* widths, const unsigned int* heights,
    unsigned numROIs)
{
    multiROIXs_.clear();
    multiROIYs_.clear();
    multiROIWidths_.clear();
    multiROIHeights_.clear();
    unsigned int minX = UINT_MAX;
    unsigned int minY = UINT_MAX;
    unsigned int maxX = 0;
    unsigned int maxY = 0;
    for (unsigned int i = 0; i < numROIs; ++i)
    {
        multiROIXs_.push_back(xs[i]);
        multiROIYs_.push_back(ys[i]);
        multiROIWidths_.push_back(widths[i]);
        multiROIHeights_.push_back(heights[i]);
        if (minX > xs[i])
        {
            minX = xs[i];
        }
        if (minY > ys[i])
        {
            minY = ys[i];
        }
        if (xs[i] + widths[i] > maxX)
        {
            maxX = xs[i] + widths[i];
        }
        if (ys[i] + heights[i] > maxY)
        {
            maxY = ys[i] + heights[i];
        }
    }
    img_.Resize(maxX - minX, maxY - minY);
    roiX_ = minX;
    roiY_ = minY;
    return DEVICE_OK;
}

/**
 * Queries for current multiple-ROI setting. May be called even if no ROIs of
 * any type have been set. Must return length of 0 in that case.
 * Optional method in the MM::Camera API; by default cameras do not support
 * multiple ROIs.
 * @param xs (Return value) X indices of upper-left corner of the ROIs.
 * @param ys (Return value) Y indices of upper-left corner of the ROIs.
 * @param widths (Return value) Widths of the ROIs, in pixels.
 * @param heights (Return value) Heights of the ROIs, in pixels.
 * @param numROIs Length of the input arrays. If there are fewer ROIs than
 *        this, then this value must be updated to reflect the new count.
 */
int CIDSPeak::GetMultiROI(unsigned* xs, unsigned* ys, unsigned* widths,
    unsigned* heights, unsigned* length)
{
    unsigned int roiCount = (unsigned int)multiROIXs_.size();
    if (roiCount > *length)
    {
        // This should never happen.
        return DEVICE_INTERNAL_INCONSISTENCY;
    }
    for (unsigned int i = 0; i < roiCount; ++i)
    {
        xs[i] = multiROIXs_[i];
        ys[i] = multiROIYs_[i];
        widths[i] = multiROIWidths_[i];
        heights[i] = multiROIHeights_[i];
    }
    *length = roiCount;
    return DEVICE_OK;
}

/**
* Returns the current exposure setting in milliseconds.
* Required by the MM::Camera API.
*/
double CIDSPeak::GetExposure() const
{
    char buf[MM::MaxStrLength];
    int nRet = GetProperty(MM::g_Keyword_Exposure, buf);
    if (nRet != DEVICE_OK) { return 0.; } // If something goes wrong, return 0. 
    return atof(buf);
}

/**
 * Returns the current exposure from a sequence and increases the sequence counter
 * Used for exposure sequences
 */
double CIDSPeak::GetSequenceExposure()
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
void CIDSPeak::SetExposure(double exp)
{
    // Convert milliseconds to microseconds (peak cameras expect time in microseconds)
    // and make exposure set multiple of increment.
    double exposureSet = ceil(exp / exposureInc_) * exposureInc_ * 1000;
    // Check if we can write to the exposure time of the camera, if not do nothing
    if (peak_ExposureTime_GetAccessStatus(hCam) == PEAK_ACCESS_READWRITE)
    {
        // Check if exposure time is less than the minimun exposure time
        // If so, set it to minimum exposure time.
        if (exp <= exposureMin_) {
            printf("Exposure time too short. Exposure time set to minimum.");
            status = peak_ExposureTime_Set(hCam, exposureMin_ * 1000);
        }
        // Check if exposure time is less than the maximum exposure time
        // If so, set it to maximum exposure time.
        else if (exp >= exposureMax_) {
            printf("Exposure time too long. Exposure time set to maximum.");
            status = peak_ExposureTime_Set(hCam, exposureMax_ * 1000);
        }
        // 
        else
        {
            status = peak_ExposureTime_Set(hCam, exposureSet);
        }
        // Update framerate range
        status = peak_FrameRate_GetRange(hCam, &framerateMin_, &framerateMax_, &framerateInc_);
        SetPropertyLimits("MDA framerate", framerateMin_, framerateMax_);

        // Set displayed exposure time
        status = peak_ExposureTime_Get(hCam, &exposureCur_);
        exposureCur_ /= 1000;
        SetProperty(MM::g_Keyword_Exposure, CDeviceUtils::ConvertToString(exposureCur_));
        GetCoreCallback()->OnExposureChanged(this, exp);
    }
}

/**
* Returns the current binning factor.
* Required by the MM::Camera API.
*/
int CIDSPeak::GetBinning() const
{
    char buf[MM::MaxStrLength];
    int nRet = GetProperty(MM::g_Keyword_Binning, buf);
    if (nRet != DEVICE_OK) { return 0; } // If something goes wrong, return 0 (unphysical binning)
    return atoi(buf);
}

/**
* Sets binning factor.
* Required by the MM::Camera API.
*/
int CIDSPeak::SetBinning(int binF)
{
    // Update binning
    status = peak_Binning_Set(hCam, (uint32_t)binF, (uint32_t)binF);
    if (status != PEAK_STATUS_SUCCESS) { return DEVICE_ERR; }
    binSize_ = binF;
    int nRet = SetProperty(MM::g_Keyword_Binning, CDeviceUtils::ConvertToString(binF));
    return nRet;
}

int CIDSPeak::IsExposureSequenceable(bool& isSequenceable) const
{
    isSequenceable = isSequenceable_;
    return DEVICE_OK;
}

int CIDSPeak::GetExposureSequenceMaxLength(long& nrEvents) const
{
    if (!isSequenceable_) {
        return DEVICE_UNSUPPORTED_COMMAND;
    }

    nrEvents = sequenceMaxLength_;
    return DEVICE_OK;
}

int CIDSPeak::StartExposureSequence()
{
    if (!isSequenceable_) {
        return DEVICE_UNSUPPORTED_COMMAND;
    }

    // may need thread lock
    sequenceRunning_ = true;
    return DEVICE_OK;
}

int CIDSPeak::StopExposureSequence()
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
int CIDSPeak::ClearExposureSequence()
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
int CIDSPeak::AddToExposureSequence(double exposureTime_ms)
{
    if (!isSequenceable_) {
        return DEVICE_UNSUPPORTED_COMMAND;
    }

    exposureSequence_.push_back(exposureTime_ms);
    return DEVICE_OK;
}

int CIDSPeak::SendExposureSequence() const {
    if (!isSequenceable_) {
        return DEVICE_UNSUPPORTED_COMMAND;
    }

    return DEVICE_OK;
}

/**
* Function that requests the possible binning values from the camera driver
* It implicitly assumes that the allowed binning factors in X and Y are the same
* It then restricts the allowed binning values to the possible ones
* It also checks whether the current binning value is allowed. If, somehow, it
* is not, it will set it to 1 (which it assumes is always available)
**/
int CIDSPeak::SetAllowedBinning()
{
    int nRet = DEVICE_OK;
    if (peak_Binning_GetAccessStatus(hCam) == PEAK_ACCESS_READWRITE)
    {
        // Get the binning factors, uses two staged data query (first get length of list, then get list)
        size_t binningFactorCount;
        status = peak_Binning_FactorY_GetList(hCam, NULL, &binningFactorCount);
        if (status != PEAK_STATUS_SUCCESS) { return DEVICE_ERR; }
        uint32_t* binningFactorList = (uint32_t*)calloc(binningFactorCount, sizeof(uint32_t));
        status = peak_Binning_FactorY_GetList(hCam, binningFactorList, &binningFactorCount);
        if (status != PEAK_STATUS_SUCCESS) { return DEVICE_ERR; }

        bool curr_bin_invalid = true;
        vector<string> binValues;
        for (size_t i = 0; i < binningFactorCount; i++)
        {
            if (binningFactorList[i] == (uint32_t)binSize_) { curr_bin_invalid = false; }

            binValues.push_back(to_string(binningFactorList[i]));
        }

        if (curr_bin_invalid)
        {
            nRet = SetBinning(1);
        }
        else
        {
            nRet = SetBinning(binSize_);
        }
        nRet = ClearAllowedValues(MM::g_Keyword_Binning);
        nRet = SetAllowedValues(MM::g_Keyword_Binning, binValues);
        return nRet;
    }
    else { return ERR_NO_WRITE_ACCESS; }
}


/**
 * Required by the MM::Camera API
 * Please implement this yourself and do not rely on the base class implementation
 * The Base class implementation is deprecated and will be removed shortly
 */
int CIDSPeak::StartSequenceAcquisition(double interval)
{
    return StartSequenceAcquisition(LONG_MAX, interval, false);
}

/**
* Stop and wait for the Sequence thread finished
*/
int CIDSPeak::StopSequenceAcquisition()
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
int CIDSPeak::StartSequenceAcquisition(long numImages, double interval_ms, bool stopOnOverflow)
{
    if (IsCapturing())
    {
        return DEVICE_CAMERA_BUSY_ACQUIRING;
    }
    int nRet = DEVICE_OK;

    // Adjust framerate to match requested interval between frames
    nRet = framerateSet(1000 / interval_ms);

    // Wait until shutter is ready
    nRet = GetCoreCallback()->PrepareForAcq(this);
    if (nRet != DEVICE_OK)
        return nRet;
    sequenceStartTime_ = GetCurrentMMTime();
    imageCounter_ = 0;
    thd_->Start(numImages, interval_ms);
    stopOnOverflow_ = stopOnOverflow;
    return DEVICE_OK;
}

/*
 * Inserts Image and MetaData into MMCore circular Buffer
 */
int CIDSPeak::InsertImage()
{
    MM::MMTime timeStamp = this->GetCurrentMMTime();
    char label[MM::MaxStrLength];
    this->GetLabel(label);

    // Important:  metadata about the image are generated here:
    Metadata md;
    md.put(MM::g_Keyword_Metadata_CameraLabel, label);
    md.put(MM::g_Keyword_Elapsed_Time_ms, CDeviceUtils::ConvertToString((timeStamp - sequenceStartTime_).getMsec()));
    md.put(MM::g_Keyword_Metadata_ROI_X, CDeviceUtils::ConvertToString((long)roiX_));
    md.put(MM::g_Keyword_Metadata_ROI_Y, CDeviceUtils::ConvertToString((long)roiY_));

    char buf[MM::MaxStrLength];
    GetProperty(MM::g_Keyword_Binning, buf);
    md.put(MM::g_Keyword_Binning, buf);

    imageCounter_++;

    MMThreadGuard g(imgPixelsLock_);
    int nRet = GetCoreCallback()->InsertImage(this, img_.GetPixels(),
        img_.Width(),
        img_.Height(),
        img_.Depth(),
        md.Serialize().c_str());

    if (!stopOnOverflow_ && nRet == DEVICE_BUFFER_OVERFLOW)
    {
        // do not stop on overflow - just reset the buffer
        GetCoreCallback()->ClearImageBuffer(this);
        return GetCoreCallback()->InsertImage(this, img_.GetPixels(),
            img_.Width(),
            img_.Height(),
            img_.Depth(),
            md.Serialize().c_str());
    }
    else { return nRet; }
}

/*
 * Do actual capturing
 * Called from inside the thread
 */
int CIDSPeak::RunSequenceOnThread()
{
    int nRet = DEVICE_ERR;
    MM::MMTime startTime = GetCurrentMMTime();

    // Trigger
    if (triggerDevice_.length() > 0) {
        MM::Device* triggerDev = GetDevice(triggerDevice_.c_str());
        if (triggerDev != 0) {
            LogMessage("trigger requested");
            triggerDev->SetProperty("Trigger", "+");
        }
    }

    uint32_t three_frame_times_timeout_ms = (uint32_t)(3000 / framerateCur_);

    peak_frame_handle hFrame;
    status = peak_Acquisition_WaitForFrame(hCam, three_frame_times_timeout_ms, &hFrame);
    if (status != PEAK_STATUS_SUCCESS) { return DEVICE_ERR; }
    else { nRet = DEVICE_OK; }

    // At this point we successfully got a frame handle. We can deal with the info now!
    nRet = transferBuffer(hFrame, img_);
    if (nRet != DEVICE_OK) { return DEVICE_ERR; }
    else { nRet = DEVICE_OK; }

    nRet = InsertImage();
    if (nRet != DEVICE_OK) { return DEVICE_ERR; }
    else { nRet = DEVICE_OK; }

    // Now we have transfered all information, we can release the frame.
    status = peak_Frame_Release(hCam, hFrame);
    if (status != PEAK_STATUS_SUCCESS) { return DEVICE_ERR; }
    else { nRet = DEVICE_OK; }

    return nRet;
};

bool CIDSPeak::IsCapturing() {
    return !thd_->IsStopped();
}

/*
 * called from the thread function before exit
 */
void CIDSPeak::OnThreadExiting() throw()
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


MySequenceThread::MySequenceThread(CIDSPeak* pCam)
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

//void MySequenceThread::SetIntervalMs(double intervalms)
//{
//    intervalMs_ = intervalms;
//}

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
    int nRet = DEVICE_ERR;
    peak_status status = PEAK_STATUS_SUCCESS;
    try
    {
        // peak_Acquisition_Start doesn't take LONG_MAX (2.1B) as near infinite, it crashes.
        // Instead, if numImages is LONG_MAX, PEAK_INFINITE is passed. This means that sometimes
        // the acquisition has to be stopped manually, but since this is properly escaped anyway
        // (in case of manual closing live view), this is all handled.
        if (numImages_ == LONG_MAX) { status = peak_Acquisition_Start(camera_->hCam, PEAK_INFINITE); }
        else { status = peak_Acquisition_Start(camera_->hCam, (uint32_t)numImages_); }

        // Check if acquisition is started properly
        if (status != PEAK_STATUS_SUCCESS) { return ERR_ACQ_START; }

        // do-while loop over numImages_
        do
        {
            nRet = camera_->RunSequenceOnThread();
        } while (nRet == DEVICE_OK && !IsStopped() && imageCounter_++ < numImages_ - 1);

        // If the acquisition is stopped manually, the acquisition has to be properly closed to
        // prevent the camera to be locked in acquisition mode.
        if (IsStopped())
        {
            status = peak_Acquisition_Stop(camera_->hCam);
            camera_->LogMessage("SeqAcquisition interrupted by the user\n");
        }
    }
    catch (...) {
        camera_->LogMessage(g_Msg_EXCEPTION_IN_THREAD, false);
    }
    stop_ = true;
    actualDuration_ = camera_->GetCurrentMMTime() - startTime_;
    camera_->OnThreadExiting();
    return nRet;
}


///////////////////////////////////////////////////////////////////////////////
// CIDSPeak Action handlers
///////////////////////////////////////////////////////////////////////////////

int CIDSPeak::OnMaxExposure(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        pProp->Set(exposureMax_);
        return DEVICE_OK;
    }
    else if (eAct == MM::AfterSet)
    {
        if (IsCapturing())
            return DEVICE_CAMERA_BUSY_ACQUIRING;

        double exposureSet;
        pProp->Get(exposureSet);

        if (peak_ExposureTime_GetAccessStatus(hCam) == PEAK_ACCESS_READWRITE)
        {
            status = peak_ExposureTime_Set(hCam, exposureMax_ * 1000);
            if (status != PEAK_STATUS_SUCCESS) { return DEVICE_ERR; } // Should not be possible
            status = peak_ExposureTime_Get(hCam, &exposureCur_);
            if (status != PEAK_STATUS_SUCCESS) { return DEVICE_ERR; } // Should not be possible
            exposureCur_ /= 1000;
            int nRet = SetProperty(MM::g_Keyword_Exposure, CDeviceUtils::ConvertToString(exposureCur_));
            GetCoreCallback()->OnExposureChanged(this, exposureCur_);
            return nRet;
        }
        else { return ERR_NO_WRITE_ACCESS; }
    }
    return DEVICE_OK; // Should not be possible, but doesn't affect anything
}

/**
* Handles "Binning" property.
*/
int CIDSPeak::OnBinning(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    int nRet = DEVICE_ERR;
    switch (eAct)
    {
    case MM::AfterSet:
    {
        // Should not change binning during acquisition
        if (IsCapturing()) { return DEVICE_CAMERA_BUSY_ACQUIRING; }

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
            img_.Resize(
                (unsigned int)(img_.Width() / factor),
                (unsigned int)(img_.Height() / factor)
            );
            binSize_ = binFactor;
            std::ostringstream os;
            os << binSize_;
            OnPropertyChanged("Binning", os.str().c_str());
            nRet = DEVICE_OK;
        }
    }break;
    case MM::BeforeGet:
    {
        nRet = DEVICE_OK;
        pProp->Set(binSize_);
    }break;
    default:
        break;
    }
    return nRet;
}

int CIDSPeak::OnFrameRate(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        pProp->Set(framerateCur_);
    }
    else if (eAct == MM::AfterSet)
    {
        double framerateTemp;
        pProp->Get(framerateTemp);
        framerateSet(framerateTemp);
    }
    return DEVICE_OK;
}

/**
* Handles "Auto whitebalance" property.
*/
int CIDSPeak::OnAutoWhiteBalance(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    int nRet = DEVICE_OK;

    if (eAct == MM::BeforeGet)
    {
        if (peak_AutoWhiteBalance_GetAccessStatus(hCam) == PEAK_ACCESS_READWRITE
            || peak_AutoWhiteBalance_GetAccessStatus(hCam) == PEAK_ACCESS_READONLY)
        {
            status = peak_AutoWhiteBalance_Mode_Get(hCam, &peakAutoWhiteBalance_);
        }
        const char* autoWB = peakAutoToString[peakAutoWhiteBalance_].c_str();
        pProp->Set(autoWB);
    }

    else if (eAct == MM::AfterSet)
    {
        if (IsCapturing())
            return DEVICE_CAMERA_BUSY_ACQUIRING;

        string autoWB;
        pProp->Get(autoWB);

        status = peak_AutoWhiteBalance_Mode_Set(hCam, (peak_auto_feature_mode)stringToPeakAuto[autoWB]);
        if (status != PEAK_STATUS_SUCCESS) { nRet = ERR_NO_WRITE_ACCESS; }
        else { peakAutoWhiteBalance_ = (peak_auto_feature_mode)stringToPeakAuto[autoWB]; }
    }
    return nRet;
}

/**
* Handles "Gain master" property.
*/
int CIDSPeak::OnGainMaster(MM::PropertyBase* pProp, MM::ActionType eAct)
{

    int nRet = DEVICE_OK;

    if (eAct == MM::BeforeGet)
    {
        status = peak_Gain_Get(hCam, PEAK_GAIN_TYPE_DIGITAL, PEAK_GAIN_CHANNEL_MASTER, &gainMaster_);
        pProp->Set(gainMaster_);
    }

    else if (eAct == MM::AfterSet)
    {
        if (IsCapturing())
            return DEVICE_CAMERA_BUSY_ACQUIRING;

        double gainMaster;
        pProp->Get(gainMaster);

        status = peak_Gain_Set(hCam, PEAK_GAIN_TYPE_DIGITAL, PEAK_GAIN_CHANNEL_RED, gainMaster);
        if (status != PEAK_STATUS_SUCCESS) { nRet = ERR_NO_WRITE_ACCESS; }
        else { gainMaster_ = gainMaster; }
    }
    return nRet;
}

/**
* Handles "Gain red" property.
*/
int CIDSPeak::OnGainRed(MM::PropertyBase* pProp, MM::ActionType eAct)
{

    int nRet = DEVICE_OK;

    if (eAct == MM::BeforeGet)
    {
        status = peak_Gain_Get(hCam, PEAK_GAIN_TYPE_DIGITAL, PEAK_GAIN_CHANNEL_RED, &gainRed_);
        pProp->Set(gainRed_);
    }

    else if (eAct == MM::AfterSet)
    {
        if (IsCapturing())
            return DEVICE_CAMERA_BUSY_ACQUIRING;

        double gainRed;
        pProp->Get(gainRed);

        status = peak_Gain_Set(hCam, PEAK_GAIN_TYPE_DIGITAL, PEAK_GAIN_CHANNEL_RED, gainRed);
        if (status != PEAK_STATUS_SUCCESS) { nRet = ERR_NO_WRITE_ACCESS; }
        else { gainRed_ = gainRed; }

    }
    return nRet;
}

/**
* Handles "Gain green" property.
*/
int CIDSPeak::OnGainGreen(MM::PropertyBase* pProp, MM::ActionType eAct)
{

    int nRet = DEVICE_OK;

    if (eAct == MM::BeforeGet)
    {
        status = peak_Gain_Get(hCam, PEAK_GAIN_TYPE_DIGITAL, PEAK_GAIN_CHANNEL_GREEN, &gainGreen_);
        pProp->Set(gainGreen_);
    }

    else if (eAct == MM::AfterSet)
    {
        if (IsCapturing())
            return DEVICE_CAMERA_BUSY_ACQUIRING;

        double gainGreen;
        pProp->Get(gainGreen);

        status = peak_Gain_Set(hCam, PEAK_GAIN_TYPE_DIGITAL, PEAK_GAIN_CHANNEL_GREEN, gainGreen);
        if (status != PEAK_STATUS_SUCCESS) { nRet = ERR_NO_WRITE_ACCESS; }
        else { gainGreen_ = gainGreen; }

    }
    return nRet;
}

/**
* Handles "Gain blue" property.
*/
int CIDSPeak::OnGainBlue(MM::PropertyBase* pProp, MM::ActionType eAct)
{

    int nRet = DEVICE_OK;

    if (eAct == MM::BeforeGet)
    {
        status = peak_Gain_Get(hCam, PEAK_GAIN_TYPE_DIGITAL, PEAK_GAIN_CHANNEL_BLUE, &gainBlue_);
        pProp->Set(gainRed_);
    }

    else if (eAct == MM::AfterSet)
    {
        if (IsCapturing())
            return DEVICE_CAMERA_BUSY_ACQUIRING;

        double gainBlue;
        pProp->Get(gainBlue);

        status = peak_Gain_Set(hCam, PEAK_GAIN_TYPE_DIGITAL, PEAK_GAIN_CHANNEL_BLUE, gainBlue);
        if (status != PEAK_STATUS_SUCCESS) { nRet = ERR_NO_WRITE_ACCESS; }
        else { gainBlue_ = gainBlue; }

    }
    return nRet;
}

/**
* Handles "PixelType" property.
*/
int CIDSPeak::OnPixelType(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    switch (eAct)
    {
    case MM::AfterSet:
    {
        if (IsCapturing())
            return DEVICE_CAMERA_BUSY_ACQUIRING;

        string pixelType;
        pProp->Get(pixelType);

        if (pixelType == g_PixelType_8bit) {
            nComponents_ = 1;
            bitDepth_ = 8;
        }
        else if (pixelType == g_PixelType_16bit) {
            nComponents_ = 1;
            bitDepth_ = 16;
        }
        else if (pixelType == g_PixelType_32bitRGBA) {
            nComponents_ = 4;
            bitDepth_ = 8;
        }
        else {
            return DEVICE_CAN_NOT_SET_PROPERTY;
        }
        pixelType_ = pixelType;

        // Resize buffer to accomodate the new image
        img_.Resize(img_.Width(), img_.Height(), nComponents_ * (bitDepth_ / 8));
    }
    break;
    case MM::BeforeGet:
    {
        if (nComponents_ == 1 && bitDepth_ == 8) {
            pProp->Set(g_PixelType_8bit);
        }
        else if (nComponents_ == 1 && bitDepth_ == 16) {
            pProp->Set(g_PixelType_16bit);
        }
        else if (nComponents_ == 4 && bitDepth_ == 8) {
            pProp->Set(g_PixelType_32bitRGBA);
        }
        else {
            return DEVICE_CAN_NOT_SET_PROPERTY;
        }
    } break;
    default:
        break;
    }
    return DEVICE_OK;
}

/**
* Handles "PixelType" property.
*/
int CIDSPeak::OnPeakPixelFormat(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    int nRet = DEVICE_OK;
    string pixelFormat;

    switch (eAct) {
    case MM::AfterSet:
        if (IsCapturing())
            return DEVICE_CAMERA_BUSY_ACQUIRING;

        pProp->Get(pixelFormat);
        if (peak_PixelFormat_GetAccessStatus(hCam) == PEAK_ACCESS_READWRITE) {
            currPixelFormat = stringToPixelFormat(pixelFormat);
            peak_PixelFormat_Set(hCam, currPixelFormat);
        }
        else { return ERR_NO_WRITE_ACCESS; }
        break;

    case MM::BeforeGet:
        pProp->Set(pixelFormatToString(currPixelFormat).c_str());
        break;
    }
    return nRet;
}

/**
* Handles "ReadoutTime" property.
* Not sure this does anything...
*/
int CIDSPeak::OnReadoutTime(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::AfterSet)
    {
        double readoutMs;
        pProp->Get(readoutMs);

        readoutUs_ = readoutMs * 1000.0;
    }
    else if (eAct == MM::BeforeGet)
    {
        pProp->Set(readoutUs_ / 1000.0);
    }

    return DEVICE_OK;
}

int CIDSPeak::OnSupportsMultiROI(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::AfterSet)
    {
        long tvalue = 0;
        pProp->Get(tvalue);
        supportsMultiROI_ = (tvalue != 0);
    }
    else if (eAct == MM::BeforeGet)
    {
        pProp->Set((long)supportsMultiROI_);
    }

    return DEVICE_OK;
}

int CIDSPeak::OnMultiROIFillValue(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::AfterSet)
    {
        long tvalue = 0;
        pProp->Get(tvalue);
        multiROIFillValue_ = (int)tvalue;
    }
    else if (eAct == MM::BeforeGet)
    {
        pProp->Set((long)multiROIFillValue_);
    }

    return DEVICE_OK;
}

int CIDSPeak::OnCameraCCDXSize(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        pProp->Set(cameraCCDXSize_);
    }
    else if (eAct == MM::AfterSet)
    {
        long value;
        pProp->Get(value);
        if ((value < 16) || (33000 < value))
            return DEVICE_ERR;  // invalid image size
        if (value != cameraCCDXSize_)
        {
            cameraCCDXSize_ = value;
            img_.Resize(cameraCCDXSize_ / binSize_, cameraCCDYSize_ / binSize_);
        }
    }
    return DEVICE_OK;

}

int CIDSPeak::OnCameraCCDYSize(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        pProp->Set(cameraCCDYSize_);
    }
    else if (eAct == MM::AfterSet)
    {
        long value;
        pProp->Get(value);
        if ((value < 16) || (33000 < value))
            return DEVICE_ERR;  // invalid image size
        if (value != cameraCCDYSize_)
        {
            cameraCCDYSize_ = value;
            img_.Resize(cameraCCDXSize_ / binSize_, cameraCCDYSize_ / binSize_);
        }
    }
    return DEVICE_OK;

}

int CIDSPeak::OnTriggerDevice(MM::PropertyBase* pProp, MM::ActionType eAct)
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


int CIDSPeak::OnCCDTemp(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    // This is a readonly function
    if (eAct == MM::BeforeGet)
    {
        status = getTemperature(&ccdT_);
        pProp->Set(ccdT_);
    }
    return DEVICE_OK;
}

int CIDSPeak::OnModelName(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    int nRet = DEVICE_OK;
    // This is a readonly function
    if (eAct == MM::BeforeGet)
    {
        pProp->Set(modelName_.c_str());
    }
    return nRet;
}

int CIDSPeak::OnSerialNumber(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    int nRet = DEVICE_OK;
    // This is a readonly function
    if (eAct == MM::BeforeGet)
    {
        pProp->Set(serialNum_.c_str());
    }
    return nRet;
}

int CIDSPeak::OnIsSequenceable(MM::PropertyBase* pProp, MM::ActionType eAct)
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

///////////////////////////////////////////////////////////////////////////////
// Private CIDSPeak methods
///////////////////////////////////////////////////////////////////////////////

/**
* Sync internal image buffer size to the chosen property values.
*/
int CIDSPeak::ResizeImageBuffer()
{
    char buf[MM::MaxStrLength];
    int nRet = GetProperty(MM::g_Keyword_Binning, buf);
    if (nRet != DEVICE_OK)
        return nRet;
    binSize_ = atol(buf);

    img_.Resize(cameraCCDXSize_ / binSize_, cameraCCDYSize_ / binSize_, nComponents_ * (bitDepth_ / 8));
    return DEVICE_OK;
}

void CIDSPeak::GenerateEmptyImage(ImgBuffer& img)
{
    MMThreadGuard g(imgPixelsLock_);
    if (img.Height() == 0 || img.Width() == 0 || img.Depth() == 0)
        return;
    unsigned char* pBuf = const_cast<unsigned char*>(img.GetPixels());
    memset(pBuf, 0, img.Height() * img.Width() * img.Depth());
}

peak_status CIDSPeak::getTemperature(double* sensorTemp)
{
    size_t enumerationEntryCount = 0;

    if (PEAK_IS_READABLE(
        peak_GFA_Feature_GetAccessStatus(hCam, PEAK_GFA_MODULE_REMOTE_DEVICE, "DeviceFirmwareVersion")))
    {
        // get the length of the feature string
        status = peak_GFA_Enumeration_GetList(hCam, PEAK_GFA_MODULE_REMOTE_DEVICE, "DeviceTemperatureSelector", NULL, &enumerationEntryCount);
        status = getGFAfloat("DeviceTemperature", sensorTemp);
    }
    else
    {
        printf("No read access to device temperature");
    }
    return status;
}

/**
* Cleanly exit camera, called during Shutdown()
**/
int CIDSPeak::cleanExit()
{
    if (hCam != PEAK_INVALID_HANDLE && hCam != NULL) // Check if camera is open
    {
        if (peak_Acquisition_IsStarted(hCam)) // Check if camera is acquiring data
        {
            status = peak_Acquisition_Stop(hCam); // Stop acquisition
        }
        status = peak_Camera_Close(hCam); // Close Camera
    }
    hCam = NULL;
    return DEVICE_OK;
}

//Returns PEAK_TRUE, if function was successful.
//Returns PEAK_FALSE, if function returned with an error. If continueExecution == PEAK_FALSE,
//the backend is exited.
peak_bool CIDSPeak::checkForSuccess(peak_status checkStatus, peak_bool continueExecution)
{
    if (PEAK_ERROR(checkStatus))
    {
        peak_status lastErrorCode = PEAK_STATUS_SUCCESS;
        size_t lastErrorMessageSize = 0;

        // Get size of error message
        status = peak_Library_GetLastError(&lastErrorCode, NULL, &lastErrorMessageSize);
        if (PEAK_ERROR(status))
        {
            // Something went wrong getting the last error!
            printf("Last-Error: Getting last error code failed! Status: %#06x\n", status);
            return PEAK_FALSE;
        }

        if (checkStatus != lastErrorCode)
        {
            // Another error occured in the meantime. Proceed with the last error.
            printf("Last-Error: Another error occured in the meantime!\n");
        }

        // Allocate and zero-initialize the char array for the error message
        char* lastErrorMessage = (char*)calloc((lastErrorMessageSize) / sizeof(char), sizeof(char));
        if (lastErrorMessage == NULL)
        {
            // Cannot allocate lastErrorMessage. Most likely not enough Memory.
            printf("Last-Error: Failed to allocate memory for the error message!\n");
            free(lastErrorMessage);
            return PEAK_FALSE;
        }

        // Get the error message
        status = peak_Library_GetLastError(&lastErrorCode, lastErrorMessage, &lastErrorMessageSize);
        if (PEAK_ERROR(status))
        {
            // Unable to get error message. This shouldn't ever happen.
            printf("Last-Error: Getting last error message failed! Status: %#06x; Last error code: %#06x\n", status,
                lastErrorCode);
            free(lastErrorMessage);
            return PEAK_FALSE;
        }

        printf("Last-Error: %s | Code: %#06x\n", lastErrorMessage, lastErrorCode);
        free(lastErrorMessage);

        if (!continueExecution)
        {
            cleanExit();
        }

        return PEAK_FALSE;
    }
    return PEAK_TRUE;
}

int CIDSPeak::getSensorInfo()
{
    // check if the feature is readable
    if (PEAK_IS_READABLE(
        peak_GFA_Feature_GetAccessStatus(hCam, PEAK_GFA_MODULE_REMOTE_DEVICE, "DeviceFirmwareVersion")))
    {
        int64_t temp_x;
        int64_t temp_y;
        status = getGFAInt("WidthMax", &temp_x);
        status = getGFAInt("HeightMax", &temp_y);
        cameraCCDXSize_ = (long)temp_x;
        cameraCCDYSize_ = (long)temp_y;
    }
    else
    {
        return ERR_NO_READ_ACCESS;
    }
    if (status == PEAK_STATUS_SUCCESS) { return DEVICE_OK; }
    else { return DEVICE_ERR; }
}

peak_status CIDSPeak::getGFAString(const char* featureName, char* stringValue)
{
    size_t stringLength = 0;

    // get the length of the feature string
    status = peak_GFA_String_Get(hCam, PEAK_GFA_MODULE_REMOTE_DEVICE, featureName, NULL, &stringLength);

    // if successful, read the firmware version
    if (checkForSuccess(status, PEAK_TRUE))
    {
        // read the string value of featureName
        status = peak_GFA_String_Get(
            hCam, PEAK_GFA_MODULE_REMOTE_DEVICE, featureName, stringValue, &stringLength);
    }
    return status;
}

peak_status CIDSPeak::getGFAInt(const char* featureName, int64_t* intValue)
{
    // read the integer value of featureName
    status = peak_GFA_Integer_Get(
        hCam, PEAK_GFA_MODULE_REMOTE_DEVICE, featureName, intValue);
    return status;
}

peak_status CIDSPeak::getGFAfloat(const char* featureName, double* floatValue)
{
    // read the float value of featureName
    status = peak_GFA_Float_Get(
        hCam, PEAK_GFA_MODULE_REMOTE_DEVICE, featureName, floatValue);
    return status;
}

void CIDSPeak::initializeAutoWBConversion()
{
    peakAutoToString.insert(pair<int, string>(PEAK_AUTO_FEATURE_MODE_OFF, "Off"));
    peakAutoToString.insert(pair<int, string>(PEAK_AUTO_FEATURE_MODE_ONCE, "Once"));
    peakAutoToString.insert(pair<int, string>(PEAK_AUTO_FEATURE_MODE_CONTINUOUS, "Continuous"));

    stringToPeakAuto.insert(pair<string, int>("Off", PEAK_AUTO_FEATURE_MODE_OFF));
    stringToPeakAuto.insert(pair<string, int>("Once", PEAK_AUTO_FEATURE_MODE_ONCE));
    stringToPeakAuto.insert(pair<string, int>("Continuous", PEAK_AUTO_FEATURE_MODE_CONTINUOUS));
}

int CIDSPeak::transferBuffer(peak_frame_handle hFrame, ImgBuffer& img) {
    peak_frame_handle hFrameConverted;
    peak_buffer peakBuffer;
    unsigned char* pBuf = (unsigned char*) const_cast<unsigned char*>(img.GetPixels());

    LogMessage(pixelType_);
    if (pixelType_ == g_PixelType_8bit) {
        if (currPixelFormat == PEAK_PIXEL_FORMAT_MONO8) {
            // Simply get framebuffer
            status = peak_Frame_Buffer_Get(hFrame, &peakBuffer);
            if (status != PEAK_STATUS_SUCCESS) { return ERR_ACQ_FRAME; }
        }
        else {
            // Get buffer and convert it to 8bit mono
            status = peak_IPL_PixelFormat_Set(hCam, PEAK_PIXEL_FORMAT_MONO8);
            if (status != PEAK_STATUS_SUCCESS) {
                LogMessage("IDS does not support the conversion of pixel format " +
                    pixelFormatToString(currPixelFormat) +
                    " to 8bit mono. Please select a different " +
                    g_keyword_Peak_PixelFormat);
                return ERR_ACQ_FRAME;
            }
            status = peak_IPL_ProcessFrame(hCam, hFrame, &hFrameConverted);
            if (status != PEAK_STATUS_SUCCESS) {
                LogMessage("Something went wrong while converting the image to 8bit mono. It is best to select the " +
                    (string)g_keyword_Peak_PixelFormat +
                    pixelFormatToString(PEAK_PIXEL_FORMAT_MONO8) +
                    " to bypass the image conversion.");
            }
            status = peak_Frame_Buffer_Get(hFrameConverted, &peakBuffer);
        }
        // Copy 8bit mono buffer to imageBuffer
        memcpy(pBuf, peakBuffer.memoryAddress, peakBuffer.memorySize);
    }
    else if (pixelType_ == g_PixelType_16bit) {
        // MONO10 and MONO12 are already 16 bit, so they don't need to be converted
        if (currPixelFormat == PEAK_PIXEL_FORMAT_MONO10 ||
            currPixelFormat == PEAK_PIXEL_FORMAT_MONO12) {
            // Simply get framebuffer
            status = peak_Frame_Buffer_Get(hFrame, &peakBuffer);
            if (status != PEAK_STATUS_SUCCESS) { return ERR_ACQ_FRAME; }
        }
        else {
            // Get buffer and convert it to 8bit mono
            status = peak_IPL_PixelFormat_Set(hCam, PEAK_PIXEL_FORMAT_MONO12);
            if (status != PEAK_STATUS_SUCCESS) {
                LogMessage("IDS does not support the conversion of pixel format " +
                    pixelFormatToString(currPixelFormat) +
                    " to 16bit mono. Please select a different " +
                    g_keyword_Peak_PixelFormat);
                return ERR_ACQ_FRAME;
            }
            status = peak_IPL_ProcessFrame(hCam, hFrame, &hFrameConverted);
            if (status != PEAK_STATUS_SUCCESS) {
                LogMessage("Something went wrong while converting the image to 16bit mono. It is best to select the " +
                    (string)g_keyword_Peak_PixelFormat +
                    pixelFormatToString(PEAK_PIXEL_FORMAT_MONO12) +
                    " to bypass the image conversion.");
            }
            status = peak_Frame_Buffer_Get(hFrameConverted, &peakBuffer);
        }
        // Copy 16bit mono buffer to imageBufer
        memcpy(pBuf, peakBuffer.memoryAddress, peakBuffer.memorySize);
    }
    else if (pixelType_ == g_PixelType_32bitRGBA) {
        // MONO10 and MONO12 are already 16 bit, so they don't need to be converted
        if (currPixelFormat == PEAK_PIXEL_FORMAT_BGRA8) {
            // Simply get framebuffer
            status = peak_Frame_Buffer_Get(hFrame, &peakBuffer);
            if (status != PEAK_STATUS_SUCCESS) { return ERR_ACQ_FRAME; }
        }
        else {
            // Get buffer and convert it to 8bit mono
            status = peak_IPL_PixelFormat_Set(hCam, PEAK_PIXEL_FORMAT_BGRA8);
            if (status != PEAK_STATUS_SUCCESS) {
                LogMessage("IDS does not support the conversion of pixel format " +
                    pixelFormatToString(currPixelFormat) +
                    " to 32bitRGBA mono. Please select a different " +
                    g_keyword_Peak_PixelFormat);
                return ERR_ACQ_FRAME;
            }
            status = peak_IPL_ProcessFrame(hCam, hFrame, &hFrameConverted);
            if (status != PEAK_STATUS_SUCCESS) {
                LogMessage("Something went wrong while converting the image to 32bitRGBA. It is best to select the " +
                    (string)g_keyword_Peak_PixelFormat +
                    pixelFormatToString(PEAK_PIXEL_FORMAT_BGRA8) +
                    " to bypass the image conversion.");
            }
            status = peak_Frame_Buffer_Get(hFrameConverted, &peakBuffer);
        }
        // Copy 16bit mono buffer to imageBufer
        memcpy(pBuf, peakBuffer.memoryAddress, peakBuffer.memorySize);
    }

    // Exit if something went wrong during the conversion/obtaining the buffer.
    if (status != PEAK_STATUS_SUCCESS) { return DEVICE_UNSUPPORTED_DATA_FORMAT; }

    return DEVICE_OK;
}

int CIDSPeak::updateAutoWhiteBalance()
{
    if (peak_AutoWhiteBalance_GetAccessStatus(hCam) == PEAK_ACCESS_READONLY
        || peak_AutoWhiteBalance_GetAccessStatus(hCam) == PEAK_ACCESS_READWRITE)
    {
        // Update the gain channels
        status = peak_Gain_Get(hCam, PEAK_GAIN_TYPE_DIGITAL, PEAK_GAIN_CHANNEL_MASTER, &gainMaster_);
        status = peak_Gain_Get(hCam, PEAK_GAIN_TYPE_DIGITAL, PEAK_GAIN_CHANNEL_RED, &gainRed_);
        status = peak_Gain_Get(hCam, PEAK_GAIN_TYPE_DIGITAL, PEAK_GAIN_CHANNEL_GREEN, &gainGreen_);
        status = peak_Gain_Get(hCam, PEAK_GAIN_TYPE_DIGITAL, PEAK_GAIN_CHANNEL_BLUE, &gainBlue_);
        // Update the auto white balance mode
        status = peak_AutoWhiteBalance_Mode_Get(hCam, &peakAutoWhiteBalance_);
    }
    else { return ERR_NO_READ_ACCESS; }

    if (status == PEAK_STATUS_SUCCESS) { return DEVICE_OK; }
    else { return DEVICE_ERR; }
}

int CIDSPeak::framerateSet(double framerate)
{
    // Check if interval doesn't exceed framerate limitations of camera
    // Else set interval to match max framerate
    if (framerate > framerateMax_)
    {
        framerate = framerateMax_;
    }
    else if (framerate < framerateMin_)
    {
        framerate = framerateMin_;
    }

    if (peak_FrameRate_GetAccessStatus(hCam) == PEAK_ACCESS_READWRITE)
    {
        status = peak_FrameRate_Set(hCam, framerate);
        framerateCur_ = framerate;
    }
    else
    {
        return ERR_NO_WRITE_ACCESS;
    }
    return DEVICE_OK;
}

// Actual initialization of the camera (is called every time camera is swapped).
int CIDSPeak::cameraChanged()
{

}

// Checks if camera supportes color image formats
bool CIDSPeak::isColorCamera()
{
    size_t pixelFormatCount = 0;
    status = peak_PixelFormat_GetList(hCam, NULL, &pixelFormatCount);
    peak_pixel_format* pixelFormatList = (peak_pixel_format*)calloc(
        pixelFormatCount, sizeof(peak_pixel_format));
    status = peak_PixelFormat_GetList(hCam, pixelFormatList, &pixelFormatCount);
    for (int i = 0; i < pixelFormatCount; i++)
    {
        if (pixelFormatList[i] == PEAK_PIXEL_FORMAT_BAYER_RG8) { return true; }
    }
    return false;
}
