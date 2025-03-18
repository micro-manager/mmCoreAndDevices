///////////////////////////////////////////////////////////////////////////////
// FILE:          POACamera.cpp
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

#include "POACamera.h"
#include "ConvFuncs.h"


#include <cstdio>
#include <string>
#include <math.h>
#include "ModuleInterface.h"
#include <sstream>
#include <algorithm>
//#include "WriteCompactTiffRGB.h"
#include <iostream>
#include <future>

double g_IntensityFactor_ = 1.0;

// External names used used by the rest of the system
// to load particular device from the "DemoCamera.dll" library
const char* g_CameraDeviceName = "POA_Camera";
const char* g_WheelDeviceName = "POA_FilterWheel";
//const char* g_AutoFocusDeviceName = "DAutoFocus"; // possible use in future

// constants for naming pixel types (allowed values of the "PixelType" property)
const char* g_PixelType_8bit = "RAW8";
const char* g_PixelType_16bit = "RAW16";
const char* g_PixelType_32bitRGB = "RGB32";
const char* g_PixelType_MONO8bit = "MONO8";
//const char* g_PixelType_64bitRGB = "64bitRGB";
//const char* g_PixelType_32bit = "32bit";  // floating point greyscale

const char* g_HubDeviceName = "DHub";
const char* g_SelectCamera = "Select Camera";
const char* g_SelectFilterWheel = "Select Filter Wheel";

//camera properties
const char* g_PropName_USBHost = "USB Host";
const char* g_PropName_USB_BW_Limit = "USB Bandwidth Limit";
const char* g_PropName_FrameRate_Limit = "Frame Rate Limit";
const char* g_PropName_Flip = "Image Flip";
const char* g_PropName_PixelBinSum = "Pixel Bin Sum";
const char* g_PropName_HardwareBin = "Hardware Bin";

const char* g_PropName_GainPreset = "Gain Preset";
const char* g_PropName_OffsetPreset = "Offset Preset";

const char* g_PropName_WB_R = "WB_Red";
const char* g_PropName_WB_G = "WB_Green";
const char* g_PropName_WB_B = "WB_Blue";
const char* g_PropName_AutoWB = "WB Auto";
const char* g_PropName_MonoBin = "Mono Bin";

const char* g_PropName_Gamma = "Gamma";
const char* g_PropName_AutoExp = "Exp Auto";
const char* g_PropName_AutoGain = "Gain Auto";
const char* g_PropName_AutoExpBrightness = "Auto Exp Brightness";


const char* g_PropName_TargetTemp = "Target Temperature";
const char* g_PropName_IsCoolerOn = "Cooler On";
const char* g_PropName_CoolPower = "Cooler Power(%)";
const char* g_PropName_FanPower = "Fan Power(%)";
const char* g_PropName_IsHeaterOn = "Anti-dew(Heater) On";
const char* g_PropName_HeaterPower = "Anti-dew(Heater) Power(%)";

const char* g_PropName_ON = "ON";
const char* g_PropName_OFF = "OFF";

// flip values
const std::string g_FlipHori("Hori");
const std::string g_FlipVert("Vert");
const std::string g_FlipBoth("Both");
const std::string g_FlipNone("None");

//gain preset
const std::string g_gainHighestDR("HighestDR Gain");
const std::string g_HCGain("HCGain");
const std::string g_unityGain("Unity Gain");
const std::string g_gainLowestRN("LowestRN Gain");

//offset preset
const std::string g_offsetHighestDR("HighestDR Offset");
const std::string g_offsetHCGain("HCGain Offset");
const std::string g_offsetUnityGain("UnityGain Offset");
const std::string g_offsetLowestRN("LowestRN Offset");

//gamma values
const double g_gamma_min = 0.01;
const double g_gamma_max = 7.99;
const double g_gamma_def = 1.0;

enum { MODE_ARTIFICIAL_WAVES, MODE_NOISE, MODE_COLOR_TEST };

//#define DEBUG_METHOD_NAMES

#ifdef DEBUG_METHOD_NAMES
#define LOG(name)              this->LogMessage(name);
#define LOG_ONPROPERTY(name,action)   this->LogMessage(string(name)+(action==MM::AfterSet?"(AfterSet)":"(BeforeGet)"));
#else
#define LOG(name)
#define LOG_ONPROPERTY(name,action)
#endif


///////////////////////////////////////////////////////////////////////////////
// Exported MMDevice API
///////////////////////////////////////////////////////////////////////////////

MODULE_API void InitializeModuleData()
{
    RegisterDevice(g_CameraDeviceName, MM::CameraDevice, "POA camera");
    RegisterDevice(g_WheelDeviceName, MM::StateDevice, "POA filter wheel");

    /*RegisterDevice("TransposeProcessor", MM::ImageProcessorDevice, "TransposeProcessor");
    RegisterDevice("ImageFlipX", MM::ImageProcessorDevice, "ImageFlipX");
    RegisterDevice("ImageFlipY", MM::ImageProcessorDevice, "ImageFlipY");
    RegisterDevice("MedianFilter", MM::ImageProcessorDevice, "MedianFilter");*/
}

MODULE_API MM::Device* CreateDevice(const char* deviceName)
{
    if (deviceName == 0)
        return 0;

    // decide which device class to create based on the deviceName parameter
    if (strcmp(deviceName, g_CameraDeviceName) == 0)
    {
        // create camera
        return new POACamera();
    }
    else if (strcmp(deviceName, g_WheelDeviceName) == 0)
    {
        // create filter wheel
        return new POAFilterWheel();
    }
    // ...supplied name not recognized
    return 0;
}

MODULE_API void DeleteDevice(MM::Device* pDevice)
{
    delete pDevice;
}

///////////////////////////////////////////////////////////////////////////////
// POACamera implementation
// ~~~~~~~~~~~~~~~~~~~~~~~~~~

/**
* POACamera constructor.
* Setup default all variables and create device properties required to exist
* before intialization. In this case, no such properties were required. All
* properties will be created in the Initialize() method.
*
* As a general guideline Micro-Manager devices do not access hardware in the
* the constructor. We should do as little as possible in the constructor and
* perform most of the initialization in the Initialize() method.
*/
POACamera::POACamera() :
    CLegacyCameraBase<POACamera>(),
    exposureMaximum_(2000000.0),
    initialized_(false),
    roiX_(0),
    roiY_(0),
    cameraCCDXSize_(512),
    cameraCCDYSize_(512),
    imgFmt_(POA_RAW8),
    bitDepth_(8),
    binSize_(1),
    ccdT_(0.0),
    gammaValue_(g_gamma_def),
    p8bitGammaTable(nullptr),
    p16bitGammaTable(nullptr),
    nominalPixelSizeUm_(1.0),
    pRGB24(nullptr),
    RGB24BufSize_(0),
    readoutUs_(0.0),
    sequenceStartTime_(0),
    isSequenceable_(false),
    sequenceMaxLength_(100),
    sequenceRunning_(false),
    sequenceIndex_(0),
    stopOnOverflow_(false),
    supportsMultiROI_(false),
    nComponents_(1),
    gainHighestDR_(0),
    HCGain_(0),
    unityGain_(0),
    gainLowestRN_(0),
    offsetHighestDR_(0),
    offsetHCGain_(0),
    offsetUnityGain_(0),
    offsetLowestRN_(0)
{

    // call the base class method to set-up default error codes/messages
    InitializeDefaultErrorMessages();

    // parent ID display
    CreateHubIDProperty();

    camProp_.cameraID = -1;
    //scan all POA cameras
    POACameraProperties camProp;
    connectCamerasName_.clear();
    int connnectCamNums = POAGetCameraCount();
    for (int i = 0; i < connnectCamNums; ++i)
    {
        camProp.cameraID = -1;
        POAGetCameraProperties(i, &camProp); //get camera properties
        if(camProp.cameraID == -1)
        {
            continue;
        }

        connectCamerasName_.push_back(std::string(camProp.cameraModelName));
    }
    
    CPropertyAction* pAct = new CPropertyAction(this, &POACamera::OnSelectCamIndex);
    if (connnectCamNums > 0)
    {
        selectCamIndex_ = 0;
        selectCamName_ = connectCamerasName_.at(selectCamIndex_); //default camera is first one
        POAGetCameraProperties(selectCamIndex_, &camProp_);
    }
    else
    {
        selectCamIndex_ = -1;
        selectCamName_ = "No POA camera found";
    }

    CreateStringProperty(g_SelectCamera, selectCamName_.c_str(), false, pAct, true);
    SetAllowedValues(g_SelectCamera, connectCamerasName_);

    CreateFloatProperty("MaximumExposureMs", exposureMaximum_, false,
        new CPropertyAction(this, &POACamera::OnMaxExposure),
        true);

    readoutStartTime_ = GetCurrentMMTime();
    thd_ = new MySequenceThread(this);
}

/**
* POACamera destructor.
* If this device used as intended within the Micro-Manager system,
* Shutdown() will be always called before the destructor. But in any case
* we need to make sure that all resources are properly released even if
* Shutdown() was not called.
*/
POACamera::~POACamera()
{
    if (camProp_.cameraID >= 0)
    {
        POACameraState camState;
        POAGetCameraState(camProp_.cameraID, &camState);

        if (camState != STATE_CLOSED)
        {
            Shutdown();
        }
    }

    if (pRGB24)
    {
        delete[] pRGB24;
        pRGB24 = nullptr;
    }

    if (p8bitGammaTable)
    {
        delete[] p8bitGammaTable;
        p8bitGammaTable = nullptr;
    }

    if (p16bitGammaTable)
    {
        delete[] p16bitGammaTable;
        p16bitGammaTable = nullptr;
    }

    delete thd_;
}

/**
* Obtains device name.
* Required by the MM::Device API.
*/
void POACamera::GetName(char* name) const
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
int POACamera::Initialize()
{
    char cameraIDStr[128];

    LOG("POACamera::Initialize");

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
    std::vector<std::string> boolStrVec;
    boolStrVec.push_back(g_PropName_ON);
    boolStrVec.push_back(g_PropName_OFF);

    // Name
    int nRet = CreateStringProperty(MM::g_Keyword_Name, g_CameraDeviceName, true);
    if (DEVICE_OK != nRet)
        return nRet;

    // Description
    nRet = CreateStringProperty(MM::g_Keyword_Description, "Player One Astronomy Camera Adapter", true);
    if (DEVICE_OK != nRet)
        return nRet;

    //operate camera
    if (camProp_.cameraID < 0)
    {
        return DEVICE_NOT_CONNECTED;
    }

    if (POAOpenCamera(camProp_.cameraID) != POA_OK) //open camera
    {
        return DEVICE_NOT_CONNECTED;
    }

    if (POAInitCamera(camProp_.cameraID) != POA_OK) // init camera
    {
        return DEVICE_NOT_CONNECTED;
    }
    
    //init gamma table
    p8bitGammaTable = new unsigned char[256];
    p16bitGammaTable = new unsigned short[65536];
    ResetGammaTable();

    nominalPixelSizeUm_ = camProp_.pixelSize;

    char* pCameraName = camProp_.cameraModelName;

    // CameraName
    nRet = CreateStringProperty(MM::g_Keyword_CameraName, pCameraName, true);
    assert(nRet == DEVICE_OK);

    // CameraID(SN)
    sprintf(cameraIDStr, "Serial number %s", camProp_.SN);
    nRet = CreateStringProperty(MM::g_Keyword_CameraID, cameraIDStr, true);
    assert(nRet == DEVICE_OK);

    //SDK version
    const char *pVer = POAGetSDKVersion();
    nRet = CreateStringProperty("Cam SDK Ver", pVer, true);
    assert(nRet == DEVICE_OK);

    //get camera current image parameters
    binSize_ = 1;
    POAGetImageBin(camProp_.cameraID, &binSize_);
    roiX_ = 0, roiY_ = 0;
    POAGetImageStartPos(camProp_.cameraID, &roiX_, &roiY_);
    cameraCCDXSize_ = camProp_.maxWidth;
    cameraCCDYSize_ = camProp_.maxHeight;
    POAGetImageSize(camProp_.cameraID, &cameraCCDXSize_, &cameraCCDYSize_);
    imgFmt_ = POA_RAW8;
    POAGetImageFormat(camProp_.cameraID, &imgFmt_);

    //get gain and offset preset values
    POAGetGainsAndOffsets(camProp_.cameraID, &gainHighestDR_, &HCGain_, &unityGain_, &gainLowestRN_,
        &offsetHighestDR_, &offsetHCGain_, &offsetUnityGain_, &offsetLowestRN_);

    // get current camera temperature
    POABool isAuto;
    POAGetConfig(camProp_.cameraID, POA_TEMPERATURE, &ccdT_, &isAuto);

    // Host (PC) USB connenction Type
    std::string strUSBType = (camProp_.isUSB3Speed == POA_TRUE) ? "USB3.0" : "USB2.0";
    nRet = CreateStringProperty(g_PropName_USBHost, strUSBType.c_str(), true);
    assert(nRet == DEVICE_OK);

    // binning
    CPropertyAction* pAct = new CPropertyAction(this, &POACamera::OnBinning);
    nRet = CreateIntegerProperty(MM::g_Keyword_Binning, binSize_, false, pAct);
    assert(nRet == DEVICE_OK);

    std::vector<std::string> binningStrVec;
    char binStr[2];
    for (int i = 0; i < 8; i++) //int bins[8]; in camera property
    {
        if (camProp_.bins[i] == 0)
        {
            break;
        }
        memset(binStr, 0, 2);
        sprintf(binStr, "%d", camProp_.bins[i]);
        binningStrVec.push_back(binStr);
    }

    nRet = SetAllowedValues(MM::g_Keyword_Binning, binningStrVec);
    if (nRet != DEVICE_OK)
        return nRet;

    // pixel type
    pAct = new CPropertyAction(this, &POACamera::OnPixelType);
    nRet = CreateStringProperty(MM::g_Keyword_PixelType, ImgFmtToString(imgFmt_), false, pAct);
    assert(nRet == DEVICE_OK);

    std::vector<std::string> pixelTypeValues;
    for (int i = 0; i < 8; i++) //POAImgFormat imgFormats[8]; in camera property
    {
        if (camProp_.imgFormats[i] == POA_RAW8)
        {
            pixelTypeValues.push_back(g_PixelType_8bit);
        }
        else if (camProp_.imgFormats[i] == POA_RAW16)
        {
            pixelTypeValues.push_back(g_PixelType_16bit);
        }
        else if (camProp_.imgFormats[i] == POA_RGB24)
        {
            pixelTypeValues.push_back(g_PixelType_32bitRGB);
        }
        else if (camProp_.imgFormats[i] == POA_MONO8)
        {
            pixelTypeValues.push_back(g_PixelType_MONO8bit);
        }
        else
        {
            break;
        }
    }
     
    nRet = SetAllowedValues(MM::g_Keyword_PixelType, pixelTypeValues);
    if (nRet != DEVICE_OK)
        return nRet;

    long minVal = 0, maxVal = 0, defVal = 0;
 
    // exposure
    double defExp;
    if (GetConfigRange(camProp_.cameraID, POA_EXPOSURE, &maxVal, &minVal, &defVal) != POA_OK)
    {
        exposureMaximum_ = 2000000.0;
        defExp = 10.0;
    }
    else
    {
        exposureMaximum_ = maxVal / 1000.0;
        defExp = defVal / 1000.0;
    }
    pAct = new CPropertyAction(this, &POACamera::OnExposure);
    nRet = CreateFloatProperty(MM::g_Keyword_Exposure, defExp, false, pAct);
    assert(nRet == DEVICE_OK);
    SetPropertyLimits(MM::g_Keyword_Exposure, 0.0, exposureMaximum_);

    //Exp auto
    pAct = new CPropertyAction(this, &POACamera::OnExpAuto);
    nRet = CreateStringProperty(g_PropName_AutoExp, g_PropName_OFF, false, pAct);
    assert(nRet == DEVICE_OK);
    SetAllowedValues(g_PropName_AutoExp, boolStrVec);

    // camera gain
    minVal = 0;
    maxVal = 500;
    defVal = 0;
    GetConfigRange(camProp_.cameraID, POA_GAIN, &maxVal, &minVal, &defVal);
    pAct = new CPropertyAction(this, &POACamera::OnGain);
    nRet = CreateIntegerProperty(MM::g_Keyword_Gain, defVal, false, pAct);
    assert(nRet == DEVICE_OK);
    SetPropertyLimits(MM::g_Keyword_Gain, minVal, maxVal);

    //auto gain
    pAct = new CPropertyAction(this, &POACamera::OnGainAuto);
    nRet = CreateStringProperty(g_PropName_AutoGain, g_PropName_OFF, false, pAct);
    assert(nRet == DEVICE_OK);
    SetAllowedValues(g_PropName_AutoGain, boolStrVec);

    //gain preset
    std::vector<std::string> gainPresetStrVec;
    gainPresetStrVec.push_back(g_gainHighestDR);
    gainPresetStrVec.push_back(g_HCGain);
    gainPresetStrVec.push_back(g_unityGain);
    gainPresetStrVec.push_back(g_gainLowestRN);
    //gainPresetStrVec.push_back(g_manually);
    pAct = new CPropertyAction(this, &POACamera::OnGainPreset);
    nRet = CreateStringProperty(g_PropName_GainPreset, "", false, pAct);
    assert(nRet == DEVICE_OK);
    SetAllowedValues(g_PropName_GainPreset, gainPresetStrVec);

    //offset preset
    std::vector<std::string> offsetPresetStrVec;
    offsetPresetStrVec.push_back(g_offsetHighestDR);
    offsetPresetStrVec.push_back(g_offsetHCGain);
    offsetPresetStrVec.push_back(g_offsetUnityGain);
    offsetPresetStrVec.push_back(g_offsetLowestRN);
    //offsetPresetStrVec.push_back(g_manually);
    pAct = new CPropertyAction(this, &POACamera::OnOffsetPreset);
    nRet = CreateStringProperty(g_PropName_OffsetPreset, "", false, pAct);
    assert(nRet == DEVICE_OK);
    SetAllowedValues(g_PropName_OffsetPreset, offsetPresetStrVec);

    //auto Exp brightness
    minVal = 50;
    maxVal = 200;
    defVal = 100;
    GetConfigRange(camProp_.cameraID, POA_AUTOEXPO_BRIGHTNESS, &maxVal, &minVal, &defVal);
    pAct = new CPropertyAction(this, &POACamera::OnAutoExpBrightness);
    nRet = CreateIntegerProperty(g_PropName_AutoExpBrightness, defVal, false, pAct);
    assert(nRet == DEVICE_OK);
    SetPropertyLimits(g_PropName_AutoExpBrightness, minVal, maxVal);

    // camera offset
    minVal = 0;
    maxVal = 200;
    defVal = 0;
    GetConfigRange(camProp_.cameraID, POA_OFFSET, &maxVal, &minVal, &defVal);
    pAct = new CPropertyAction(this, &POACamera::OnOffset);
    nRet = CreateIntegerProperty(MM::g_Keyword_Offset, defVal, false, pAct);
    assert(nRet == DEVICE_OK);
    SetPropertyLimits(MM::g_Keyword_Offset, minVal, maxVal);

    //gamma
    pAct = new CPropertyAction(this, &POACamera::OnGamma);
    nRet = CreateFloatProperty(g_PropName_Gamma, g_gamma_def, false, pAct);
    assert(nRet == DEVICE_OK);
    SetPropertyLimits(g_PropName_Gamma, g_gamma_min, g_gamma_max);

    // camera temperature
    pAct = new CPropertyAction(this, &POACamera::OnCCDTemp);
    nRet = CreateFloatProperty(MM::g_Keyword_CCDTemperature, 0, true, pAct);//read only
    assert(nRet == DEVICE_OK);
    SetPropertyLimits(MM::g_Keyword_CCDTemperature, -100, 100);

    //camera USB bandwidth limit
    minVal = 35;
    maxVal = 100;
    defVal = 100;
    GetConfigRange(camProp_.cameraID, POA_USB_BANDWIDTH_LIMIT, &maxVal, &minVal, &defVal);
    pAct = new CPropertyAction(this, &POACamera::OnUSBBandwidthLimit);
    nRet = CreateIntegerProperty(g_PropName_USB_BW_Limit, defVal, false, pAct);
    assert(nRet == DEVICE_OK);
    SetPropertyLimits(g_PropName_USB_BW_Limit, minVal, maxVal);

    //camera frame rate limit
    minVal = 0;
    maxVal = 2000;
    defVal = 0;
    GetConfigRange(camProp_.cameraID, POA_FRAME_LIMIT, &maxVal, &minVal, &defVal);
    pAct = new CPropertyAction(this, &POACamera::OnFrameRateLimit);
    nRet = CreateIntegerProperty(g_PropName_FrameRate_Limit, defVal, false, pAct);
    assert(nRet == DEVICE_OK);
    SetPropertyLimits(g_PropName_FrameRate_Limit, minVal, maxVal);

    // camera image flip
    std::vector<std::string> imgFlipStrVec;
    imgFlipStrVec.push_back(g_FlipHori);
    imgFlipStrVec.push_back(g_FlipVert);
    imgFlipStrVec.push_back(g_FlipBoth);
    imgFlipStrVec.push_back(g_FlipNone);
    pAct = new CPropertyAction(this, &POACamera::OnFlip);
    nRet = CreateStringProperty(g_PropName_Flip, g_FlipNone.c_str(), false, pAct);
    assert(nRet == DEVICE_OK);
    SetAllowedValues(g_PropName_Flip, imgFlipStrVec);

    //camera hardware bin
    if (camProp_.isSupportHardBin == POA_TRUE)
    {
        pAct = new CPropertyAction(this, &POACamera::OnHardBin);
        nRet = CreateStringProperty(g_PropName_HardwareBin, g_PropName_OFF, false, pAct);
        assert(nRet == DEVICE_OK);
        SetAllowedValues(g_PropName_HardwareBin, boolStrVec);
    }

    //Pixel Bin Sum
    pAct = new CPropertyAction(this, &POACamera::OnPixelBinSum);
    nRet = CreateStringProperty(g_PropName_PixelBinSum, g_PropName_OFF, false, pAct);
    assert(nRet == DEVICE_OK);
    SetAllowedValues(g_PropName_PixelBinSum, boolStrVec);

    //camera white balance
    if (camProp_.isColorCamera == POA_TRUE)
    {
        minVal = -1200;
        maxVal = 1200;
        defVal = 0;
        
        //Red
        GetConfigRange(camProp_.cameraID, POA_WB_R, &maxVal, &minVal, &defVal);
        pAct = new CPropertyAction(this, &POACamera::OnWB_Red);
        nRet = CreateIntegerProperty(g_PropName_WB_R, defVal, false, pAct);
        assert(nRet == DEVICE_OK);
        SetPropertyLimits(g_PropName_WB_R, minVal, maxVal);

        //Green
        GetConfigRange(camProp_.cameraID, POA_WB_G, &maxVal, &minVal, &defVal);
        pAct = new CPropertyAction(this, &POACamera::OnWB_Green);
        nRet = CreateIntegerProperty(g_PropName_WB_G, defVal, false, pAct);
        assert(nRet == DEVICE_OK);
        SetPropertyLimits(g_PropName_WB_G, minVal, maxVal);

        //Blue
        GetConfigRange(camProp_.cameraID, POA_WB_B, &maxVal, &minVal, &defVal);
        pAct = new CPropertyAction(this, &POACamera::OnWB_Blue);
        nRet = CreateIntegerProperty(g_PropName_WB_B, defVal, false, pAct);
        assert(nRet == DEVICE_OK);
        SetPropertyLimits(g_PropName_WB_B, minVal, maxVal);

        //Auto
        pAct = new CPropertyAction(this, &POACamera::OnAutoWB);
        nRet = CreateStringProperty(g_PropName_AutoWB, g_PropName_OFF, false, pAct);
        assert(nRet == DEVICE_OK);
        SetAllowedValues(g_PropName_AutoWB, boolStrVec);

        // MONO bin
        pAct = new CPropertyAction(this, &POACamera::OnMonoBin);
        nRet = CreateStringProperty(g_PropName_MonoBin, g_PropName_OFF, false, pAct);
        assert(nRet == DEVICE_OK);
        SetAllowedValues(g_PropName_MonoBin, boolStrVec);
    }

    // cooled camera settings
    if (camProp_.isHasCooler == POA_TRUE)
    {
        //Target Temperature
        minVal = -50;
        maxVal = 50;
        defVal = 0;
        GetConfigRange(camProp_.cameraID, POA_TARGET_TEMP, &maxVal, &minVal, &defVal);
        pAct = new CPropertyAction(this, &POACamera::OnTargetTEMP);
        nRet = CreateIntegerProperty(g_PropName_TargetTemp, defVal, false, pAct);
        assert(nRet == DEVICE_OK);
        SetPropertyLimits(g_PropName_TargetTemp, minVal, maxVal);
       
        // Cooler On
        pAct = new CPropertyAction(this, &POACamera::OnCoolerOn);
        nRet = CreateStringProperty(g_PropName_IsCoolerOn, g_PropName_OFF, false, pAct);
        assert(nRet == DEVICE_OK);
        SetAllowedValues(g_PropName_IsCoolerOn, boolStrVec);

        //Cooler Power
        minVal = 0;
        maxVal = 100;
        defVal = 0;
        GetConfigRange(camProp_.cameraID, POA_COOLER_POWER, &maxVal, &minVal, &defVal);
        pAct = new CPropertyAction(this, &POACamera::OnCoolerPower);
        nRet = CreateIntegerProperty(g_PropName_CoolPower, defVal, true, pAct); //read only
        assert(nRet == DEVICE_OK);
        SetPropertyLimits(g_PropName_CoolPower, minVal, maxVal);

        //Fan Power
        minVal = 0;
        maxVal = 100;
        defVal = 70;
        GetConfigRange(camProp_.cameraID, POA_FAN_POWER, &maxVal, &minVal, &defVal);
        pAct = new CPropertyAction(this, &POACamera::OnFanPower);
        nRet = CreateIntegerProperty(g_PropName_FanPower, defVal, false, pAct);
        assert(nRet == DEVICE_OK);
        SetPropertyLimits(g_PropName_FanPower, minVal, maxVal);

        // Anti-dew(Heater) //deprecated, set Cooler On / Off , then Heater On / Off
        /*pAct = new CPropertyAction(this, &POACamera::OnHeaterOn);
        nRet = CreateStringProperty(g_PropName_IsHeaterOn, g_PropName_OFF, false, pAct);
        assert(nRet == DEVICE_OK);
        SetAllowedValues(g_PropName_IsHeaterOn, boolStrVec);*/

        // Heater Power
        minVal = 0;
        maxVal = 100;
        defVal = 10;
        GetConfigRange(camProp_.cameraID, POA_HEATER_POWER, &maxVal, &minVal, &defVal);
        pAct = new CPropertyAction(this, &POACamera::OnHeaterPower);
        nRet = CreateIntegerProperty(g_PropName_HeaterPower, defVal, false, pAct);
        assert(nRet == DEVICE_OK);
        SetPropertyLimits(g_PropName_HeaterPower, minVal, maxVal);
    }
    
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

    LOG("Init POACamera OK!");
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
int POACamera::Shutdown()
{
    LOG("Shut down camera!");
    initialized_ = false;
    StopSequenceAcquisition();

    POACameraState camState = STATE_CLOSED;
    POAGetCameraState(camProp_.cameraID, &camState);
    if (camState == STATE_EXPOSING)
    {
        POAStopExposure(camProp_.cameraID);
    }
    POACloseCamera(camProp_.cameraID);

    if (pRGB24)
    {
        delete[] pRGB24;
        pRGB24 = nullptr;
    }

    if (p8bitGammaTable)
    {
        delete[] p8bitGammaTable;
        p8bitGammaTable = nullptr;
    }

    if (p16bitGammaTable)
    {
        delete[] p16bitGammaTable;
        p16bitGammaTable = nullptr;
    }

    return DEVICE_OK;
}

/**
* Performs exposure and grabs a single image.
* This function should block during the actual exposure and return immediately afterwards
* (i.e., before readout).  This behavior is needed for proper synchronization with the shutter.
* Required by the MM::Camera API.
*/
int POACamera::SnapImage()
{
    LOG("Snap Image");

    if (camProp_.cameraID < 0)
        return DEVICE_NOT_CONNECTED;

    double exp = GetExposure();

    POACameraState camState = STATE_CLOSED;
    POAGetCameraState(camProp_.cameraID, &camState);
    if (camState == STATE_EXPOSING)
    {
        return DEVICE_OK;
    }

    if (POAStartExposure(camProp_.cameraID, POA_TRUE) != POA_OK)
    {
        return DEVICE_SNAP_IMAGE_FAILED;
    }

    m_bIsToStopExposure = false;
    
    do
    {
        if (m_bIsToStopExposure)
        {
            break;
        }
     
        POAGetCameraState(camProp_.cameraID, &camState);
    } while (camState == STATE_EXPOSING);

    POABool pIsReady = POA_FALSE;

    POAImageReady(camProp_.cameraID, &pIsReady);

    if (m_bIsToStopExposure)
    {
        return DEVICE_OK;
    }

    if (pIsReady == POA_TRUE)
    {
        MMThreadGuard g(imgPixelsLock_);
        if (img_.Height() == 0 || img_.Width() == 0 || img_.Depth() == 0)
            return DEVICE_OUT_OF_MEMORY;

        unsigned char* pImgBuffer = img_.GetPixelsRW();
        long lBufSize = GetImageBufferSize();
        POAErrors error = POAGetImageData(camProp_.cameraID, pImgBuffer, lBufSize, (int)exp+500);

        if (error != POA_OK)
        {
            return DEVICE_SNAP_IMAGE_FAILED;
        }

        bool isEnableGamma = (gammaValue_ != g_gamma_def);

        if (imgFmt_ == POA_RGB24)
        {
            if (!pRGB24) // if RGB24 buffer is nullptr
            {
                return DEVICE_SNAP_IMAGE_FAILED;
            }
            std::memcpy(pRGB24, pImgBuffer, RGB24BufSize_);
            BGR888ToRGB32(pRGB24, pImgBuffer, (int) RGB24BufSize_, isEnableGamma);
        }
        else
        {
            if (isEnableGamma)
            {
                unsigned int pixNum = img_.Width() * img_.Height();

                if (imgFmt_ == POA_RAW16)
                {
                    unsigned short* pU16DataBuffer = reinterpret_cast<unsigned short*>(pImgBuffer);
                    for (unsigned int i = 0; i < pixNum; i++)
                    {
                        pU16DataBuffer[i] = p16bitGammaTable[ pU16DataBuffer[i] ];
                    }
                }
                else //raw8 and mono8
                {
                    for (unsigned int i = 0; i < pixNum; i++)
                    {
                        pImgBuffer[i] = p8bitGammaTable[pImgBuffer[i]];
                    }
                }
            }
        }
    }
    else
    {
        return DEVICE_SNAP_IMAGE_FAILED;
    }

    return DEVICE_OK;

    //------------------------------------------------------
    //static int callCounter = 0;
    //++callCounter;

    //MM::MMTime startTime = GetCurrentMMTime();
    //
    //if (sequenceRunning_ && IsCapturing())
    //{
    //    exp = GetSequenceExposure();
    //}

    //if (!fastImage_)
    //{
    //    //GenerateSyntheticImage(img_, exp);
    //}

    //MM::MMTime s0(0, 0);
    //if (s0 < startTime)
    //{
    //    while (exp > (GetCurrentMMTime() - startTime).getMsec())
    //    {
    //        CDeviceUtils::SleepMs(1);
    //    }
    //}
    //else
    //{
    //    std::cerr << "You are operating this device adapter without setting the core callback, timing functions aren't yet available" << std::endl;
    //    // called without the core callback probably in off line test program
    //    // need way to build the core in the test program

    //}
    //readoutStartTime_ = GetCurrentMMTime();
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
const unsigned char* POACamera::GetImageBuffer()
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
unsigned POACamera::GetImageWidth() const
{
    return img_.Width();
}

/**
* Returns image buffer Y-size in pixels.
* Required by the MM::Camera API.
*/
unsigned POACamera::GetImageHeight() const
{
    return img_.Height();
}

/**
* Returns image buffer pixel depth in bytes.
* Required by the MM::Camera API.
*/
unsigned POACamera::GetImageBytesPerPixel() const
{
    return img_.Depth();
}

/**
* Returns the bit depth (dynamic range) of the pixel.
* This does not affect the buffer size, it just gives the client application
* a guideline on how to interpret pixel values.
* Required by the MM::Camera API.
*/
unsigned POACamera::GetBitDepth() const
{
    return bitDepth_;
}

/**
* Returns the size in bytes of the image buffer.
* Required by the MM::Camera API.
*/
long POACamera::GetImageBufferSize() const
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
int POACamera::SetROI(unsigned x, unsigned y, unsigned xSize, unsigned ySize)
{
    if (camProp_.cameraID < 0)
        return DEVICE_NOT_CONNECTED;

    if (xSize == 0 && ySize == 0)
    {
        // effectively clear ROI
        ClearROI();
    }
    else
    {
        POABool isFlipHori = POA_FALSE, isFlipVert = POA_FALSE;
        POAGetFlip(camProp_.cameraID, &isFlipHori, &isFlipVert);

        int curBinMaxWidth = camProp_.maxWidth / binSize_ / 4 * 4;
        int curBinMaxHeight = camProp_.maxHeight / binSize_ / 2 * 2;

        if ((isFlipHori == POA_TRUE) && (isFlipVert == POA_TRUE))
        {
            x = curBinMaxWidth - x - xSize;
            y = curBinMaxHeight - y - ySize;
        }
        else if (isFlipHori == POA_TRUE)
        {
            x = curBinMaxWidth - x - xSize;
        }
        else if (isFlipVert == POA_TRUE)
        {
            y = curBinMaxHeight - y - ySize;
        }

        POAErrors error = POASetImageSize(camProp_.cameraID, xSize, ySize);
        if (error != POA_OK)
        {
            return DEVICE_ERR;
        }

        error = POASetImageStartPos(camProp_.cameraID, x, y);
        if (error != POA_OK)
        {
            return DEVICE_ERR;
        }

        RefreshCurrROIParas();

        // apply ROI
        img_.Resize(cameraCCDXSize_, cameraCCDYSize_);
        ManageRGB24Memory();
    }
    return DEVICE_OK;
}

/**
* Returns the actual dimensions of the current ROI.
* If multiple ROIs are set, then the returned ROI should encompass all of them.
* Required by the MM::Camera API.
*/
int POACamera::GetROI(unsigned& x, unsigned& y, unsigned& xSize, unsigned& ySize)
{
    x = roiX_;
    y = roiY_;

    xSize = img_.Width();
    ySize = img_.Height();

    POABool isFlipHori = POA_FALSE, isFlipVert = POA_FALSE;
    POAGetFlip(camProp_.cameraID, &isFlipHori, &isFlipVert);

    int curBinMaxWidth = camProp_.maxWidth / binSize_ / 4 * 4;
    int curBinMaxHeight = camProp_.maxHeight / binSize_ / 2 * 2;

    if ((isFlipHori == POA_TRUE) && (isFlipVert == POA_TRUE))
    {
        x = curBinMaxWidth - roiX_ - xSize;
        y = curBinMaxHeight - roiY_ - ySize;
    }
    else if (isFlipHori == POA_TRUE)
    {
        x = curBinMaxWidth - roiX_ - xSize;
    }
    else if (isFlipVert == POA_TRUE)
    {
        y = curBinMaxHeight - roiY_ - ySize;
    }

    return DEVICE_OK;
}

/**
* Resets the Region of Interest to full frame.
* Required by the MM::Camera API.
*/
int POACamera::ClearROI()
{
    if (camProp_.cameraID < 0)
        return DEVICE_NOT_CONNECTED;

    roiX_ = 0;
    roiY_ = 0;

    cameraCCDXSize_ = camProp_.maxWidth / binSize_ / 4 * 4;
    cameraCCDYSize_ = camProp_.maxHeight / binSize_ / 2 * 2;

    POASetImageStartPos(camProp_.cameraID, roiX_, roiY_); //move (0, 0) fisrt

    POAErrors error = POASetImageSize(camProp_.cameraID, cameraCCDXSize_, cameraCCDYSize_);
    if (error != POA_OK)
    {
        return DEVICE_ERR;
    }

    error = POASetImageStartPos(camProp_.cameraID, roiX_, roiY_);
    if (error != POA_OK)
    {
        return DEVICE_ERR;
    }

    ResizeImageBuffer();
    

    return DEVICE_OK;
}

/**
 * Queries if the camera supports multiple simultaneous ROIs.
 * Optional method in the MM::Camera API; by default cameras do not support
 * multiple ROIs.
 */
bool POACamera::SupportsMultiROI()
{
    return supportsMultiROI_;
}


/**
* Returns the current exposure setting in milliseconds.
* Required by the MM::Camera API.
*/
double POACamera::GetExposure() const
{
    long lExp = 0;
    POABool isAuto = POA_FALSE;
    POAGetConfig(camProp_.cameraID, POA_EXPOSURE, &lExp, &isAuto);
    return lExp / 1000.0;
}

/**
 * Returns the current exposure from a sequence and increases the sequence counter
 * Used for exposure sequences
 */
double POACamera::GetSequenceExposure()
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
void POACamera::SetExposure(double exp)
{
    if (exp < 0.0)
    {
        exp = 0.0;
    }
    else if (exp > exposureMaximum_) {
        exp = exposureMaximum_;
    }
    SetProperty(MM::g_Keyword_Exposure, CDeviceUtils::ConvertToString(exp));
    GetCoreCallback()->OnExposureChanged(this, exp);
}

/**
* Returns the current binning factor.
* Required by the MM::Camera API.
*/
int POACamera::GetBinning() const
{
    //POAGetImageBin(camProp_.cameraID, &binSize_);
    return binSize_;
}

/**
* Sets binning factor.
* Required by the MM::Camera API.
*/
int POACamera::SetBinning(int binF)
{
    return SetProperty(MM::g_Keyword_Binning, CDeviceUtils::ConvertToString(binF));
}

int POACamera::IsExposureSequenceable(bool& isSequenceable) const
{
    isSequenceable = isSequenceable_;
    return DEVICE_OK;
}

int POACamera::GetExposureSequenceMaxLength(long& nrEvents) const
{
    if (!isSequenceable_) {
        return DEVICE_UNSUPPORTED_COMMAND;
    }

    nrEvents = sequenceMaxLength_;
    return DEVICE_OK;
}

int POACamera::StartExposureSequence()
{
    if (!isSequenceable_) {
        return DEVICE_UNSUPPORTED_COMMAND;
    }

    // may need thread lock
    sequenceRunning_ = true;
    return DEVICE_OK;
}

int POACamera::StopExposureSequence()
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
int POACamera::ClearExposureSequence()
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
int POACamera::AddToExposureSequence(double exposureTime_ms)
{
    if (!isSequenceable_) {
        return DEVICE_UNSUPPORTED_COMMAND;
    }

    exposureSequence_.push_back(exposureTime_ms);
    return DEVICE_OK;
}

int POACamera::SendExposureSequence() const {
    if (!isSequenceable_) {
        return DEVICE_UNSUPPORTED_COMMAND;
    }

    return DEVICE_OK;
}


int POACamera::PrepareSequenceAcqusition()
{
    LOG("PrepareSequenceAcqusition")
    if (IsCapturing())
        return DEVICE_CAMERA_BUSY_ACQUIRING;

    return DEVICE_OK;
}

/**
 * Required by the MM::Camera API
 * Please implement this yourself and do not rely on the base class implementation
 * The Base class implementation is deprecated and will be removed shortly
 */
int POACamera::StartSequenceAcquisition(double interval)
{
    return StartSequenceAcquisition(LONG_MAX, interval, false);
}

/**
* Stop and wait for the Sequence thread finished
*/
int POACamera::StopSequenceAcquisition()
{
    LOG("StopSequenceAcquisition")

    m_bIsToStopExposure = true;
    if (!thd_->IsStopped())
    {
        thd_->Stop();
        thd_->wait();
    }

    POAStopExposure(camProp_.cameraID);

    return DEVICE_OK;
}

/**
* Simple implementation of Sequence Acquisition
* A sequence acquisition should run on its own thread and transport new images
* coming of the camera into the MMCore circular buffer.
*/
int POACamera::StartSequenceAcquisition(long numImages, double interval_ms, bool stopOnOverflow)
{
    LOG("StartSequenceAcquisition")

    if (IsCapturing())
        return DEVICE_CAMERA_BUSY_ACQUIRING;

    m_bIsToStopExposure = false;
    POAErrors error = POAStartExposure(camProp_.cameraID, POA_FALSE); // video mode
    if (error != POA_OK)
        return DEVICE_ERR;

    int ret = GetCoreCallback()->PrepareForAcq(this);
    if (ret != DEVICE_OK)
        return ret;
    sequenceStartTime_ = GetCurrentMMTime();
    imageCounter_ = 0;
    thd_->Start(numImages, interval_ms);
    stopOnOverflow_ = stopOnOverflow;
    return DEVICE_OK;
}


bool POACamera::IsCapturing() 
{
    POACameraState camState = STATE_CLOSED;
    POAGetCameraState(camProp_.cameraID, &camState);
    
    return camState == STATE_EXPOSING;
    
    //return !thd_->IsStopped();
}

/*
 * Inserts Image and MetaData into MMCore circular Buffer
 */
int POACamera::InsertImage()
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

    int ret = GetCoreCallback()->InsertImage(this, pI, w, h, b, md.Serialize().c_str());

    if (!stopOnOverflow_ && ret == DEVICE_BUFFER_OVERFLOW)
    {
        // do not stop on overflow - just reset the buffer
        GetCoreCallback()->ClearImageBuffer(this);
        // don't process this same image again...
        return GetCoreCallback()->InsertImage(this, pI, w, h, b, md.Serialize().c_str(), false);
    }
    else
        return ret;
}

/*
 * Do actual capturing
 * Called from inside the thread
 */
int POACamera::RunSequenceOnThread(MM::MMTime /* startTime */)
{
    LOG("RunSequenceOnThread")

    int ret = DEVICE_ERR;

    double exp = GetExposure();
    unsigned char* pImgBuffer = img_.GetPixelsRW();
    long lBufSize = GetImageBufferSize();

    POABool pIsReady = POA_FALSE;
    while (pIsReady == POA_FALSE)
    { 
        if (m_bIsToStopExposure)
            break;

        POAImageReady(camProp_.cameraID, &pIsReady);
    }

    if (m_bIsToStopExposure)
        return ret;

    POAErrors error = POAGetImageData(camProp_.cameraID, pImgBuffer, lBufSize, (int)exp + 500);
    if (error == POA_OK)
    {
        bool isEnableGamma = (gammaValue_ != g_gamma_def);

        if (imgFmt_ == POA_RGB24)
        {
            if (!pRGB24) // if RGB24 buffer is nullptr
            {
                return DEVICE_ERR;
            }
            std::memcpy(pRGB24, pImgBuffer, RGB24BufSize_);
            BGR888ToRGB32(pRGB24, pImgBuffer, (int) RGB24BufSize_, isEnableGamma);
        }
        else
        {
            if (isEnableGamma)
            {
                unsigned int pixNum = img_.Width() * img_.Height();

                if (imgFmt_ == POA_RAW16)
                {
                    unsigned short* pU16DataBuffer = reinterpret_cast<unsigned short*>(pImgBuffer);
                    for (unsigned int i = 0; i < pixNum; i++)
                    {
                        pU16DataBuffer[i] = p16bitGammaTable[pU16DataBuffer[i]];
                    }
                }
                else //raw8 and mono8
                {
                    for (unsigned int i = 0; i < pixNum; i++)
                    {
                        pImgBuffer[i] = p8bitGammaTable[pImgBuffer[i]];
                    }
                }
            }
        }

        ret = InsertImage();
    }
    
    return ret;
}

/*
 * called from the thread function before exit
 */
void POACamera::OnThreadExiting() throw()
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

MySequenceThread::MySequenceThread(POACamera* pCam)
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
            ret = camera_->RunSequenceOnThread(startTime_);
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
// POACamera Action handlers
///////////////////////////////////////////////////////////////////////////////

/**
* Handles "CamIndex" property.
*/
int POACamera::OnSelectCamIndex(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    std::string str;
    if (eAct == MM::AfterSet)
    {
        pProp->Get(str);
        for (int i = 0; i < connectCamerasName_.size(); i++)
        {
            if (str == connectCamerasName_.at(i))
            {
                POAGetCameraProperties(i, &camProp_);
                selectCamIndex_ = i;
                selectCamName_ = connectCamerasName_.at(i);
                break;
            }
        }
    }
    else if (eAct == MM::BeforeGet)
    {
        pProp->Set(selectCamName_.c_str());
    }

    return DEVICE_OK;
}

int POACamera::OnMaxExposure(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        pProp->Set(exposureMaximum_);
    }
    else if (eAct == MM::AfterSet)
    {
        pProp->Get(exposureMaximum_);
    }
    return DEVICE_OK;
}


void POACamera::SlowPropUpdate(std::string leaderValue)
{
    // wait in order to simulate a device doing something slowly
    // in a thread
    long delay; GetProperty("AsyncPropertyDelayMS", delay);
    CDeviceUtils::SleepMs(delay);
    {
        MMThreadGuard g(asyncFollowerLock_);
        asyncFollower_ = leaderValue;
    }
    OnPropertyChanged("AsyncPropertyFollower", leaderValue.c_str());
}

int POACamera::OnAsyncFollower(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet) {
        MMThreadGuard g(asyncFollowerLock_);
        pProp->Set(asyncFollower_.c_str());
    }
    // no AfterSet as this is a readonly property
    return DEVICE_OK;
}

int POACamera::OnAsyncLeader(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet) {
        pProp->Set(asyncLeader_.c_str());
    }
    if (eAct == MM::AfterSet)
    {
        pProp->Get(asyncLeader_);
        fut_ = std::async(std::launch::async, &POACamera::SlowPropUpdate, this, asyncLeader_);
    }
    return DEVICE_OK;
}

/**
* Handles "Binning" property.
*/
int POACamera::OnBinning(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (camProp_.cameraID < 0)
        return DEVICE_NOT_CONNECTED;

    int ret = DEVICE_ERR;
    switch (eAct)
    {
    case MM::AfterSet:
    {
        if (IsCapturing())
            return DEVICE_CAMERA_BUSY_ACQUIRING;

        long binFactor;
        pProp->Get(binFactor);

        POAErrors error = POASetImageBin(camProp_.cameraID, binFactor);

        if (error == POA_OK)
        {
            RefreshCurrROIParas();

            img_.Resize(cameraCCDXSize_, cameraCCDYSize_);
            ManageRGB24Memory();

            std::ostringstream os;
            os << binSize_;
            OnPropertyChanged(MM::g_Keyword_Binning, os.str().c_str());
            ret = DEVICE_OK;
        }
    }
    break;
    case MM::BeforeGet:
    {   
        POAGetImageBin(camProp_.cameraID, &binSize_);
        pProp->Set((long)binSize_);
        ret = DEVICE_OK;
    }
    break;
    default:
        break;
    }
    return ret;
}

/**
* Handles "PixelType" property.
*/
int POACamera::OnPixelType(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (camProp_.cameraID < 0)
        return DEVICE_NOT_CONNECTED;

    int ret = DEVICE_ERR;
    switch (eAct)
    {
    case MM::AfterSet:
    {
        if (IsCapturing())
            return DEVICE_CAMERA_BUSY_ACQUIRING;

        std::string pixelType;
        pProp->Get(pixelType);
        POAImgFormat imgFmt;

        if (pixelType.compare(g_PixelType_8bit) == 0)
        {
            imgFmt = POA_RAW8;
        }
        else if (pixelType.compare(g_PixelType_16bit) == 0)
        {
            imgFmt = POA_RAW16;
        }
        else if (pixelType.compare(g_PixelType_32bitRGB) == 0)
        {
            imgFmt = POA_RGB24;
        }
        else if (pixelType.compare(g_PixelType_MONO8bit) == 0)
        {
            imgFmt = POA_MONO8;  
        }
        else
        {
            imgFmt = POA_END;  
        }

        if (imgFmt == POA_END)
        {
            ret = ERR_UNKNOWN_MODE;
        }
        else
        {
            POAErrors error = POASetImageFormat(camProp_.cameraID, imgFmt);
            if (error == POA_OK)
            {
                imgFmt_ = imgFmt;

                if (imgFmt == POA_RAW16)
                {
                    nComponents_ = 1;
                    img_.Resize(img_.Width(), img_.Height(), 2);
                    bitDepth_ = 16;
                }
                else if (imgFmt == POA_RGB24)
                {
                    nComponents_ = 4;
                    img_.Resize(img_.Width(), img_.Height(), 4);
                    bitDepth_ = 8;
                }
                else
                {
                    nComponents_ = 1;
                    img_.Resize(img_.Width(), img_.Height(), 1);
                    bitDepth_ = 8;
                }
                ret = DEVICE_OK;
            }

            ManageRGB24Memory();
        }
 
    }
    break;
    case MM::BeforeGet:
    {
        POAGetImageFormat(camProp_.cameraID, &imgFmt_);
        pProp->Set(ImgFmtToString(imgFmt_));
 
        ret = DEVICE_OK;
    } break;
    default:
        break;
    }
    return ret;
}

/**
* Handles "Exposure" property.
*/
int POACamera::OnExposure(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (camProp_.cameraID < 0)
        return DEVICE_NOT_CONNECTED;

    int ret = DEVICE_ERR;
    switch (eAct)
    {
    case MM::AfterSet:
    {
        double dblExp;
        pProp->Get(dblExp);

        if (dblExp < 0.0)
        {
            dblExp = 0.0;
        }
        else if (dblExp > exposureMaximum_) {
            dblExp = exposureMaximum_;
        }

        long lExp = (long) (dblExp * 1000.0);
        POASetConfig(camProp_.cameraID, POA_EXPOSURE, lExp, POA_FALSE);

        ret = DEVICE_OK;
    }
    break;
    case MM::BeforeGet:
    {
        long lExp = 0;
        POABool isAuto = POA_FALSE;
        POAGetConfig(camProp_.cameraID, POA_EXPOSURE, &lExp, &isAuto);

        pProp->Set(lExp / 1000.0);

        ret = DEVICE_OK;
    }
    break;
    default:
        break;
    }

    return ret;
}

int POACamera::OnExpAuto(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (camProp_.cameraID < 0)
        return DEVICE_NOT_CONNECTED;

    int ret = DEVICE_ERR;
    switch (eAct)
    {
    case MM::AfterSet:
    {
        std::string strExpAuto;
        pProp->Get(strExpAuto);
        long lExp = 0;
        POABool isExpAuto = POA_FALSE;
        POAGetConfig(camProp_.cameraID, POA_EXPOSURE, &lExp, &isExpAuto); // get current exp value first

        isExpAuto = (strExpAuto == g_PropName_ON) ? POA_TRUE : POA_FALSE;
        POASetConfig(camProp_.cameraID, POA_EXPOSURE, lExp, isExpAuto);
        ret = DEVICE_OK;
    }
    break;
    case MM::BeforeGet:
    {
        long lExp = 0;
        POABool isExpAuto = POA_FALSE;
        POAGetConfig(camProp_.cameraID, POA_EXPOSURE, &lExp, &isExpAuto);

        pProp->Set((isExpAuto == POA_TRUE) ? g_PropName_ON : g_PropName_OFF);

        ret = DEVICE_OK;
    }
    break;
    default:
        break;
    }

    return ret;
}

/**
* Handles "Gain" property.
*/
int POACamera::OnGain(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (camProp_.cameraID < 0)
        return DEVICE_NOT_CONNECTED;

    int ret = DEVICE_ERR;
    switch (eAct)
    {
    case MM::AfterSet:
    {
        long lGain;
        pProp->Get(lGain);

        POASetConfig(camProp_.cameraID, POA_GAIN, lGain, POA_FALSE);

        ret = DEVICE_OK;
    }
    break;
    case MM::BeforeGet:
    {
        long lGain = 0;
        POABool isAuto = POA_FALSE;
        POAGetConfig(camProp_.cameraID, POA_GAIN, &lGain, &isAuto);

        pProp->Set(lGain);

        ret = DEVICE_OK;
    }
    break;
    default:
        break;
    }

    return ret;
}

int POACamera::OnGainAuto(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (camProp_.cameraID < 0)
        return DEVICE_NOT_CONNECTED;

    int ret = DEVICE_ERR;
    switch (eAct)
    {
    case MM::AfterSet:
    {
        std::string strGainAuto;
        pProp->Get(strGainAuto);
        long lGain = 0;
        POABool isGainAuto = POA_FALSE;
        POAGetConfig(camProp_.cameraID, POA_GAIN, &lGain, &isGainAuto); // get current gain value first

        isGainAuto = (strGainAuto == g_PropName_ON) ? POA_TRUE : POA_FALSE;
        POASetConfig(camProp_.cameraID, POA_GAIN, lGain, isGainAuto);
        ret = DEVICE_OK;
    }
    break;
    case MM::BeforeGet:
    {
        long lGain = 0;
        POABool isGainAuto = POA_FALSE;
        POAGetConfig(camProp_.cameraID, POA_GAIN, &lGain, &isGainAuto);

        pProp->Set((isGainAuto == POA_TRUE) ? g_PropName_ON : g_PropName_OFF);

        ret = DEVICE_OK;
    }
    break;
    default:
        break;
    }

    return ret;
}

int POACamera::OnAutoExpBrightness(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (camProp_.cameraID < 0)
        return DEVICE_NOT_CONNECTED;

    int ret = DEVICE_ERR;
    switch (eAct)
    {
    case MM::AfterSet:
    {
        long lBrighntness;
        pProp->Get(lBrighntness);

        POASetConfig(camProp_.cameraID, POA_AUTOEXPO_BRIGHTNESS, lBrighntness, POA_FALSE);

        ret = DEVICE_OK;
    }
    break;
    case MM::BeforeGet:
    {
        long lBrighntness = 0;
        POABool isAuto = POA_FALSE;
        POAGetConfig(camProp_.cameraID, POA_AUTOEXPO_BRIGHTNESS, &lBrighntness, &isAuto);

        pProp->Set(lBrighntness);

        ret = DEVICE_OK;
    }
    break;
    default:
        break;
    }

    return ret;
}

/**
* Handles "Offset" property.
*/
int POACamera::OnOffset(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (camProp_.cameraID < 0)
        return DEVICE_NOT_CONNECTED;

    int ret = DEVICE_ERR;
    switch (eAct)
    {
    case MM::AfterSet:
    {
        long lOffset;
        pProp->Get(lOffset);

        POASetConfig(camProp_.cameraID, POA_OFFSET, lOffset, POA_FALSE);

        ret = DEVICE_OK;
    }
    break;
    case MM::BeforeGet:
    {
        long lOffset = 0;
        POABool isAuto = POA_FALSE;
        POAGetConfig(camProp_.cameraID, POA_OFFSET, &lOffset, &isAuto);

        pProp->Set(lOffset);

        ret = DEVICE_OK;
    }
    break;
    default:
        break;
    }

    return ret;
}

int POACamera::OnGamma(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (camProp_.cameraID < 0)
        return DEVICE_NOT_CONNECTED;

    int ret = DEVICE_ERR;
    switch (eAct)
    {
    case MM::AfterSet:
    {
        pProp->Get(gammaValue_);

        //reset gamma table
        ResetGammaTable();

        ret = DEVICE_OK;
    }
    break;
    case MM::BeforeGet:
    {  
        pProp->Set(gammaValue_);

        ret = DEVICE_OK;
    }
    break;
    default:
        break;
    }

    return ret;
}

/**
* Handles "USB Bandwidth Limit" property.
*/
int POACamera::OnUSBBandwidthLimit(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (camProp_.cameraID < 0)
        return DEVICE_NOT_CONNECTED;

    int ret = DEVICE_ERR;
    switch (eAct)
    {
    case MM::AfterSet:
    {
        long lUSB_BW;
        pProp->Get(lUSB_BW);

        POASetConfig(camProp_.cameraID, POA_USB_BANDWIDTH_LIMIT, lUSB_BW, POA_FALSE);

        ret = DEVICE_OK;
    }
    break;
    case MM::BeforeGet:
    {
        long lUSB_BW = 0;
        POABool isAuto = POA_FALSE;
        POAGetConfig(camProp_.cameraID, POA_USB_BANDWIDTH_LIMIT, &lUSB_BW, &isAuto);

        pProp->Set(lUSB_BW);

        ret = DEVICE_OK;
    }
    break;
    default:
        break;
    }

    return ret;
}

/**
* Handles "Frame Rate Limit" property.
*/
int POACamera::OnFrameRateLimit(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (camProp_.cameraID < 0)
        return DEVICE_NOT_CONNECTED;

    int ret = DEVICE_ERR;
    switch (eAct)
    {
    case MM::AfterSet:
    {
        long lFrmRateLimit;
        pProp->Get(lFrmRateLimit);

        POASetConfig(camProp_.cameraID, POA_FRAME_LIMIT, lFrmRateLimit, POA_FALSE);

        ret = DEVICE_OK;
    }
    break;
    case MM::BeforeGet:
    {
        long lFrmRateLimit = 0;
        POABool isAuto = POA_FALSE;
        POAGetConfig(camProp_.cameraID, POA_FRAME_LIMIT, &lFrmRateLimit, &isAuto);

        pProp->Set(lFrmRateLimit);

        ret = DEVICE_OK;
    }
    break;
    default:
        break;
    }

    return ret;
}

/**
* Handles "Image Flip" property.
*/
int POACamera::OnFlip(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (camProp_.cameraID < 0)
        return DEVICE_NOT_CONNECTED;

    int ret = DEVICE_ERR;
    switch (eAct)
    {
    case MM::AfterSet:
    {
        std::string strFlip;
        pProp->Get(strFlip);

        POAConfig flipCfg = POA_FLIP_NONE;
        if (strFlip == g_FlipHori)
        {
            flipCfg = POA_FLIP_HORI;
        }
        else if (strFlip == g_FlipVert)
        {
            flipCfg = POA_FLIP_VERT;
        }
        else if (strFlip == g_FlipBoth)
        {
            flipCfg = POA_FLIP_BOTH;
        }

        POASetConfig(camProp_.cameraID, flipCfg, POA_TRUE);

        ret = DEVICE_OK;
    }
    break;
    case MM::BeforeGet:
    {
        std::string strFlip = g_FlipNone;

        POABool isEnableBoth = POA_FALSE, isEnableHori = POA_FALSE, isEnableVert = POA_FALSE;
        
        POAGetConfig(camProp_.cameraID, POA_FLIP_BOTH, &isEnableBoth);
        POAGetConfig(camProp_.cameraID, POA_FLIP_HORI, &isEnableHori);
        POAGetConfig(camProp_.cameraID, POA_FLIP_VERT, &isEnableVert);

        if (isEnableBoth == POA_TRUE)
        {
            strFlip = g_FlipBoth;
        }
        else if(isEnableHori == POA_TRUE)
        {
            strFlip = g_FlipHori;
        }
        else if (isEnableVert == POA_TRUE)
        {
            strFlip = g_FlipVert;
        }

        pProp->Set(strFlip.c_str());

        ret = DEVICE_OK;
    }
    break;
    default:
        break;
    }

    return ret;
}

int POACamera::OnCCDTemp(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (camProp_.cameraID < 0)
        return DEVICE_NOT_CONNECTED;

    if (eAct == MM::BeforeGet)
    {
        POABool isAuto;
        POAGetConfig(camProp_.cameraID, POA_TEMPERATURE, &ccdT_, &isAuto);
        pProp->Set(ccdT_);
    }
    // no AfterSet as this is a readonly property
    return DEVICE_OK;
}


int POACamera::OnHardBin(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (camProp_.cameraID < 0)
        return DEVICE_NOT_CONNECTED;

    int ret = DEVICE_ERR;
    switch (eAct)
    {
    case MM::AfterSet:
    {
        std::string strHardBin;
        pProp->Get(strHardBin);

        POABool isHardBin = (strHardBin == g_PropName_ON) ? POA_TRUE : POA_FALSE;
        POASetConfig(camProp_.cameraID, POA_HARDWARE_BIN, isHardBin);      
        ret = DEVICE_OK;
    }
    break;
    case MM::BeforeGet:
    {
        POABool isHardBin;
        POAGetConfig(camProp_.cameraID, POA_HARDWARE_BIN, &isHardBin);

        pProp->Set((isHardBin == POA_TRUE) ? g_PropName_ON : g_PropName_OFF);

        ret = DEVICE_OK;
    }
    break;
    default:
        break;
    }

    return ret;
}

int POACamera::OnPixelBinSum(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (camProp_.cameraID < 0)
        return DEVICE_NOT_CONNECTED;

    int ret = DEVICE_ERR;
    switch (eAct)
    {
    case MM::AfterSet:
    {
        std::string strPixelBinSum;
        pProp->Get(strPixelBinSum);

        POABool isPixBinSum = (strPixelBinSum == g_PropName_ON) ? POA_TRUE : POA_FALSE;
        POASetConfig(camProp_.cameraID, POA_PIXEL_BIN_SUM, isPixBinSum);
        ret = DEVICE_OK;
    }
    break;
    case MM::BeforeGet:
    {
        POABool isPixBinSum;
        POAGetConfig(camProp_.cameraID, POA_PIXEL_BIN_SUM, &isPixBinSum);

        pProp->Set((isPixBinSum == POA_TRUE) ? g_PropName_ON : g_PropName_OFF);

        ret = DEVICE_OK;
    }
    break;
    default:
        break;
    }

    return ret;
}

int POACamera::OnWB_Red(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (camProp_.cameraID < 0)
        return DEVICE_NOT_CONNECTED;

    int ret = DEVICE_ERR;
    switch (eAct)
    {
    case MM::AfterSet:
    {
        long lWB_R;
        pProp->Get(lWB_R);

        POASetConfig(camProp_.cameraID, POA_WB_R, lWB_R, POA_FALSE);

        ret = DEVICE_OK;
    }
    break;
    case MM::BeforeGet:
    {
        long lWB_R = 0;
        POABool isAuto = POA_FALSE;
        POAGetConfig(camProp_.cameraID, POA_WB_R, &lWB_R, &isAuto);

        pProp->Set(lWB_R);

        ret = DEVICE_OK;
    }
    break;
    default:
        break;
    }

    return ret;
}

int POACamera::OnWB_Green(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (camProp_.cameraID < 0)
        return DEVICE_NOT_CONNECTED;

    int ret = DEVICE_ERR;
    switch (eAct)
    {
    case MM::AfterSet:
    {
        long lWB_G;
        pProp->Get(lWB_G);

        POASetConfig(camProp_.cameraID, POA_WB_G, lWB_G, POA_FALSE);

        ret = DEVICE_OK;
    }
    break;
    case MM::BeforeGet:
    {
        long lWB_G = 0;
        POABool isAuto = POA_FALSE;
        POAGetConfig(camProp_.cameraID, POA_WB_G, &lWB_G, &isAuto);

        pProp->Set(lWB_G);

        ret = DEVICE_OK;
    }
    break;
    default:
        break;
    }

    return ret;
}

int POACamera::OnWB_Blue(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (camProp_.cameraID < 0)
        return DEVICE_NOT_CONNECTED;

    int ret = DEVICE_ERR;
    switch (eAct)
    {
    case MM::AfterSet:
    {
        long lWB_B;
        pProp->Get(lWB_B);

        POASetConfig(camProp_.cameraID, POA_WB_B, lWB_B, POA_FALSE);

        ret = DEVICE_OK;
    }
    break;
    case MM::BeforeGet:
    {
        long lWB_B = 0;
        POABool isAuto = POA_FALSE;
        POAGetConfig(camProp_.cameraID, POA_WB_B, &lWB_B, &isAuto);

        pProp->Set(lWB_B);

        ret = DEVICE_OK;
    }
    break;
    default:
        break;
    }

    return ret;
}

int POACamera::OnAutoWB(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (camProp_.cameraID < 0)
        return DEVICE_NOT_CONNECTED;

    int ret = DEVICE_ERR;
    switch (eAct)
    {
    case MM::AfterSet:
    {
        std::string strAutoWB;
        pProp->Get(strAutoWB);

        long lWB_R = 0;
        POABool isAuto = POA_FALSE;
        POAGetConfig(camProp_.cameraID, POA_WB_R, &lWB_R, &isAuto);

        isAuto = (strAutoWB == g_PropName_ON) ? POA_TRUE : POA_FALSE;
        POASetConfig(camProp_.cameraID, POA_WB_R, lWB_R, isAuto); // just set R is OK
        ret = DEVICE_OK;
    }
    break;
    case MM::BeforeGet:
    {
        long lWB_R = 0;
        POABool isAuto = POA_FALSE;
        POAGetConfig(camProp_.cameraID, POA_WB_R, &lWB_R, &isAuto);

        pProp->Set((isAuto == POA_TRUE) ? g_PropName_ON : g_PropName_OFF);

        ret = DEVICE_OK;
    }
    break;
    default:
        break;
    }

    return ret;
}

int POACamera::OnMonoBin(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (camProp_.cameraID < 0)
        return DEVICE_NOT_CONNECTED;

    int ret = DEVICE_ERR;
    switch (eAct)
    {
    case MM::AfterSet:
    {
        std::string strMonoBin;
        pProp->Get(strMonoBin);

        POABool isMonoBin = (strMonoBin == g_PropName_ON) ? POA_TRUE : POA_FALSE;
        POASetConfig(camProp_.cameraID, POA_MONO_BIN, isMonoBin);
        ret = DEVICE_OK;
    }
    break;
    case MM::BeforeGet:
    {
        POABool isMonoBin;
        POAGetConfig(camProp_.cameraID, POA_MONO_BIN, &isMonoBin);

        pProp->Set((isMonoBin == POA_TRUE) ? g_PropName_ON : g_PropName_OFF);

        ret = DEVICE_OK;
    }
    break;
    default:
        break;
    }

    return ret;
}

int POACamera::OnTargetTEMP(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (camProp_.cameraID < 0)
        return DEVICE_NOT_CONNECTED;

    int ret = DEVICE_ERR;
    switch (eAct)
    {
    case MM::AfterSet:
    {
        long lTargetTemp;
        pProp->Get(lTargetTemp);

        POASetConfig(camProp_.cameraID, POA_TARGET_TEMP, lTargetTemp, POA_FALSE);

        ret = DEVICE_OK;
    }
    break;
    case MM::BeforeGet:
    {
        long lTargetTemp = 0;
        POABool isAuto = POA_FALSE;
        POAGetConfig(camProp_.cameraID, POA_TARGET_TEMP, &lTargetTemp, &isAuto);

        pProp->Set(lTargetTemp);

        ret = DEVICE_OK;
    }
    break;
    default:
        break;
    }

    return ret;
}

int POACamera::OnCoolerOn(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (camProp_.cameraID < 0)
        return DEVICE_NOT_CONNECTED;

    int ret = DEVICE_ERR;
    switch (eAct)
    {
    case MM::AfterSet:
    {
        std::string strCoolerOn;
        pProp->Get(strCoolerOn);

        POABool isCoolerOn = (strCoolerOn == g_PropName_ON) ? POA_TRUE : POA_FALSE;
        POASetConfig(camProp_.cameraID, POA_COOLER, isCoolerOn);
        ret = DEVICE_OK;
    }
    break;
    case MM::BeforeGet:
    {
        POABool isCoolerOn;
        POAGetConfig(camProp_.cameraID, POA_COOLER, &isCoolerOn);

        pProp->Set((isCoolerOn == POA_TRUE) ? g_PropName_ON : g_PropName_OFF);

        ret = DEVICE_OK;
    }
    break;
    default:
        break;
    }

    return ret;
}

int POACamera::OnCoolerPower(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (camProp_.cameraID < 0)
        return DEVICE_NOT_CONNECTED;

    if (eAct == MM::BeforeGet)
    {
        long lCoolerPower = 0;
        POABool isAuto = POA_FALSE;
        POAGetConfig(camProp_.cameraID, POA_COOLER_POWER, &lCoolerPower, &isAuto);
        pProp->Set(lCoolerPower);
    }
    // no AfterSet as this is a readonly property
    return DEVICE_OK;
}

int POACamera::OnFanPower(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (camProp_.cameraID < 0)
        return DEVICE_NOT_CONNECTED;

    int ret = DEVICE_ERR;
    switch (eAct)
    {
    case MM::AfterSet:
    {
        long lFanPower;
        pProp->Get(lFanPower);

        POASetConfig(camProp_.cameraID, POA_FAN_POWER, lFanPower, POA_FALSE);

        ret = DEVICE_OK;
    }
    break;
    case MM::BeforeGet:
    {
        long lFanPower = 0;
        POABool isAuto = POA_FALSE;
        POAGetConfig(camProp_.cameraID, POA_FAN_POWER, &lFanPower, &isAuto);

        pProp->Set(lFanPower);

        ret = DEVICE_OK;
    }
    break;
    default:
        break;
    }

    return ret;
}

int POACamera::OnHeaterPower(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (camProp_.cameraID < 0)
        return DEVICE_NOT_CONNECTED;

    int ret = DEVICE_ERR;
    switch (eAct)
    {
    case MM::AfterSet:
    {
        long lHeaterPower;
        pProp->Get(lHeaterPower);

        POASetConfig(camProp_.cameraID, POA_HEATER_POWER, lHeaterPower, POA_FALSE);

        ret = DEVICE_OK;
    }
    break;
    case MM::BeforeGet:
    {
        long lHeaterPower = 0;
        POABool isAuto = POA_FALSE;
        POAGetConfig(camProp_.cameraID, POA_HEATER_POWER, &lHeaterPower, &isAuto);

        pProp->Set(lHeaterPower);

        ret = DEVICE_OK;
    }
    break;
    default:
        break;
    }

    return ret;
}

int POACamera::OnGainPreset(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (camProp_.cameraID < 0)
        return DEVICE_NOT_CONNECTED;

    int ret = DEVICE_ERR;
    switch (eAct)
    {
    case MM::AfterSet:
    {
        std::string strGainPreset;
        pProp->Get(strGainPreset);

        long lGain;
        if (strGainPreset == g_gainHighestDR)
        {
            lGain = gainHighestDR_;
        }
        else if (strGainPreset == g_HCGain)
        {
            lGain = HCGain_;
        }
        else if (strGainPreset == g_unityGain)
        {
            lGain = unityGain_;
        }
        else if (strGainPreset == g_gainLowestRN)
        {
            lGain = gainLowestRN_;
        }
        else
        {
            POABool isAuto = POA_FALSE;
            POAGetConfig(camProp_.cameraID, POA_GAIN, &lGain, &isAuto);
        }

        SetProperty(MM::g_Keyword_Gain, CDeviceUtils::ConvertToString(lGain));
        //GetCoreCallback()->OnPropertyChanged(this, MM::g_Keyword_Gain, CDeviceUtils::ConvertToString(lGain));

        ret = DEVICE_OK;
    }
    break;
    case MM::BeforeGet:
    {
        std::string strGainPreset;
        long lGain;
        POABool isAuto = POA_FALSE;
        POAGetConfig(camProp_.cameraID, POA_GAIN, &lGain, &isAuto);

        if (lGain == gainHighestDR_)
        {
            strGainPreset = g_gainHighestDR;
        }
        else if (lGain == HCGain_)
        {
            strGainPreset = g_HCGain;
        }
        else if (lGain == unityGain_)
        {
            strGainPreset = g_unityGain;
        }
        else if (lGain == gainLowestRN_)
        {
            strGainPreset = g_gainLowestRN;
        }
        else
        {
            strGainPreset = "";
        }
        

        pProp->Set(strGainPreset.c_str());

        ret = DEVICE_OK;
    }
    break;
    default:
        break;
    }

    return ret;
}
int POACamera::OnOffsetPreset(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (camProp_.cameraID < 0)
        return DEVICE_NOT_CONNECTED;

    int ret = DEVICE_ERR;
    switch (eAct)
    {
    case MM::AfterSet:
    {
        std::string strOffsetPreset;
        pProp->Get(strOffsetPreset);

        long lOffset;
        if (strOffsetPreset == g_offsetHighestDR)
        {
            lOffset = offsetHighestDR_;
        }
        else if (strOffsetPreset == g_offsetHCGain)
        {
            lOffset = offsetHCGain_;
        }
        else if (strOffsetPreset == g_offsetUnityGain)
        {
            lOffset = offsetUnityGain_;
        }
        else if (strOffsetPreset == g_offsetLowestRN)
        {
            lOffset = offsetLowestRN_;
        }
        else
        {
            POABool isAuto = POA_FALSE;
            POAGetConfig(camProp_.cameraID, POA_OFFSET, &lOffset, &isAuto);
        }

        SetProperty(MM::g_Keyword_Offset, CDeviceUtils::ConvertToString(lOffset));
        //GetCoreCallback()->OnPropertyChanged(this, MM::g_Keyword_Offset, CDeviceUtils::ConvertToString(lOffset));

        ret = DEVICE_OK;
    }
    break;
    case MM::BeforeGet:
    {
        std::string strOffsetPreset;
        long lOffset;
        POABool isAuto = POA_FALSE;
        POAGetConfig(camProp_.cameraID, POA_OFFSET, &lOffset, &isAuto);

        if (lOffset == offsetHighestDR_)
        {
            strOffsetPreset = g_offsetHighestDR;
        }
        else if (lOffset == offsetHCGain_)
        {
            strOffsetPreset = g_offsetHCGain;
        }
        else if (lOffset == offsetUnityGain_)
        {
            strOffsetPreset = g_offsetUnityGain;
        }
        else if (lOffset == offsetLowestRN_)
        {
            strOffsetPreset = g_offsetLowestRN;
        }
        else
        {
            strOffsetPreset = "";
        }


        pProp->Set(strOffsetPreset.c_str());
        
        ret = DEVICE_OK;
    }
    break;
    default:
        break;
    }

    return ret;
}

int POACamera::OnIsSequenceable(MM::PropertyBase* pProp, MM::ActionType eAct)
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
// Private POACamera methods
///////////////////////////////////////////////////////////////////////////////

/**
* Sync internal image buffer size to the chosen property values.
*/
int POACamera::ResizeImageBuffer()
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
    else if (pixelType.compare(g_PixelType_MONO8bit) == 0)
    {
        byteDepth = 1;
    }

    img_.Resize(cameraCCDXSize_, cameraCCDYSize_, byteDepth);
    ManageRGB24Memory();

    return DEVICE_OK;
}

void POACamera::GenerateEmptyImage(ImgBuffer& img)
{
    MMThreadGuard g(imgPixelsLock_);
    if (img.Height() == 0 || img.Width() == 0 || img.Depth() == 0)
        return;
    unsigned char* pBuf = const_cast<unsigned char*>(img.GetPixels());
    memset(pBuf, 0, img.Height() * img.Width() * img.Depth());
}

bool POACamera::GenerateColorTestPattern(ImgBuffer& img)
{
    unsigned width = img.Width(), height = img.Height();
    switch (img.Depth())
    {
    case 1:
    {
        const unsigned char maxVal = 255;
        unsigned char* rawBytes = img.GetPixelsRW();
        for (unsigned y = 0; y < height; ++y)
        {
            for (unsigned x = 0; x < width; ++x)
            {
                if (y == 0)
                {
                    rawBytes[x] = (unsigned char)(maxVal * (x + 1) / (width - 1));
                }
                else {
                    rawBytes[x + y * width] = rawBytes[x];
                }
            }
        }
        return true;
    }
    case 2:
    {
        const unsigned short maxVal = 65535;
        unsigned short* rawShorts =
            reinterpret_cast<unsigned short*>(img.GetPixelsRW());
        for (unsigned y = 0; y < height; ++y)
        {
            for (unsigned x = 0; x < width; ++x)
            {
                if (y == 0)
                {
                    rawShorts[x] = (unsigned short)(maxVal * (x + 1) / (width - 1));
                }
                else {
                    rawShorts[x + y * width] = rawShorts[x];
                }
            }
        }
        return true;
    }
    case 4:
    {
        const unsigned long maxVal = 255;
        unsigned* rawPixels = reinterpret_cast<unsigned*>(img.GetPixelsRW());
        for (unsigned section = 0; section < 8; ++section)
        {
            unsigned ystart = section * (height / 8);
            unsigned ystop = section == 7 ? height : ystart + (height / 8);
            for (unsigned y = ystart; y < ystop; ++y)
            {
                for (unsigned x = 0; x < width; ++x)
                {
                    rawPixels[x + y * width] = 0;
                    for (unsigned component = 0; component < 4; ++component)
                    {
                        unsigned sample = 0;
                        if (component == section ||
                            (section >= 4 && section - 4 != component))
                        {
                            sample = maxVal * (x + 1) / (width - 1);
                        }
                        sample &= 0xff; // Just in case
                        rawPixels[x + y * width] |= sample << (8 * component);
                    }
                }
            }
        }
        return true;
    }
    }
    return false;
}


void POACamera::TestResourceLocking(const bool recurse)
{
    if (recurse)
        TestResourceLocking(false);
}


const char* POACamera::ImgFmtToString(const POAImgFormat& imgFmt)
{
    switch (imgFmt)
    {
    case POA_RAW8:
        return g_PixelType_8bit;
    case POA_RAW16:
        return g_PixelType_16bit;
    case POA_RGB24:
        return g_PixelType_32bitRGB;
    case POA_MONO8:
        return g_PixelType_MONO8bit;
    default:
        return "";
    }
}

void POACamera::RefreshCurrROIParas()
{
    POAGetImageBin(camProp_.cameraID, &binSize_);

    POAGetImageStartPos(camProp_.cameraID, &roiX_, &roiY_);

    POAGetImageSize(camProp_.cameraID, &cameraCCDXSize_, &cameraCCDYSize_);

    POAGetImageFormat(camProp_.cameraID, &imgFmt_);
}

void POACamera::ManageRGB24Memory()
{
    //malloc RGB24 memory
    if (imgFmt_ == POA_RGB24)
    {
        std::size_t size = (std::size_t)img_.Width() * img_.Height() * 3;
        if (!pRGB24)
        {
            pRGB24 = new unsigned char[size];
            RGB24BufSize_ = size;
        }
        else
        {
            if (size != RGB24BufSize_)
            {
                delete[] pRGB24;
                pRGB24 = nullptr;

                pRGB24 = new unsigned char[size];
                RGB24BufSize_ = size;
            }
        }
    }
    else
    {
        if (pRGB24)
        {
            delete[] pRGB24;
            pRGB24 = nullptr;
        }
    }
}

void POACamera::ResetGammaTable()
{
    if (!p8bitGammaTable || !p16bitGammaTable)
    {
        return;
    }

    for (int i = 0; i < 256; i++)
    {
        double gamma_f = 1 / gammaValue_;
        double f = std::pow((i + 0.5) / 255.0, gamma_f);
        p8bitGammaTable[i] = (unsigned char)std::min(255.0, f * 255.0 - 0.5);
    }

    for (int i = 0; i < 65536; i++)
    {
        double gamma_f = 1 / gammaValue_;
        double f = std::pow((i + 0.5) / 65535.0, gamma_f);
        p16bitGammaTable[i] = (unsigned short)std::min(65535.0, f * 65535.0 - 0.5);
    }
}

void POACamera::BGR888ToRGB32(unsigned char* pBGR888Buf, unsigned char* pRGB32Buf, int bgr888Len, bool isEnableGamma)
{
    int pixNum = bgr888Len / 3;
    for (int i = 0; i < pixNum; i++)
    {
        unsigned char value_B = pBGR888Buf[i * 3];
        unsigned char value_G = pBGR888Buf[i * 3 + 1];
        unsigned char value_R = pBGR888Buf[i * 3 + 2];

        if (isEnableGamma)
        {
            value_B = p8bitGammaTable[value_B];
            value_G = p8bitGammaTable[value_G];
            value_R = p8bitGammaTable[value_R];
        }

        pRGB32Buf[i * 4] = value_B;
        pRGB32Buf[i * 4 + 1] = value_G;
        pRGB32Buf[i * 4 + 2] = value_R;
        pRGB32Buf[i * 4 + 3] = 255;
    }
}

///////////////////////////////////////////////////////////////////////////////
// POAFilterWheel implementation
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

POAFilterWheel::POAFilterWheel() :
    initialized_(false),
    changedTime_(0.0),
    position_(0),
    isBusyWait(false)
{
    InitializeDefaultErrorMessages();
    SetErrorText(ERR_UNKNOWN_POSITION, "Requested position not available in this device");
    EnableDelay(); // signals that the delay setting will be used
    // parent ID display
    CreateHubIDProperty();

    PWProp_.Handle = -1;

    //scan all POA wheels
    int PW_count = POAGetPWCount();
    PWProperties PWProp;
    connectPWsName_.clear();
    for (int i = 0; i < PW_count; i++)
    {
        PWProp.Handle = -1;
        POAGetPWProperties(i, &PWProp);
        if (PWProp.Handle == -1)
        {
            continue;
        }
        connectPWsName_.push_back(std::string(PWProp.Name));
    }

    CPropertyAction* pAct = new CPropertyAction(this, &POAFilterWheel::OnSelectPWIndex);

    if (PW_count > 0)
    {
        selectPWIndex_ = 0;
        selectPWName_ = connectPWsName_.at(selectPWIndex_); //default PW is first one
        POAGetPWProperties(selectPWIndex_, &PWProp_);
    }
    else
    {
        selectPWIndex_ = -1;
        selectPWName_ = "No POA filter wheel found";
    }

    CreateStringProperty(g_SelectFilterWheel, selectPWName_.c_str(), false, pAct, true);
    SetAllowedValues(g_SelectFilterWheel, connectPWsName_);
}

POAFilterWheel::~POAFilterWheel()
{
    Shutdown();
}

void POAFilterWheel::GetName(char* Name) const
{
    CDeviceUtils::CopyLimitedString(Name, g_WheelDeviceName);
}


int POAFilterWheel::Initialize()
{
    DemoHub* pHub = static_cast<DemoHub*>(GetParentHub());
    if (pHub)
    {
        char hubLabel[MM::MaxStrLength];
        pHub->GetLabel(hubLabel);
        SetParentID(hubLabel); // for backward comp.
    }
    else
        LogMessage(NoHubError);

    if (initialized_)
        return DEVICE_OK;

    // set property list
    // -----------------

    // Name
    int ret = CreateStringProperty(MM::g_Keyword_Name, g_WheelDeviceName, true);
    if (DEVICE_OK != ret)
        return ret;

    // Description
    ret = CreateStringProperty(MM::g_Keyword_Description, "Player One Astronomy Filter Wheel Driver", true);
    if (DEVICE_OK != ret)
        return ret;

    // Set timer for the Busy signal, or we'll get a time-out the first time we check the state of the shutter, for good measure, go back 'delay' time into the past
    changedTime_ = GetCurrentMMTime();

    // Gate Closed Position
    //ret = CreateIntegerProperty(MM::g_Keyword_Closed_Position, 0, false);
   //if (ret != DEVICE_OK)
        //return ret;

    if (PWProp_.Handle < 0)
        return DEVICE_NOT_CONNECTED;

    if (POAOpenPW(PWProp_.Handle) != PW_OK)
    {
        return DEVICE_NOT_CONNECTED;
    }

    // create default positions and labels
    const int bufSize = 128;
    char buf[bufSize];
    for (int i = 0; i < PWProp_.PositionCount; i++)
    {
        snprintf(buf, bufSize, "position-%d", i);
        SetPositionLabel(i, buf);
        //snprintf(buf, bufSize, "%ld", i);
        //AddAllowedValue(MM::g_Keyword_Closed_Position, buf);
    }

    // State
    // -----
    POAGetCurrentPosition(PWProp_.Handle, &position_);

    CPropertyAction* pAct = new CPropertyAction(this, &POAFilterWheel::OnState);
    ret = CreateIntegerProperty(MM::g_Keyword_State, position_, false, pAct);
    SetPropertyLimits(MM::g_Keyword_State, 0, (PWProp_.PositionCount - 1));
    if (ret != DEVICE_OK)
        return ret;

    // Label
    // -----
    pAct = new CPropertyAction(this, &CStateBase::OnLabel);
    ret = CreateStringProperty(MM::g_Keyword_Label, "", false, pAct);
    if (ret != DEVICE_OK)
        return ret;

    ret = UpdateStatus();
    if (ret != DEVICE_OK)
        return ret;

    initialized_ = true;

    return DEVICE_OK;
}

bool POAFilterWheel::Busy()
{
    if (isBusyWait) //Prevent frequent queries
    {
        MM::MMTime interval = GetCurrentMMTime() - changedTime_;
        if (interval < MM::MMTime::fromMs(200))
            return true;
    }

    PWState pw_state;
    POAGetPWState(PWProp_.Handle, &pw_state);
    if (pw_state == PW_STATE_MOVING)
    {
        changedTime_ = GetCurrentMMTime();
        isBusyWait = true;
        return true;
    }
    else
    {
        isBusyWait = false;
        return false;
    }
}


int POAFilterWheel::Shutdown()
{
    if (initialized_)
    {
        initialized_ = false;
    }

    POAClosePW(PWProp_.Handle);
    return DEVICE_OK;
}

///////////////////////////////////////////////////////////////////////////////
// Action handlers
///////////////////////////////////////////////////////////////////////////////

int POAFilterWheel::OnState(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        POAGetCurrentPosition(PWProp_.Handle, &position_);
        pProp->Set((long)position_);
        // nothing to do, let the caller to use cached property
    }
    else if (eAct == MM::AfterSet)
    {
        // Set timer for the Busy signal
        changedTime_ = GetCurrentMMTime();

        long pos;
        pProp->Get(pos);
        if (pos >= PWProp_.PositionCount || pos < 0)
        {
            return ERR_UNKNOWN_POSITION;
        }
        else
        {
            PWErrors err = POAGotoPosition(PWProp_.Handle, (int)pos);
            if (err != PW_OK)
            {
                return DEVICE_ERR;
            }
        }

        position_ = pos;
    }

    return DEVICE_OK;
}

int POAFilterWheel::OnSelectPWIndex(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    std::string str;
    if (eAct == MM::AfterSet)
    {
        pProp->Get(str);
        for (int i = 0; i < connectPWsName_.size(); i++)
        {
            if (str == connectPWsName_.at(i))
            {
                POAGetPWProperties(i, &PWProp_);
                selectPWIndex_ = i;
                selectPWName_ = connectPWsName_.at(i);
                break;
            }
        }
    }
    else if (eAct == MM::BeforeGet)
    {
        pProp->Set(selectPWName_.c_str());
    }

    return DEVICE_OK;
}

unsigned long POAFilterWheel::GetNumberOfPositions() const
{
    return PWProp_.PositionCount;
}


///////////////////////////////////////////////////////////////////////////////
// Demo hub
///////////////////////////////////////////////////////////////////////////////
int DemoHub::Initialize()
{
    initialized_ = true;

    return DEVICE_OK;
}

int DemoHub::DetectInstalledDevices()
{
    ClearInstalledDevices();

    // make sure this method is called before we look for available devices
    InitializeModuleData();

    char hubName[MM::MaxStrLength];
    GetName(hubName); // this device name
    for (unsigned i = 0; i < GetNumberOfDevices(); i++)
    {
        char deviceName[MM::MaxStrLength];
        bool success = GetDeviceName(i, deviceName, MM::MaxStrLength);
        if (success && (strcmp(hubName, deviceName) != 0))
        {
            MM::Device* pDev = CreateDevice(deviceName);
            AddInstalledDevice(pDev);
        }
    }
    return DEVICE_OK;
}

void DemoHub::GetName(char* pName) const
{
    CDeviceUtils::CopyLimitedString(pName, g_HubDeviceName);
}