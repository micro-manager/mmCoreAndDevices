///////////////////////////////////////////////////////////////////////////////
// FILE:          NikonKsCam.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Device adapter for Nikon DS-Ri2 and DS-Qi2
//                Based on several other DeviceAdapters,
//				  especially ThorLabsUSBCamera
//
// AUTHOR:        Andrew Gomella, andrewgomella@gmail.com, 2015
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

#include "NikonKsCam.h"
#include "ModuleInterface.h"
#include <cstdio>
#include <string>
#include <sstream>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>

#define KSCAM_BUFFER_NUM       5

using namespace std;

// External names used used by the rest of the system
const char* g_CameraDeviceName = "NikonKsCam";

// Feature names that aren't dynamically set
const char* g_RoiPositionX = "ROI Position X";
const char* g_RoiPositionY = "ROI Position Y";
const char* g_TriggerFrameCt = "Trigger Frame Count";
const char* g_TriggerFrameDelay = "Trigger Frame Delay";
const char* g_MeteringAreaLeft = "Metering Area Left";
const char* g_MeteringAreaTop = "Metering Area Top";
const char* g_MeteringAreaWidth = "Metering Area Width";
const char* g_MeteringAreaHeight = "Metering Area Height";

///////////////////////////////////////////////////////////////////////////////
// Exported MMDevice API
///////////////////////////////////////////////////////////////////////////////

MODULE_API void InitializeModuleData()
{
    RegisterDevice(g_CameraDeviceName, MM::CameraDevice, "Nikon Ks Camera Adapter");
}

MODULE_API MM::Device* CreateDevice(const char* deviceName)
{
    if (deviceName == 0)
        return 0;

    // decide which device class to create based on the deviceName parameter
    if (strcmp(deviceName, g_CameraDeviceName) == 0)
    {
        // create camera
        return new NikonKsCam();
    }

    // ...supplied name not recognized
    return 0;
}

MODULE_API void DeleteDevice(MM::Device* pDevice)
{
    delete pDevice;
}

///////////////////////////////////////////////////////////////////////////////
// NikonKsCam implementation
///////////////////////////////////////////////////////////////////////////////

/* Pointer for KsCam callback function to use */
NikonKsCam* g_pDlg = NULL;

/* Camera event callback function */
FCAM_EventCallback EventCallback(const lx_uint32 eventCameraHandle,
                                 CAM_Event* pEvent, void* pTransData)
{
    g_pDlg->DoEvent(eventCameraHandle, pEvent, pTransData);
    return nullptr;
}

/* Camera event callback function handler */
void NikonKsCam::DoEvent(const lx_uint32 eventCameraHandle, CAM_Event* pEvent, void* /*pTransData */)
{
    lx_uint32 result = LX_OK;
    std::ostringstream os;
    char strWork[CAM_FEA_COMMENT_MAX];

    if ( eventCameraHandle != this->cameraHandle_ )
    {
        LogMessage("DoEvent Error, Invalid Camera Handle \n");
        return;
    }
    switch(pEvent->eEventType)
    {
    case    ecetImageReceived:
        os << "ImageRecieved Frameno=" << pEvent->stImageReceived.uiFrameNo << " uiRemain= " << pEvent->stImageReceived.uiRemained << endl;
        LogMessage(os.str().c_str());
        /* Signal frameDoneEvent_ so we know an image has been recieved */
        frameDoneEvent_.Set();
        break;
    case    ecetFeatureChanged:
        strcpy(strWork, ConvFeatureIdToName(pEvent->stFeatureChanged.uiFeatureId));
        os << "Feature Changed Callback: " << strWork << endl;
        LogMessage(os.str().c_str());
        /* Update vectFeatureValue_ to have the new stVariant */
        vectFeatureValue_.pstFeatureValue[mapFeatureIndex_[pEvent->stFeatureChanged.uiFeatureId]].stVariant
            = pEvent->stFeatureChanged.stVariant;
        /* Update FeatureDesc because it may have changed */
        result = CAM_GetFeatureDesc(cameraHandle_, pEvent->stFeatureChanged.uiFeatureId,
                                    featureDesc_[mapFeatureIndex_[pEvent->stFeatureChanged.uiFeatureId]]);
        if (result != LX_OK) {
            LogMessage("Error updating featuredesc after callback");
        }
        switch (pEvent->stFeatureChanged.uiFeatureId) {
        case eExposureTime:
            UpdateProperty((char*)MM::g_Keyword_Exposure);
            break;
        default:
            UpdateProperty(strWork);
        }
        break;
    case    ecetExposureEnd:
        break;
    case    ecetTriggerReady:
        break;
    case    ecetDeviceCapture:
        break;
    case    ecetAeStay:
        break;
    case    ecetAeRunning:
        break;
    case    ecetAeDisable:
        break;
    case    ecetTransError:
        os << "Transmit Error T" << pEvent->stTransError.uiTick << " UsbErrorCode " << pEvent->stTransError.uiUsbErrorCode << " DriverErrorCode "
           << pEvent->stTransError.uiDriverErrorCode << " RecievedSize " << pEvent->stTransError.uiReceivedSize << " SettingSize "
           << pEvent->stTransError.uiReceivedSize << endl;
        LogMessage(os.str().c_str());
        break;
    case    ecetBusReset:
        os << "Bus Reset Error Code:" << pEvent->stBusReset.eBusResetCode << " ImageCleared: " << pEvent->stBusReset.bImageCleared << endl;
        LogMessage(os.str().c_str());
        break;
    default:
        LogMessage("Error: Unknown Event");
        break;
    }
}

/**
* NikonKsCam constructor.
* Setup default all variables and create device properties required to exist
* before intialization. In this case, no such properties were required. All
* properties will be created in the Initialize() method.
*
* As a general guideline Micro-Manager devices do not access hardware in the
* the constructor. We should do as little as possible in the constructor and
* perform most of the initialization in the Initialize() method.
*/
NikonKsCam::NikonKsCam() :
    CCameraBase<NikonKsCam>(),
    featureDesc_(NULL),
    isOpened_(false),
    isInitialized_(false),
    isRi2_(false),
    deviceIndex_(0),
    deviceCount_(0),
    bitDepth_(8),
    byteDepth_(0),
    cameraHandle_(0),
    ptrEventData_(nullptr),
    imageWidth_(0),
    imageHeight_(0),
    numComponents_(1),
    sequenceStartTime_(0),
    roiX_(0),
    roiY_(0),
    roiWidth_(0),
    roiHeight_(0),
    binSize_(1),
    readoutUs_(0.0),
    framesPerSecond_(0.0),
    cameraBuf_(nullptr),
    cameraBufId_(0)
{
    // call the base class method to set-up default error codes/messages
    InitializeDefaultErrorMessages();
    readoutStartTime_ = GetCurrentMMTime();
    thd_ = new MySequenceThread(this);

    /*Initialize image data buffer*/
    image_.pDataBuffer = new BYTE[(4908 * (3264 + 1) * 3)];

    // Create a pre-initialization property and list all the available cameras
    // Demo cameras will be included in the list
    auto pAct = new CPropertyAction(this, &NikonKsCam::OnCameraSelection);
    CreateProperty("Camera", "(None)", MM::String, false, pAct, true);
    SearchDevices();
}

/* Search for cameras and populate the pre-init camera selection list */
void NikonKsCam::SearchDevices()
{
    auto camName = new char[CAM_NAME_MAX];
    auto result = LX_OK;
    CAM_Device* ptrDeviceTemp;

    if (deviceCount_ > 0)
    {
        result = CAM_CloseDevices();
        if (result != LX_OK)
        {
            LogMessage("Close device error!");
            return;
        }

        ptrDeviceTemp = nullptr;
        deviceCount_ = 0;
    }

    result = CAM_OpenDevices(deviceCount_, &ptrDeviceTemp);
    if (result != LX_OK)
    {
        LogMessage("Error calling CAM_OpenDevices().");
        return;
    }

    if (deviceCount_ < 0)
    {
        LogMessage("Device is not connected");
        return;
    }

    for (size_t i = 0; i < deviceCount_; i++)
    {
        wcstombs(camName, reinterpret_cast<wchar_t const*>(ptrDeviceTemp[i].wszCameraName), CAM_NAME_MAX);
        AddAllowedValue("Camera", camName);
    }
}

/**
* NikonKsCam destructor.
* If this device used as intended within the Micro-Manager system,
* Shutdown() will be always called before the destructor. But in any case
* we need to make sure that all resources are properly released even if
* Shutdown() was not called.
*/
NikonKsCam::~NikonKsCam()
{
    StopSequenceAcquisition();
    delete thd_;
}

/**
* Obtains device name.
* Required by the MM::Device API.
*/
void NikonKsCam::GetName(char* name) const
{
    // Return the name used to refer to this device adapter
    CDeviceUtils::CopyLimitedString(name, g_CameraDeviceName);
}

/**
* Initializes the hardware.
* Required by the MM::Device API.
* Typically we access and initialize hardware at this point.
* Device properties are typically created here as well, except
* the ones we need to use for defining initialization parameters.
* Such pre-initialization properties are created in the constructor.
* (This device does not have any pre-initialization properties)
*/
int NikonKsCam::Initialize()
{

    if (isInitialized_)
    {
        return DEVICE_OK;
    }

    auto camName = new char[CAM_NAME_MAX];
    char strWork[CAM_FEA_COMMENT_MAX];
    auto result = LX_OK;
    CAM_Device* ptrDeviceTemp;
    lx_uint32 i;
    lx_wchar szError[CAM_ERRMSG_MAX];
    string camID_string = camID_;
    ostringstream os;


    /* Rescan device list */
    result = CAM_OpenDevices(deviceCount_, &ptrDeviceTemp);
    if (result != LX_OK)
    {
        LogMessage("Error calling CAM_OpenDevices().");
    }

    os << "Opening Camera: " << camID_ << endl;
    LogMessage(os.str().c_str());

    /* Find camera in list */
    for (i = 0; i < deviceCount_; i++)
    {
        wcstombs(strWork, reinterpret_cast<wchar_t const*>(ptrDeviceTemp[i].wszCameraName), CAM_NAME_MAX);

        if (camID_string.compare(strWork) == 0)
        {
            deviceIndex_ = i;
            break;
        }
    }

    ZeroMemory(szError, sizeof(szError));

    /* Open the camera using the deviceIndex number */
    result = CAM_Open(deviceIndex_, cameraHandle_, (sizeof(szError) / sizeof((szError)[0])), szError);
    if (result != LX_OK)
    {
        os << "Error calling CAM_Open():" << szError << " DeviceIndex: " << deviceIndex_ << endl;
        LogMessage(os.str().c_str());
        throw DEVICE_ERR;
    }

    /* Connection was succesful, set Device info so we can use it later */
    this->device_ = ptrDeviceTemp[this->deviceIndex_];
    this->isOpened_ = TRUE;

    /* Setup callback function for event notification and handling */
    result = CAM_SetEventCallback(cameraHandle_, reinterpret_cast<FCAM_EventCallback>(EventCallback),
                                  ptrEventData_);
    if (result != LX_OK)
    {
        LogMessage("Error calling CAM_SetEventCallback().");
        throw DEVICE_ERR ;
    }

    /* Needed to setup callback access */
    g_pDlg = this;

    /* Get all feature values and descriptions */
    GetAllFeatures();

    GetAllFeaturesDesc();

    /* Ri2 has additional features, check if camera is Ri2 */
    switch (device_.eCamDeviceType)
    {
    case eRi2:
        isRi2_ = TRUE;
        break;
    case eQi2:
        break;
    case eRi2_Simulator:
        isRi2_ = TRUE;
        break;
    case eQi2_Simulator:
        break;
    default:
        break;
    }

    // set property list
    // -----------------
    // CameraName
    wcstombs(camName, reinterpret_cast<wchar_t const*>(device_.wszCameraName), CAM_NAME_MAX);
    auto nRet = CreateProperty(MM::g_Keyword_CameraName, camName, MM::String, true);
    assert(nRet == DEVICE_OK);

    os.str("");
    // Read Only Camera Info
    os << device_.uiSerialNo << endl;
    nRet |= CreateProperty("Serial Number", os.str().c_str(), MM::String, true);
    wcstombs(strWork, reinterpret_cast<wchar_t const*>(device_.wszFwVersion), CAM_VERSION_MAX);
    nRet |= CreateProperty("FW Version", strWork, MM::String, true);
    wcstombs(strWork, reinterpret_cast<wchar_t const*>(device_.wszFpgaVersion), CAM_VERSION_MAX);
    nRet |= CreateProperty("FPGA Version", strWork, MM::String, true);
    wcstombs(strWork, reinterpret_cast<wchar_t const*>(device_.wszUsbDcVersion), CAM_VERSION_MAX);
    nRet |= CreateProperty("USBDc Version", strWork, MM::String, true);
    wcstombs(strWork, reinterpret_cast<wchar_t const*>(device_.wszUsbVersion), CAM_VERSION_MAX);
    nRet |= CreateProperty("USB Version", strWork, MM::String, true);
    wcstombs(strWork, reinterpret_cast<wchar_t const*>(device_.wszDriverVersion), CAM_VERSION_MAX);
    nRet |= CreateProperty("Driver Version", strWork, MM::String, true);
    assert(nRet == DEVICE_OK);

    //Binning is handled by the Image Format setting and this camera only allows hardware bin 3
    nRet |= CreateProperty(MM::g_Keyword_Binning, "1", MM::Integer, false);
    nRet |= AddAllowedValue(MM::g_Keyword_Binning, "1");
    assert(nRet == DEVICE_OK);

    //Exposure
    auto* pAct = new CPropertyAction(this, &NikonKsCam::OnExposureTime);
    nRet = CreateKsProperty(eExposureTime, pAct);
    assert(nRet == DEVICE_OK);

    //Camera gain
    pAct = new CPropertyAction(this, &NikonKsCam::OnHardwareGain);
    nRet = CreateKsProperty(eGain, pAct);
    assert(nRet == DEVICE_OK);

    //Exposure Limit (for auto exposure)
    pAct = new CPropertyAction(this, &NikonKsCam::OnExposureTimeLimit);
    nRet = CreateKsProperty(eExposureTimeLimit, pAct);
    assert(nRet == DEVICE_OK);

    //Gain Limit (for auto exposure)
    pAct = new CPropertyAction(this, &NikonKsCam::OnGainLimit);
    nRet = CreateKsProperty(eGainLimit, pAct);
    assert(nRet == DEVICE_OK);

    //Brightness
    pAct = new CPropertyAction(this, &NikonKsCam::OnBrightness);
    nRet = CreateKsProperty(eBrightness, pAct);
    assert(nRet == DEVICE_OK);

    //Create properties that are only found on Ri2
    //(All pertain to color post-processing)
    if (isRi2_ == TRUE)
    {
        pAct = new CPropertyAction(this, &NikonKsCam::OnSharpness);
        nRet = CreateKsProperty(eSharpness, pAct);
        assert(nRet == DEVICE_OK);

        pAct = new CPropertyAction(this, &NikonKsCam::OnHue);
        nRet = CreateKsProperty(eHue, pAct);
        assert(nRet == DEVICE_OK);

        pAct = new CPropertyAction(this, &NikonKsCam::OnSaturation);
        nRet = CreateKsProperty(eSaturation, pAct);
        assert(nRet == DEVICE_OK);

        pAct = new CPropertyAction(this, &NikonKsCam::OnWhiteBalanceRed);
        nRet = CreateKsProperty(eWhiteBalanceRed, pAct);
        assert(nRet == DEVICE_OK);

        pAct = new CPropertyAction(this, &NikonKsCam::OnWhiteBalanceBlue);
        nRet = CreateKsProperty(eWhiteBalanceBlue, pAct);
        assert(nRet == DEVICE_OK);

        pAct = new CPropertyAction(this, &NikonKsCam::OnPresets);
        nRet = CreateKsProperty(ePresets, pAct);
        assert(nRet == DEVICE_OK);
    }

    //ROI Position (range is subject to change depending on format!)
    auto* featureDesc = &featureDesc_[mapFeatureIndex_[eRoiPosition]];
    pAct = new CPropertyAction(this, &NikonKsCam::OnRoiX);
    nRet = CreateProperty(g_RoiPositionX, "", MM::Integer, false, pAct);
    assert(nRet == DEVICE_OK);

    pAct = new CPropertyAction(this, &NikonKsCam::OnRoiY);
    nRet = CreateProperty(g_RoiPositionY, "", MM::Integer, false, pAct);
    assert(nRet == DEVICE_OK);
    SetROILimits();

    //Trigger Options
    featureDesc = &featureDesc_[mapFeatureIndex_[eTriggerOption]];
    pAct = new CPropertyAction(this, &NikonKsCam::OnTriggerFrame);
    nRet = CreateProperty(g_TriggerFrameCt, "", MM::Integer, false, pAct);
    nRet |= SetPropertyLimits(g_TriggerFrameCt, featureDesc->stTriggerOption.stRangeFrameCount.stMin.ui32Value, featureDesc->stTriggerOption.stRangeFrameCount.stMax.ui32Value);
    assert(nRet == DEVICE_OK);

    pAct = new CPropertyAction(this, &NikonKsCam::OnTriggerDelay);
    nRet = CreateProperty(g_TriggerFrameDelay, "", MM::Integer, false, pAct);
    nRet |= SetPropertyLimits(g_TriggerFrameDelay, featureDesc->stTriggerOption.stRangeDelayTime.stMin.i32Value, featureDesc->stTriggerOption.stRangeDelayTime.stMax.i32Value);
    assert(nRet == DEVICE_OK);

    //Metering Area (range is subject to change depending on format!)
    pAct = new CPropertyAction(this, &NikonKsCam::OnMeteringAreaLeft);
    nRet = CreateProperty(g_MeteringAreaLeft, "", MM::Integer, false, pAct);
    pAct = new CPropertyAction(this, &NikonKsCam::OnMeteringAreaTop);
    nRet |= CreateProperty(g_MeteringAreaTop, "", MM::Integer, false, pAct);
    pAct = new CPropertyAction(this, &NikonKsCam::OnMeteringAreaWidth);
    nRet |= CreateProperty(g_MeteringAreaWidth, "", MM::Integer, false, pAct);
    pAct = new CPropertyAction(this, &NikonKsCam::OnMeteringAreaHeight);
    nRet |= CreateProperty(g_MeteringAreaHeight, "", MM::Integer, false, pAct);
    assert(nRet == DEVICE_OK);
    SetMeteringAreaLimits();

    //List Settings
    //Image Format
    pAct = new CPropertyAction(this, &NikonKsCam::OnImageFormat);
    nRet = CreateKsProperty(eFormat, pAct);
    assert(nRet == DEVICE_OK);

    pAct = new CPropertyAction(this, &NikonKsCam::OnCaptureMode);
    nRet = CreateKsProperty(eCaptureMode, pAct);
    assert(nRet == DEVICE_OK);

    pAct = new CPropertyAction(this, &NikonKsCam::OnMeteringMode);
    nRet = CreateKsProperty(eMeteringMode, pAct);
    assert(nRet == DEVICE_OK);

    //Trigger Mode
    pAct = new CPropertyAction(this, &NikonKsCam::OnTriggerMode);
    nRet = CreateKsProperty(eTriggerMode, pAct);
    assert(nRet == DEVICE_OK);

    //Exposure Mode
    pAct = new CPropertyAction(this, &NikonKsCam::OnExposureMode);
    nRet = CreateKsProperty(eExposureMode, pAct);
    assert(nRet == DEVICE_OK);

    //Exposure Bias
    pAct = new CPropertyAction(this, &NikonKsCam::OnExposureBias);
    nRet = CreateKsProperty(eExposureBias, pAct);
    assert(nRet == DEVICE_OK);

    //SignalExposureEnd
    pAct = new CPropertyAction(this, &NikonKsCam::OnSignalExposureEnd);
    nRet = CreateKsProperty(eSignalExposureEnd, pAct);
    assert(nRet == DEVICE_OK);

    //SignalTriggerReady
    pAct = new CPropertyAction(this, &NikonKsCam::OnSignalTriggerReady);
    nRet = CreateKsProperty(eSignalTriggerReady, pAct);
    assert(nRet == DEVICE_OK);

    //SignalDeviceCapture
    pAct = new CPropertyAction(this, &NikonKsCam::OnSignalDeviceCapture);
    nRet = CreateKsProperty(eSignalDeviceCapture, pAct);
    assert(nRet == DEVICE_OK);

    //ExposureOutput
    pAct = new CPropertyAction(this, &NikonKsCam::OnExposureOutput);
    nRet = CreateKsProperty(eExposureOutput, pAct);
    assert(nRet == DEVICE_OK);

    // setup the buffer
    // ----------------
    UpdateImageSettings();

    // synchronize all properties
    // --------------------------
    nRet = DEVICE_OK;
    nRet = UpdateStatus();
    if (nRet != DEVICE_OK)
        return nRet;

    isInitialized_ = true;

    return DEVICE_OK;
}

/* Create MM Property for a given FeatureId */
int NikonKsCam::CreateKsProperty(lx_uint32 FeatureId, CPropertyAction *pAct)
{
    auto    featureIndex = mapFeatureIndex_[FeatureId];
    auto*	featureValue = &vectFeatureValue_.pstFeatureValue[featureIndex];
    auto*   featureDesc = &featureDesc_[featureIndex];
    char	strWork[50];
    const char*	strTitle;
    auto nRet = DEVICE_OK;


    /* strTitle is readable name of feature */
    strTitle = ConvFeatureIdToName(featureValue->uiFeatureId);

    switch (featureDesc_[featureIndex].eFeatureDescType) {
    case edesc_Range:
        switch (featureValue->stVariant.eVarType) {
        case	evrt_int32:
            nRet |= CreateProperty(strTitle, "", MM::Integer, false, pAct);
            nRet |= SetPropertyLimits(strTitle, featureDesc->stRange.stMin.i32Value, featureDesc->stRange.stMax.i32Value);
            break;
        case	evrt_uint32:
            if(featureValue->uiFeatureId == eExposureTime)
            {
                strTitle = const_cast<char*>(MM::g_Keyword_Exposure);
                CreateProperty(strTitle, "", MM::Float, false, pAct);
                SetPropertyLimits(strTitle, featureDesc->stRange.stMin.ui32Value/1000, featureDesc->stRange.stMax.ui32Value/1000);
            }
            else if (featureValue->uiFeatureId == eExposureTimeLimit)
            {
                CreateProperty(strTitle, "", MM::Float, false, pAct);
                SetPropertyLimits(strTitle, featureDesc->stRange.stMin.ui32Value/1000, featureDesc->stRange.stMax.ui32Value/1000);
            }
            else
            {
                CreateProperty(strTitle, "", MM::Integer, false, pAct);
                SetPropertyLimits(strTitle, featureDesc->stRange.stMin.ui32Value, featureDesc->stRange.stMax.ui32Value);
            }
            break;
        default:
            break;
        }
        break;
    case edesc_Area:
        //Metering Area handled in initialize
        break;
    case edesc_Position:
        //ROI position handled in initialize
        break;
    case edesc_TriggerOption:
        //TriggerOption handled in initialize
        break;
    case edesc_ElementList:
        nRet |= CreateProperty(strTitle, "", MM::String, false, pAct);
        for (lx_uint32 Index = 0; Index < featureDesc->uiListCount; Index++)
        {
            wcstombs(strWork, reinterpret_cast<wchar_t const*>(featureDesc->stElementList[Index].wszComment), CAM_FEA_COMMENT_MAX);
            /* OnePushAE is not allowed to be set by User */
            if (strcmp(strWork, "OnePushAE") != 0)
                nRet |= AddAllowedValue(strTitle, strWork);
        }
        break;
    case edesc_FormatList:
        nRet |= CreateProperty(strTitle, "", MM::String, false, pAct);
        for (lx_uint32 formatIndex = 0; formatIndex < featureDesc->uiListCount; formatIndex++)
        {
            wcstombs(strWork, reinterpret_cast<wchar_t const*>(featureDesc->stFormatList[formatIndex].wszComment), CAM_FEA_COMMENT_MAX);
            nRet |= AddAllowedValue(strTitle, strWork);
        }
        break;
    case edesc_unknown:
        break;
    default:
        break;
    }

    return nRet;
}

/* convert FeatureId to readable name */
const char* NikonKsCam::ConvFeatureIdToName(const lx_uint32 featureId)
{
    lx_uint32       i;
    auto* featureName= new char[30];

    for (i = 0;; i++)
    {
        if (stFeatureNameRef[i].uiFeatureId ==  eUnknown || stFeatureNameRef[i].uiFeatureId == featureId)
        {
            break;
        }
    }
    wcstombs(featureName, (wchar_t const *)stFeatureNameRef[i].wszName, 30);
    return featureName;
}

/* This function populates vectFeatureValue_ with all features */
void NikonKsCam::GetAllFeatures()
{
    auto result = LX_OK;

    Free_Vector_CAM_FeatureValue(vectFeatureValue_);

    vectFeatureValue_.uiCapacity = CAM_FEA_CAPACITY;
    vectFeatureValue_.pstFeatureValue = new
    CAM_FeatureValue[vectFeatureValue_.uiCapacity];

    if ( !vectFeatureValue_.pstFeatureValue )
    {
        LogMessage("GetAllFeatures() Memory allocation error. \n");
        return;
    }

    result = CAM_GetAllFeatures(cameraHandle_, vectFeatureValue_);
    if ( result != LX_OK )
    {
        LogMessage("GetAllFeatures() error. \n");
        return;
    }

    if ( vectFeatureValue_.uiCountUsed == 0 )
    {
        LogMessage("Error: GetAllFeatures() returned no features.\n");
        return;
    }
    return;
}

/* This function creates the feature map and calls featureChanged for every feature. populates all featureDesc_ */
void NikonKsCam::GetAllFeaturesDesc()
{
    lx_uint32   uiFeatureId, i;

    mapFeatureIndex_.clear();

    featureDesc_ = new CAM_FeatureDesc[vectFeatureValue_.uiCountUsed];
    if ( !featureDesc_ )
    {
        LogMessage("GetAllFeaturesDesc memory allocate Error.[] \n");
        return;
    }

    /* This loops through the total number of features on the device */
    for( i=0; i<vectFeatureValue_.uiCountUsed; i++ )
    {
        uiFeatureId = vectFeatureValue_.pstFeatureValue[i].uiFeatureId;
        /* map the FeatureId to i */
        mapFeatureIndex_.insert(std::make_pair(uiFeatureId, i));
        auto result = CAM_GetFeatureDesc(cameraHandle_, uiFeatureId, featureDesc_[i]);
        if (result != LX_OK)
        {
            LogMessage("CAM_GetFeatureDesc Error");
            return;
        }
    }
}

/* This function calls SetFeature for a given uiFeatureId */
void NikonKsCam::SetFeature(lx_uint32 uiFeatureId)
{
    auto result = LX_OK;
    lx_uint32                   index;
    Vector_CAM_FeatureValue     vectFeatureValue;

    /* Prepare the vectFeatureValue structure to use in the CAM_setFeatures command */
    vectFeatureValue.uiCountUsed = 1;
    vectFeatureValue.uiCapacity = 1;
    vectFeatureValue.uiPauseTransfer = 0;
    vectFeatureValue.pstFeatureValue = new CAM_FeatureValue[1];
    if (vectFeatureValue.pstFeatureValue == nullptr)
    {
        LogMessage("Error allocating memory vecFeatureValue.");
        return;
    }

    index = mapFeatureIndex_[uiFeatureId];
    vectFeatureValue.pstFeatureValue[0] = vectFeatureValue_.pstFeatureValue[index];

    result = CAM_SetFeatures(cameraHandle_, vectFeatureValue);
    Free_Vector_CAM_FeatureValue(vectFeatureValue);
    if (result != LX_OK)
    {
        LogMessage("CAM_SetFeatures Error");
        GetAllFeatures();
        return;
    }

    LogMessage("SetFeature() Success");
    return;
}

/* This function calls CAM_Command */
void NikonKsCam::Command(const lx_wchar* wszCommand)
{
    auto result = LX_OK;

    if (!_wcsicmp(reinterpret_cast<wchar_t const *>(wszCommand), CAM_CMD_START_FRAMETRANSFER))
    {
        CAM_CMD_StartFrameTransfer      stCmd;

        /* Start frame transfer */
        stCmd.uiImageBufferNum = KSCAM_BUFFER_NUM;
        result = CAM_Command(cameraHandle_, CAM_CMD_START_FRAMETRANSFER, &stCmd);
        if (result != LX_OK)
        {
            LogMessage("CAM_Command start frame transfer error");
            return;
        }
    }
    else
    {
        result = CAM_Command(cameraHandle_, wszCommand, nullptr);
        if (result != LX_OK)
        {
            LogMessage("CAM_Command error");
            return;
        }
    }
    return;
}

/* This should be called at initialization after features have been received, as well as whenever imgFormat is changed
/* it should update the image buffer to have the proper width/height/depth as well as update relevant Properties */
void NikonKsCam::UpdateImageSettings()
{
    auto result = LX_OK;

    switch(vectFeatureValue_.pstFeatureValue[mapFeatureIndex_[eFormat]].stVariant.stFormat.eColor)
    {
    case ecfcUnknown:
        LogMessage("Error: unknown image type.");
        break;
    case ecfcRgb24:
        numComponents_ = 4;
        byteDepth_ = 4;
        bitDepth_ = 8;
        color_ = true;
        break;
    case ecfcYuv444:
        numComponents_ = 4;
        byteDepth_ = 4;
        bitDepth_ = 8;
        color_ = true;
        break;
    case ecfcMono16:
        numComponents_ = 1;
        byteDepth_ = 2;
        bitDepth_ = 16;
        color_ = false;
        break;
    }

    switch(vectFeatureValue_.pstFeatureValue[mapFeatureIndex_[eFormat]].stVariant.stFormat.eMode)
    {
    case ecfmUnknown:
        LogMessage("Error: unknown image resolution.");
        break;
    case ecfm4908x3264:
        imageWidth_=4908;
        imageHeight_=3264;
        break;
    case ecfm2454x1632:
        imageWidth_=2454;
        imageHeight_=1632;
        break;
    case ecfm1636x1088:
        imageWidth_=1636;
        imageHeight_=1088;
        break;
    case ecfm818x544:
        imageWidth_=818;
        imageHeight_=544;
        break;
    case ecfm1608x1608:
        imageWidth_=1608;
        imageHeight_=1608;
        break;
    case ecfm804x804:
        imageWidth_=804;
        imageHeight_=804;
        break;
    case ecfm536x536:
        imageWidth_=536;
        imageHeight_=536;
        break;
    }

    /* Update the buffer to have the proper width height and depth */
    img_.Resize(imageWidth_, imageHeight_, byteDepth_);

    /* Update frameSize_ so we know how to size image_ in the GetImage() calls to driver*/
    result = CAM_Command(cameraHandle_, CAM_CMD_GET_FRAMESIZE, &frameSize_);
    if ( result != LX_OK )
    {
        LogMessage("GetFrameSize Error.");
    }

}

/* Update ROI Property x and y limits */
void NikonKsCam::SetROILimits()
{
    auto result = CAM_GetFeatureDesc(cameraHandle_, eRoiPosition,
                                     featureDesc_[mapFeatureIndex_[eRoiPosition]]);
    if (result != LX_OK)
    {
        LogMessage("CAM_GetFeatureDesc Error");
        return;
    }

    auto roiFeatureDesc = &featureDesc_[mapFeatureIndex_[eRoiPosition]];

    /* If not in an ROI format setting (e.g. full frame), SDK will return min=max=1 */
    /* which will cause an error in micromanager SetPropertyLimits() function */
    /* for now just set min to 0 and max to 1 */
    if (roiFeatureDesc->stPosition.stMin.uiX == roiFeatureDesc->stPosition.stMax.uiX)
    {
        SetPropertyLimits(g_RoiPositionX, 0, 1);
        SetPropertyLimits(g_RoiPositionY, 0, 1);
    }
    else {
        SetPropertyLimits(g_RoiPositionX, roiFeatureDesc->stPosition.stMin.uiX, roiFeatureDesc->stPosition.stMax.uiX);
        SetPropertyLimits(g_RoiPositionY, roiFeatureDesc->stPosition.stMin.uiY, roiFeatureDesc->stPosition.stMax.uiY);
    }
    UpdateProperty(g_RoiPositionX);
    UpdateProperty(g_RoiPositionY);
}

/* Update Metering Area limits */
void NikonKsCam::SetMeteringAreaLimits()
{
    auto result = CAM_GetFeatureDesc(cameraHandle_, eMeteringArea,	featureDesc_[mapFeatureIndex_[eMeteringArea]]);
    if (result != LX_OK)
    {
        LogMessage("CAM_GetFeatureDesc Error");
        return;
    }
    auto featureDesc = &featureDesc_[mapFeatureIndex_[eMeteringArea]];

    SetPropertyLimits(g_MeteringAreaLeft, featureDesc->stArea.stMin.uiLeft, featureDesc->stArea.stMax.uiLeft);
    SetPropertyLimits(g_MeteringAreaTop, featureDesc->stArea.stMin.uiTop, featureDesc->stArea.stMax.uiTop);
    SetPropertyLimits(g_MeteringAreaWidth, featureDesc->stArea.stMin.uiWidth, featureDesc->stArea.stMax.uiWidth);
    SetPropertyLimits(g_MeteringAreaHeight, featureDesc->stArea.stMin.uiHeight, featureDesc->stArea.stMax.uiHeight);

    UpdateProperty(g_MeteringAreaLeft);
    UpdateProperty(g_MeteringAreaTop);
    UpdateProperty(g_MeteringAreaWidth);
    UpdateProperty(g_MeteringAreaHeight);
}

/**
* Shuts down (unloads) the device.
* Required by the MM::Device API.
* Ideally this method will completely unload the device and release all resources.
* Shutdown() may be called multiple times in a row.
* After Shutdown() we should be allowed to call Initialize() again to load the device
* without causing problems.
*/
int NikonKsCam::Shutdown()
{
    auto result = LX_OK;

    if ( this->isOpened_ )
    {
        result = CAM_Close(cameraHandle_);
        if ( result != LX_OK )
        {
            LogMessage("Error Closing Camera.");
        }
        Free_Vector_CAM_FeatureValue(vectFeatureValue_);
        if (featureDesc_ != NULL)
        {
            delete [] featureDesc_;
            featureDesc_ = NULL;
        }
        this->deviceIndex_ = 0;
        this->cameraHandle_ = 0;
        this->isOpened_ = FALSE;
        this->isInitialized_ = FALSE;
        this->isRi2_ = FALSE;
        g_pDlg = nullptr;
    }

    return DEVICE_OK;
}

/**
* Performs exposure and grabs a single image.
* This function should block during the actual exposure and return immediately afterwards
* (i.e., before readout).  This behavior is needed for proper synchronization with the shutter.
* Required by the MM::Camera API.
*/
int NikonKsCam::SnapImage()
{
    //Determine exposureLength so we know a reasonable time to wait for frame arrival
    auto exposureLength = vectFeatureValue_.pstFeatureValue[mapFeatureIndex_[eExposureTime]].stVariant.ui32Value / 1000;
    char buf[MM::MaxStrLength];
    //Determine current trigger mode
    GetProperty(ConvFeatureIdToName(eTriggerMode), buf);

    Command(CAM_CMD_START_FRAMETRANSFER);
    //If in soft trigger mode we need to send the signal to capture.
    if (!strcmp(buf, "Soft"))
        Command(CAM_CMD_ONEPUSH_SOFTTRIGGER);
    //Wait for frameDoneEvent from callback method
    // (time out after exposure length + 100 ms)
    frameDoneEvent_.Wait(exposureLength + 100);
    Command(CAM_CMD_STOP_FRAMETRANSFER);
    GrabFrame();

    return DEVICE_OK;
}

//Call after a frame is recieved to get the image from camera and copy to img_ buffer
void NikonKsCam::GrabFrame()
{
    lx_result           result;
    lx_uint32           uiRemained;

    /* Set image_ buffer to appropriate size */
    image_.uiDataBufferSize = this->frameSize_.uiFrameSize;

    /* Grab the Image */
    result = CAM_GetImage(cameraHandle_, true, image_, uiRemained);
    if (result != LX_OK)
    {
        LogMessage("CAM_GetImage error.");
    }

    if (color_)
        Bgr8ToBGRA8(img_.GetPixelsRW(), (uint8_t*)image_.pDataBuffer, img_.Width(), img_.Height());
    else
        memcpy(img_.GetPixelsRW(), image_.pDataBuffer, img_.Width()*img_.Height()*img_.Depth());

}

//copied from MM dc1394.cpp driver file
//EF: converts bgr image to Micromanager BGRA
// It is the callers responsibility that both src and destination exist
void NikonKsCam::Bgr8ToBGRA8(unsigned char* dest, unsigned char* src, unsigned int width, unsigned int height)
{
    for (register uint64_t i = 0, j = 0; i < (width * height * 3); i += 3, j += 4)
    {
        dest[j] = src[i];
        dest[j + 1] = src[i + 1];
        dest[j + 2] = src[i + 2];
        dest[j + 3] = 0;
    }
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
const unsigned char* NikonKsCam::GetImageBuffer()
{
    auto pB = const_cast<unsigned char*>(img_.GetPixels());
    return pB;
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
* @param x - top-left corner coordinate
* @param y - top-left corner coordinate
* @param xSize - width
* @param ySize - height
*/
/* KsCam only has fixed ROI settings which are handled by the format setting */
int NikonKsCam::SetROI(unsigned /*x*/, unsigned /*y*/, unsigned /*xSize*/, unsigned /*ySize*/)
{
    return DEVICE_UNSUPPORTED_COMMAND;
}

/**
* Returns the actual dimensions of the current ROI.
* Required by the MM::Camera API.
*/
/* KsCam only has fixed ROI settings which are handled by the format setting */
int NikonKsCam::GetROI(unsigned& x, unsigned& y, unsigned& xSize, unsigned& ySize)
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
int NikonKsCam::ClearROI()
{
    roiX_ = 0;
    roiY_ = 0;
    roiWidth_ = 4908; //true for both ri2 and qi2
    roiHeight_ = 3264;

    return DEVICE_OK;
}

/**
* Returns the current exposure setting in milliseconds.
* Required by the MM::Camera API.
*/
double NikonKsCam::GetExposure() const
{
    char buf[MM::MaxStrLength];
    int ret = GetProperty(MM::g_Keyword_Exposure, buf);
    if (ret != DEVICE_OK)
        return 0.0;
    return atof(buf);
}

/**
* Sets exposure in milliseconds.
* Required by the MM::Camera API.
*/
void NikonKsCam::SetExposure(double exp)
{
    SetProperty(MM::g_Keyword_Exposure, CDeviceUtils::ConvertToString(exp));
}

/**
* Returns the current binning factor.
* Required by the MM::Camera API.
*/
int NikonKsCam::GetBinning() const
{
    return DEVICE_OK;
}

/**
* Sets binning factor.
* Required by the MM::Camera API.
*/
int NikonKsCam::SetBinning(int /* binF */)
{
    return DEVICE_OK;
}

int NikonKsCam::GetComponentName(unsigned comp, char* name)
{
    if (comp > 4)
    {
        name = "invalid comp";
        return DEVICE_ERR;
    }

    std::string rgba("RGBA");
    CDeviceUtils::CopyLimitedString(name, &rgba.at(comp));

    return DEVICE_OK;
}


//Sequence functions, mostly copied from other drivers with slight modification
/**
 * Required by the MM::Camera API
 * Please implement this yourself and do not rely on the base class implementation
 * The Base class implementation is deprecated and will be removed shortly
 */
int NikonKsCam::StartSequenceAcquisition(double interval) {
    return StartSequenceAcquisition(LONG_MAX, interval, false);
}

/**
* Stop and wait for the Sequence thread finished
*/
int NikonKsCam::StopSequenceAcquisition()
{
    Command(CAM_CMD_STOP_FRAMETRANSFER);
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
int NikonKsCam::StartSequenceAcquisition(long numImages, double interval_ms, bool stopOnOverflow)
{
    char triggerOffChar[10];
    if (IsCapturing())
        return DEVICE_CAMERA_BUSY_ACQUIRING;

    auto ret = GetCoreCallback()->PrepareForAcq(this);
    if (ret != DEVICE_OK)
        return ret;
    sequenceStartTime_ = GetCurrentMMTime();
    imageCounter_ = 0;

    auto*   featureDesc = &featureDesc_[mapFeatureIndex_[eTriggerMode]];
    wcstombs(triggerOffChar, reinterpret_cast<wchar_t const*>(featureDesc->stElementList[ectmOff].wszComment), CAM_FEA_COMMENT_MAX);

    char triggerMode[MM::MaxStrLength];
    /* if not in trigger mode:off set to trigger mode: off (for "live view") */
    GetProperty(ConvFeatureIdToName(eTriggerMode), triggerMode);
    if (strcmp(triggerMode, triggerOffChar) != 0)
    {
        SetProperty(ConvFeatureIdToName(eTriggerMode), "OFF");
    }

    Command(CAM_CMD_START_FRAMETRANSFER);

    thd_->Start(numImages,interval_ms);
    stopOnOverFlow_ = stopOnOverflow;

    return DEVICE_OK;
}

/*
 * Inserts Image and MetaData into MMCore circular Buffer
 */
int NikonKsCam::InsertImage()
{

    // Image metadata
    Metadata md;
    char label[MM::MaxStrLength];
    this->GetLabel(label);
    md.put("Camera", label);
    // md.put(MM::g_Keyword_Metadata_StartTime, CDeviceUtils::ConvertToString(sequenceStartTime_.getMsec()));
    md.put(MM::g_Keyword_Elapsed_Time_ms, CDeviceUtils::ConvertToString((GetCurrentMMTime() - sequenceStartTime_).getMsec()));
    md.put(MM::g_Keyword_Metadata_ImageNumber, CDeviceUtils::ConvertToString(imageCounter_));

    imageCounter_++;

    MMThreadGuard g(imgPixelsLock_);

    return GetCoreCallback()->InsertImage(this, img_.GetPixels(),
              img_.Width(),
              img_.Height(),
              img_.Depth(),
              md.Serialize().c_str());
}

/*
 * Do actual capturing
 * Called from inside the thread
 */
int NikonKsCam::ThreadRun (void)
{
    MM::MMTime startFrame = GetCurrentMMTime();

    auto exposureLength = vectFeatureValue_.pstFeatureValue[mapFeatureIndex_[eExposureTime]].stVariant.ui32Value / 1000;
    DWORD dwRet = frameDoneEvent_.Wait(exposureLength + 300);//wait up to exposure length + 250 ms

    if (dwRet == MM_WAIT_TIMEOUT)
    {
        LogMessage("Timeout");
        return 0;
    }
    else if (dwRet == MM_WAIT_OK)
    {
        GrabFrame();

        auto ret = InsertImage();

        MM::MMTime frameInterval = GetCurrentMMTime() - startFrame;
        if (frameInterval.getMsec() > 0.0)
            framesPerSecond_ = 1000.0 / frameInterval.getMsec();

        return ret;
    }
    else
    {
        ostringstream os;
        os << "Unknown event status " << dwRet;
        LogMessage(os.str());
        return 0;
    }

    return 1;
};

bool NikonKsCam::IsCapturing() {
    return !thd_->IsStopped();
}

/*
 * called from the thread function before exit
 */
void NikonKsCam::OnThreadExiting() throw()
{
    try
    {
        LogMessage(g_Msg_SEQUENCE_ACQUISITION_THREAD_EXITING);
        GetCoreCallback()?GetCoreCallback()->AcqFinished(this,0):DEVICE_OK;
    }
    catch(...)
    {
        LogMessage(g_Msg_EXCEPTION_IN_ON_THREAD_EXITING, false);
    }
}


MySequenceThread::MySequenceThread(NikonKsCam* pCam)
    :stop_(true)
    ,suspend_(false)
    ,numImages_(default_numImages)
    ,imageCounter_(0)
    ,intervalMs_(default_intervalMS)
    ,startTime_(0)
    ,actualDuration_(0)
    ,lastFrameTime_(0)
    ,camera_(pCam)
{};

MySequenceThread::~MySequenceThread() {};

void MySequenceThread::Stop() {
    MMThreadGuard(this->stopLock_);
    stop_=true;
}

void MySequenceThread::Start(long numImages, double intervalMs)
{
    MMThreadGuard(this->stopLock_);
    MMThreadGuard(this->suspendLock_);
    numImages_=numImages;
    intervalMs_=intervalMs;
    imageCounter_=0;
    stop_ = false;
    suspend_=false;
    activate();
    actualDuration_ = actualDuration_.fromUs(0);
    //startTime_= camera_->GetCurrentMMTime();
    lastFrameTime_ = lastFrameTime_.fromUs(0);
}

bool MySequenceThread::IsStopped() {
    MMThreadGuard(this->stopLock_);
    return stop_;
}

void MySequenceThread::Suspend() {
    MMThreadGuard(this->suspendLock_);
    suspend_ = true;
}

bool MySequenceThread::IsSuspended() {
    MMThreadGuard(this->suspendLock_);
    return suspend_;
}

void MySequenceThread::Resume() {
    MMThreadGuard(this->suspendLock_);
    suspend_ = false;
}

int MySequenceThread::svc(void) throw()
{
    int ret = DEVICE_ERR;

    try
    {
        do
        {
            ret = camera_->ThreadRun();
        } while (DEVICE_OK == ret && !IsStopped() && imageCounter_++ < numImages_-1);

        if (IsStopped())
            camera_->LogMessage("SeqAcquisition interrupted by the user\n");

    } catch(...) {
        camera_->LogMessage(g_Msg_EXCEPTION_IN_THREAD, false);
    }
    stop_=true;
    actualDuration_ = camera_->GetCurrentMMTime() - startTime_;
    camera_->OnThreadExiting();

    return ret;
}


///////////////////////////////////////////////////////////////////////////////
// NikonKsCam Action handlers
///////////////////////////////////////////////////////////////////////////////

/*Pre-initialization property which shows cameras (and simulators) detected */
int NikonKsCam::OnCameraSelection(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        pProp->Set(camID_);
    }
    else if (eAct == MM::AfterSet)
    {
        string value;
        pProp->Get(value);
        strcpy(camID_, value.c_str());
    }
    return DEVICE_OK;
}

/*Nikon Ks cam only has hardware bin 3 which is handled by Image Format setting*/
int NikonKsCam::OnBinning(MM::PropertyBase* /*pProp*/, MM::ActionType /*eAct*/)
{
    return DEVICE_OK;
}

int NikonKsCam::OnExposureTime(MM::PropertyBase* pProp , MM::ActionType eAct)
{
    return OnExposureChange(pProp, eAct, eExposureTime);
}

int NikonKsCam::OnTriggerMode(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    return OnList(pProp, eAct, eTriggerMode);
}

int NikonKsCam::OnExposureMode(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    return OnList(pProp, eAct, eExposureMode);
}

int NikonKsCam::OnExposureBias(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    return OnList(pProp, eAct, eExposureBias);
}

int NikonKsCam::OnCaptureMode(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    return OnList(pProp, eAct, eCaptureMode);
}

int NikonKsCam::OnMeteringMode(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    return OnList(pProp, eAct, eMeteringMode);
}

int NikonKsCam::OnSignalExposureEnd(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    return OnList(pProp, eAct, eSignalExposureEnd);
}

int NikonKsCam::OnSignalTriggerReady(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    return OnList(pProp, eAct, eSignalTriggerReady);
}

int NikonKsCam::OnSignalDeviceCapture(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    return OnList(pProp, eAct, eSignalDeviceCapture);
}

int NikonKsCam::OnExposureOutput(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    return OnList(pProp, eAct, eExposureOutput);
}

int NikonKsCam::OnHardwareGain(MM::PropertyBase* pProp , MM::ActionType eAct)
{
    return OnRange(pProp, eAct, eGain);
}

int NikonKsCam::OnBrightness(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    return OnRange(pProp, eAct, eBrightness);
}

int NikonKsCam::OnSharpness(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    return OnRange(pProp, eAct, eSharpness);
}

int NikonKsCam::OnHue(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    return OnRange(pProp, eAct, eHue);
}

int NikonKsCam::OnSaturation(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    return OnRange(pProp, eAct, eSaturation);
}

int NikonKsCam::OnWhiteBalanceRed(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    return OnRange(pProp, eAct, eWhiteBalanceRed);
}

int NikonKsCam::OnWhiteBalanceBlue(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    return OnRange(pProp, eAct, eWhiteBalanceBlue);
}

int NikonKsCam::OnPresets(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    return OnList(pProp, eAct, ePresets);
}

int NikonKsCam::OnExposureTimeLimit(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    return OnExposureChange(pProp, eAct, eExposureTimeLimit);
}

int NikonKsCam::OnGainLimit(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    return OnRange(pProp, eAct, eGainLimit);
}

int NikonKsCam::OnMeteringAreaLeft(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    long value;
    lx_uint32 index = mapFeatureIndex_[eMeteringArea];
    CAM_FeatureValue*   featureValue = &vectFeatureValue_.pstFeatureValue[index];

    if (eAct == MM::AfterSet)
    {
        pProp->Get(value);
        featureValue->stVariant.stArea.uiLeft = value;
        SetFeature(featureValue->uiFeatureId);
    }

    if (eAct == MM::BeforeGet || eAct == MM::AfterSet)
    {
        pProp->Set((long)vectFeatureValue_.pstFeatureValue[index].stVariant.stArea.uiLeft);
    }
    return DEVICE_OK;
}

int NikonKsCam::OnMeteringAreaTop(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    long value;
    lx_uint32 index = mapFeatureIndex_[eMeteringArea];
    CAM_FeatureValue*   featureValue = &vectFeatureValue_.pstFeatureValue[index];

    if (eAct == MM::AfterSet)
    {
        pProp->Get(value);
        featureValue->stVariant.stArea.uiTop = value;
        SetFeature(featureValue->uiFeatureId);
    }

    if (eAct == MM::BeforeGet || eAct == MM::AfterSet)
    {
        pProp->Set((long)vectFeatureValue_.pstFeatureValue[index].stVariant.stArea.uiTop);
    }
    return DEVICE_OK;
}

int NikonKsCam::OnMeteringAreaWidth(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    long value;
    lx_uint32 index = mapFeatureIndex_[eMeteringArea];
    CAM_FeatureValue*   featureValue = &vectFeatureValue_.pstFeatureValue[index];

    if (eAct == MM::AfterSet)
    {
        pProp->Get(value);
        featureValue->stVariant.stArea.uiWidth = value;
        SetFeature(featureValue->uiFeatureId);
    }

    if (eAct == MM::BeforeGet || eAct == MM::AfterSet)
    {
        pProp->Set((long)vectFeatureValue_.pstFeatureValue[index].stVariant.stArea.uiWidth);
    }
    return DEVICE_OK;
}

int NikonKsCam::OnMeteringAreaHeight(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    lx_uint32 index = mapFeatureIndex_[eMeteringArea];
    CAM_FeatureValue*   featureValue = &vectFeatureValue_.pstFeatureValue[index];

    if (eAct == MM::AfterSet)
    {
        long value;
        pProp->Get(value);
        featureValue->stVariant.stArea.uiHeight = value;
        SetFeature(featureValue->uiFeatureId);
    }

    if (eAct == MM::BeforeGet || eAct == MM::AfterSet)
    {
        pProp->Set((long)vectFeatureValue_.pstFeatureValue[index].stVariant.stArea.uiHeight);
    }
    return DEVICE_OK;
}

int NikonKsCam::OnRoiX(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    lx_uint32 index = mapFeatureIndex_[eRoiPosition];
    CAM_FeatureValue*   featureValue = &vectFeatureValue_.pstFeatureValue[index];

    if (eAct == MM::AfterSet)
    {
        long value;
        pProp->Get(value);
        featureValue->stVariant.stPosition.uiX = value;
        SetFeature(featureValue->uiFeatureId);
    }

    if (eAct == MM::BeforeGet || eAct == MM::AfterSet)
    {
        pProp->Set((long)vectFeatureValue_.pstFeatureValue[index].stVariant.stPosition.uiX);
    }
    return DEVICE_OK;
}

int NikonKsCam::OnRoiY(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    lx_uint32 index = mapFeatureIndex_[eRoiPosition];
    CAM_FeatureValue*   featureValue = &vectFeatureValue_.pstFeatureValue[index];

    if (eAct == MM::AfterSet)
    {
        long value;
        pProp->Get(value);
        featureValue->stVariant.stPosition.uiY = value;
        SetFeature(featureValue->uiFeatureId);
    }

    if (eAct == MM::BeforeGet || eAct == MM::AfterSet)
    {
        pProp->Set((long)vectFeatureValue_.pstFeatureValue[index].stVariant.stPosition.uiY);
    }

    return DEVICE_OK;
}

int NikonKsCam::OnTriggerFrame(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    long value;
    lx_uint32 index = mapFeatureIndex_[eTriggerOption];
    CAM_FeatureValue*   featureValue = &vectFeatureValue_.pstFeatureValue[index];

    if (eAct == MM::BeforeGet)
    {
        pProp->Set((long)vectFeatureValue_.pstFeatureValue[index].stVariant.stTriggerOption.iDelayTime);
    }
    else if (eAct == MM::AfterSet)
    {
        pProp->Get(value);
        featureValue->stVariant.stTriggerOption.uiFrameCount = value;
        SetFeature(featureValue->uiFeatureId);
    }
    return DEVICE_OK;
}

int NikonKsCam::OnTriggerDelay(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    lx_uint32 index = mapFeatureIndex_[eTriggerOption];
    long value;
    CAM_FeatureValue*   featureValue = &vectFeatureValue_.pstFeatureValue[index];

    if (eAct == MM::BeforeGet)
    {
        pProp->Set((long)vectFeatureValue_.pstFeatureValue[index].stVariant.stTriggerOption.iDelayTime);
    }
    else if (eAct == MM::AfterSet)
    {
        pProp->Get(value);
        featureValue->stVariant.stTriggerOption.iDelayTime = value;
        SetFeature(featureValue->uiFeatureId);
    }
    return DEVICE_OK;
}

int NikonKsCam::OnImageFormat(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    char strWork[30];
    lx_uint32 uiFeatureId = eFormat;
    lx_uint32 index = mapFeatureIndex_[uiFeatureId];
    CAM_FeatureDesc*    featureDesc;
    CAM_FeatureValue*   featureValue;
    featureValue = &vectFeatureValue_.pstFeatureValue[index];
    featureDesc = &featureDesc_[index];

    if (eAct == MM::AfterSet)
    {
        string value;
        pProp->Get(value);
        for (lx_uint32 i = 0; i < featureDesc->uiListCount; i++)
        {
            wcstombs(strWork, featureDesc->stFormatList[i].wszComment, CAM_FEA_COMMENT_MAX);
            if (value.compare(strWork) == 0)
            {
                LogMessage(strWork);
                featureValue->stVariant.stFormat = featureDesc->stFormatList[i].stFormat;
                SetFeature(featureValue->uiFeatureId);
                UpdateImageSettings();
                //Update ROI, MeteringArea limits, they change with format setting
                SetROILimits();
                SetMeteringAreaLimits();
                break;
            }
        }
    }
    if (eAct == MM::BeforeGet || eAct == MM::AfterSet )
    {
        for (lx_uint32 i = 0; i < featureDesc->uiListCount; i++)
        {
            if (featureDesc->stFormatList[i].stFormat == featureValue->stVariant.stFormat)
            {
                wcstombs(strWork, featureDesc->stFormatList[i].wszComment, CAM_FEA_COMMENT_MAX);
                pProp->Set(strWork);
                break;
            }
        }
    }
    return DEVICE_OK;
}

//Generic - Handle "Range" feature
int NikonKsCam::OnRange(MM::PropertyBase* pProp, MM::ActionType eAct, lx_uint32 uiFeatureId)
{
    lx_uint32 index = mapFeatureIndex_[uiFeatureId];
    long value;
    CAM_FeatureValue*   featureValue = &vectFeatureValue_.pstFeatureValue[index];
    switch (featureValue->stVariant.eVarType) {
    case evrt_uint32:
        if (eAct == MM::BeforeGet)
        {
            pProp->Set((long)vectFeatureValue_.pstFeatureValue[index].stVariant.ui32Value);
        }
        else if (eAct == MM::AfterSet)
        {
            pProp->Get(value);
            featureValue->stVariant.ui32Value = value;
            SetFeature(featureValue->uiFeatureId);
        }
    case evrt_int32:
        if (eAct == MM::BeforeGet)
        {
            pProp->Set((long)vectFeatureValue_.pstFeatureValue[index].stVariant.i32Value);
        }
        else if (eAct == MM::AfterSet)
        {
            pProp->Get(value);
            featureValue->stVariant.i32Value = value;
            SetFeature(featureValue->uiFeatureId);
        }
    }
    return DEVICE_OK;
}

//Generic - Handle "List" feature
int NikonKsCam::OnList(MM::PropertyBase* pProp, MM::ActionType eAct, lx_uint32 uiFeatureId)
{
    char strWork[30];
    lx_uint32 index = mapFeatureIndex_[uiFeatureId];
    CAM_FeatureDesc*    featureDesc = &featureDesc_[index];
    CAM_FeatureValue*   featureValue = &vectFeatureValue_.pstFeatureValue[index];

    if (eAct == MM::BeforeGet)
    {
        for (lx_uint32 i = 0; i < featureDesc->uiListCount; i++)
        {
            if (featureDesc->stElementList[i].varValue.ui32Value == featureValue->stVariant.ui32Value)
            {
                wcstombs(strWork, featureDesc->stElementList[i].wszComment, CAM_FEA_COMMENT_MAX);
                pProp->Set(strWork);
                break;
            }
        }
    }
    else if (eAct == MM::AfterSet)
    {
        string value;
        lx_uint32 i;
        pProp->Get(value);
        for (i = 0; i < featureDesc->uiListCount; i++)
        {
            wcstombs(strWork, featureDesc->stElementList[i].wszComment, CAM_FEA_COMMENT_MAX);
            if (value.compare(strWork) == 0)
            {
                featureValue->stVariant.ui32Value = featureDesc->stElementList[i].varValue.ui32Value;
                SetFeature(featureValue->uiFeatureId);
                UpdateImageSettings();
                break;
            }
        }


    }
    return DEVICE_OK;
}

//Generic- Set either exposure time or exposure time limit
int NikonKsCam::OnExposureChange(MM::PropertyBase* pProp, MM::ActionType eAct, lx_uint32 uiFeatureId)
{
    lx_uint32 index = mapFeatureIndex_[uiFeatureId];

    if (eAct == MM::BeforeGet)
    {
        pProp->Set(vectFeatureValue_.pstFeatureValue[index].stVariant.ui32Value / 1000.);
    }
    else if (eAct == MM::AfterSet)
    {
        double value;
        CAM_FeatureValue*   featureValue;
        pProp->Get(value);
        value = (value * 1000)+0.5;
        featureValue = &vectFeatureValue_.pstFeatureValue[index];
        //value = AdjustExposureTime((unsigned int)value);

        featureValue->stVariant.ui32Value = (lx_uint32) value;
        SetFeature(featureValue->uiFeatureId);
    }
    return DEVICE_OK;
}
