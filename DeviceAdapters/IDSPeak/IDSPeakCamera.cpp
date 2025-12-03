////////////////////////////////////////////////////////////////////////////////
// FILE:          IDSPeakCamera.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Driver for IDS peak series cameras
//
//                Based on IDS peak SDK and Micro-manager DemoCamera example
//                tested with SDK version 2.5
//
// AUTHOR:        Lars Kool, Institut Pierre-Gilles de Gennes
//
// YEAR:          2025
//                
// VERSION:       2.0.1
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
// LAST UPDATE:   13.11.2025 LK
////////////////////////////////////////////////////////////////////////////////

#include "IDSPeakHub.h"
#include "IDSPeakCamera.h"
#include "ModuleInterface.h"

#include <cstdio>
#include <string>
#include <math.h>
#include <sstream>
#include <algorithm>
#include <iostream>
#include <future>
#include <unordered_map>

#include <peak/peak.hpp>
#include <peak_ipl/peak_ipl.hpp>
#include <peak/converters/peak_buffer_converter_ipl.hpp>

using AccessNode = peak::core::nodes::NodeAccessStatus;
using CommandNode = peak::core::nodes::CommandNode;
using EnumNode = peak::core::nodes::EnumerationNode;
using FloatNode = peak::core::nodes::FloatNode;
using IntNode = peak::core::nodes::IntegerNode;

const char* g_PixelType = "PixelType";
const char* g_PixelType_8bit = "8bit";
const char* g_PixelType_16bit = "16bit";
const char* g_PixelType_32bitBGRA = "32bit BGRA";

const char* g_keyword_Peak_PixelFormat = "IDS Pixel Format";


////////////////////////////////////////////////////////////////////////////////
// Constructor/Destructor
////////////////////////////////////////////////////////////////////////////////

/**
* CIDSPeakCamera constructor.
* Setup default all variables and create device properties required to exist
* before intialization. Basically, it is used to allow the user to select
* which "advanced" features to enable.
*
* The constructor does not access the hardware at all. The verification whether
* these features are actually supported happens during the initialization.
* Features are simply skipped if they turn out not to be supported.
*
* @params idx - Unique Camera id
*/
CIDSPeakCamera::CIDSPeakCamera(int idx) :
    CCameraBase<CIDSPeakCamera>(),
    initialized_(false)
{
    // Enable autowhitebalance
    std::vector<std::string> allowedValues = { "true", "false" };
    CPropertyAction* pAct = new CPropertyAction(this, &CIDSPeakCamera::OnEnableAutoWhitebalance);
    CreateStringProperty("Enable auto whitebalance", allowedValues[1].c_str(), false, pAct, true);
    SetAllowedValues("Enable auto whitebalance", allowedValues);

    // Enable analog gain
    pAct = new CPropertyAction(this, &CIDSPeakCamera::OnEnableAnalogGain);
    CreateStringProperty("Enable analog gain", allowedValues[1].c_str(), false, pAct, true);
    SetAllowedValues("Enable analog gain", allowedValues);

    // Enable digital gain
    pAct = new CPropertyAction(this, &CIDSPeakCamera::OnEnableDigitalGain);
    CreateStringProperty("Enable digital gain", allowedValues[1].c_str(), false, pAct, true);
    SetAllowedValues("Enable digitgal gain", allowedValues);

    // Enable temperature monitor
    pAct = new CPropertyAction(this, &CIDSPeakCamera::OnEnableTemperature);
    CreateStringProperty("Enable temperature", allowedValues[1].c_str(), false, pAct, true);
    SetAllowedValues("Enable temperature", allowedValues);

    // Enable Trigger
    pAct = new CPropertyAction(this, &CIDSPeakCamera::OnEnableTrigger);
    CreateStringProperty("Enable trigger", allowedValues[1].c_str(), false, pAct, true);
    SetAllowedValues("Enable trigger", allowedValues);

    // call the base class method to set-up default error codes/messages
    InitializeDefaultErrorMessages();
    readoutStartTime_ = GetCurrentMMTime();
    thd_ = new MySequenceThread(this);
    deviceIdx_ = idx;
    deviceName_ = g_IDSPeakCameraName + std::to_string(deviceIdx_);

    // parent ID display
    CreateHubIDProperty();
}

/**
* CIDSPeakCamera destructor.
* If this device used as intended within the Micro-Manager system,
* Shutdown() will be always called before the destructor. But in any case
* we need to make sure that all resources are properly released even if
* Shutdown() was not called.
*/
CIDSPeakCamera::~CIDSPeakCamera()
{
    StopSequenceAcquisition();
    delete thd_;

    Shutdown();
}

////////////////////////////////////////////////////////////////////////////////
// MM::Device API
////////////////////////////////////////////////////////////////////////////////

/**
* Copies the name of this device into the provided, preallocated char buffer.
* @param name - Pointer to char buffer
*/
void CIDSPeakCamera::GetName(char* name) const
{
    CDeviceUtils::CopyLimitedString(name, deviceName_.c_str());
}

/**
* Returns whether camera is busy executing a command. Since the API is not
* asynchronous, it should never be busy.
* @returns Boolean - Always false
*/
bool CIDSPeakCamera::Busy()
{
    return false;
}

/**
* Intializes the hardware.
* There the features are initialized, and the device properties created.
* See individiual initializers for more details.
* @return Integer status code - Returns DEVICE_OK on success
*/
int CIDSPeakCamera::Initialize()
{
    if (initialized_) { return DEVICE_OK; }

    // Get handle to Hub
    IDSPeakHub* pHub = static_cast<IDSPeakHub*>(GetParentHub());
    if (pHub)
    {
        char hubLabel[MM::MaxStrLength];
        pHub->GetLabel(hubLabel);
        SetParentID(hubLabel);
    }
    else
    {
        LogMessage("No hub was found.");
    }

    // Open camera
    try
    {
        auto& deviceManager = peak::DeviceManager::Instance(); // Get device manager
        descriptor = deviceManager.Devices().at(deviceIdx_); // set device
        device = descriptor->OpenDevice(peak::core::DeviceAccessType::Control); // Open camera
        nodeMapRemoteDevice = device->RemoteDevice()->NodeMaps().at(0);
    }
    catch (std::exception& e)
    {
        LogMessage("IDS exception: Could not open camera.");
        LogMessage(e.what());
        return DEVICE_ERR;
    }

    // Compulsory properties
    //----------------------

    // Various camera properties
    int ret = InitializeCameraDescription();
    if (DEVICE_OK != ret) { return ret; }

    // ExposureTime
    ret = InitializeExposureTime();
    if (DEVICE_OK != ret) { return ret; }

    // Framerate
    ret = InitializeFramerate();
    if (DEVICE_OK != ret) { return ret; }

    // Binning
    ret = InitializeBinning();
    if (DEVICE_OK != ret) { return ret; }

    // Pixel types
    ret = InitializePixelTypes();
    if (DEVICE_OK != ret) { return ret; }

    // ROI
    ret = InitializeROI();
    if (DEVICE_OK != ret) { return ret; }

    // Buffer
    ret = InitializeBuffer();
    if (DEVICE_OK != ret) { return ret; }


    // Optional properties
    //--------------------

    // Auto white balance
    if (enableAutoWhitebalance_)
    {
        ret = InitializeAutoWhiteBalance();
        if (DEVICE_OK != ret) { return ret; }
    }

    // Analog gain
    if (enableAnalogGain_)
    {
        ret = InitializeAnalogGain();
        if (DEVICE_OK != ret) { return ret; }
    }

    // Digital gain
    if (enableDigitalGain_)
    {
        ret = InitializeDigitalGain();
        if (DEVICE_OK != ret) { return ret; }
    }

    // Camera temperature ReadOnly
    if (enableTemperature_)
    {
        ret = InitializeTemperature();
        if (DEVICE_OK != ret) { return ret; }
    }

    // Trigger
    if (enableTrigger_)
    {
        ret = InitializeTrigger();
        if (DEVICE_OK != ret) { return ret; }
    }

    // synchronize all properties
    ret = UpdateStatus();
    if (DEVICE_OK != ret) { return ret; }

    initialized_ = true;
    return DEVICE_OK;
}

/**
* Shuts down (unloads) the device and releases buffer resources.
* @returns Integer status code - Returns DEVICE_OK on success
*/
int CIDSPeakCamera::Shutdown()
{
    if (initialized_ == false) { return DEVICE_OK; }

    // Release any buffer resources
    int ret = ClearBuffer();
    if (DEVICE_OK != ret) { return ret; }

    initialized_ = false;
    return DEVICE_OK;
}

////////////////////////////////////////////////////////////////////////////////
// MM::Camera API
////////////////////////////////////////////////////////////////////////////////

/**
* Performs exposure and grabs a single image.
* This function blocks the main thread while called.
* @returns Integer status code - Returns DEVICE_OK on success.
*/
int CIDSPeakCamera::SnapImage()
{
    int ret = DEVICE_OK;
    static int callCounter = 0;
    ++callCounter;

    // Start acquisition of single image
    PrepareBuffer();
    ret = StartAcquisition(1);
    if (DEVICE_OK != ret)
    {
        StopAcquisition();
        return ret;
    }

    // This function blocks the main thread, meaning that waiting for a trigger will cause
    // MM to hang until a trigger is received. This is undesirable behavior, hence we ignore the trigger
    if ("On" == triggerMode_)
    {
        LogMessage("IDS warning: Cannot use trigger with SnapImage. Image is directly captured.");
        LogMessage("IDS warning: See https://micro-manager.org/IDSPeak on how to use the trigger function.");
        try
        {
            nodeMapRemoteDevice->FindNode<CommandNode>("TriggerSoftware")->Execute();
            nodeMapRemoteDevice->FindNode<CommandNode>("TriggerSoftware")->WaitUntilDone();
        }
        catch (std::exception& e)
        {
            LogMessage("IDS exception: Could not activate trigger.");
            LogMessage(e.what());
            return DEVICE_ERR;
        }
    }

    // Acquire and transfer the image to MM
    uint64_t timeout_ms = exposureCur_ * 3;
    ret = AcquireAndTransferImage(timeout_ms, true);
    // Unblock the camera
    StopAcquisition();
    if (DEVICE_OK != ret)
    {
        LogMessage("IDS warning: Could not snap image. Acquisition stopped automatically.");
    }

    readoutStartTime_ = GetCurrentMMTime();
    return DEVICE_OK;
}


/**
* Get pointer to the pixel data. Note that no metadata is passed with the buffer.
* This should be queried using the GetImageWidth(), GetImageHeight(), and
* GetImageBytesPerPixel() functions.
* @returns unsigned char pointer - Pointer to first byte of first pixel in image
*/
const unsigned char* CIDSPeakCamera::GetImageBuffer()
{
    // Not sure about this ThreadGuard
    MMThreadGuard g(imgPixelsLock_);
    unsigned char* pB = (unsigned char*)(img_.GetPixels());
    return pB;
}

/**
* Get image buffer X-size in pixels.
* @returns unsigned int - Width of image in pixels
*/
unsigned CIDSPeakCamera::GetImageWidth() const
{
    return img_.Width();
}

/**
* Get image buffer Y-size in pixels.
* @returns unsigned int - Height of image in pixels
*/
unsigned CIDSPeakCamera::GetImageHeight() const
{
    return img_.Height();
}

/**
* Get image buffer pixel depth in bytes.
* @returns unsigned int - Depth of image in bytes
*/
unsigned CIDSPeakCamera::GetImageBytesPerPixel() const
{
    return img_.Depth();
}

/**
* Get the bit depth (dynamic range) of the pixel.
* This does not affect the buffer size, it just gives the client application
* a guideline on how to interpret pixel values.
* @returns unsigned int - Bit depth of the pixels
*/
unsigned CIDSPeakCamera::GetBitDepth() const
{
    return 8 * img_.Depth();
}

/**
* Get the number of components (1 for Mono, 4 for BGRA)
* @ returns unsigned int - Number of components
*/
unsigned CIDSPeakCamera::GetNumberOfComponents() const
{
    return nComponents_;
}

/**
* Get the size in bytes of the image buffer.
* @returns long - Size of image buffer in bytes
*/
long CIDSPeakCamera::GetImageBufferSize() const
{
    return img_.Width() * img_.Height() * GetImageBytesPerPixel();
}

/**
* Sets the camera Region Of Interest.
* This command will change the dimensions of the image.
* Depending on the hardware capabilities the camera may not be able to configure the
* exact dimensions requested - but should try do as close as possible.
* If both xSize and ySize are set to 0, the ROI is set to the entire sensor.
* @param x - top-left corner coordinate
* @param y - top-left corner coordinate
* @param xSize - width
* @param ySize - height
* @returns Integer status code - Returns DEVICE_OK on success
*/
int CIDSPeakCamera::SetROI(unsigned x, unsigned y, unsigned xSize, unsigned ySize)
{
    try
    {
        if (!CheckAccess(nodeMapRemoteDevice->FindNode<IntNode>("OffsetX"), AccessTypes::READWRITE))
        {
            LogMessage("IDS error: Could not set ROI, due to lack of READWRITE access.");
            return DEVICE_CAN_NOT_SET_PROPERTY;
        }
    }
    catch (std::exception& e)
    {
        LogMessage("IDS exception: Could not determine whether ROI is writeable.");
        LogMessage(e.what());
        return DEVICE_ERR;
    }

    // if xSize and ySize are 0, set ROI to full frame
    if (0 == xSize && 0 == ySize)
    {
        x = 0;
        y = 0;
        xSize = sensorWidth_ / binSize_;
        ySize = sensorHeight_ / binSize_;
    }

    xSize = multipleOfIncrement(xSize, roiWidthIncrement_);
    ySize = multipleOfIncrement(ySize, roiHeightIncrement_);
    if (xSize < roiWidthMin_)
    {
        std::stringstream ss;
        ss << "IDS warning: The provided ROI width (" << xSize << ") is less than the ";
        ss << "minimum ROI width for this camera: " << roiWidthMin_ << ".\n";
        ss << "    The ROI width is set to the ROI minimum.";
        LogMessage(ss.str());
        xSize = roiWidthMin_;
    }
    if (ySize < roiHeightMin_)
    {
        std::stringstream ss;
        ss << "IDS warning: The provided ROI height (" << ySize << ") is less than the ";
        ss << "minimum ROI height for this camera: " << roiHeightMin_ << ".\n";
        ss << "    The ROI height is set to the ROI minimum.";
        LogMessage(ss.str());
        ySize = roiHeightMin_;
    }

    x = multipleOfIncrement(x + roiX_, roiOffsetXIncrement_);
    if (x + xSize > sensorWidth_ / binSize_)
    {
        LogMessage("IDS warning: After adjusting the ROI width to meet the increment requirement, the ROI exceeds the sensor boundaries.");
        LogMessage("IDS warning: The ROI is displaced so it stays within the sensor boundaries.");
        x = (sensorWidth_ / binSize_) - xSize;
    }
    y = multipleOfIncrement(y + roiY_, roiOffsetYIncrement_);
    if (y + ySize > sensorHeight_ / binSize_)
    {
        LogMessage("IDS warning: After adjusting the ROI height to meet the increment requirement, the ROI exceeds the sensor boundaries.");
        LogMessage("IDS warning: The ROI is displaced so it stays within the sensor boundaries.");
        y = (sensorHeight_ / binSize_) - ySize;
    }

    try
    {
        // Adjust offset to make sure window doesn't exceed sensor
        nodeMapRemoteDevice->FindNode<IntNode>("OffsetX")->SetValue(0);
        nodeMapRemoteDevice->FindNode<IntNode>("OffsetY")->SetValue(0);

        // Actually set ROI, first set ROI size, to make sure it doesn't exceed window size
        nodeMapRemoteDevice->FindNode<IntNode>("Width")->SetValue(xSize);
        nodeMapRemoteDevice->FindNode<IntNode>("Height")->SetValue(ySize);
        nodeMapRemoteDevice->FindNode<IntNode>("OffsetX")->SetValue(x);
        nodeMapRemoteDevice->FindNode<IntNode>("OffsetY")->SetValue(y);
    }
    catch (std::exception& e)
    {
        LogMessage("IDS exception: An error occurred while setting the ROI.");
        LogMessage(e.what());
        return DEVICE_CAN_NOT_SET_PROPERTY;
    }

    // Update framerate limits
    int ret = GetBoundaries(nodeMapRemoteDevice->FindNode<FloatNode>("AcquisitionFrameRate"), frameRateMin_, frameRateMax_, frameRateInc_);
    SetPropertyLimits("MDA framerate", frameRateMin_, frameRateMax_);

    img_.Resize(xSize, ySize);
    roiX_ = x;
    roiY_ = y;
    return DEVICE_OK;
}

/**
* Get the dimensions and offset of the current ROI.
* @param x - x-offset of the ROI
* @param y - y-offset of the ROI
* @param xSize - size of ROI in x-direction
* @param ySize - size of ROI in y-direction
* @returns Integer status code - Returns DEVICE_OK on success
*/
int CIDSPeakCamera::GetROI(unsigned& x, unsigned& y, unsigned& xSize, unsigned& ySize)
{
    x = roiX_;
    y = roiY_;
    xSize = img_.Width();
    ySize = img_.Height();
    return DEVICE_OK;
}

/**
* Resets the Region of Interest to full frame.
* @returns Integer status code - Returns DEVICE_OK on success
*/
int CIDSPeakCamera::ClearROI()
{
    // Passing all zeros to SetROI sets the ROI to the full frame
    int ret = SetROI(0, 0, 0, 0);
    return ret;
}

/**
* Get the current binning factor.
* @returns int - Binning factor
*/
int CIDSPeakCamera::GetBinning() const
{
    char buf[MM::MaxStrLength];
    int ret = GetProperty(MM::g_Keyword_Binning, buf);
    if (DEVICE_OK != ret) { return 0; } // If something goes wrong, return 0 (unphysical binning)
    return atoi(buf);
}

/**
* Set binning factor.
* @returns Integer status code - Returns DEVICE_OK on success
*/
int CIDSPeakCamera::SetBinning(int binF)
{
    try
    {
        nodeMapRemoteDevice->FindNode<EnumNode>("BinningSelector")
            ->SetCurrentEntry(binningSelector_);
        nodeMapRemoteDevice->FindNode<IntNode>("BinningHorizontal")
            ->SetValue(binF);
        nodeMapRemoteDevice->FindNode<IntNode>("BinningVertical")
            ->SetValue(binF);
    }
    catch (std::exception& e)
    {
        LogMessage("IDS exception: An error occurred while setting binning.");
        LogMessage(e.what());
        return DEVICE_CAN_NOT_SET_PROPERTY;
    }

    binSize_ = binF;
    return DEVICE_OK;
}

/**
* Get the current exposure time.
* @returns double - Exposure time in milliseconds
*/
double CIDSPeakCamera::GetExposure() const
{
    char buf[MM::MaxStrLength];
    int ret = GetProperty(MM::g_Keyword_Exposure, buf);
    if (DEVICE_OK != ret) { return 0.; } // If something goes wrong, return 0. 
    return atof(buf);
}

/**
 * Get the current exposure from a sequence and increases the sequence counter
 * Used for exposure sequences.
 * For now, this device adapter does not support sequencing, even though some
 * IDS cameras support sequencing.
 * @returns double - Next exposure time in sequence
 */
double CIDSPeakCamera::GetSequenceExposure()
{
    if (exposureSequence_.size() == 0)
    {
        return this->GetExposure();
    }

    double exposure = exposureSequence_[sequenceIndex_];

    sequenceIndex_++;
    if (sequenceIndex_ >= exposureSequence_.size())
    {
        sequenceIndex_ = 0;
    }

    return exposure;
}

/**
* Sets exposure in milliseconds.
* @param exp - Exposure time in milliseconds
*/
void CIDSPeakCamera::SetExposure(double exp)
{
    // Check for access
    auto node = nodeMapRemoteDevice->FindNode<FloatNode>("ExposureTime");
    if (node->AccessStatus() != peak::core::nodes::NodeAccessStatus::ReadWrite)
    {
        LogMessage("IDS error: No write access to ExposureTime.");
        return;
    }

    // Check if exposure time is less than the minimun exposure time
    // If so, set it to minimum exposure time.
    if (exp <= exposureMin_)
    {
        exp = exposureMin_;
        LogMessage("IDS warning: Exposure time too short. Exposure time set to minimum.");
    }
    // Check if exposure time is less than the maximum exposure time
    // If so, set it to maximum exposure time.
    else if (exp >= exposureMax_)
    {
        exp = exposureMax_;
        LogMessage("IDS warning: Exposure time too long. Exposure time set to maximum.");
    }

    // Convert milliseconds to microseconds (peak cameras expect time in microseconds)
    // and make exposure set multiple of increment.
    double exposureSet = multipleOfIncrement(exp * 1000, exposureInc_);

    // Set new exposure time and get actual value after set
    try
    {
        node->SetValue(exposureSet);
        exposureCur_ = node->Value() / 1000;
    }
    catch (std::exception& e)
    {
        LogMessage("IDS exception: Could not set exposure time.");
        LogMessage(e.what());
        return;
    }

    // Update framerate range
    try
    {
        auto nodeFrameRate = nodeMapRemoteDevice->FindNode<FloatNode>("AcquisitionFrameRate");
        frameRateMin_ = nodeFrameRate->Minimum();
        frameRateMax_ = nodeFrameRate->Maximum();
    }
    catch (std::exception& e)
    {
        LogMessage("IDS exception: Could not read acquisition framerate");
        LogMessage(e.what());
    }
    SetPropertyLimits("MDA framerate", frameRateMin_, frameRateMax_);

    SetProperty(MM::g_Keyword_Exposure, CDeviceUtils::ConvertToString(exposureCur_));
    GetCoreCallback()->OnExposureChanged(this, exp);
}

/**
* Returns whether the camera supports sequencing. For now, this feature is
* not supported by the IDSPeak device adapter, hence it will always return
* false.
* @param isSequenceable - Reference to bool used as output
* @returns Integer status code - Returns DEVICE_OK on success
*/
int CIDSPeakCamera::IsExposureSequenceable(bool& isSequenceable) const
{
    isSequenceable = isSequenceable_;
    return DEVICE_OK;
}

/**
* Get the maximum allowed length of the exposure sequence.
* Since isSequenceable_ is always false, it will always return the
* DEVICE_UNSUPPORTED_COMMAND errorcode.
* @param nrEvents - Reference to long used as output for the maximum sequence length
* @returns Integer status code - Returns DEVICE_OK on success
*/
int CIDSPeakCamera::GetExposureSequenceMaxLength(long& nrEvents) const
{
    if (!isSequenceable_) { return DEVICE_UNSUPPORTED_COMMAND; }
    nrEvents = sequenceMaxLength_;
    return DEVICE_OK;
}

/**
* Start the exposure sequence. Since isSequenceable_ is always false,
* it will always return the DEVICE_UNSUPPORTED_COMMAND errorcode.
* @returns Integer status code - Returns DEVICE_OK on success
*/
int CIDSPeakCamera::StartExposureSequence()
{
    if (!isSequenceable_) { return DEVICE_UNSUPPORTED_COMMAND; }
    // may need thread lock
    sequenceRunning_ = true;
    return DEVICE_OK;
}

/**
* Stop the exposure sequence. Since isSequenceable_ is always false,
* it will always return the DEVIC_UNSUPPORTED_COMMAND errorcode.
* @returns Integer status code - Returns DEVICE_OK on success
*/
int CIDSPeakCamera::StopExposureSequence()
{
    if (!isSequenceable_) { return DEVICE_UNSUPPORTED_COMMAND; }
    // may need thread lock
    sequenceRunning_ = false;
    sequenceIndex_ = 0;
    return DEVICE_OK;
}

/**
 * Clears the list of exposures used in sequences. Since isSequenceable_
 * is always false, it will always return the DEVICE_UNSUPPORTED_COMMAND
 * errorcode.
 * @returns Integer status code - Returns DEVICE_OK on success
 */
int CIDSPeakCamera::ClearExposureSequence()
{
    if (!isSequenceable_) { return DEVICE_UNSUPPORTED_COMMAND; }
    exposureSequence_.clear();
    return DEVICE_OK;
}

/**
 * Adds an exposure to a list of exposures used in sequences. Since
 * isSequenceable_ is always false, it will always return the
 * DEVICE_UNSUPPORTED_COMMAND errocode.
 * @returns Integer status code - Returns DEVICE_OK on success
 */
int CIDSPeakCamera::AddToExposureSequence(double exposureTime_ms)
{
    if (!isSequenceable_) { return DEVICE_UNSUPPORTED_COMMAND; }
    exposureSequence_.push_back(exposureTime_ms);
    return DEVICE_OK;
}

/**
* Sends exposure sequence to hardware. Since isSequenceable_ is always
* false, it will always return the DEVICE_UNSUPPORTED_COMMAND errorcode.
* @returns Integer status code - Returns DEVICE_OK on success
*/
int CIDSPeakCamera::SendExposureSequence() const
{
    if (!isSequenceable_) {
        return DEVICE_UNSUPPORTED_COMMAND;
    }

    return DEVICE_OK;
}

/**
* Starts the acquisition. To the best of my knowledge, this function is called
* when the "Live" button in the main MM menu is pressed.
* This function is just a wrapper setting the numImages to "LONG_MAX",
* indicating that the camera should just keep acquiring images.
* @params interval - double indicating the interval between the images.
* @returns Integer status code - Returns DEVICE_OK on success
*/
int CIDSPeakCamera::StartSequenceAcquisition(double interval)
{
    if ("On" == triggerMode_)
    {
        LogMessage("IDS warning: Cannot use \"Live\" mode while TriggerMode is On." +
            std::string(" Turned off TriggerMode automatically.")
        );
        nodeMapRemoteDevice->FindNode<EnumNode>("TriggerMode")->SetCurrentEntry("Off");
        triggerMode_ = nodeMapRemoteDevice->FindNode<EnumNode>("TriggerMode")->CurrentEntry()->SymbolicValue();
    }
    return StartSequenceAcquisition(LONG_MAX, interval, false);
}


/**
* Starts the acquisition. To the best of my knowledge, this function is called
* when the "MDA" acquisition is started. But only when there is more than one
* image to be acquired, and "interval_ms" <= "exposureTime_". Otherwise,
* a series of SnapImage calls is made.
* The acquisition is run on a separate thread, meaning that MDA acquisition
* supports a Trigger.
* @params numImages - number of images to be acquired (LONG_MAX for infinite)
* @params interval_ms - interval between images in milliseconds
* @params stopOnOverflow - flag on whether to stop acquisition on buffer overflow
* @returns Integer status code - Returns DEVICE_OK on success
*/
int CIDSPeakCamera::StartSequenceAcquisition(long numImages, double interval_ms, bool stopOnOverflow)
{
    if (IsCapturing()) { return DEVICE_CAMERA_BUSY_ACQUIRING; }
    int ret = DEVICE_OK;

    // Adjust framerate to match requested interval between frames
    if (interval_ms > 0)
    {
        ret = SetFrameRate(1000 / interval_ms);
        if (DEVICE_OK != ret) { return ret; }
    }

    // Set frame count
    ret = SetFrameCount(numImages);
    if (DEVICE_OK != ret) { return ret; }

    ret = PrepareBuffer();
    if (DEVICE_OK != ret) { return ret; }

    // Wait until shutter is ready
    ret = GetCoreCallback()->PrepareForAcq(this);
    if (DEVICE_OK != ret) { return ret; }

    // Start sequence
    sequenceStartTime_ = GetCurrentMMTime();
    imageCounter_ = 0;
    thd_->Start(numImages, interval_ms);
    stopOnOverflow_ = stopOnOverflow;
    return DEVICE_OK;
}

/**
* Stop and wait for the Sequence thread finished.
* @returns Integer status code - Returns DEVICE_OK on success
*/
int CIDSPeakCamera::StopSequenceAcquisition()
{
    if (!thd_->IsStopped())
    {
        thd_->Stop();
        thd_->wait();
    }
    return DEVICE_OK;
}

////////////////////////////////////////////////////////////////////////////////
// MM Action Handlers
////////////////////////////////////////////////////////////////////////////////

/**
* Handles EnableTemperature pre-init property, allowing the user to select
* whether to include a read-only temperature property.
* @param pProp - pointer to property
* @param eAct - type of action performed on property (eg get/set)
* @returns Integer status code - Returns DEVICE_OK on success
*/
int CIDSPeakCamera::OnEnableTemperature(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        pProp->Set((enableTemperature_) ? "true" : "false");
    }
    else if (eAct == MM::AfterSet)
    {
        std::string temp;
        pProp->Get(temp);
        enableTemperature_ = (temp == "true");
    }
    return DEVICE_OK;
}

/**
* Handles EnableAnalogGain pre-init property, allowing the used to select
* whether to include analog gain properties (if they are supported by the device).
* @param pProp - pointer to property
* @param eAct - type of action performed on property (eg get/set)
* @returns Integer status code - Returns DEVICE_OK on success
*/
int CIDSPeakCamera::OnEnableAnalogGain(MM::PropertyBase* pProp, MM::ActionType eAct) {
    if (eAct == MM::BeforeGet)
    {
        pProp->Set((enableAnalogGain_) ? "true" : "false");
    }
    else if (eAct == MM::AfterSet)
    {
        std::string temp;
        pProp->Get(temp);
        enableAnalogGain_ = (temp == "true");
    }
    return DEVICE_OK;
}

/**
* Handles EnableDigitalGain pre-init property, allowing the used to select
* whether to include digital gain properties (if they are supported by the device).
* @param pProp - pointer to property
* @param eAct - type of action performed on property (eg get/set)
* @returns Integer status code - Returns DEVICE_OK on success
*/
int CIDSPeakCamera::OnEnableDigitalGain(MM::PropertyBase* pProp, MM::ActionType eAct) {
    if (eAct == MM::BeforeGet)
    {
        pProp->Set((enableDigitalGain_) ? "true" : "false");
    }
    else if (eAct == MM::AfterSet)
    {
        std::string temp;
        pProp->Get(temp);
        enableDigitalGain_ = (temp == "true");
    }
    return DEVICE_OK;
}

/**
* Handles EnableAutoWhitebalance pre-init property, allowing the user to select
* whether to include auto whitebalance property (if it is supported by the device).
* @param pProp - pointer to property
* @param eAct - type of action performed on property (eg get/set)
* @returns Integer status code - Returns DEVICE_OK on success
*/
int CIDSPeakCamera::OnEnableAutoWhitebalance(MM::PropertyBase* pProp, MM::ActionType eAct) {
    if (eAct == MM::BeforeGet)
    {
        pProp->Set((enableAutoWhitebalance_) ? "true" : "false");
    }
    else if (eAct == MM::AfterSet)
    {
        std::string temp;
        pProp->Get(temp);
        enableAutoWhitebalance_ = (temp == "true");
    }
    return DEVICE_OK;
}

/**
* Handles EnableTrigger pre-init property, allowing the user to select
* whether to include a trigger property (if it is supported by the device).
* @param pProp - pointer to property
* @param eAct - type of action performed on property (eg get/set)
* @returns Integer status code - Returns DEVICE_OK on success
*/
int CIDSPeakCamera::OnEnableTrigger(MM::PropertyBase* pProp, MM::ActionType eAct) {
    if (eAct == MM::BeforeGet)
    {
        pProp->Set((enableTrigger_) ? "true" : "false");
    }
    else if (eAct == MM::AfterSet)
    {
        std::string temp;
        pProp->Get(temp);
        enableTrigger_ = (temp == "true");
    }
    return DEVICE_OK;
}

/**
* Handles the Binning property. The OnPropertyChanged call will
* call the SetBinning function.
* @param pProp - pointer to property
* @param eAct - type of action performed on property (eg get/set)
* @returns Integer status code - Returns DEVICE_OK on success
*/
int CIDSPeakCamera::OnBinning(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    int ret = DEVICE_ERR;
    switch (eAct)
    {
    case MM::AfterSet:
    {
        // Should not change binning during acquisition
        if (IsCapturing()) { return DEVICE_CAMERA_BUSY_ACQUIRING; }

        long binFactor;
        pProp->Get(binFactor);

        ret = SetBinning((int)binFactor);
        if (DEVICE_OK != ret) { return ret; }

        std::stringstream ss;
        ss << "Image Width: " << nodeMapRemoteDevice->FindNode<IntNode>("Width")->Value();
        ss << " Image Height: " << nodeMapRemoteDevice->FindNode<IntNode>("Height")->Value();
        LogMessage(ss.str());

        // Get the new ROI limits
        ret = GetBoundaries(nodeMapRemoteDevice->FindNode<IntNode>("OffsetX"),
            roiOffsetXMin_, roiOffsetXMax_, roiOffsetXIncrement_);
        if (DEVICE_OK != ret) { return ret; }
        ret = GetBoundaries(nodeMapRemoteDevice->FindNode<IntNode>("OffsetY"),
            roiOffsetYMin_, roiOffsetYMax_, roiOffsetYIncrement_);
        if (DEVICE_OK != ret) { return ret; }
        ret = GetBoundaries(nodeMapRemoteDevice->FindNode<IntNode>("Width"),
            roiWidthMin_, roiWidthMax_, roiWidthIncrement_);
        if (DEVICE_OK != ret) { return ret; }
        ret = GetBoundaries(nodeMapRemoteDevice->FindNode<IntNode>("Height"),
            roiHeightMin_, roiHeightMax_, roiHeightIncrement_);
        if (DEVICE_OK != ret) { return ret; }

        // Request new ROI values
        try
        {
            roiX_ = nodeMapRemoteDevice->FindNode<IntNode>("OffsetX")->Value();
            roiY_ = nodeMapRemoteDevice->FindNode<IntNode>("OffsetY")->Value();
            unsigned width = (unsigned)nodeMapRemoteDevice->FindNode<IntNode>("Width")->Value();
            unsigned height = (unsigned)nodeMapRemoteDevice->FindNode<IntNode>("Height")->Value();
            img_.Resize(width, height);
            binSize_ = binFactor;
        }
        catch (std::exception& e)
        {
            LogMessage("IDS exception: Could not determine ROI parameters.");
            LogMessage(e.what());
        }


        ret = GetBoundaries(nodeMapRemoteDevice->FindNode<FloatNode>("AcquisitionFrameRate"), frameRateMin_, frameRateMax_, frameRateInc_);
        SetPropertyLimits("MDA framerate", frameRateMin_, frameRateMax_);

        OnPropertyChanged("Binning", std::to_string(binSize_).c_str());
        ret = DEVICE_OK;
        break;
    }
    case MM::BeforeGet:
    {
        ret = DEVICE_OK;
        pProp->Set(std::to_string(binSize_).c_str());
        break;
    }
    default:
        break;
    }
    return ret;
}


/**
* Handles the FrameRate property. The main way for the user to set the frame rate.
* @param pProp - pointer to property
* @param eAct - type of action performed on property (eg get/set)
* @returns Integer status code - Returns DEVICE_OK on success
*/
int CIDSPeakCamera::OnFrameRate(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        pProp->Set(frameRateCur_);
    }
    else if (eAct == MM::AfterSet)
    {
        double framerateTemp;
        pProp->Get(framerateTemp);
        SetFrameRate(framerateTemp);
    }
    return DEVICE_OK;
}

/**
* Handles "PixelFormat" property. The main way for the user to set the dataformat
* used by the IDS camera. It is unclear if the IDS IPL can convert any PixelFormat
* to any PixelFormat. Therefore, we try to check whether this is possible, and inform
* the user of any problems (either the conversion of the new pixelformat to the
* active pixeltype is not supported, and the pixeltype is set to an available one, or
* there is no conversion possible, and the pixelformat change is reverted).
* @param pProp - pointer to property
* @param eAct - type of action performed on property (eg get/set)
* @returns Integer status code - Returns DEVICE_OK on success
*/
int CIDSPeakCamera::OnPixelFormat(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    int ret = DEVICE_OK;
    switch (eAct) {
    case MM::AfterSet: {
        if (IsCapturing())
            return DEVICE_CAMERA_BUSY_ACQUIRING;

        // Get requested pixelFormat
        std::string temp;
        pProp->Get(temp);

        // Check if pixelFormat didn't change
        if (temp == pixelFormat_) { return DEVICE_OK; }

        try
        {
            // Check for write access
            auto node = nodeMapRemoteDevice->FindNode<EnumNode>("PixelFormat");
            if (node->AccessStatus() != peak::core::nodes::NodeAccessStatus::ReadWrite) {
                LogMessage("IDS error: Currently no write access to PixelFormat.");
                return DEVICE_CAN_NOT_SET_PROPERTY;
            }

            // Write pixelFormat
            node->SetCurrentEntry(temp);

            // Get available pixelTypes
            std::vector<std::string> pixelTypes = GetAvailablePixelTypes(temp);
            if (pixelTypes.size() == 0)
            {
                std::stringstream ss;
                ss << "IDS warning:\n";
                ss << "    The IDS Image Processing Library(IPL) does not support the conversion of\n";
                ss << "    PixelFormat " << temp << " to any of the format that Micro-Manager accepts: Mono8, Mono16, BGRA8.\n";
                ss << "    Therefore, the change of PixelType is ignored.";
                LogMessage(ss.str());
                return DEVICE_ERR;
            }

            if (std::find(pixelTypes.begin(), pixelTypes.end(), pixelType_) == pixelTypes.end())
            {
                std::stringstream ss;
                ss << "IDS warning:\n";
                ss << "    The IDS Image Processing Library(IPL) does not support the conversion of\n";
                ss << "    the selected PixelFormat: " << temp << " to the currently active PixelType: " << pixelType_ << ".\n";
                ss << "    Therefore, the PixelType was set to the first available pixelType: " << pixelTypes[0] << ".\n";
                LogMessage(ss.str());
                SetPixelType(pixelTypes[0]);
            }
            SetAllowedValues(g_PixelType, pixelTypes);
            pixelFormat_ = temp;
        }
        catch (std::exception& e)
        {
            LogMessage("IDS exception: Could not set PixelFormat.");
            LogMessage(e.what());
            return DEVICE_CAN_NOT_SET_PROPERTY;
        }
    }
    case MM::BeforeGet:
        pProp->Set(pixelFormat_.c_str());
        break;
    }
    return ret;
}

/**
* Handles "PixelType" property. This is the main way the user sets the dataformat.
* There are 3 available formats: Mono8, Mono16, and BGRA8.
* @param pProp - pointer to property
* @param eAct - type of action performed on property (eg get/set)
* @returns Integer status code - Returns DEVICE_OK on success
*/
int CIDSPeakCamera::OnPixelType(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    switch (eAct)
    {
    case MM::AfterSet:
    {
        if (IsCapturing())
            return DEVICE_CAMERA_BUSY_ACQUIRING;

        std::string pixelType;
        pProp->Get(pixelType);
        return SetPixelType(pixelType);
    }
    case MM::BeforeGet:
    {
        if (nComponents_ == 1 && bitDepth_ == 8)
        {
            pProp->Set(g_PixelType_8bit);
        }
        else if (nComponents_ == 1 && bitDepth_ == 16)
        {
            pProp->Set(g_PixelType_16bit);
        }
        else if (nComponents_ == 4 && bitDepth_ == 8)
        {
            pProp->Set(g_PixelType_32bitBGRA);
        }
        else
        {
            return DEVICE_CAN_NOT_SET_PROPERTY;
        }
        break;
    }
    default:
        break;
    }
    return DEVICE_OK;
}

/**
* Handles "Auto whitebalance" property.
* @param pProp - pointer to property
* @param eAct - type of action performed on property (eg get/set)
* @returns Integer status code - Returns DEVICE_OK on success
*/
int CIDSPeakCamera::OnAutoWhitebalance(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    int ret = DEVICE_OK;
    if (eAct == MM::BeforeGet)
    {
        auto node = nodeMapRemoteDevice->FindNode<EnumNode>("BalanceWhiteAuto");
        if (!CheckAccess(node, AccessTypes::READONLY))
        {
            LogMessage("IDS error: Auto whitebalance read mode not available currently.");
            return DEVICE_ERR;
        }

        try
        {
            whitebalanceCurr_ = node->CurrentEntry()->SymbolicValue();
        }
        catch (std::exception& e)
        {
            LogMessage("IDS exception: Could not read the auto whitebalance mode.");
            LogMessage(e.what());
        }
        pProp->Set(whitebalanceCurr_.c_str());
    }
    else if (eAct == MM::AfterSet)
    {
        if (IsCapturing()) { return DEVICE_CAMERA_BUSY_ACQUIRING; }

        std::string autoWB;
        pProp->Get(autoWB);

        // Already in that mode
        if (autoWB == whitebalanceCurr_) { return DEVICE_OK; }

        auto node = nodeMapRemoteDevice->FindNode<EnumNode>("BalanceWhiteAuto");
        if (!CheckAccess(node, AccessTypes::READWRITE))
        {
            LogMessage("IDS error: Auto whitebalance readwrite mode not available currently.");
            return DEVICE_ERR;
        }

        try
        {
            node->SetCurrentEntry(autoWB);
            whitebalanceCurr_ = node->CurrentEntry()->SymbolicValue();
        }
        catch (std::exception& e)
        {
            LogMessage(("IDS exception: Could not set the auto whitebalance to: " + autoWB).c_str());
            LogMessage(e.what());
            return DEVICE_ERR;
        }
    }
    return ret;
}

/**
* Handles Analog Gain Master property.
* @param pProp - pointer to property
* @param eAct - type of action performed on property (eg get/set)
* @returns Integer status code - Returns DEVICE_OK on success
*/
int CIDSPeakCamera::OnAnalogMaster(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    int ret = DEVICE_OK;
    if (eAct == MM::BeforeGet)
    {
        ret = GetGain("AnalogAll", gainAnalogMaster_);
        if (DEVICE_OK != ret) { return ret; }
        pProp->Set(gainAnalogMaster_);
    }
    else if (eAct == MM::AfterSet)
    {
        if (IsCapturing()) { return DEVICE_CAMERA_BUSY_ACQUIRING; }

        double gain;
        pProp->Get(gain);

        ret = SetGain("AnalogAll", gain);
        if (DEVICE_OK != ret) { return ret; }
        gainAnalogMaster_ = gain;
    }
    return ret;
}

/**
* Handles Analog Gain Red property.
* @param pProp - pointer to property
* @param eAct - type of action performed on property (eg get/set)
* @returns Integer status code - Returns DEVICE_OK on success
*/
int CIDSPeakCamera::OnAnalogRed(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    int ret = DEVICE_OK;
    if (eAct == MM::BeforeGet)
    {
        ret = GetGain("AnalogRed", gainAnalogRed_);
        if (DEVICE_OK != ret) { return ret; }
        pProp->Set(gainAnalogRed_);
    }
    else if (eAct == MM::AfterSet)
    {
        if (IsCapturing()) { return DEVICE_CAMERA_BUSY_ACQUIRING; }

        double gain;
        pProp->Get(gain);

        ret = SetGain("AnalogRed", gain);
        if (DEVICE_OK != ret) { return ret; }
        gainAnalogRed_ = gain;
    }
    return ret;
}

/**
* Handles Analog Gain Green property.
* @param pProp - pointer to property
* @param eAct - type of action performed on property (eg get/set)
* @returns Integer status code - Returns DEVICE_OK on success
*/
int CIDSPeakCamera::OnAnalogGreen(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    int ret = DEVICE_OK;
    if (eAct == MM::BeforeGet)
    {
        ret = GetGain("AnalogGreen", gainAnalogGreen_);
        if (DEVICE_OK != ret) { return ret; }
        pProp->Set(gainAnalogGreen_);
    }
    else if (eAct == MM::AfterSet)
    {
        if (IsCapturing()) { return DEVICE_CAMERA_BUSY_ACQUIRING; }

        double gain;
        pProp->Get(gain);

        ret = SetGain("AnalogGreen", gain);
        if (DEVICE_OK != ret) { return ret; }
        gainAnalogGreen_ = gain;
    }
    return ret;
}

/**
* Handles Analog Gain Blue property.
* @param pProp - pointer to property
* @param eAct - type of action performed on property (eg get/set)
* @returns Integer status code - Returns DEVICE_OK on success
*/
int CIDSPeakCamera::OnAnalogBlue(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    int ret = DEVICE_OK;
    if (eAct == MM::BeforeGet)
    {
        ret = GetGain("AnalogBlue", gainAnalogBlue_);
        if (DEVICE_OK != ret) { return ret; }
        pProp->Set(gainAnalogBlue_);
    }
    else if (eAct == MM::AfterSet)
    {
        if (IsCapturing()) { return DEVICE_CAMERA_BUSY_ACQUIRING; }

        double gain;
        pProp->Get(gain);

        ret = SetGain("AnalogBlue", gain);
        if (DEVICE_OK != ret) { return ret; }
        gainAnalogBlue_ = gain;
    }
    return ret;
}

/**
* Handles Digital Gain Master property.
* @param pProp - pointer to property
* @param eAct - type of action performed on property (eg get/set)
* @returns Integer status code - Returns DEVICE_OK on success
*/
int CIDSPeakCamera::OnDigitalMaster(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    int ret = DEVICE_OK;
    if (eAct == MM::BeforeGet)
    {
        ret = GetGain("DigitalAll", gainDigitalMaster_);
        if (DEVICE_OK != ret) { return ret; }
        pProp->Set(gainDigitalMaster_);
    }
    else if (eAct == MM::AfterSet)
    {
        if (IsCapturing()) { return DEVICE_CAMERA_BUSY_ACQUIRING; }

        double gain;
        pProp->Get(gain);

        ret = SetGain("DigitalAll", gain);
        if (DEVICE_OK != ret) { return ret; }
        gainDigitalMaster_ = gain;
    }
    return ret;
}

/**
* Handles Digital Gain Red property.
* @param pProp - pointer to property
* @param eAct - type of action performed on property (eg get/set)
* @returns Integer status code - Returns DEVICE_OK on success
*/
int CIDSPeakCamera::OnDigitalRed(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    int ret = DEVICE_OK;
    if (eAct == MM::BeforeGet)
    {
        ret = GetGain("DigitalRed", gainDigitalRed_);
        if (DEVICE_OK != ret) { return ret; }
        pProp->Set(gainDigitalRed_);
    }
    else if (eAct == MM::AfterSet)
    {
        if (IsCapturing()) { return DEVICE_CAMERA_BUSY_ACQUIRING; }

        double gain;
        pProp->Get(gain);

        ret = SetGain("DigitalRed", gain);
        if (DEVICE_OK != ret) { return ret; }
        gainDigitalRed_ = gain;
    }
    return ret;
}

/**
* Handles Digital Gain Green property.
* @param pProp - pointer to property
* @param eAct - type of action performed on property (eg get/set)
* @returns Integer status code - Returns DEVICE_OK on success
*/
int CIDSPeakCamera::OnDigitalGreen(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    int ret = DEVICE_OK;
    if (eAct == MM::BeforeGet)
    {
        ret = GetGain("DigitalGreen", gainDigitalGreen_);
        if (DEVICE_OK != ret) { return ret; }
        pProp->Set(gainDigitalGreen_);
    }
    else if (eAct == MM::AfterSet)
    {
        if (IsCapturing()) { return DEVICE_CAMERA_BUSY_ACQUIRING; }

        double gain;
        pProp->Get(gain);

        ret = SetGain("DigitalGreen", gain);
        if (DEVICE_OK != ret) { return ret; }
        gainDigitalGreen_ = gain;
    }
    return ret;
}

/**
* Handles Digital Gain Blue property.
* @param pProp - pointer to property
* @param eAct - type of action performed on property (eg get/set)
* @returns Integer status code - Returns DEVICE_OK on success
*/
int CIDSPeakCamera::OnDigitalBlue(MM::PropertyBase* pProp, MM::ActionType eAct)
{

    int ret = DEVICE_OK;
    if (eAct == MM::BeforeGet)
    {
        ret = GetGain("DigitalBlue", gainDigitalBlue_);
        if (DEVICE_OK != ret) { return ret; }
        pProp->Set(gainDigitalBlue_);
    }
    else if (eAct == MM::AfterSet)
    {
        if (IsCapturing()) { return DEVICE_CAMERA_BUSY_ACQUIRING; }

        double gain;
        pProp->Get(gain);

        ret = SetGain("DigitalBlue", gain);
        if (DEVICE_OK != ret) { return ret; }
        gainDigitalBlue_ = gain;
    }
    return ret;
}

/**
* Handles TriggerMode property. "On" means the camera waits for a trigger
* before starting the capture, whereas "Off" causes the camera to acquire
* the images immediately after calling the function.
* @param pProp - pointer to property
* @param eAct - type of action performed on property (eg get/set)
* @returns Integer status code - Returns DEVICE_OK on success
*/
int CIDSPeakCamera::OnTriggerMode(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        pProp->Set(triggerMode_.c_str());
    }
    else if (eAct == MM::AfterSet)
    {
        std::string temp;
        pProp->Get(temp);
        if (temp == triggerMode_) { return DEVICE_OK; }

        auto node = nodeMapRemoteDevice->FindNode<EnumNode>("TriggerMode");
        if (!CheckAccess(node, AccessTypes::READWRITE))
        {
            LogMessage("IDS error: Could not set TriggerMode, due to lack of READWRITE access.");
            return DEVICE_CAN_NOT_SET_PROPERTY;
        }
        try
        {
            node->SetCurrentEntry(temp); // Actually set node
            triggerMode_ = node->CurrentEntry()->SymbolicValue(); // Verify value
        }
        catch (std::exception& e)
        {
            LogMessage("IDS exception: Could not set Trigger Mode.");
            LogMessage(e.what());
            return DEVICE_CAN_NOT_SET_PROPERTY;
        }
    }
    return DEVICE_OK;
}

/**
* Handles TriggerSelector property. Allows the user to select what happens when
* a trigger event happens (eg start acquisition, aquire 1 frame, aquire 1 line, stop acquisition, etc).
* @param pProp - pointer to property
* @param eAct - type of action performed on property (eg get/set)
* @returns Integer status code - Returns DEVICE_OK on success
*/
int CIDSPeakCamera::OnTriggerSelector(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        pProp->Set(triggerSelector_.c_str());
    }
    else if (eAct == MM::AfterSet)
    {
        std::string temp;
        pProp->Get(temp);
        if (temp == triggerSelector_) { return DEVICE_OK; }

        auto node = nodeMapRemoteDevice->FindNode<EnumNode>("TriggerSelector");
        if (!CheckAccess(node, AccessTypes::READWRITE))
        {
            LogMessage("IDS error: Could not set TriggerSelector, due to lack of READWRITE access.");
            return DEVICE_CAN_NOT_SET_PROPERTY;
        }
        try
        {
            node->SetCurrentEntry(temp);
            triggerSelector_ = node->CurrentEntry()->SymbolicValue();
        }
        catch (std::exception& e)
        {
            LogMessage("IDS exception: Could not set TriggerSelector.");
            LogMessage(e.what());
            return DEVICE_CAN_NOT_SET_PROPERTY;
        }
    }
    return DEVICE_OK;
}

/**
* Handles TriggerSource property. Allows the user to set the source of a trigger event,
* like a Software trigger, or any of the IO-Pins of the camera.
* @param pProp - pointer to property
* @param eAct - type of action performed on property (eg get/set)
* @returns Integer status code - Returns DEVICE_OK on success
*/
int CIDSPeakCamera::OnTriggerSource(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        pProp->Set(triggerSource_.c_str());
    }
    else if (eAct == MM::AfterSet)
    {
        std::string temp;
        pProp->Get(temp);
        if (temp == triggerSource_) { return DEVICE_OK; }

        auto node = nodeMapRemoteDevice->FindNode<EnumNode>("TriggerSource");
        if (!CheckAccess(node, AccessTypes::READWRITE))
        {
            LogMessage("IDS error: Could not set TriggerSource, due to lack of READWRITE access.");
            return DEVICE_CAN_NOT_SET_PROPERTY;
        }
        try
        {
            node->SetCurrentEntry(temp);
            triggerSource_ = node->CurrentEntry()->SymbolicValue();
        }
        catch (std::exception& e)
        {
            LogMessage("IDS exception: Could not set TriggerSouce.");
            LogMessage(e.what());
            return DEVICE_CAN_NOT_SET_PROPERTY;
        }
    }
    return DEVICE_OK;
}

/**
* Handles Trigger Activation property. Setting this property to 1 launches a trigger event.
* @param pProp - pointer to property
* @param eAct - type of action performed on property (eg get/set)
* @returns Integer status code - Returns DEVICE_OK on success
*/
int CIDSPeakCamera::OnTriggerActivation(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        pProp->Set(triggerActivate_);
    }
    else if (eAct == MM::AfterSet)
    {
        long temp = 0;
        pProp->Get(temp);
        if (temp == triggerActivate_) { return DEVICE_OK; }
        if (temp == 0) { return DEVICE_OK; }

        auto node = nodeMapRemoteDevice->FindNode<CommandNode>("TriggerSoftware");
        if (!CheckAccess(node, AccessTypes::WRITEONLY))
        {
            LogMessage("IDS error: Could not activate trigger, due to lack of READWRITE access.");
            return DEVICE_CAN_NOT_SET_PROPERTY;
        }
        try
        {
            node->Execute();
            node->WaitUntilDone();
        }
        catch (std::exception& e)
        {
            LogMessage("IDS exception: Could not activate trigger.");
            LogMessage(e.what());
            return DEVICE_CAN_NOT_SET_PROPERTY;
        }
    }
    return DEVICE_OK;
}

/**
* Handles Trigger Edge property. Allows the user to set when to launch the trigger,
* eg on RisingEdge, FallingEdge, AnyEdge, etc
* @param pProp - pointer to property
* @param eAct - type of action performed on property (eg get/set)
* @returns Integer status code - Returns DEVICE_OK on success
*/
int CIDSPeakCamera::OnTriggerEdge(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        pProp->Set(triggerEdge_.c_str());
    }
    else if (eAct == MM::AfterSet)
    {
        std::string temp;
        pProp->Get(temp);
        if (temp == triggerEdge_) { return DEVICE_OK; }

        auto node = nodeMapRemoteDevice->FindNode<EnumNode>("TriggerActivation");
        if (!CheckAccess(node, AccessTypes::READWRITE))
        {
            LogMessage("IDS error: Could not set TriggerEdge, due to lack of READWRITE access.");
            return DEVICE_CAN_NOT_SET_PROPERTY;
        }
        try
        {
            node->SetCurrentEntry(temp);
            triggerEdge_ = node->CurrentEntry()->SymbolicValue();
        }
        catch (std::exception& e)
        {
            LogMessage("IDS exception: Could not set TriggerEdge.");
            LogMessage(e.what());
            return DEVICE_CAN_NOT_SET_PROPERTY;
        }
    }
    return DEVICE_OK;
}

/**
* Handles Sensor Temperature property.
* @param pProp - pointer to property
* @param eAct - type of action performed on property (eg get/set)
* @returns Integer status code - Returns DEVICE_OK on success
*/
int CIDSPeakCamera::OnSensorTemp(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    // This is a readonly function
    if (eAct == MM::BeforeGet)
    {
        try
        {
            auto node = nodeMapRemoteDevice->FindNode<FloatNode>("DeviceTemperature");
            if (!CheckAccess(node, AccessTypes::READONLY))
            {
                LogMessage("IDS error: Could not read temperature.");
                return DEVICE_ERR;
            }
            sensorTemp_ = node->Value();
        }
        catch (std::exception& e)
        {
            LogMessage("IDS exception: Could not read temperature.");
            LogMessage(e.what());
            return DEVICE_ERR;
        }
        pProp->Set(sensorTemp_);
    }
    return DEVICE_OK;
}

///////////////////////////////////////////////////////////////////////////////
// CIDSPeakCamera Initializers
///////////////////////////////////////////////////////////////////////////////

/**
* Create general camera description properties
* @returns Integer status code - Returns DEVICE_OK on success
*/
int CIDSPeakCamera::InitializeCameraDescription()
{
    // Name
    int ret = CreateStringProperty(MM::g_Keyword_Name, g_IDSPeakCameraName, true);
    if (DEVICE_OK != ret) { return ret; }

    // Description
    ret = CreateStringProperty(MM::g_Keyword_Description, "IDS Peak Camera Adapter", true);
    if (DEVICE_OK != ret) { return ret; }

    // CameraName and SerialNumber
    try
    {
        modelName_ = descriptor->ModelName();
        serialNumber_ = descriptor->SerialNumber();
    }
    catch (std::exception& e)
    {
        LogMessage("IDS exception: Could not get model name or serial number.");
        LogMessage(e.what());
        return DEVICE_ERR;
    }
    ret = CreateStringProperty(MM::g_Keyword_CameraName, modelName_.c_str(), true);
    if (DEVICE_OK != ret) { return ret; }
    ret = CreateStringProperty("Serial Number", serialNumber_.c_str(), true);
    if (DEVICE_OK != ret) { return ret; }

    // Sensor information
    try
    {
        sensorHeight_ = (long)nodeMapRemoteDevice->FindNode<IntNode>("SensorHeight")->Value();
        sensorWidth_ = (long)nodeMapRemoteDevice->FindNode<IntNode>("SensorWidth")->Value();
    }
    catch (std::exception& e)
    {
        LogMessage("IDS exception: Could not get sensor information (heigt/width).");
        LogMessage(e.what());
        return DEVICE_ERR;
    }
    ret = CreateIntegerProperty("Sensor height", sensorHeight_, true);
    ret = CreateIntegerProperty("Sensor width", sensorWidth_, true);
    if (DEVICE_OK != ret) { return ret; }

    // Get current acquisition Mode
    try
    {
        acquisitionMode_ = nodeMapRemoteDevice->FindNode<EnumNode>("AcquisitionMode")
            ->CurrentEntry()->SymbolicValue();
    }
    catch (std::exception& e)
    {
        LogMessage("IDS exception: Could not get acquisition mode.");
        LogMessage(e.what());
        return DEVICE_ERR;
    }
    return ret;
}

/**
* Initialize exposure time property. NOTE: Micro-Manager expects the
* exposure time in milliseconds, whereas IDS expects them in microseconds.
* @returns Integer status code - Returns DEVICE_OK on success
*/
int CIDSPeakCamera::InitializeExposureTime()
{
    // Exposure time, divide by 1000 to convert us to ms.
    try
    {
        auto node = nodeMapRemoteDevice->FindNode<FloatNode>("ExposureTime");
        exposureCur_ = node->Value() / 1000;
        exposureMin_ = node->Minimum() / 1000;
        exposureMax_ = node->Maximum() / 1000;
        exposureInc_ = node->Increment() / 1000;
    }
    catch (std::exception& e)
    {
        LogMessage("IDS exception: Could not read exposure time.");
        LogMessage(e.what());
        return DEVICE_ERR;
    }
    int ret = CreateFloatProperty(MM::g_Keyword_Exposure, exposureCur_, false);
    if (DEVICE_OK != ret) { return ret; }
    ret = SetPropertyLimits(MM::g_Keyword_Exposure, exposureMin_, exposureMax_);
    return ret;
}

/**
* Initialize FrameRate property
* @returns Integer status code - Returns DEVICE_OK on success
*/
int CIDSPeakCamera::InitializeFramerate()
{
    try
    {
        auto node = nodeMapRemoteDevice->FindNode<FloatNode>("AcquisitionFrameRate");
        frameRateCur_ = node->Value();
        frameRateMin_ = node->Minimum();
        frameRateMax_ = node->Maximum();
        if (node->HasConstantIncrement())
        {
            frameRateInc_ = node->Increment();
        }
        else
        {
            frameRateInc_ = 0.001;
        }
    }
    catch (std::exception& e) {
        LogMessage("IDS exception: Could not read acquisition framerate");
        LogMessage(e.what());
        return DEVICE_ERR;
    }
    CPropertyAction* pAct = new CPropertyAction(this, &CIDSPeakCamera::OnFrameRate);
    int ret = CreateFloatProperty("MDA framerate", 1, false, pAct);
    if (DEVICE_OK != ret) { return ret; }
    ret = SetPropertyLimits("MDA framerate", frameRateMin_, frameRateMax_);
    return ret;
}

///**
//* Initialize the Binning property, and set it to 1.
//* @returns Integer status code - Returns DEVICE_OK on success
//*/
//int CIDSPeakCamera::InitializeBinning()
//{
//    std::vector<std::string> binningValues;
//    try
//    {
//        binningSelector_ = nodeMapRemoteDevice->FindNode<EnumNode>("BinningSelector")->CurrentEntry()->SymbolicValue();
//
//        int64_t maxVal = nodeMapRemoteDevice->FindNode<IntNode>("BinningHorizontal")->Maximum();
//        int64_t i = 1;
//        while (i <= maxVal)
//        {
//            binningValues.push_back(std::to_string(i));
//            i *= 2;
//        }
//        nodeMapRemoteDevice->FindNode<IntNode>("BinningVertical")->SetValue(1);
//        nodeMapRemoteDevice->FindNode<IntNode>("BinningHorizontal")->SetValue(1);
//        binSize_ = 1;
//    }
//    catch (std::exception& e)
//    {
//        LogMessage("IDS exception: An error occurred while getting available binning options.");
//        LogMessage(e.what());
//        return DEVICE_ERR;
//    }
//
//    CPropertyAction* pAct = new CPropertyAction(this, &CIDSPeakCamera::OnBinning);
//    int ret = CreateIntegerProperty(MM::g_Keyword_Binning, 1, false, pAct);
//    if (DEVICE_OK != ret) { return ret; }
//    ret = SetAllowedValues(MM::g_Keyword_Binning, binningValues);
//    return ret;
//}


/**
* Initialize the Binning property, including the new Engine selector (FPGA vs Sensor).
* @returns Integer status code - Returns DEVICE_OK on success
*/
int CIDSPeakCamera::InitializeBinning()
{
    std::vector<std::string> binningValues;
    try
    {
        binningSelector_ = nodeMapRemoteDevice->FindNode<EnumNode>("BinningSelector")->CurrentEntry()->SymbolicValue();

        nodeMapRemoteDevice->FindNode<IntNode>("BinningVertical")->SetValue(1);
        nodeMapRemoteDevice->FindNode<IntNode>("BinningHorizontal")->SetValue(1);
        binSize_ = 1;
        // Calculate valid binning factors for the CURRENT engine (Sensor or FPGA)
        int64_t maxVal = nodeMapRemoteDevice->FindNode<IntNode>("BinningHorizontal")->Maximum();
        int64_t i = 1;
        while (i <= maxVal)
        {
            binningValues.push_back(std::to_string(i));
            i *= 2;
        }
    }
    catch (std::exception& e)
    {
        LogMessage("IDS exception: An error occurred while getting available binning options.");
        LogMessage(e.what());
        return DEVICE_ERR;
    }
    CPropertyAction* pAct = new CPropertyAction(this, &CIDSPeakCamera::OnBinning);
    int ret = CreateIntegerProperty(MM::g_Keyword_Binning, 1, false, pAct);
    if (DEVICE_OK != ret) {
        return ret; 
    }
    ret = SetAllowedValues(MM::g_Keyword_Binning, binningValues);
    if (DEVICE_OK != ret) { return ret; }

    CPropertyAction* pActEngine = new CPropertyAction(this, &CIDSPeakCamera::OnBinningEngine);
    std::string initialEngine = (binningSelector_ == "Region0") ? "FPGA" : "Sensor";
    ret = CreateStringProperty("BinningEngine", initialEngine.c_str(), false, pActEngine);
    if (DEVICE_OK != ret) { 
        return ret; 
    }
    std::vector<std::string> binningDrivers;
    binningDrivers.push_back("Sensor");
    binningDrivers.push_back("FPGA");
    ret = SetAllowedValues("BinningEngine", binningDrivers);

    return ret;
}



/**
* Initialize IDS PixelFormat and MM PixelType and the conversion between them.
* @returns Integer status code - Returns DEVICE_OK on success
*/
int CIDSPeakCamera::InitializePixelTypes()
{
    // IDS PixelFormat
    std::vector<std::string> availablePixelFormats = GetAvailableEntries("PixelFormat");
    try
    {
        auto node = nodeMapRemoteDevice->FindNode<EnumNode>("PixelFormat");
        pixelFormat_ = node->CurrentEntry()->SymbolicValue();
    }
    catch (std::exception& e)
    {
        LogMessage("IDS exception: An error occurred while reading all available PixelFormats");
        LogMessage(e.what());
        return DEVICE_ERR;
    }
    CPropertyAction* pAct = new CPropertyAction(this, &CIDSPeakCamera::OnPixelFormat);
    int ret = CreateStringProperty("IDS PixelFormat", pixelFormat_.c_str(), false, pAct);
    if (DEVICE_OK != ret) { return ret; }
    ret = SetAllowedValues("IDS PixelFormat", availablePixelFormats);
    if (DEVICE_OK != ret) { return ret; }

    // MM PixelType
    std::vector<std::string> availablePixelTypes = GetAvailablePixelTypes(pixelFormat_);
    pAct = new CPropertyAction(this, &CIDSPeakCamera::OnPixelType);
    ret = CreateStringProperty(g_PixelType, g_PixelType_8bit, false, pAct);
    if (DEVICE_OK != ret) { return ret; }
    ret = SetAllowedValues(g_PixelType, availablePixelTypes);
    return ret;
}

/**
* Initialize the ROI property and set it to full frame.
* @returns Integer status code - Returns DEVICE_OK on success
*/
int CIDSPeakCamera::InitializeROI()
{
    try
    {
        // Get minimum roi values
        roiOffsetXMin_ = (unsigned int)nodeMapRemoteDevice->FindNode<IntNode>("OffsetX")->Minimum();
        roiOffsetYMin_ = (unsigned int)nodeMapRemoteDevice->FindNode<IntNode>("OffsetY")->Minimum();
        roiWidthMin_ = (unsigned int)nodeMapRemoteDevice->FindNode<IntNode>("Width")->Minimum();
        roiHeightMin_ = (unsigned int)nodeMapRemoteDevice->FindNode<IntNode>("Height")->Minimum();

        // Set the minimum ROI. This removes any size restrictions due to previous ROI settings
        nodeMapRemoteDevice->FindNode<IntNode>("OffsetX")->SetValue(roiOffsetXMin_);
        nodeMapRemoteDevice->FindNode<IntNode>("OffsetY")->SetValue(roiOffsetYMin_);
        nodeMapRemoteDevice->FindNode<IntNode>("Width")->SetValue(roiWidthMin_);
        nodeMapRemoteDevice->FindNode<IntNode>("Height")->SetValue(roiHeightMin_);

        // Get increment roi values (stepsize in which roi can be changed)
        roiOffsetXIncrement_ = (unsigned int)nodeMapRemoteDevice->FindNode<IntNode>("OffsetX")->Increment();
        roiOffsetYIncrement_ = (unsigned int)nodeMapRemoteDevice->FindNode<IntNode>("OffsetY")->Increment();
        roiWidthIncrement_ = (unsigned int)nodeMapRemoteDevice->FindNode<IntNode>("Width")->Increment();
        roiHeightIncrement_ = (unsigned int)nodeMapRemoteDevice->FindNode<IntNode>("Height")->Increment();

        // Get maximum roi values (basically sensor size)
        roiOffsetXMax_ = (unsigned int)nodeMapRemoteDevice->FindNode<IntNode>("OffsetX")->Maximum();
        roiOffsetYMax_ = (unsigned int)nodeMapRemoteDevice->FindNode<IntNode>("OffsetY")->Maximum();
        roiWidthMax_ = (unsigned int)nodeMapRemoteDevice->FindNode<IntNode>("Width")->Maximum();
        roiHeightMax_ = (unsigned int)nodeMapRemoteDevice->FindNode<IntNode>("Height")->Maximum();

        // Maximize ROI
        nodeMapRemoteDevice->FindNode<IntNode>("Width")->SetValue(roiWidthMax_);
        nodeMapRemoteDevice->FindNode<IntNode>("Height")->SetValue(roiHeightMax_);
    }
    catch (std::exception& e)
    {
        LogMessage("IDS exception: An error occurred while getting the minimum and maximum ROI");
        LogMessage(e.what());
        return DEVICE_ERR;
    }
    return DEVICE_OK;
}

/**
* Initializes the buffer from the MM side, and device side.
* @returns Integer status code - Returns DEVICE_OK on success
*/
int CIDSPeakCamera::InitializeBuffer()
{
    img_ = ImgBuffer(roiWidthMax_, roiHeightMax_, 1);
    try
    {
        auto dataStreams = device->DataStreams();
        if (dataStreams.empty())
        {
            LogMessage("IDS error: Could not find any data streams.");
            return DEVICE_ERR;
        }
        dataStream = device->DataStreams().at(0)->OpenDataStream();
        nodeMapDataStream = dataStream->NodeMaps().at(0);
        nBuffers_ = 20;
    }
    catch (std::exception& e)
    {
        LogMessage("IDS exception: Could not prepare camera buffer");
        LogMessage(e.what());
        return DEVICE_ERR;
    }
    return DEVICE_OK;
}

/**
* Initializes the auto whitebalance property.
* @returns Integer status code - Returns DEVICE_OK on success
*/
int CIDSPeakCamera::InitializeAutoWhiteBalance()
{
    if (!nodeMapRemoteDevice->HasNode<EnumNode>("GainAuto"))
    {
        LogMessage("IDS warning: This IDS camera does not seem support auto whitebalance.");
        LogMessage("\tPlease verify this with the specifications on the manufacturers website. Look for \"Auto Gain\"");
        return DEVICE_OK;
    }
    CPropertyAction* pAct = new CPropertyAction(this, &CIDSPeakCamera::OnAutoWhitebalance);
    int ret = CreateStringProperty("Auto white balance", whitebalanceCurr_.c_str(), false, pAct);
    if (DEVICE_OK != ret) { return ret; }
    std::vector<std::string> whitebalanceValues = GetAvailableEntries("GainAuto");
    ret = SetAllowedValues("Auto white balance", whitebalanceValues);
    return ret;
}

/**
* Initializes the analog gain.
* @returns Integer status code - Returns DEVICE_OK on success
*/
int CIDSPeakCamera::InitializeAnalogGain()
{
    try
    {
        if (!nodeMapRemoteDevice->HasNode<EnumNode>("GainSelector"))
        {
            LogMessage("IDS warning: This camera does not support Analog gain.");
            return DEVICE_OK;
        }
    }
    catch (std::exception& e)
    {
        LogMessage("IDS exception: Could not verify if Analog gain is supported.");
        LogMessage(e.what());
        return DEVICE_ERR;
    }
    int ret = CreateGain("AnalogAll", gainAnalogMaster_, &CIDSPeakCamera::OnAnalogMaster);
    if (DEVICE_OK != ret) { return ret; }
    ret = CreateGain("AnalogRed", gainAnalogRed_, &CIDSPeakCamera::OnAnalogRed);
    if (DEVICE_OK != ret) { return ret; }
    ret = CreateGain("AnalogGreen", gainAnalogGreen_, &CIDSPeakCamera::OnAnalogGreen);
    if (DEVICE_OK != ret) { return ret; }
    ret = CreateGain("AnalogBlue", gainAnalogBlue_, &CIDSPeakCamera::OnAnalogBlue);
    return ret;
}

/**
* Initializes the digital gain.
* @returns Integer status code - Returns DEVICE_OK on success
*/
int CIDSPeakCamera::InitializeDigitalGain()
{
    try
    {
        if (!nodeMapRemoteDevice->HasNode<EnumNode>("GainSelector"))
        {
            LogMessage("IDS warning: This camera does not support Digital gain.");
            return DEVICE_OK;
        }
    }
    catch (std::exception& e)
    {
        LogMessage("IDS exception: Could not verify if Digital gain is supported.");
        LogMessage(e.what());
        return DEVICE_ERR;
    }

    int ret = CreateGain("DigitalAll", gainDigitalMaster_, &CIDSPeakCamera::OnDigitalMaster);
    if (DEVICE_OK != ret) { return ret; }
    ret = CreateGain("DigitalRed", gainDigitalRed_, &CIDSPeakCamera::OnDigitalRed);
    if (DEVICE_OK != ret) { return ret; }
    ret = CreateGain("DigitalGreen", gainDigitalGreen_, &CIDSPeakCamera::OnDigitalGreen);
    if (DEVICE_OK != ret) { return ret; }
    ret = CreateGain("DigitalBlue", gainDigitalBlue_, &CIDSPeakCamera::OnDigitalBlue);
    return ret;
}

/**
* Initializes the temperature property.
* @returns Integer status code - Returns DEVICE_OK on success
*/
int CIDSPeakCamera::InitializeTemperature()
{
    try
    {
        if (!nodeMapRemoteDevice->HasNode<FloatNode>("DeviceTemperature"))
        {
            LogMessage(std::string("IDS warning: This device does not support temperature monitoring. ") +
                "The temperature property is skipped."
            );
            return DEVICE_OK;
        }
        auto node = nodeMapRemoteDevice->FindNode<FloatNode>("DeviceTemperature");
        if (!CheckAccess(node, AccessTypes::READONLY))
        {
            LogMessage("IDS warning: Temperature not available");
            return DEVICE_OK;
        }
        sensorTemp_ = node->Value();
    }
    catch (std::exception& e)
    {
        LogMessage("IDS exception: An error occurred while reading the temperature");
        LogMessage(e.what());
        return DEVICE_ERR;
    }
    CPropertyAction* pAct = new CPropertyAction(this, &CIDSPeakCamera::OnSensorTemp);
    int ret = CreateFloatProperty("Sensor temperature", sensorTemp_, true, pAct);
    return ret;
}

/**
* Initializes the trigger property.
* @returns Integer status code - Returns DEVICE_OK on success
*/
int CIDSPeakCamera::InitializeTrigger()
{
    try
    {
        if (!nodeMapRemoteDevice->HasNode<EnumNode>("TriggerMode"))
        {
            LogMessage("IDS warning: This camera does not support triggers. Please verify");
        }
        // Trigger Enable/Disable
        triggerMode_ = nodeMapRemoteDevice->FindNode<EnumNode>("TriggerMode")->CurrentEntry()->SymbolicValue();
        CPropertyAction* pAct = new CPropertyAction(this, &CIDSPeakCamera::OnTriggerMode);
        int ret = CreateStringProperty("TriggerMode", triggerMode_.c_str(), false, pAct, false);
        if (DEVICE_OK != ret) { return ret; }
        std::vector<std::string> allowedModeValues = GetAvailableEntries("TriggerMode");
        ret = SetAllowedValues("TriggerMode", allowedModeValues);
        if (DEVICE_OK != ret) { return ret; }

        // Trigger mode selector
        pAct = new CPropertyAction(this, &CIDSPeakCamera::OnTriggerSelector);
        triggerSelector_ = nodeMapRemoteDevice->FindNode<EnumNode>("TriggerSelector")->CurrentEntry()->SymbolicValue();
        ret = CreateStringProperty("TriggerSelector", triggerSelector_.c_str(), false, pAct, false);
        if (DEVICE_OK != ret) { return ret; }
        std::vector<std::string> allowedSelectorValues = GetAvailableEntries("TriggerSelector");
        ret = SetAllowedValues("TriggerSelector", allowedSelectorValues);
        if (DEVICE_OK != ret) { return ret; }

        // Trigger source
        pAct = new CPropertyAction(this, &CIDSPeakCamera::OnTriggerSource);
        triggerSource_ = nodeMapRemoteDevice->FindNode<EnumNode>("TriggerSource")->CurrentEntry()->SymbolicValue();
        ret = CreateStringProperty("TriggerSource", triggerSource_.c_str(), false, pAct, false);
        if (DEVICE_OK != ret) { return ret; }
        std::vector<std::string> allowedSourceValues = GetAvailableEntries("TriggerSource");
        ret = SetAllowedValues("TriggerSource", allowedSourceValues);
        if (DEVICE_OK != ret) { return ret; }

        // Trigger activation
        pAct = new CPropertyAction(this, &CIDSPeakCamera::OnTriggerActivation);
        ret = CreateIntegerProperty("TriggerActivate", 0, false, pAct, false);
        if (DEVICE_OK != ret) { return ret; }
        std::vector<std::string> allowedActivateValues = { "0", "1" };
        ret = SetAllowedValues("TriggerActivate", allowedActivateValues);
        if (DEVICE_OK != ret) { return ret; }

        // Trigger Edge
        pAct = new CPropertyAction(this, &CIDSPeakCamera::OnTriggerEdge);
        triggerEdge_ = nodeMapRemoteDevice->FindNode<EnumNode>("TriggerActivation")->CurrentEntry()->SymbolicValue();
        ret = CreateStringProperty("TriggerEdge", triggerEdge_.c_str(), false, pAct, false);
        if (DEVICE_OK != ret) { return ret; }
        std::vector<std::string> allowedEdgeValues = GetAvailableEntries("TriggerActivation");
        ret = SetAllowedValues("TriggerEdge", allowedEdgeValues);
        return ret;
    }
    catch (std::exception& e)
    {
        LogMessage("IDS exception: Could not initialize Trigger.");
        LogMessage(e.what());
        return DEVICE_ERR;
    }
    return DEVICE_OK;
}

////////////////////////////////////////////////////////////////////////////////
// Gain helper functions
////////////////////////////////////////////////////////////////////////////////

/**
* Helper function that creates a specific gain property, and reports its current value
* @param gainType - String name of gain to be created
* @param gain - double ouput of the current value of the gain
* @param fpt - function pointer to the associated On... Action Handler
* @returns Integer status code - Returns DEVICE_OK on success
*/
int CIDSPeakCamera::CreateGain(std::string gainType, double& gain,
    int(CIDSPeakCamera::* fpt)(MM::PropertyBase* pProp, MM::ActionType eAct))
{
    double min = 0, max = 0, increment = 0;
    try
    {
        if (!nodeMapRemoteDevice->FindNode<EnumNode>("GainSelector")->HasEntry(gainType))
        {
            LogMessage("IDS message: " + gainType + " not supported by this camera.");
            return DEVICE_OK;
        }

        nodeMapRemoteDevice->FindNode<EnumNode>("GainSelector")
            ->SetCurrentEntry(gainType);
        auto node = nodeMapRemoteDevice->FindNode<FloatNode>("Gain");
        if (!CheckAccess(node, AccessTypes::READWRITE))
        {
            LogMessage(("IDS warning: " + gainType + " not available").c_str());
            return DEVICE_OK;
        }
        int ret = GetBoundaries(node, min, max, increment);
        if (DEVICE_OK != ret) { return ret; }
        min = node->Minimum();
        max = node->Maximum();
        gain = node->Value();
    }
    catch (std::exception& e)
    {
        LogMessage("IDS exception: Error occurred during creation of: " + gainType);
        LogMessage(e.what());
        return DEVICE_ERR;
    }

    CPropertyAction* pAct = new CPropertyAction(this, fpt);
    int ret = CreateFloatProperty(("Gain " + gainType).c_str(), gain, false, pAct);
    if (DEVICE_OK != ret) { return ret; }
    SetPropertyLimits(("Gain " + gainType).c_str(), min, max);
    if (DEVICE_OK != ret) { return ret; }
    return DEVICE_OK;
}

/**
* Get gain of a gaintype.
* @param gainType - Name of the gain type to be retrieved
* @param gain - double output value
* @returns Integer status code - Returns DEVICE_OK on success
*/
int CIDSPeakCamera::GetGain(const std::string& gainType, double& gain)
{
    try
    {
        nodeMapRemoteDevice->FindNode<EnumNode>("GainSelector")->SetCurrentEntry(gainType);
        gain = nodeMapRemoteDevice->FindNode<FloatNode>("Gain")->Value();
    }
    catch (std::exception& e)
    {
        LogMessage("IDS exception: Could not get " + gainType);
        LogMessage(e.what());
        return DEVICE_ERR;
    }
    return DEVICE_OK;
}

/**
* Set gain of a gaintype
* @param gainType - Name of the gain type to be set
* @param gain - Value to be set
* @returns Integer status code - Returns DEVICE_OK on success
*/
int CIDSPeakCamera::SetGain(const std::string& gainType, double gain)
{
    if (IsCapturing()) { return DEVICE_CAMERA_BUSY_ACQUIRING; }

    try
    {
        nodeMapRemoteDevice->FindNode<EnumNode>("GainSelector")->SetCurrentEntry(gainType);
        nodeMapRemoteDevice->FindNode<FloatNode>("Gain")->SetValue(gain);
    }
    catch (std::exception& e)
    {
        LogMessage("IDS exception: Could not set " + gainType);
        LogMessage(e.what());
        return DEVICE_CAN_NOT_SET_PROPERTY;
    }
    return DEVICE_OK;
}

////////////////////////////////////////////////////////////////////////////////
// Image/buffer helper functions
////////////////////////////////////////////////////////////////////////////////

/**
* Sets the char buffer of the image to all zeros.
* @params img - ImgBuffer with already initialized buffer
*/
void CIDSPeakCamera::GenerateEmptyImage(ImgBuffer& img)
{
    if (img.Height() == 0 || img.Width() == 0 || img.Depth() == 0)
        return;
    MMThreadGuard g(imgPixelsLock_);
    unsigned char* pBuf = const_cast<unsigned char*>(img.GetPixels());
    memset(pBuf, 0, img.Height() * img.Width() * img.Depth());
}

/*
 * Inserts Image and MetaData into MMCore circular Buffer
 * @returns Integer status code - Returns DEVICE_OK on success
 */
int CIDSPeakCamera::InsertImage()
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
    int ret = GetCoreCallback()->InsertImage(this, img_.GetPixels(),
        img_.Width(),
        img_.Height(),
        img_.Depth(),
        md.Serialize().c_str());
    return ret;
}

/**
* Flushes the camera datastream and revokes the allocated buffers.
* @returns Integer status code - Returns DEVICE_OK on success
*/
int CIDSPeakCamera::ClearBuffer()
{
    if (!dataStream)
    {
        LogMessage("IDS error: No data stream to clear buffer. This error is unrecoverable.");
        return DEVICE_ERR;
    }

    try
    {
        // Flush queue (input and output)
        dataStream->Flush(peak::core::DataStreamFlushMode::DiscardAll);
        // Revoke all old buffers
        for (const auto& buffer : dataStream->AnnouncedBuffers())
        {
            dataStream->RevokeBuffer(buffer);
        }
    }
    catch (std::exception& e)
    {
        LogMessage("IDS exception: Could not clear buffer.");
        LogMessage(e.what());
        return DEVICE_ERR;
    }
    return DEVICE_OK;
}

/**
* Prepares the buffers for acquisition. It allocates memory for the camera to transfer its data.
* @returns Integer status code - Returns DEVICE_OK on success
*/
int CIDSPeakCamera::PrepareBuffer()
{
    int ret = ClearBuffer();
    if (DEVICE_OK != ret) { return ret; }

    try
    {
        payloadSize_ = nodeMapRemoteDevice->FindNode<IntNode>("PayloadSize")->Value();

        // Allocate buffers
        for (size_t count = 0; count < nBuffers_; count++)
        {
            auto buffer = dataStream->AllocAndAnnounceBuffer(static_cast<size_t>(payloadSize_), nullptr);
            dataStream->QueueBuffer(buffer);
        }
    }
    catch (std::exception& e)
    {
        LogMessage("IDS exception: Could not allocate buffers.");
        LogMessage(e.what());
        return DEVICE_ERR;
    }
    return DEVICE_OK;
}

/**
* Retrieves image from camera, transforms it to the correct PixelType, and transfers it to the
* Micro-Manager circular buffer.
* @param timeout_ms - Timeout for the image acquisition
* @param insertImage - Insert the image into Micro-Manager's circular buffer
* @returns Integer status code - Returns DEVICE_OK on success
*/
int CIDSPeakCamera::AcquireAndTransferImage(uint64_t timeout_ms, bool insertImage)
{
    try
    {
        MMThreadGuard g(imgPixelsLock_);
        unsigned char* pBuf = (unsigned char*) const_cast<unsigned char*>(img_.GetPixels());

        // Get image from camera buffer
        const auto buffer = dataStream->WaitForFinishedBuffer(
            ("On" == triggerMode_) ? peak::core::Timeout::INFINITE_TIMEOUT : timeout_ms
        );

        // Convert buffer to expected format
        if (pixelFormat_ != destinationFormat_.Name())
        {
            converter.Convert(
                peak::BufferTo<peak::ipl::Image>(buffer),
                destinationFormat_,
                pBuf,
                img_.Width() * img_.Height() * img_.Depth()
            );
        }
        else
        {
            const auto image = peak::BufferTo<peak::ipl::Image>(buffer);
            memcpy(pBuf, image.Data(), payloadSize_);
        }

        // Requeue (release) buffer
        dataStream->QueueBuffer(buffer);
    }
    catch (std::exception& e)
    {
        LogMessage("IDS exception: Could not acquire image or transfer it to the Micro-Manager buffer.");
        LogMessage(e.what());
        return DEVICE_ERR;
    }

    // Insert image in circular buffer
    if (insertImage) { InsertImage(); }

    return DEVICE_OK;
}

/**
* This function is called from inside the thread to do the actual capturing
* @returns Integer status code - Returns DEVICE_OK on success
*/
int CIDSPeakCamera::RunSequenceOnThread()
{
    MM::MMTime startTime = GetCurrentMMTime();
    uint64_t timeout_ms = (uint64_t)(3000 / frameRateCur_);
    return AcquireAndTransferImage(timeout_ms, true);
};

/**
* Returns whether the camera is currently running an acquisition.
* @returns bool - Returns true when camera is running acquisition
*/
bool CIDSPeakCamera::IsCapturing() {
    return !thd_->IsStopped();
}

/*
 * Function is called when exiting the thread. It should let the core know
 * the acquisition is finished, and release any thread related resources if needed.
 */
void CIDSPeakCamera::OnThreadExiting()
{
    try
    {
        LogMessage(g_Msg_SEQUENCE_ACQUISITION_THREAD_EXITING);
        GetCoreCallback() ? GetCoreCallback()->AcqFinished(this, 0) : DEVICE_OK;
    }
    catch (std::exception& e)
    {
        LogMessage(e.what());
        LogMessage(g_Msg_EXCEPTION_IN_ON_THREAD_EXITING, false);
    }
}

/**
* Function called by the acquisition thread to prepare the camera for acquisition, and start the acquisition
* @param numImages - Number of images to be acquired. LONG_MAX means collected until manually stopped
* @returns Integer status code - Returns DEVICE_OK on success
*/
int CIDSPeakCamera::StartAcquisition(long numImages)
{
    try
    {
        if (numImages == LONG_MAX)
        {
            dataStream->StartAcquisition(peak::core::AcquisitionStartMode::Default, PEAK_INFINITE_NUMBER);
        }
        else
        {
            dataStream->StartAcquisition(peak::core::AcquisitionStartMode::Default, numImages);
        }
        nodeMapRemoteDevice->FindNode<IntNode>("TLParamsLocked")->SetValue(1); // Lock acq params
        nodeMapRemoteDevice->FindNode<CommandNode>("AcquisitionStart")->Execute();
    }
    catch (std::exception& e)
    {
        LogMessage("IDS exception: Could not start acquisition");
        LogMessage(e.what());
        return DEVICE_ERR;
    }
    return DEVICE_OK;
}

/**
* Function called by the acquisition thread to stop the acquisition. It also unlocks the camera,
* allowing parameters to be changed.
* @returns Integer status code - Returns DEVICE_OK on success
*/
int CIDSPeakCamera::StopAcquisition()
{
    try
    {
        auto node = nodeMapRemoteDevice->FindNode<CommandNode>("AcquisitionStop");
        node->Execute();
        node->WaitUntilDone();
        nodeMapRemoteDevice->FindNode<IntNode>("TLParamsLocked")->SetValue(0); // Unlock acq params
        dataStream->StopAcquisition(peak::core::AcquisitionStopMode::Default);
    }
    catch (std::exception& e)
    {
        LogMessage("IDS exception: Could not stop acquisition.");
        LogMessage(e.what());
        return DEVICE_ERR;
    }
    return DEVICE_OK;
}

////////////////////////////////////////////////////////////////////////////////
// IDS Setters
////////////////////////////////////////////////////////////////////////////////

int CIDSPeakCamera::SetFrameRate(double frameRate)
{
    auto node = nodeMapRemoteDevice->FindNode<FloatNode>("AcquisitionFrameRate");
    // Check if we have access
    if (node->AccessStatus() != peak::core::nodes::NodeAccessStatus::ReadWrite)
    {
        LogMessage("IDS error: Could not set frame rate, due to lack of write access.");
        return DEVICE_CAN_NOT_SET_PROPERTY;
    }

    // Make sure frameRate is multiple of increment
    frameRate = std::floor(frameRate / frameRateInc_) * frameRateInc_;
    try
    {
        node->SetValue(frameRate);
        frameRateCur_ = node->Value();
    }
    catch (std::exception& e)
    {
        LogMessage("IDS exception: Could not set frame rate.");
        LogMessage(e.what());
        return DEVICE_CAN_NOT_SET_PROPERTY;
    }
    return DEVICE_OK;
}

int CIDSPeakCamera::SetFrameCount(long count)
{
    auto node = nodeMapRemoteDevice->FindNode<EnumNode>("AcquisitionMode");
    if (node->AccessStatus() != peak::core::nodes::NodeAccessStatus::ReadWrite)
    {
        return DEVICE_CAN_NOT_SET_PROPERTY;
    }

    try
    {
        if (count == 1)
        {
            node->SetCurrentEntry("SingleFrame");
        }
        else if (count == LONG_MAX)
        {
            node->SetCurrentEntry("Continuous");
        }
        else
        {
            node->SetCurrentEntry("MultiFrame");
            nodeMapRemoteDevice->FindNode<IntNode>("AcquisitionFrameCount")->SetValue(count);
        }
    }
    catch (std::exception& e)
    {
        LogMessage("IDS exception: An error occurred while setting acquisition mode.");
        LogMessage(e.what());
        return DEVICE_CAN_NOT_SET_PROPERTY;
    }
    return DEVICE_OK;
}

int CIDSPeakCamera::SetPixelType(const std::string& pixelType)
{
    if (IsCapturing()) { return DEVICE_CAMERA_BUSY_ACQUIRING; }

    if (pixelType == g_PixelType_8bit)
    {
        nComponents_ = 1;
        bitDepth_ = 8;
        destinationFormat_ = peak::ipl::PixelFormatName::Mono8;
    }
    else if (pixelType == g_PixelType_16bit)
    {
        nComponents_ = 1;
        bitDepth_ = 16;
        destinationFormat_ = peak::ipl::PixelFormatName::Mono16;
    }
    else if (pixelType == g_PixelType_32bitBGRA)
    {
        nComponents_ = 4;
        bitDepth_ = 8;
        destinationFormat_ = peak::ipl::PixelFormatName::BGRa8;
    }
    else
    {
        return DEVICE_CAN_NOT_SET_PROPERTY;
    }
    pixelType_ = pixelType;

    // Resize buffer to accomodate the new image
    img_.Resize(img_.Width(), img_.Height(), nComponents_ * (bitDepth_ / 8));
    return DEVICE_OK;
}

////////////////////////////////////////////////////////////////////////////////
// Utility functions
////////////////////////////////////////////////////////////////////////////////

/**
* Get boundary values of a IntNode
* @param node - Pointer to IntNode
* @param minVal - Reference to int of minimum value
* @param maxVal - Reference to int of maximum value
* @param increment - Reference to int of increment value
* @returns Integer status code - Returns DEVICE_OK on success
*/
int CIDSPeakCamera::GetBoundaries(std::shared_ptr<IntNode> node, unsigned& minVal, unsigned& maxVal, unsigned& increment)
{
    try
    {
        minVal = (unsigned)node->Minimum();
        maxVal = (unsigned)node->Maximum();
        increment = (unsigned)node->Increment();
    }
    catch (std::exception& e)
    {
        LogMessage("IDS exception: Could not get the bounds of node ");
        LogMessage(e.what());
        return DEVICE_ERR;
    }
    return DEVICE_OK;
}

/**
* Get boundary values of a FloatNode
* @param node - Pointer to FloatNode
* @param minVal - Reference to double of minimum value
* @param maxVal - Reference to double of maximum value
* @param increment - Reference to double of increment value
* @returns Integer status code - Returns DEVICE_OK on success
*/
int CIDSPeakCamera::GetBoundaries(std::shared_ptr<FloatNode> node, double& minVal, double& maxVal, double& increment)
{
    try
    {
        minVal = node->Minimum();
        maxVal = node->Maximum();
        increment = (node->HasConstantIncrement()) ? node->Increment() : 0.001;
    }
    catch (std::exception& e)
    {
        LogMessage("IDS exception: Could not get the bounds of node ");
        LogMessage(e.what());
        return DEVICE_ERR;
    }
    return DEVICE_OK;
}

/**
* Gets all available entries from an EnumNode. Available means accessible with
* any perimission (READONLY, WRITEONLY, READWRITE).
* @params nodeName - Name of EnumNode
* @returns vector with available entries
*/
std::vector<std::string> CIDSPeakCamera::GetAvailableEntries(const std::string& nodeName)
{
    auto allEntries = nodeMapRemoteDevice->FindNode<EnumNode>(nodeName)->Entries();
    std::vector<std::string> stringEntries = {};
    for (const auto& entry : allEntries)
    {
        if (!CheckAccess(entry, AccessTypes::ANY)) { continue; }
        stringEntries.push_back(entry->StringValue());
    }
    return stringEntries;
}

/**
* Gets a list of all available PixelTypes available with provided pixelFormat.
* @params pixelFormat - Name of the PixelFormat
* @returns list of available pixelTypes
*/
std::vector<std::string> CIDSPeakCamera::GetAvailablePixelTypes(const std::string& pixelFormat)
{
    const peak::ipl::PixelFormat inputFormat =
        (const peak::ipl::PixelFormatName)nodeMapRemoteDevice->FindNode<EnumNode>("PixelFormat")->FindEntry(pixelFormat)->NumericValue();
    std::vector<std::string> output = {};
    std::vector<peak::ipl::PixelFormatName> outputFormats = converter.SupportedOutputPixelFormatNames(inputFormat);
    for (auto& outputFormat : outputFormats)
    {
        if (peak::ipl::PixelFormatName::Mono8 == outputFormat)
        {
            output.push_back(g_PixelType_8bit);
        }
        else if (peak::ipl::PixelFormatName::Mono16 == outputFormat)
        {
            output.push_back(g_PixelType_16bit);
        }
        else if (peak::ipl::PixelFormatName::BGRa8 == outputFormat)
        {
            output.push_back(g_PixelType_32bitBGRA);
        }
    }
    return output;
}

/**
* Verifies if it has the right access to a node.
* @params node - NodeType pointer to the node
* @params type - Access required (READ/WRITE/READWRITE)
* @returns bool whether or not it has the right access
*/
template <class NodeType, typename std::enable_if<std::is_base_of<peak::core::nodes::Node, NodeType>::value, int>::type>
bool CIDSPeakCamera::CheckAccess(std::shared_ptr<NodeType> node, AccessTypes type)
{
    switch (node->AccessStatus())
    {
    case peak::core::nodes::NodeAccessStatus::ReadWrite:
        return true;
    case peak::core::nodes::NodeAccessStatus::WriteOnly:
        return (type == AccessTypes::WRITEONLY || type == AccessTypes::ANY);
    case peak::core::nodes::NodeAccessStatus::ReadOnly:
        return (type == AccessTypes::READONLY || type == AccessTypes::ANY);
    default:
        return false;
    }
}

/**
* Limits a value between a minimum and maximum.
* @param val - Value to be bound
* @param minVal - Minimum value
* @param maxVal - Maximum value
* @returns Value limited between minVal and maxVal
*/
template <typename T>
T CIDSPeakCamera::boundValue(T val, T minVal, T maxVal) {
    return min(minVal, max(maxVal, val));
}

/**
* Returns value that is (as) close to the input val, but divisible by the increment
* @param val - Value to be made a multiple of the increment
* @param increment - Step size in val
* @returns value that is close to the input val but divisible by the increment
*/
template <typename T>
T CIDSPeakCamera::multipleOfIncrement(T val, T increment)
{
    if (0 == increment) { return val; }
    return (T)std::round(val / increment) * increment;
}




///////////////////////////////////////////////////////////////////////////////
// MySequenceThread
///////////////////////////////////////////////////////////////////////////////

MySequenceThread::MySequenceThread(CIDSPeakCamera* pCam)
    :intervalMs_(default_intervalMS)
    , numImages_(default_numImages)
    , imageCounter_(0)
    , stop_(true)
    , suspend_(false)
    , camera_(pCam)
    , startTime_(0)
    , actualDuration_(0)
    , lastFrameTime_(0)
{
};

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
    try {
        // Start acquisition
        ret = camera_->StartAcquisition(numImages_);
        if (DEVICE_OK != ret) { return ret; }

        // do-while loop over numImages_
        do {
            ret = camera_->RunSequenceOnThread();
        } while (ret == DEVICE_OK && !IsStopped() && imageCounter_++ < numImages_ - 1);

        // If the acquisition is stopped manually, the acquisition has to be properly closed to
        // prevent the camera to be locked in acquisition mode.
        if (IsStopped()) {
            camera_->LogMessage("SeqAcquisition interrupted by the user\n");
        }
        else if (imageCounter_ == numImages_) {
            camera_->LogMessage("Number of images reached.");
        }

        // Stop and unlock camera after acquisition
        camera_->StopAcquisition();
    }
    catch (std::exception& e) {
        camera_->LogMessage("IDS exception: Something went wrong during image acquisition");
        camera_->LogMessage(e.what());
        camera_->LogMessage(g_Msg_EXCEPTION_IN_THREAD, false);
    }
    stop_ = true;
    actualDuration_ = camera_->GetCurrentMMTime() - startTime_;
    camera_->OnThreadExiting();
    return ret;
}

// helper function for binning
int CIDSPeakCamera::OnBinningEngine(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        try {
            std::string currentSel = nodeMapRemoteDevice->FindNode<EnumNode>("BinningSelector")->CurrentEntry()->SymbolicValue();
            if (currentSel == "Region0")
                pProp->Set("FPGA");
            else
                pProp->Set("Sensor");

            binningSelector_ = currentSel;
        }
        catch (...) {
            return DEVICE_ERR;
        }
    }
    else if (eAct == MM::AfterSet)
    {
        if (IsCapturing()) return DEVICE_CAMERA_BUSY_ACQUIRING;

        std::string val;
        pProp->Get(val);
        std::string genicamSelector = (val == "FPGA") ? "Region0" : "Sensor";

        try {
            nodeMapRemoteDevice->FindNode<EnumNode>("BinningSelector")->SetCurrentEntry(genicamSelector);
            binningSelector_ = genicamSelector;
            int64_t maxVal = nodeMapRemoteDevice->FindNode<IntNode>("BinningHorizontal")->Maximum();
            std::vector<std::string> binningValues;
            int64_t i = 1;
            while (i <= maxVal) {
                binningValues.push_back(std::to_string(i));
                i *= 2;
            }
            int ret = SetAllowedValues(MM::g_Keyword_Binning, binningValues);
            if (ret != DEVICE_OK) return ret;

            int targetBinning = binSize_;
            if (targetBinning > maxVal) {
                targetBinning = 1; // Reset to 1 if current is too high (e.g. 8x -> Sensor)
            }
            SetBinning(targetBinning);
            SetProperty(MM::g_Keyword_Binning, std::to_string(targetBinning).c_str());

            ClearROI();
        }
        catch (std::exception& e) {
            LogMessage("IDS Exception in OnBinningEngine");
            LogMessage(e.what());
            return DEVICE_ERR;
        }
    }
    return DEVICE_OK;
}