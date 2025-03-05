///////////////////////////////////////////////////////////////////////////////
// FILE:          Camera2.cpp
// PROJECT:       Micro-Manager 2.0
// SUBSYSTEM:     DeviceAdapters
//  
//-----------------------------------------------------------------------------
// DESCRIPTION:   SIGMA-KOKI device adapter 2.0
//                
// AUTHOR   :    Hiroki Kibata, Abed Toufik  Release Date :  05/02/2023
//
// COPYRIGHT:     SIGMA KOKI CO.,LTD, Tokyo, 2023
#pragma once

#include <cstdio>
#include <string>
#include <math.h>
#include <sstream>
#include <algorithm>
#include <iostream>
#include "Camera.h"

#define NOMINMAX
#include <Windows.h> // Needed by StCamD.h
#include "StCamD.h"

using namespace std;

// Constants 
// 
// Property name

const char* g_CameraDeviceName = "SKCam";
const char* g_CameraProductName = "ProductName";
const char* g_CameraALCMode = "ALC_Mode";
const char* g_CameraShutterGain = "Shutter Gain";
const char* g_CameraShutterGainMode = "Shutter Gain Mode";
const char* g_CameraShutterGainAutoMax = "Shutter Gain Auto Max";
const char* g_CameraShutterGainAutoMin = "Shutter Gain Auto Min";
const char* g_CameraExposure = "Exposure (msec)";
const char* g_CameraClockSpeed = "ClockSpeed";
const char* g_CameraFPS = "FPS";
const char* g_CameraImageDeviceSizeH = "ImageDeviceSize_H(pixel)";
const char* g_CameraImageDeviceSizeV = "ImageDeviceSize_V(pixel)";
const char* g_CameraWBMode = "WhiteBalance_Mode";
const char* g_CameraWBGainR = "WhiteBalance_GainR";
const char* g_CameraWBGainGr = "WhiteBalance_GainGr";
const char* g_CameraWBGainGb = "WhiteBalance_GainGb";
const char* g_CameraWBGainB = "WhiteBalance_GainB";
const char* g_CameraDelaySnap = "SnapDelay (msec)";

// scan mode 
const char* g_ScanMode = "Binning Scan Mode";


// ALC(Auto Luminance Control) mode
const char* g_ALCMode_FS_MG = "Fixed Shutter/Manual Gain";
const char* g_ALCMode_AS_AGC = "Auto Shutter/AGC";
const char* g_ALCMode_AS_MG = "Auto Shutter/Manual Gain";
const char* g_ALCMode_FS_AGC = "Fixed Shutter/AGC";

// clock-speed types
const char* g_ClockSpeed_Reference = "Reference clock";
const char* g_ClockSpeed_Div2 = "1/2 clock";
const char* g_ClockSpeed_Div4 = "1/4 clock";

//Mirror orienation 
const char* g_Mirror_Mode = "Mirror Mode";


// pixel type
const char* g_PixelType_8bitMONO = "8bit_MONO";
const char* g_PixelType_32bitBGR = "32bit_BGR";

// white balance mode
const char* g_WB_Mode_Off = "Off";
const char* g_WB_Mode_Auto = "Auto";
const char* g_WB_Mode_Manual = "Manual";
const char* g_WB_Mode_OneShot = "OneShot";

///////////////////////////////////////////////////////////////////////////////
// Exported MMDevice API
///////////////////////////////////////////////////////////////////////////////

// General utility function:
int ClearPort(MM::Device& device, MM::Core& core, std::string port)
{
	//Clear contents of serial port 
	const int bufSize = 255;
	unsigned char clear[bufSize];
	unsigned long read = bufSize;
	int ret;
	while (read == (unsigned)bufSize)
	{
		ret = core.ReadFromSerial(&device, port.c_str(), clear, bufSize, read);
		if (ret != DEVICE_OK)
			return ret;
	}
	return DEVICE_OK;
}

#pragma region Camera
/////////////////////////////////////////////////////////////////////////////////
//// Camera
////
////Supported models �c [STC-MC202USB, STC-STC-MCA5MUSB3, SK-TC202USB-AT](old model)
////Added Cameras USB 3 [STC-MCS122BU3V, STC-MCCM401U3V, STC-MBCM200U3V/NIR, other USB3 cameras were not tested yet  ]

/**
* Camera constructor.
* Setup default all variables and create device properties required to exist
* before intialization. In this case, no such properties were required. All
* properties will be created in the Initialize() method.
*
* As a general guideline Micro-Manager devices do not access hardware in the
* the constructor. We should do as little as possible in the constructor and
* perform most of the initialization in the Initialize() method.
*/
Camera::Camera() :
	// initialization of parameters
	SigmaBase(this),
	CLegacyCameraBase<Camera>(),
	initialized_(false),
	isMonochrome_(false),
	enabledROI_(false),
	readoutUs_(0.0),
	bitDepth_(8),
	roiX_(0),
	roiY_(0),
	dwSize_(0),
	dwLinePitch_(0),
	dwPreviewPixelFormat_(0),
	gain_(0),
	gainMode_("Digital All"),
	autoGainMax_(0),
	autoGainMin_(0),
	exposureMsec_(0.0),
	exposureMaxMsec_(0.0),
	exposureMinMsec_(0.0),
	exposureClock_(0),
	exposureMaxClock_(0),
	clockFreq_(0),
	fps_(0.0),
	wbGainR_(0),
	wbGainGr_(0),
	wbGainGb_(0),
	wbGainB_(0),
	delayMsec_(0),
	sequenceStartTime_(0),
	isSequenceable_(false),
	binFac_(1),
	imageSizeH_(0),
	imageSizeV_(0),
	mirrorMode_("MIRROR OFF"),
	stopOnOverflow_(false),
	nComponents_(1)
{
	// Error Msgs
	InitializeDefaultErrorMessages();
	SetErrorText(ERR_CAMERA_OPEN_FAILED, "Failed to open.");
	SetErrorText(ERR_CAMERA_GET_PREVIEWDATASIZE_FAILED, "Failed to get the preview data size.");
	SetErrorText(ERR_CAMERA_SET_PIXELFORMAT_FAILED, "Failed to set the pixel format.");
	SetErrorText(ERR_CAMERA_SET_TARGET_BRIGHTNESS_FAILED, "Failed to set the target brightness.");
	SetErrorText(ERR_CAMERA_GET_SHUTTER_GAIN_CONTROL_RANGE_FAILED, "Failed to get the auto gain control range.");
	SetErrorText(ERR_CAMERA_SET_ALCMODE_FAILED, "Failed to set the ALC-mode.");
	SetErrorText(ERR_CAMERA_GET_ALCMODE_FAILED, "Failed to get the ALC-mode.");
	SetErrorText(ERR_CAMERA_SET_SHUTTER_GAIN_FAILED, "Failed to set the shutter gain value.");
	SetErrorText(ERR_CAMERA_GET_SHUTTER_GAIN_FAILED, "Failed to get the shutter gain value.");
	SetErrorText(ERR_CAMERA_SET_EXPOSURE_TIME_FAILED, "Failed to set the exposure time.");
	SetErrorText(ERR_CAMERA_GET_EXPOSURE_TIME_FAILED, "Failed to get the exposure time.");
	SetErrorText(ERR_CAMERA_GET_CLOCK_FAILED, "Failed to get the clock mode.");
	SetErrorText(ERR_CAMERA_SET_CLOCK_FAILED, "Failed to set the clock mode.");
	SetErrorText(ERR_CAMERA_GET_FPS_FAILED, "Failed to get the FPS.");
	SetErrorText(ERR_CAMERA_START_TRANSFER_FAILED, "Failed to transfer the start of the image data.");
	SetErrorText(ERR_CAMERA_STOP_TRANSFER_FAILED, "Failed to transfer the stop of the image data.");
	SetErrorText(ERR_CAMERA_SNAPSHOT_FAILED, "Failed to snapshot.");
	SetErrorText(ERR_CAMERA_LIVE_STOP_UNKNOWN, "Live has been stopped with an unknown error.");
	SetErrorText(ERR_CAMERA_GET_PRODUCT_NAME_FAILED, "Failed to get the product name.");
	SetErrorText(ERR_CAMERA_SET_BINNING_SCAN_MODE_FAILED, "Failed to set binning scan mode.");
	SetErrorText(ERR_CAMERA_GET_BINNING_SCAN_MODE_FAILED, "Failed to get binning scan mode.");
	SetErrorText(ERR_CAMERA_SET_WB_MODE_FAILED, "Failed to set white balance mode.");
	SetErrorText(ERR_CAMERA_GET_WB_MODE_FAILED, "Failed to get white balance mode.");
	SetErrorText(ERR_CAMERA_SET_WB_GAIN_FAILED, "Failed to set white balance gain.");
	SetErrorText(ERR_CAMERA_GET_WB_GAIN_FAILED, "Failed to get white balance gain.");
	SetErrorText(ERR_CAMERA_GET_COLOR_ARRAY_FAILED, "Failed to get color array.");
	SetErrorText(ERR_CAMERA_SET_ROI_FAILED, "Failed to set ROI.");
	SetErrorText(ERR_CAMERA_GET_ROI_COUNT_FAILED, "Failed to get ROI count.");
	SetErrorText(ERR_CAMERA_ALCMODE_UNAVAILABLE_FUNCTION, "It can not be set in the current ALC mode.");
	SetErrorText(ERR_CAMERA_WBMODE_UNAVAILABLE_FUNCTION, "It can not be set in the current White Balance mode.");
	SetErrorText(ERR_CAMERA_SCAN_MODE_FAILED_SETTING, "Failed to set scan mode");
	SetErrorText(ERR_CAMERA_SCAN_MODE_PROHIBTED, "Cannot Set Data in Actual Normal Scan Mode");
	readoutStartTime_ = GetCurrentMMTime();
	thd_ = new MySequenceThread(this);

	// Name
	CreateStringProperty(MM::g_Keyword_Name, g_CameraDeviceName, true);

	// Description
	CreateStringProperty(MM::g_Keyword_Description, "SIGMA-KOKI Camera Device Adapter", true);
}

Camera::~Camera()
{
	StopSequenceAcquisition();
	delete thd_;
}

#pragma endregion Camera

void Camera::GetName(char* Name) const
{
	CDeviceUtils::CopyLimitedString(Name, g_CameraDeviceName);
}

int Camera::Initialize()
{
	if (initialized_)
		return DEVICE_OK;
	
	// Create a system object for device scan and connection.
	// Here we use CIStSystemPtr instead of IStSystemReleasable for automatically managing the IStSystemReleasable class with auto initial/deinitial.
	//CIStSystemPtr pIStSystem(CreateIStSystem());
	// Create a camera device object and connect to first detected device by using the function of system object.
	// We use CIStDevicePtr instead of IStDeviceReleasable for automatically managing the IStDeviceReleasable class with auto initial/deinitial.
	//CIStDevicePtr pIStDevice(pIStSystem->CreateFirstIStDevice());

	// Open and product name 
	handle_ = StCam_Open(0);
	if (handle_ == 0) { return ERR_CAMERA_OPEN_FAILED; }

	BOOL ans = TRUE;

	// Product name(read only)
	ans = StCam_GetProductNameA(handle_, &name_, 255);
	if (!ans) { return ERR_CAMERA_GET_PRODUCT_NAME_FAILED; }

	int ret = CreateStringProperty(g_CameraProductName, &name_, true);
	if (ret != DEVICE_OK)
		return ret;

	productName_ = productName_.append(&name_);
	cout << "Name product " << productName_ << endl;

	// ROI Check Changed in MM 2.0
	/*Check of ROI function
	DWORD count = 0;
	ans = StCam_GetMaxROICount(handle_, &count);
	if (!ans) { return ERR_CAMERA_GET_ROI_COUNT_FAILED; }
	if (count == 1)
	{
		enabledROI_ = false;	// USB2.0 type
	}
	else
	{
		enabledROI_ = true;		// USB3.0 type
	}*/

	// ColorArray (color or monochrome)
	WORD colorArray;
	ans = StCam_GetColorArray(handle_, &colorArray);
	if (!ans) { return ERR_CAMERA_GET_COLOR_ARRAY_FAILED; }

	if (colorArray == STCAM_COLOR_ARRAY_MONO)
	{
		isMonochrome_ = true;	//mono
	}
	else
	{
		isMonochrome_ = false;	//color
	}

	// Pixel format (read only.)
	const char* pixelType;
	DWORD pixelFormat;
	if (isMonochrome_)
	{
		pixelFormat = STCAM_PIXEL_FORMAT_08_MONO_OR_RAW;
		pixelType = g_PixelType_8bitMONO;
	}
	else
	{
		pixelFormat = STCAM_PIXEL_FORMAT_32_BGR;
		pixelType = g_PixelType_32bitBGR;
	}
	ans = StCam_SetPreviewPixelFormat(handle_, pixelFormat);
	if (!ans) { return ERR_CAMERA_SET_PIXELFORMAT_FAILED; }

	ret = CreateStringProperty(MM::g_Keyword_PixelType, pixelType, true);
	if (ret != DEVICE_OK)
		return ret;

	// scan mode 
	ans = StCam_GetEnableImageSize(handle_, 0, &scanMode_);
	if (!ans) {return ERR_CAMERA_SCAN_MODE_FAILED_SETTING;}
	CPropertyAction* pAct = new CPropertyAction(this, &Camera::OnScanMode);
	ret = CreateProperty(g_ScanMode, scanModeType_.c_str(), MM::String, false, pAct);
	if (ret != DEVICE_OK)
		return ret;


	vector<string> ScanMode;
	ScanMode.push_back("SCAN MODE NORMAL");
	
	if ((scanMode_ & STCAM_SCAN_MODE_ROI) != 0)
	{
		ScanMode.push_back("SCAN MODE ROI");
		SetProperty(g_ScanMode, "SCAN MODE ROI");
		ClearROI();
	}
	SetAllowedValues(g_ScanMode, ScanMode);
	

	// Image device Size 
	DWORD imageSizeHorizontal_, imageSizeVertical_;
	ans = StCam_GetMaximumImageSize(handle_, &imageSizeHorizontal_, &imageSizeVertical_);
	if (!ans) { return ERR_CAMERA_GET_PREVIEWDATASIZE_FAILED; }
	ans = StCam_GetPreviewDataSize(handle_, &dwSize_, &imageSizeH_, &imageSizeV_, &dwLinePitch_);
	if (!ans) { return ERR_CAMERA_GET_PREVIEWDATASIZE_FAILED; }
	
	//img_.Resize((unsigned int)imageSizeHorizontal_ / binFac_, (unsigned int)imageSizeVertical_ / binFac_);

	pAct = new CPropertyAction(this, &Camera::OnImageH);
	ret = CreateProperty(g_CameraImageDeviceSizeH, to_string(imageSizeH_).c_str(), MM::Float, false, pAct);
	if (ret != DEVICE_OK)
		return ret;
	SetPropertyLimits(g_CameraImageDeviceSizeH, 0, (long)imageSizeHorizontal_);

	pAct = new CPropertyAction(this, &Camera::OnImageV);
	ret = CreateProperty(g_CameraImageDeviceSizeV, to_string(imageSizeV_).c_str(), MM::Float, false, pAct);
	if (ret != DEVICE_OK)
		return ret;
	SetPropertyLimits(g_CameraImageDeviceSizeV, 0, (long)imageSizeVertical_);
	img_.Resize((unsigned int)imageSizeH_ / binFac_, (unsigned int)imageSizeV_ / binFac_);

	// Before sizing MM 1.4 
	/*ans = StCam_GetPreviewDataSize(handle_, &dwSize_, &imageSizeH_, &imageSizeV_, &dwLinePitch_);
	if (!ans) { return ERR_CAMERA_GET_PREVIEWDATASIZE_FAILED; }
	img_.Resize((unsigned int)imageSizeH_ / binFac_, (unsigned int)imageSizeV_ / binFac_);

	ret = CreateIntegerProperty(g_CameraImageDeviceSizeH, (long)imageSizeH_, true);
	if (ret != DEVICE_OK)
		return ret;
	ret = CreateIntegerProperty(g_CameraImageDeviceSizeV, (long)imageSizeV_, true);
	if (ret != DEVICE_OK)
		return ret;*/
	

	// Binning chnaged in MM 2.0
	// NOTE: USB2.0 type is valid only full scan ("1"), binning scan can not be used.
	// NOTE: Monochrome type or USB3.0 type is available.
	vector<string> binValues;
	if (productName_ == "STC-MC202USB" || productName_ == "STC-MCCM401U3V")
	{
		binValues.push_back("1");
		ret = CreateIntegerProperty(MM::g_Keyword_Binning, binFac_, true);
	}
	else
	{
		binValues.push_back("1");
		binValues.push_back("2");
		pAct = new CPropertyAction(this, &Camera::OnBinning);
		ret = CreateIntegerProperty(MM::g_Keyword_Binning, binFac_, false, pAct);
		if (ret != DEVICE_OK)
			return ret;
		ret = SetAllowedValues(MM::g_Keyword_Binning, binValues);
		if (ret != DEVICE_OK)
			return ret;
	}

	// Target brightness
	ans = StCam_SetTargetBrightness(handle_, 128, 0, 0);
	if (!ans) { return ERR_CAMERA_SET_TARGET_BRIGHTNESS_FAILED; }

	// ALC�iAuto Luminance Control�j mode
	pAct = new CPropertyAction(this, &Camera::OnALCMode);
	ret = CreateStringProperty(g_CameraALCMode, alcMode_.c_str(), false, pAct);
	if (ret != DEVICE_OK)
		return ret;

	// Mirror Orientation
	pAct = new CPropertyAction(this, &Camera::OnMirrorRotation);
	ret = CreateProperty(g_Mirror_Mode, mirrorMode_.c_str(), MM::String, false, pAct);
	if (ret != DEVICE_OK)
		return ret;
	AddAllowedValue(g_Mirror_Mode, "MIRROR OFF");
	AddAllowedValue(g_Mirror_Mode, "MIRROR HORIZONTAL");
	AddAllowedValue(g_Mirror_Mode, "MIRROR VERTICAL");
	AddAllowedValue(g_Mirror_Mode, "MIRROR HORIZONTAL VERTICAL");

	vector<string> alcmode;
	alcmode.push_back(g_ALCMode_FS_MG);
	alcmode.push_back(g_ALCMode_AS_AGC);
	alcmode.push_back(g_ALCMode_AS_MG);
	alcmode.push_back(g_ALCMode_FS_AGC);

	ret = SetAllowedValues(g_CameraALCMode, alcmode);
	if (ret != DEVICE_OK)
		return ret;

	// Gain Mode (Added in MM 2.0) 
	pAct = new CPropertyAction(this, &Camera::OnGainMode);
	ret = CreateProperty(g_CameraShutterGainMode, gainMode_.c_str(), MM::String, false, pAct);
	if (ret != DEVICE_OK)
		return ret;
	AddAllowedValue(g_CameraShutterGainMode, "Digital All");
	ans = StCam_GetMaxDigitalGain(handle_, &digitalGainMax_);
	ans = StCam_GetDigitalGainSettingValueFromGainTimes(handle_, 1.0F, &digitalGainMin_);
	BOOL analogFlag = false;
	ans = StCam_HasFunction(handle_, STCAM_CAMERA_FUNCTION_DISABLED_ANALOG_GAIN, &analogFlag);
	if (analogFlag == FALSE)
	{
		AddAllowedValue(g_CameraShutterGainMode, "Analog All");
		ans = StCam_GetMaxGain(handle_, &analogGainMax_);
	}

	// Shutter gain
	pAct = new CPropertyAction(this, &Camera::OnShutterGain);
	ret = CreateProperty(g_CameraShutterGain, to_string(gain_).c_str(),MM::Integer ,  false, pAct);
	if (ret != DEVICE_OK)
		return ret;
	SetPropertyLimits(g_CameraShutterGain, digitalGainMin_, digitalGainMax_);

	//AutoGain Max and Min
	ans = StCam_GetGainControlRange(handle_, &autoGainMin_, &autoGainMax_); 
	if (!ans) { return ERR_CAMERA_GET_SHUTTER_GAIN_CONTROL_RANGE_FAILED; }

	pAct = new CPropertyAction(this, &Camera::OnAutoGainMax);
	ret = CreateProperty(g_CameraShutterGainAutoMax, to_string(autoGainMax_).c_str(),MM::Integer, false, pAct);
	if (ret != DEVICE_OK)
		return ret;
	SetPropertyLimits(g_CameraShutterGainAutoMax, 0, 255);

	pAct = new CPropertyAction(this, &Camera::OnAutoGainMin);
	ret = CreateProperty(g_CameraShutterGainAutoMin, to_string(autoGainMin_).c_str(),MM::Integer, false, pAct);
	if (ret != DEVICE_OK)
		return ret;
	SetPropertyLimits(g_CameraShutterGainAutoMin, 0, 255);

	if (productName_ == "STC-MC202USB")
	{
		// Clock speed (USB2.0 only)
		pAct = new CPropertyAction(this, &Camera::OnClockSpeed);
		ret = CreateStringProperty(g_CameraClockSpeed, g_ClockSpeed_Reference, false, pAct);
		if (ret != DEVICE_OK)
			return ret;

		vector<string> clockType;
		clockType.push_back(g_ClockSpeed_Reference);
		clockType.push_back(g_ClockSpeed_Div2);
		clockType.push_back(g_ClockSpeed_Div4);

		ret = SetAllowedValues(g_CameraClockSpeed, clockType);
		if (ret != DEVICE_OK)
			return ret;
	}

	// FPS
	ans = StCam_GetOutputFPS(handle_, &fps_);
	if (!ans) { return ERR_CAMERA_GET_FPS_FAILED; }

	char s[256] = { '\0' };
	sprintf(s, "%.*f", 2, fps_);

	ret = CreateStringProperty(g_CameraFPS, s, false);
	if (ret != DEVICE_OK)
		return ret;

	AddAllowedValue(g_CameraFPS, s);

	// Exposure time (msec)
	ans = StCam_GetMaxShortExposureClock(handle_, &exposureMaxClock_);
	if (!ans) { return ERR_CAMERA_GET_EXPOSURE_TIME_FAILED; }

	FLOAT value = 0.0;
	ans = StCam_GetExposureTimeFromClock(handle_, exposureMaxClock_, &value);
	if (!ans) { return ERR_CAMERA_GET_EXPOSURE_TIME_FAILED; }

	exposureMaxMsec_ = (double)value * 1000;

	if (enabledROI_)
	{
		// USB3.0 type
		exposureMinMsec_ = 0.032;
	}
	else
	{
		// USB2.0 type
		if (clockMode_ == g_ClockSpeed_Reference)
		{
			exposureMinMsec_ = 0.000027;
		}
		else if (clockMode_ == g_ClockSpeed_Div2 || clockMode_ == g_ClockSpeed_Div4)
		{
			exposureMinMsec_ = 0.0;
		}
	}

	pAct = new CPropertyAction(this, &Camera::OnExposure);
	ret = CreateFloatProperty(g_CameraExposure, exposureMaxMsec_, false, pAct);
	if (ret != DEVICE_OK)
		return ret;

	SetPropertyLimits(g_CameraExposure, exposureMinMsec_, exposureMaxMsec_);

	// White Balance 
	if (!isMonochrome_)
	{
		// White balance mode�icolor type only�j
		pAct = new CPropertyAction(this, &Camera::OnWhiteBalanceMode);
		ret = CreateStringProperty(g_CameraWBMode, wbMode_.c_str(), false, pAct);
		if (ret != DEVICE_OK)
			return ret;

		vector<string> mode;
		mode.push_back(g_WB_Mode_Off);
		mode.push_back(g_WB_Mode_Auto);
		mode.push_back(g_WB_Mode_Manual);
		mode.push_back(g_WB_Mode_OneShot);

		ret = SetAllowedValues(g_CameraWBMode, mode);
		if (ret != DEVICE_OK)
			return ret;

		// Get white balance gain
		ans = StCam_GetWhiteBalanceGain(handle_, &wbGainR_, &wbGainGr_, &wbGainGb_, &wbGainB_);
		if (!ans) { return ERR_CAMERA_GET_WB_GAIN_FAILED; }

		// White Balance Gain R
		pAct = new CPropertyAction(this, &Camera::OnWBGainR);
		ret = CreateIntegerProperty(g_CameraWBGainR, wbGainR_, false, pAct);
		if (ret != DEVICE_OK)
			return ret;

		SetPropertyLimits(g_CameraWBGainR, 128, 640);

		// White Balance Gain Gr
		pAct = new CPropertyAction(this, &Camera::OnWBGainGr);
		ret = CreateIntegerProperty(g_CameraWBGainGr, wbGainGr_, false, pAct);
		if (ret != DEVICE_OK)
			return ret;

		SetPropertyLimits(g_CameraWBGainGr, 128, 640);

		// White Balance Gain Gb
		pAct = new CPropertyAction(this, &Camera::OnWBGainGb);
		ret = CreateIntegerProperty(g_CameraWBGainGb, wbGainGb_, false, pAct);
		if (ret != DEVICE_OK)
			return ret;

		SetPropertyLimits(g_CameraWBGainGb, 128, 640);

		// White Balance Gain B
		pAct = new CPropertyAction(this, &Camera::OnWBGainB);
		ret = CreateIntegerProperty(g_CameraWBGainB, wbGainB_, false, pAct);
		if (ret != DEVICE_OK)
			return ret;

		SetPropertyLimits(g_CameraWBGainB, 128, 640);
	}

	// readout time
	pAct = new CPropertyAction(this, &Camera::OnReadoutTime);
	ret = CreateFloatProperty(MM::g_Keyword_ReadoutTime, 0, false, pAct);
	if (ret != DEVICE_OK)
		return ret;


	// {synchronize all properties
	ret = UpdateStatus();
	if (ret != DEVICE_OK)
		return ret;

	// setup the buffer
	ret = ResizeImageBuffer();
	if (ret != DEVICE_OK)
		return ret;


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
int Camera::Shutdown()
{
	StCam_Close(handle_);

	initialized_ = false;

	return DEVICE_OK;
}

/**
* Performs exposure and grabs a single image.
*
* SnapImage should start the image exposure in the camera and block until
* the exposure is finished.  It should not wait for read-out and transfer of data.
* Return DEVICE_OK on succes, error code otherwise.
*
* This function should block during the actual exposure and return immediately afterwards
* (i.e., before readout).  This behavior is needed for proper synchronization with the shutter.
*
* Required by the MM::Camera API.
*/
int Camera::SnapImage()
{
	//Delay
	MMThreadGuard g(imgPixelsLock_);
	MM::MMTime start = GetCurrentMMTime();
	MM::TimeoutMs tout(start, delayMsec_);
	while (!tout.expired(GetCurrentMMTime())) {}

	ResizeImageBuffer();
	//Allocate Memory
	PBYTE imgBuff = (PBYTE) new BYTE[GetImageBufferSize()];

	cout << "Before StCam_TakePreviewSnapShot    " << img_.Height() << "X" << img_.Width() << endl;
	//Take Snap Shot
	BOOL ans = TRUE;
	DWORD dwNumberOfByteTrans, dwFrameNo;
	DWORD dwMilliseconds = 1000;
	ans = StCam_TakePreviewSnapShot(handle_, imgBuff, GetImageBufferSize(), &dwNumberOfByteTrans, &dwFrameNo, dwMilliseconds);
	cout << "last error     " << ans << endl;
	if (!ans) { return ERR_CAMERA_SNAPSHOT_FAILED; }
	cout << "Before memcpy imageSizeH_ X imageSizeV_    " << imageSizeH_ << "X" << imageSizeV_ << endl;
	cout << "Before memcpy img_.Width() X img_.Height()   " << img_.Width() << "X" << img_.Height() << endl;
	memcpy(img_.GetPixelsRW(), imgBuff, GetImageBufferSize());
	cout << "after memcpy img_.Width() X img_.Height()   " << img_.Width() << "X" << img_.Height() << endl;

	delete[] imgBuff;

	return DEVICE_OK;
}

/**
* Returns pixel data.
* Required by the MM::Camera API.
* GetImageBuffer will be called shortly after SnapImage returns.
* Use it to wait for camera read-out and transfer of data into memory
* Return a pointer to a buffer containing the image data
* The calling program will assume the size of the buffer based on the values
* obtained from GetImageBufferSize(), which in turn should be consistent with
* values returned by GetImageWidth(), GetImageHight() and GetImageBytesPerPixel().
* The calling program allso assumes that camera never changes the size of
* the pixel buffer on its own. In other words, the buffer can change only if
* appropriate properties are set (such as binning, pixel type, etc.)
* Multi-Channel cameras should return the content of the first channel in this call.
*/
const unsigned char* Camera::GetImageBuffer()
{
	MMThreadGuard g(imgPixelsLock_);
	MM::MMTime readoutTime(readoutUs_);
	while (readoutTime > (GetCurrentMMTime() - readoutStartTime_)) {}
	unsigned char *pB = (unsigned char*)(img_.GetPixels());

	return pB;
}

//* Returns image buffer X-size in pixels.
//* Required by the MM::Camera API.
//*/
unsigned Camera::GetImageWidth() const
{
	return img_.Width();
}

//* Returns image buffer Y-size in pixels.
//* Required by the MM::Camera API.
//*/
unsigned Camera::GetImageHeight() const
{
	return img_.Height();
}

//* Returns image buffer pixel depth in bytes.
//* Required by the MM::Camera API.
//*/
unsigned Camera::GetImageBytesPerPixel() const
{
	return img_.Depth();
}

//* Returns the bit depth (dynamic range) of the pixel.
//* This does not affect the buffer size, it just gives the client application
//* a guideline on how to interpret pixel values.
//* Required by the MM::Camera API.
//*/
unsigned Camera::GetBitDepth() const
{
	return bitDepth_;
}

///**
//* Returns the size in bytes of the image buffer.
//* Required by the MM::Camera API.
//*/
long Camera::GetImageBufferSize() const
{
	return img_.Width() * img_.Height() * GetImageBytesPerPixel();
}

///**
//* Sets the camera Region Of Interest.
//* Required by the MM::Camera API.
//* This command will change the dimensions of the image.
//* Depending on the hardware capabilities the camera may not be able to configure the
//* exact dimensions requested - but should try do as close as possible.
//* If the hardware does not have this capability the software should simulate the ROI by
//* appropriately cropping each frame.
//* This demo implementation ignores the position coordinates and just crops the buffer.
//* @param x - top-left corner coordinate
//* @param y - top-left corner coordinate
//* @param xSize - width
//* @param ySize - height
//*/
int Camera::SetROI(unsigned x, unsigned y, unsigned xSize, unsigned ySize)
{
	BOOL ans = TRUE;


	if (scanModeType_ == "SCAN MODE NORMAL")
	{
		DWORD imageSizeHorizontal_, imageSizeVertical_;
		ans = StCam_GetMaximumImageSize(handle_, &imageSizeHorizontal_, &imageSizeVertical_);
		imageSizeH_ = imageSizeHorizontal_ / binFac_;
		imageSizeV_ = imageSizeVertical_ / binFac_;
		ResizeImageBuffer();
		return DEVICE_OK;
	}
	
	ans = StCam_StopTransfer(handle_);
	if (!ans) { return ERR_CAMERA_STOP_TRANSFER_FAILED; }



	if (xSize == 0 && ySize == 0)
	{

		cout << "Effectivelly Clear ROI    =" << imageSizeH_ << " X " << imageSizeV_ << endl;
		// effectively clear ROI
		ResizeImageBuffer();
		roiX_ = 0;
		roiY_ = 0;
	}
	else
	{
		cout << "apply ROI xoffset and yoffset beofre    =" << x << "X" << y << endl;
		cout << "apply ROI RoiX_ and RoiY_    =" << roiX_ << "X" << roiY_ << endl;
		
		unsigned x1= x; unsigned y1=y;
		
		if (binFac_ == 2)
		{
			x1 = x* binFac_;
			y1 = y *binFac_;
		}
		DWORD /*size_x, size_y, */offset_x, offset_y; WORD mode = 0;
		ans = StCam_SetImageSize(handle_, 0, STCAM_SCAN_MODE_ROI, x1, y1 , xSize , ySize);
		if (!ans) { return ERR_CAMERA_SET_ROI_FAILED; }
		
		ans = StCam_GetImageSize(handle_, nullptr, &mode, &offset_x, &offset_y, &/*size_x*/imageSizeH_, &/*size_y*/imageSizeV_);
		if (!ans) { return ERR_CAMERA_SCAN_MODE_FAILED_SETTING; }

		// apply ROI
		img_.Resize(imageSizeH_ , imageSizeV_);
		roiX_ = x;
		roiY_ = y;
		cout << "apply ROI imageSizeH_  imageSizeV_     =" << imageSizeH_ << "X" << imageSizeV_ << endl;

	}
	ans = StCam_StartTransfer(handle_);
	if (!ans) { return ERR_CAMERA_START_TRANSFER_FAILED; }

	return DEVICE_OK;
}

///**
//* Returns the actual dimensions of the current ROI.
//* Required by the MM::Camera API.
//*/
int Camera::GetROI(unsigned& x, unsigned& y, unsigned& xSize, unsigned& ySize)
{
	x = roiX_;
	y = roiY_;

	xSize = img_.Width();
	ySize = img_.Height();

	cout << "Get ROI     =" << img_.Width() << "X" << img_.Height() << endl;
	cout << "Get ROI offset    =" << roiX_ << "X" << roiY_ << endl;

	return DEVICE_OK;
}

///**
//* Resets the Region of Interest to full frame.
//* Required by the MM::Camera API.
//*/
int Camera::ClearROI()
{
	if (scanModeType_ == "SCAN MODE NORMAL")
		return ERR_CAMERA_SCAN_MODE_PROHIBTED;  // error for cameras that not support Roi settings

	BOOL ans = TRUE;
	ans = StCam_StopTransfer(handle_);
	if (!ans) { return ERR_CAMERA_STOP_TRANSFER_FAILED; }
	DWORD imageSizeHorizontal_, imageSizeVertical_;
	ans = StCam_GetMaximumImageSize(handle_, &imageSizeHorizontal_, &imageSizeVertical_); // Size of Camera with binning = 1 
	if (!ans) { return ERR_CAMERA_GET_PREVIEWDATASIZE_FAILED; }

	ans = StCam_SetImageSize(handle_, STCAM_IMAGE_SIZE_MODE_UXGA, STCAM_SCAN_MODE_ROI, 0, 0, imageSizeHorizontal_/binFac_, imageSizeVertical_/binFac_);
	if (!ans) { return ERR_CAMERA_SET_ROI_FAILED; }

	imageSizeH_ = imageSizeHorizontal_/ binFac_;
	imageSizeV_ = imageSizeVertical_/binFac_;
	
	cout << "BinFac_ check     =" << binFac_ << endl;
	cout << "Clear  ROI max size      =" << imageSizeHorizontal_ << "X" << imageSizeVertical_ << endl;
	cout << "Clear  ROI  img_ size   =" << imageSizeH_ << "X" << imageSizeV_ << endl;
	ResizeImageBuffer();
	roiX_ = 0;
	roiY_ = 0;

	ans = StCam_StartTransfer(handle_);
	if (!ans) { return ERR_CAMERA_START_TRANSFER_FAILED; }
	return DEVICE_OK;
}

/**
* Sets exposure in milliseconds through presetting
* Required by the MM::Camera API.
*/
void Camera::SetExposure(double exp)
{
	SetProperty(g_CameraExposure, CDeviceUtils::ConvertToString(exp));
	GetCoreCallback()->OnExposureChanged(this, exp);;
}

/**
* Returns the current exposure setting in milliseconds through presetting.
* Required by the MM::Camera API.
*/
double Camera::GetExposure() const
{
	char buf[MM::MaxStrLength];
	int ret = GetProperty(g_CameraExposure, buf);
	if (ret != DEVICE_OK)
		return 0.0;
	return atof(buf);
}

/**
* Returns the current binning factor through presetting.
* Required by the MM::Camera API.
*/
int Camera::GetBinning() const
{
	char buf[MM::MaxStrLength];
	int ret = GetProperty(MM::g_Keyword_Binning, buf);
	if (ret != DEVICE_OK)
		return 1;
	return atoi(buf);
}

/**
* Sets binning factor through presetting.
* Required by the MM::Camera API.
*/
int Camera::SetBinning(int binF)
{
	return SetProperty(MM::g_Keyword_Binning, CDeviceUtils::ConvertToString(binF));
}

int Camera::IsExposureSequenceable(bool& isSequenceable) const
{
	isSequenceable = isSequenceable_;
	return DEVICE_OK;
}

/**
* Required by the MM::Camera API
* Please implement this yourself and do not rely on the base class implementation
* The Base class implementation is deprecated and will be removed shortly
*/
int Camera::StartSequenceAcquisition(double interval)
{
	return StartSequenceAcquisition(LONG_MAX, interval, false);
}

/**
* Stop and wait for the Sequence thread finished
*/
int Camera::StopSequenceAcquisition()
{
	
	BOOL ans = TRUE;
	ans = StCam_StopTransfer(handle_);
	if (!ans) { return ERR_CAMERA_STOP_TRANSFER_FAILED; }

	cout << "Stop Transfer = " << ans << endl;
	if (!thd_->IsStopped()) {
		cout << "Thd entred = " << ans << endl;
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
int Camera::StartSequenceAcquisition(long numImages, double interval_ms, bool stopOnOverflow)
{
	if (IsCapturing())
		return DEVICE_CAMERA_BUSY_ACQUIRING;

	int ret = GetCoreCallback()->PrepareForAcq(this);
	if (ret != DEVICE_OK)
		return ret;
	sequenceStartTime_ = GetCurrentMMTime();
	imageCounter_ = 0;

	BOOL ans = TRUE;
	ans = StCam_StartTransfer(handle_);
	if (!ans) { return ERR_CAMERA_START_TRANSFER_FAILED; }


	cout << "interval = " << interval_ms << endl;
	thd_->Start(numImages, interval_ms);
	stopOnOverflow_ = stopOnOverflow;

	return DEVICE_OK;
}

/*
* Inserts Image and MetaData into MMCore circular Buffer
*/
int Camera::InsertImage()
{
	MM::MMTime timeStamp = this->GetCurrentMMTime();
	char label[MM::MaxStrLength];
	this->GetLabel(label);

	// Important:  metadata about the image are generated here:
	Metadata md;
	md.put(MM::g_Keyword_Metadata_CameraLabel, label);
	md.put(MM::g_Keyword_Elapsed_Time_ms, CDeviceUtils::ConvertToString((timeStamp - sequenceStartTime_).getMsec()));
	md.put(MM::g_Keyword_Metadata_ImageNumber, CDeviceUtils::ConvertToString(imageCounter_));

	imageCounter_++;

	MMThreadGuard g(imgPixelsLock_);

	const unsigned char* pI;
	pI = GetImageBuffer();   // img.pixels 
	
	unsigned int w = GetImageWidth();
	unsigned int h = GetImageHeight();
	unsigned int b = GetImageBytesPerPixel();
	cout << "bytes per pixel    = " << b << endl;
	int ret = GetCoreCallback()->InsertImage(this, pI, w, h, b, md.Serialize().c_str());

	if (!stopOnOverflow_ && ret == DEVICE_BUFFER_OVERFLOW)
	{
		// do not stop on overflow - just reset the buffer
		GetCoreCallback()->ClearImageBuffer(this);
		// don't process this same image again...
		cout << "Stop On overflow   = " << stopOnOverflow_ << endl;
		return GetCoreCallback()->InsertImage(this, pI, w, h, b, md.Serialize().c_str());
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
int Camera::ThreadRun(void)
{
	int ret = SnapImage();
	if (ret != DEVICE_OK)
	return ERR_CAMERA_LIVE_STOP_UNKNOWN;

	/*ResizeImageBuffer();
	PBYTE imgBuff = (PBYTE) new BYTE[GetImageBufferSize()];
	BOOL ans = TRUE;
	DWORD dwNumberOfByteTrans, dwFrameNo;
	DWORD dwMilliseconds = 1000;
	ans = StCam_TakePreviewSnapShot(handle_, imgBuff, GetImageBufferSize(), &dwNumberOfByteTrans, &dwFrameNo, dwMilliseconds);
	if (!ans) { return ERR_CAMERA_SNAPSHOT_FAILED; }

	memcpy(img_.GetPixelsRW(), imgBuff, GetImageBufferSize());

	delete[] imgBuff;*/
	cout << " inserting " << imageCounter_ << endl;
	ret = InsertImage();
	cout << " inserting finish " << imageCounter_ << endl;
	if (ret != DEVICE_OK)
		return ERR_CAMERA_LIVE_STOP_UNKNOWN;

	

	return ret;
}

bool Camera::IsCapturing() {
	return !thd_->IsStopped();
}

/*
* called from the thread function before exit
*/
void Camera::OnThreadExiting() throw()
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

MySequenceThread::MySequenceThread(Camera* pCam)
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
	MMThreadGuard(this->stopLock_);
	stop_ = true;
}

void MySequenceThread::Start(long numImages, double intervalMs)
{
	MMThreadGuard(this->stopLock_);
	MMThreadGuard(this->suspendLock_);
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

			camera_->LogMessage("Inside try, the result is   = " + to_string(ret), false);
			camera_->LogMessage(" numImages is =  " + to_string(numImages_), false);
			camera_->LogMessage(" numCounter is =  " + to_string(imageCounter_++), false);
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
// Private Camera methods
///////////////////////////////////////////////////////////////////////////////
/**
* Sync internal image buffer size to the chosen property values.
*/
int Camera::ResizeImageBuffer()
{
	char buf[MM::MaxStrLength];

	int ret = GetProperty(MM::g_Keyword_PixelType, buf);
	if (ret != DEVICE_OK)
		return ret;

	std::string pixelType(buf);

	if (pixelType.compare(g_PixelType_8bitMONO) == 0)
	{
		nComponents_ = 1;
	}
	else if (pixelType.compare(g_PixelType_32bitBGR) == 0)
	{
		nComponents_ = 4;
	}

	img_.Resize((unsigned int)imageSizeH_ , (unsigned int)imageSizeV_ , nComponents_);

	return DEVICE_OK;
}

///////////////////////////////////////////////////////////////////////////////
//  Action handlers
///////////////////////////////////////////////////////////////////////////////

/**
* Handles "Binning" property.
*/
int Camera::OnBinning(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	BOOL ans = TRUE;
	WORD mode = 0;
	BYTE skippH, skippV, hBin, vBin;

	cout << "Product is   " << productName_ << endl;

	if (eAct == MM::BeforeGet)
	{
		ans = StCam_GetSkippingAndBinning(handle_, &skippH, &skippV, &hBin, &vBin);
		if (!ans) { return ERR_CAMERA_GET_BINNING_SCAN_MODE_FAILED; }
		binFac_ = long(skippH);

		
		if (productName_ == "STC-MCA5MUSB3")  // Set and get binning for this camera is different than the others (ご注意)
			binFac_ += 1;
		//
		cout << "Get Skipp and Bin =   " << (long)skippH << " / " << (long)skippV << " / " << (long)hBin << "  / " << (long)vBin << endl;
		pProp->Set(binFac_);
	}
	else if (eAct == MM::AfterSet)
	{
		if (IsCapturing())
			return DEVICE_CAMERA_BUSY_ACQUIRING;

		if (scanModeType_ == "SCAN MODE NORMAL")
			return ERR_CAMERA_SCAN_MODE_PROHIBTED;
		pProp->Get(binFac_); 
		if (productName_ == "STC-MCA5MUSB3")  // Set and get binning for this camera is different than the others (ご注意) [カメラ　CSマウント]
		{
			if (binFac_ == 1)
			{ans = StCam_SetSkippingAndBinning(handle_, 0, 0, 0, 0);}
			else
				ans = StCam_SetSkippingAndBinning(handle_, 1, 1, 0, 0);
		}
		else
		{
			if (binFac_ == 1)
			{ans = StCam_SetSkippingAndBinning(handle_, 1, 1, 1, 1);}
			else
				ans = StCam_SetSkippingAndBinning(handle_, 2, 2, 1, 1);
		}
		

		if (!ans) { return ERR_CAMERA_SET_BINNING_SCAN_MODE_FAILED; }
		
		
	}
	if (scanModeType_ == "SCAN MODE NORMAL")
	{
		mode = STCAM_SCAN_MODE_NORMAL;
	}
	else
		mode = STCAM_SCAN_MODE_ROI;

	DWORD /*size_x, size_y, */offset_x, offset_y;
	ans = StCam_GetImageSize(handle_, nullptr, &mode, &offset_x, &offset_y, &/*size_x*/imageSizeH_, &/*size_y*/imageSizeV_);
	if (!ans) { return ERR_CAMERA_SCAN_MODE_FAILED_SETTING; }
	ResizeImageBuffer();
	cout << "After Set Binning offsetx and y      " << offset_x << "X" << offset_y << endl;
	cout << "After Set Binning img_ size     " << img_.Height() << "X" << img_.Width() << endl;
	return DEVICE_OK;
}

/**
* Handles "ReadoutTime" property.
*/
int Camera::OnReadoutTime(MM::PropertyBase* pProp, MM::ActionType eAct)
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

/**
* Handles "ALC_Mode" property. ALC = Auto Luminance Control
*/
int Camera::OnALCMode(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	BOOL ans = TRUE;
	BYTE mode = 0;
	if (eAct == MM::BeforeGet)
	{
		ans = StCam_GetALCMode(handle_, &mode);
		if (!ans) { return ERR_CAMERA_GET_ALCMODE_FAILED; }

		switch (mode)
		{
		case STCAM_ALCMODE_FIXED_SHUTTER_AGC_OFF:
			alcMode_ = g_ALCMode_FS_MG;
			break;
		case STCAM_ALCMODE_AUTO_SHUTTER_ON_AGC_ON:
			alcMode_ = g_ALCMode_AS_AGC;
			break;
		case STCAM_ALCMODE_AUTO_SHUTTER_ON_AGC_OFF:
			alcMode_ = g_ALCMode_AS_MG;
			break;
		case STCAM_ALCMODE_FIXED_SHUTTER_AGC_ON:
			alcMode_ = g_ALCMode_FS_AGC;
			break;
		default:
			ans = StCam_SetALCMode(handle_, STCAM_ALCMODE_FIXED_SHUTTER_AGC_OFF);
			if (!ans) { return ERR_CAMERA_SET_ALCMODE_FAILED; }
			alcMode_ = g_ALCMode_FS_MG;
			break;
		}

		pProp->Set(alcMode_.c_str());
	}
	else if (eAct == MM::AfterSet)
	{
		pProp->Get(alcMode_);

		if (alcMode_ == g_ALCMode_FS_MG)
		{
			mode = STCAM_ALCMODE_FIXED_SHUTTER_AGC_OFF;
		}
		else if (alcMode_ == g_ALCMode_AS_AGC)
		{
			mode = STCAM_ALCMODE_AUTO_SHUTTER_ON_AGC_ON;
		}
		else if (alcMode_ == g_ALCMode_AS_MG)
		{
			mode = STCAM_ALCMODE_AUTO_SHUTTER_ON_AGC_OFF;
		}
		else if (alcMode_ == g_ALCMode_FS_AGC)
		{
			mode = STCAM_ALCMODE_FIXED_SHUTTER_AGC_ON;
		}
		ans = StCam_SetALCMode(handle_, mode);
		if (!ans) { return ERR_CAMERA_SET_ALCMODE_FAILED; }

		// Re-acquisition of the shutter gain
		if (alcMode_ == g_ALCMode_FS_MG || alcMode_ == g_ALCMode_AS_MG)
		{
			if (gainMode_ == "Analog All")
			{
				ans = StCam_GetGain(handle_, &gain_);
				if (!ans) { return ERR_CAMERA_GET_SHUTTER_GAIN_FAILED; }
			}
			else
			{
				ans = StCam_GetDigitalGain(handle_, &gain_);
				if (!ans) { return ERR_CAMERA_GET_SHUTTER_GAIN_FAILED; }
			}

			SetProperty(g_CameraShutterGain, CDeviceUtils::ConvertToString(gain_));
		}

		// Re-acquisition of the exposure time
		if (alcMode_ == g_ALCMode_FS_MG || alcMode_ == g_ALCMode_FS_AGC)
		{
			ans = StCam_GetExposureClock(handle_, &exposureClock_);
			if (!ans) { return ERR_CAMERA_GET_EXPOSURE_TIME_FAILED; }

			FLOAT value;
			ans = StCam_GetExposureTimeFromClock(handle_, (DWORD)exposureClock_, &value);
			if (!ans) { return ERR_CAMERA_GET_EXPOSURE_TIME_FAILED; }

			exposureMsec_ = value * 1000;

			SetProperty(g_CameraExposure, CDeviceUtils::ConvertToString(exposureMsec_));
		}
	}
	return DEVICE_OK;
}

// Gain mode added in MM2.0 
int Camera::OnGainMode(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set(gainMode_.c_str());
	}
	else if (eAct == MM::AfterSet)
	{
		pProp->Get(gainMode_);
	}

	return DEVICE_OK;
}

/*
* Handles "ShutterGain" property. different from 1.4 [can set both Gain (Digital and Analog)]
*/
int Camera::OnShutterGain(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	BOOL ans = TRUE;
	if (eAct == MM::BeforeGet)
	{
		if (gainMode_ == "Digital All")
		{
			ans = StCam_GetDigitalGain(handle_, &gain_);//-->> for USB3, USB2?
			SetPropertyLimits(g_CameraShutterGain, digitalGainMin_, digitalGainMax_);
		}
		else
		{
			ans = StCam_GetGain(handle_, &gain_);//-->>for USB2
			SetPropertyLimits(g_CameraShutterGain, 0, analogGainMax_);
		}
		if (!ans) { return ERR_CAMERA_GET_SHUTTER_GAIN_FAILED; }

		pProp->Set((long)gain_);
	}
	else if (eAct == MM::AfterSet)
	{
		long value;
		pProp->Get(value);
		gain_ = (WORD)value;

		if (gainMode_ == "Analog All")
		{
			if (alcMode_ == g_ALCMode_AS_AGC || alcMode_ == g_ALCMode_FS_AGC)
			{
				return ERR_CAMERA_ALCMODE_UNAVAILABLE_FUNCTION;
			}
			ans = StCam_SetGain(handle_, gain_);
			SetPropertyLimits(g_CameraShutterGain, 0, analogGainMax_);
		}
		else if (gainMode_ == "Digital All")
		{
			ans = StCam_SetDigitalGain(handle_, gain_);
			SetPropertyLimits(g_CameraShutterGain, digitalGainMin_, digitalGainMax_);
		}
		
		if (!ans) { return ERR_CAMERA_SET_SHUTTER_GAIN_FAILED; }
	}
	return DEVICE_OK;
}

// Auto Gain Max Ana Min added in MM 2.0 
int Camera::OnAutoGainMax(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set((double)autoGainMax_);
	}
	else if (eAct == MM::AfterSet)
	{
		if (alcMode_ == g_ALCMode_FS_MG || alcMode_ == g_ALCMode_AS_MG)
		{
			return ERR_CAMERA_ALCMODE_UNAVAILABLE_FUNCTION;
		}

		double t;
		pProp->Get(t);

		autoGainMax_ = t;
		if (autoGainMax_ < autoGainMin_)
		{
			autoGainMax_ = autoGainMin_;
		}

		StCam_SetGainControlRange(handle_, autoGainMin_, autoGainMax_);
	}
	return DEVICE_OK;
}

// Auto Gain Max Ana Min added in MM 2.0 
int Camera::OnAutoGainMin(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set((double)autoGainMin_);
	}
	else if (eAct == MM::AfterSet)
	{
		if (alcMode_ == g_ALCMode_FS_MG || alcMode_ == g_ALCMode_AS_MG)
		{
			return ERR_CAMERA_ALCMODE_UNAVAILABLE_FUNCTION;
		}
		double t;
		pProp->Get(t);

		autoGainMin_ = t;
		if (autoGainMin_ > autoGainMax_)
		{
			autoGainMin_ = autoGainMax_;
		}

		StCam_SetGainControlRange(handle_, autoGainMin_, autoGainMax_);
	}
	

	return DEVICE_OK;
}

/**
* Handles "Exposure(msec)" time property.
*/
int Camera::OnExposure(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	BOOL ans = TRUE;
	if (eAct == MM::BeforeGet)
	{
		ans = StCam_GetExposureClock(handle_, &exposureClock_);
		if (!ans) { return ERR_CAMERA_GET_EXPOSURE_TIME_FAILED; }

		FLOAT value;
		ans = StCam_GetExposureTimeFromClock(handle_, (DWORD)exposureClock_, &value);
		if (!ans) { return ERR_CAMERA_GET_EXPOSURE_TIME_FAILED; }

		exposureMsec_ = value * 1000;

		pProp->Set(exposureMsec_);
	}
	else if (eAct == MM::AfterSet)
	{
		if (alcMode_ == g_ALCMode_AS_AGC || alcMode_ == g_ALCMode_AS_MG)
		{
			return ERR_CAMERA_ALCMODE_UNAVAILABLE_FUNCTION;
		}

		pProp->Get(exposureMsec_);

		FLOAT expTime = (FLOAT)(exposureMsec_ / 1000);
		ans = StCam_GetExposureClockFromTime(handle_, expTime, &exposureClock_);
		if (!ans) { return ERR_CAMERA_SET_EXPOSURE_TIME_FAILED; }

		ans = StCam_SetExposureClock(handle_, (DWORD)exposureClock_);
		if (!ans) { return ERR_CAMERA_SET_EXPOSURE_TIME_FAILED; }
	}
	return DEVICE_OK;
}

/**
* Handles "ClockSpeed" property.
*/
int Camera::OnClockSpeed(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	BOOL ans = TRUE;
	DWORD mode = 0;
	if (eAct == MM::BeforeGet)
	{
		ans = StCam_GetClock(handle_, &mode, &clockFreq_);
		if (!ans) { return ERR_CAMERA_GET_CLOCK_FAILED; }

		switch (mode)
		{
		case STCAM_CLOCK_MODE_NORMAL:
			clockMode_ = g_ClockSpeed_Reference;
			break;
		case STCAM_CLOCK_MODE_DIV_2:
			clockMode_ = g_ClockSpeed_Div2;
			break;
		case STCAM_CLOCK_MODE_DIV_4:
			clockMode_ = g_ClockSpeed_Div4;
			break;
		default:
			clockMode_ = g_ClockSpeed_Reference;
			mode = STCAM_CLOCK_MODE_NORMAL;
			ans = StCam_SetClock(handle_, mode, clockFreq_);
			if (!ans) { return ERR_CAMERA_SET_CLOCK_FAILED; }
			break;
		}
		pProp->Set(clockMode_.c_str());
	}
	else if (eAct == MM::AfterSet)
	{
		pProp->Get(clockMode_);

		if (clockMode_ == g_ClockSpeed_Reference)
		{
			mode = STCAM_CLOCK_MODE_NORMAL;
		}
		else if (clockMode_ == g_ClockSpeed_Div2)
		{
			mode = STCAM_CLOCK_MODE_DIV_2;
		}
		else if (clockMode_ == g_ClockSpeed_Div4)
		{
			mode = STCAM_CLOCK_MODE_DIV_4;
		}

		ans = StCam_SetClock(handle_, mode, clockFreq_);
		if (!ans) { return ERR_CAMERA_SET_CLOCK_FAILED; }

		// Re-acquisition of FPS
		ans = StCam_GetOutputFPS(handle_, &fps_);
		if (!ans) { return ERR_CAMERA_GET_FPS_FAILED; }

		ClearAllowedValues(g_CameraFPS);
		char s[256] = { '\0' };
		sprintf(s, "%.*f", 2, fps_);
		AddAllowedValue(g_CameraFPS, s);

		// Re-setting of the exposure time range.
		ans = StCam_GetMaxShortExposureClock(handle_, &exposureMaxClock_);
		if (!ans) { return ERR_CAMERA_GET_EXPOSURE_TIME_FAILED; }

		FLOAT expSec = 0.0;
		ans = StCam_GetExposureTimeFromClock(handle_, exposureMaxClock_, &expSec);
		if (!ans) { return ERR_CAMERA_GET_EXPOSURE_TIME_FAILED; }

		exposureMaxMsec_ = (double)expSec * 1000;

		if (enabledROI_)
		{
			// USB3.0 type
			exposureMinMsec_ = 0.032;
		}
		else
		{
			// USB2.0 type
			if (clockMode_ == g_ClockSpeed_Reference)
			{
				exposureMinMsec_ = 0.000027;
			}
			else if (clockMode_ == g_ClockSpeed_Div2 || clockMode_ == g_ClockSpeed_Div4)
			{
				exposureMinMsec_ = 0.0;
			}
		}

		SetPropertyLimits(g_CameraExposure, exposureMinMsec_, exposureMaxMsec_);
	}
	return DEVICE_OK;
}

/**
* Handles "WhiteBalance_Mode" property.
*/
int Camera::OnWhiteBalanceMode(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	BOOL ans = TRUE;
	BYTE mode = 0;
	if (eAct == MM::BeforeGet)
	{
		ans = StCam_GetWhiteBalanceMode(handle_, &mode);
		if (!ans) { return ERR_CAMERA_GET_WB_MODE_FAILED; }

		switch (mode)
		{
		case STCAM_WB_OFF:
			wbMode_ = g_WB_Mode_Off;
			break;
		case STCAM_WB_MANUAL:
			wbMode_ = g_WB_Mode_Manual;
			break;
		case STCAM_WB_FULLAUTO:
			wbMode_ = g_WB_Mode_Auto;
			break;
		case STCAM_WB_ONESHOT:
			wbMode_ = g_WB_Mode_OneShot;
			break;
		}

		pProp->Set(wbMode_.c_str());
	}
	else if (eAct == MM::AfterSet)
	{
		pProp->Get(wbMode_);

		if (wbMode_ == g_WB_Mode_Off)
		{
			mode = STCAM_WB_OFF;
		}
		else if (wbMode_ == g_WB_Mode_Manual)
		{
			mode = STCAM_WB_MANUAL;
		}
		else if (wbMode_ == g_WB_Mode_Auto)
		{
			mode = STCAM_WB_FULLAUTO;
		}
		else if (wbMode_ == g_WB_Mode_OneShot)
		{
			mode = STCAM_WB_ONESHOT;
		}
		ans = StCam_SetWhiteBalanceMode(handle_, mode);
		if (!ans) { return ERR_CAMERA_SET_WB_MODE_FAILED; }

		CDeviceUtils::SleepMs(1);

		ans = StCam_GetWhiteBalanceGain(handle_, &wbGainR_, &wbGainGr_, &wbGainGb_, &wbGainB_);
		if (!ans) { return ERR_CAMERA_GET_WB_GAIN_FAILED; }

		SetProperty(g_CameraWBGainR, CDeviceUtils::ConvertToString(wbGainR_));
		SetProperty(g_CameraWBGainGr, CDeviceUtils::ConvertToString(wbGainGr_));
		SetProperty(g_CameraWBGainGb, CDeviceUtils::ConvertToString(wbGainGb_));
		SetProperty(g_CameraWBGainB, CDeviceUtils::ConvertToString(wbGainB_));
	}
	return DEVICE_OK;
}

/**
* Handles "WhiteBalance_GainR" property.
*/
int Camera::OnWBGainR(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set((long)wbGainR_);
	}
	else if (eAct == MM::AfterSet)
	{
		if (wbMode_ == g_WB_Mode_Off || wbMode_ == g_WB_Mode_OneShot || wbMode_ == g_WB_Mode_Auto)
		{
			return ERR_CAMERA_WBMODE_UNAVAILABLE_FUNCTION;
		}

		long value;
		pProp->Get(value);

		wbGainR_ = (WORD)value;

		BOOL ans = TRUE;
		ans = StCam_SetWhiteBalanceGain(handle_, wbGainR_, wbGainGr_, wbGainGb_, wbGainB_);
		if (!ans) { return ERR_CAMERA_SET_WB_GAIN_FAILED; }
	}
	return DEVICE_OK;
}

/**
* Handles "WhiteBalance_GainGr" property.
*/
int Camera::OnWBGainGr(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set((long)wbGainGr_);
	}
	else if (eAct == MM::AfterSet)
	{
		if (wbMode_ == g_WB_Mode_Off)
		{
			return ERR_CAMERA_WBMODE_UNAVAILABLE_FUNCTION;
		}

		long value;
		pProp->Get(value);

		wbGainGr_ = (WORD)value;

		BOOL ans = TRUE;
		ans = StCam_SetWhiteBalanceGain(handle_, wbGainR_, wbGainGr_, wbGainGb_, wbGainB_);
		if (!ans) { return ERR_CAMERA_SET_WB_GAIN_FAILED; }
	}
	return DEVICE_OK;
}

/**
* Handles "WhiteBalance_GainGb" property.
*/
int Camera::OnWBGainGb(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set((long)wbGainGb_);
	}
	else if (eAct == MM::AfterSet)
	{
		if (wbMode_ == g_WB_Mode_Off)
		{
			return ERR_CAMERA_WBMODE_UNAVAILABLE_FUNCTION;
		}

		long value;
		pProp->Get(value);

		wbGainGb_ = (WORD)value;

		BOOL ans = TRUE;
		ans = StCam_SetWhiteBalanceGain(handle_, wbGainR_, wbGainGr_, wbGainGb_, wbGainB_);
		if (!ans) { return ERR_CAMERA_SET_WB_GAIN_FAILED; }
	}
	return DEVICE_OK;
}
/**
* Handles "WhiteBalance_GainB" property.
*/
int Camera::OnWBGainB(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set((long)wbGainB_);
	}
	else if (eAct == MM::AfterSet)
	{
		if (wbMode_ == g_WB_Mode_Off || wbMode_ == g_WB_Mode_OneShot || wbMode_ == g_WB_Mode_Auto)
		{
			return ERR_CAMERA_WBMODE_UNAVAILABLE_FUNCTION;
		}

		long value;
		pProp->Get(value);

		wbGainB_ = (WORD)value;

		BOOL ans = TRUE;
		ans = StCam_SetWhiteBalanceGain(handle_, wbGainR_, wbGainGr_, wbGainGb_, wbGainB_);
		if (!ans) { return ERR_CAMERA_SET_WB_GAIN_FAILED; }
	}
	return DEVICE_OK;
}

int Camera::OnIsSequenceable(MM::PropertyBase* pProp, MM::ActionType eAct)
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

// Mirror flip, Added in MM 2.0 
int Camera::OnMirrorRotation(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	BYTE mirrorH;
	if (eAct == MM::BeforeGet)
	{
		StCam_GetMirrorMode(handle_, &mirrorH);
		
		if (int(mirrorH) == 0)
		{
			mirrorMode_ = "MIRROR OFF";
		}
		else if (int(mirrorH) == 1)
		{
			mirrorMode_ = "MIRROR HORIZONTAL";
		}
		else if (int(mirrorH) == 2)
		{
			mirrorMode_ = "MIRROR VERTICAL";
		}
		else if (int(mirrorH) == 3)
		{
			mirrorMode_ = "MIRROR HORIZONTAL VERTICAL";
		}

		pProp->Set(mirrorMode_.c_str());
	}
	else if (eAct == MM::AfterSet)
	{
		pProp->Get(mirrorMode_);

		if (mirrorMode_ == "MIRROR OFF")
		{
			StCam_SetMirrorMode(handle_, 0);
		}
		else if (mirrorMode_ == "MIRROR HORIZONTAL")
		{
			StCam_SetMirrorMode(handle_, 1);
		}
		else if (mirrorMode_ == "MIRROR VERTICAL")
		{
			StCam_SetMirrorMode(handle_, 2);
		}
		else if (mirrorMode_ == "MIRROR HORIZONTAL VERTICAL")
		{
			StCam_SetMirrorMode(handle_, 3);
		}
	}


	return DEVICE_OK;
}

// Scan Mode, added in MM 2.0 {Normal Mode and Roi Mode} 
int Camera::OnScanMode(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	bool ans = false;
	WORD mode = 0;
	if (eAct == MM::BeforeGet)
	{
		DWORD /*size_x, size_y, */offset_x, offset_y;
		ans = StCam_GetImageSize(handle_, nullptr, &mode, &offset_x, &offset_y, &/*size_x*/imageSizeH_, &/*size_y*/imageSizeV_);
		if (!ans) { return ERR_CAMERA_SCAN_MODE_FAILED_SETTING; }

		cout << "mode type Before Get " << /*size_x*/imageSizeH_ <<" X "<<  /*size_y*/imageSizeV_ << endl;
		cout << "mode type Before Get " << mode << endl;
		if (mode == STCAM_SCAN_MODE_NORMAL)
		{
			scanModeType_ = "SCAN MODE NORMAL";
			enabledROI_ = false;
		}
		else if (mode == STCAM_SCAN_MODE_ROI)
		{
			scanModeType_ = "SCAN MODE ROI";
			enabledROI_ = true;
		}
		else
		{
			scanModeType_ = "SCAN MODE NORMAL";
			enabledROI_ = false;
			ans = StCam_SetImageSize(handle_, STCAM_IMAGE_SIZE_MODE_UXGA, STCAM_SCAN_MODE_NORMAL, 0, 0, imageSizeH_, imageSizeV_);
			if (!ans) { return ERR_CAMERA_SCAN_MODE_FAILED_SETTING; }
		}

		pProp->Set(scanModeType_.c_str());
	}
	else if (eAct == MM::AfterSet)
	{
		pProp->Get(scanModeType_);
		cout << "mode type After set " << scanModeType_ << endl;
		if (scanModeType_ == "SCAN MODE NORMAL")
		{
			enabledROI_ = false;
			mode = STCAM_SCAN_MODE_NORMAL;
			ans = StCam_SetImageSize(handle_, STCAM_IMAGE_SIZE_MODE_UXGA, mode, 0, 0, imageSizeH_, imageSizeV_);
			if (!ans) { return ERR_CAMERA_SCAN_MODE_FAILED_SETTING; }
			cout << "mode type After set " << mode << endl;
		}
		if (scanModeType_ == "SCAN MODE ROI")
		{
			enabledROI_ = true;
			mode = STCAM_SCAN_MODE_ROI;
			ans = StCam_SetImageSize(handle_, STCAM_IMAGE_SIZE_MODE_UXGA, mode, 0, 0, imageSizeH_, imageSizeV_);
			if (!ans) { return ERR_CAMERA_SCAN_MODE_FAILED_SETTING; }
			cout << "mode type After set " << mode << endl;
			
		}
	}



	return DEVICE_OK;
}

// OnImage H and OnImage V, Added in MM 2.0, allows camera sizing (works only in Roi Mode)   
int Camera::OnImageH(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	bool ans = false;
	WORD mode = 0;
	if (eAct == MM::BeforeGet)
	{
		
		DWORD size_x, size_y, offset_x, offset_y;
		mode = STCAM_SCAN_MODE_ROI;
		ans = StCam_GetImageSize(handle_, nullptr, &mode, &offset_x, &offset_y, &size_x, &size_y);

		imageSizeH_ = size_x;
		pProp->Set((long)imageSizeH_);
	}
	else if (eAct == MM::AfterSet)
	{
		if (IsCapturing())
			return DEVICE_CAMERA_BUSY_ACQUIRING;
		if (scanModeType_ == "SCAN MODE NORMAL")
			return ERR_CAMERA_SCAN_MODE_PROHIBTED;
		double x = 0;
		pProp->Get(x);
		imageSizeH_ = x;
		mode = STCAM_SCAN_MODE_ROI;
		ans = StCam_SetImageSize(handle_, 0, mode, 0, 0, imageSizeH_, imageSizeV_);
		if (!ans) { return ERR_CAMERA_SCAN_MODE_FAILED_SETTING; }


	}
	return DEVICE_OK;
}

// OnImage H and OnImage V, Added in MM 2.0, allows camera sizing (works only in Roi Mode)   
int Camera::OnImageV(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	bool ans = false;
	WORD mode = 0;
	if (eAct == MM::BeforeGet)
	{
	
		DWORD size_x, size_y, offset_x, offset_y;
		mode = STCAM_SCAN_MODE_ROI;
		ans = StCam_GetImageSize(handle_, nullptr, &mode, &offset_x, &offset_y, &size_x, &size_y);

		imageSizeV_ = size_y;
		pProp->Set((long)imageSizeV_);
	}
	else if (eAct == MM::AfterSet)
	{
		if (IsCapturing())
			return DEVICE_CAMERA_BUSY_ACQUIRING;
		if (scanModeType_ == "SCAN MODE NORMAL")
			return ERR_CAMERA_SCAN_MODE_PROHIBTED;
		double y = 0;
		pProp->Get(y);
		imageSizeV_ = y;
		mode = STCAM_SCAN_MODE_ROI;
		ans = StCam_SetImageSize(handle_, 0, mode, 0, 0, imageSizeH_, imageSizeV_);
		if (!ans) { return ERR_CAMERA_SCAN_MODE_FAILED_SETTING; }
	}
	return DEVICE_OK;
}
