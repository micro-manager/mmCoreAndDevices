///////////////////////////////////////////////////////////////////////////////
// FILE:          LumeneraAce.h
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Adapter for Lumenera  Cameras
//
// Written by Henry Pinkard (Photomics, Inc.)
//
// Redistribution and use in source and binary forms, with or without modification, 
// are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this 
// list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice, this
// list of conditions and the following disclaimer in the documentation and/or other 
// materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors may
// be used to endorse or promote products derived from this software without specific 
// prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES 
// OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT 
// SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, 
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
// TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR 
// BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN 
// ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH 
// DAMAGE.



//#include <pylon/PylonIncludes.h>
// Include file to use pylon universal instant camera parameters.
//#include <pylon/LumeneraUniversalInstantCamera.h>



#include "LumeneraCamera.h"
//#include <sstream>
#include <math.h>
#include "ModuleInterface.h"
#include "DeviceUtils.h"
#include <vector>
#include "../MMCore/Error.h"

// These are from LuXApps/shared
#include "Cameras/Camera.h"
#include "Cameras/LucamCamera/LucamCamera.h"
#include "LucamInterface.h"
#include "PropertyNames.h"
#include "CameraOperations.h"
#include "Formatting.h"
#include "Utilities.hxx"
//#include "Tools/Debugger.h"
#include "Properties/PropertyValueEnumerations.h"
#include "Images/ImageConversion.h" 

#include "VideoSequenceThread.h"

using namespace std;

const char* g_LumeneraCameraDeviceName = "LumeneraCamera";
const char* g_Device_Name = "LuCam";
const char* g_Camera_Description = "Lumenera INFINITY Adapter";

static const char* g_Keyword_USB = "USB";
static const char* g_Keyword_GIGE = "GigE Vision";
std::vector<std::string> HARDWARE_INTERFACES
{
	g_Keyword_USB,
	g_Keyword_GIGE,
};

std::vector<std::string> FILTERED_LIGHT_SOURCES
{
	to_string(LuXApps::CORRECTION_MATRIX::NONE),
	to_string(LuXApps::CORRECTION_MATRIX::CUSTOM),
};

static const char* g_PixelType_8bit_MONO = "8bit-MONO";
static const char* g_PixelType_16bit_MONO = "16bit-MONO";
static const char* g_PixelType_32bit_COLOR = "8bit-COLOR";
static const char* g_PixelType_64bit_COLOR = "16bit-COLOR";

constexpr uint32_t BIN_SIZE_1 = 1;
constexpr uint32_t BIN_SIZE_2 = 2;
constexpr uint32_t BIN_SIZE_4 = 4;
constexpr uint32_t BIN_SIZE_8 = 8;

static const char* g_CameraIndex = "CameraIndex";
static const char* g_Camera_Index = "Camera Index";
static const char* g_Camera_API = "Camera API";
static const char* g_Camera_Driver = "Camera Driver";
static const char* g_Camera_Firmware = "Camera Firmware";
static const char* g_Camera_FPGA = "Camera FPGA";
static const char* g_Camera_Hardware_Revision = "Camera Hardware Revision";
static const char* g_Camera_Hardware_Interface = "Interface";
static const char* g_Camera_Sensor_Width = "Sensor Width";
static const char* g_Camera_Sensor_Height = "Sensor Height";
static const char* g_Camera_BitDepth = "Bit Depth";
static const char* g_Camera_Tap_Configuration = "Tap Configuration";
static const char* g_Camera_Demosaic_Method = "Demosaic Method";
static const char* g_Camera_Light_Source = "Light Source";
static const char* g_Camera_Flip = "Flipping";
static const char* g_Camera_Mirror = "Mirror";
static const char* g_Camera_Gamma = "Gamma";
static const char* g_Camera_Saturation = "Saturation";
static const char* g_Camera_Hue = "Hue";
static const char* g_Camera_Brigthness = "Brightness";
static const char* g_Camera_Contrast = "Contrast";
static const char* g_Camera_High_Conversion_Gain = "High Conversion Gain";

static const char* g_Camera_Gain_Red = "RGB Red Gain";
static const char* g_Camera_Gain_Green1 = "RGB Green1 Gain";
static const char* g_Camera_Gain_Green2 = "RGB Green2 Gain";
static const char* g_Camera_Gain_Blue = "RGB Blue Gain";

static const char* g_Camera_Temperature = "Camera Temperature";
static const char* g_Camera_Cooling = "Camera Cooling";

static const char* g_Camera_Iris = "Iris";
static const char* g_Camera_Focus = "Focus";
static const char* g_Camera_Absolute_Focus = "Absolute Focus";

static const char* g_Keyword_White_Balance = "One Shot White Balance";
static const char* g_Keyword_White_Balance_Target_Red = "White Balance Target (Red)";
static const char* g_Keyword_White_Balance_Target_Green = "White Balance Target (Green)";
static const char* g_Keyword_White_Balance_Target_Blue = "White Balance Target (Blue)";


const char* g_Camera_Timeout = "Timeout";
const char* g_keyword_Red_Over_Green = "WB -> Red/Green background";
const char* g_keyword_Blue_Over_Green = "WB -> Blue/Green background";

const char* g_Camera_Pixel_Size = "Camera Pixels (um)";

static const char* TRUE_STRING = "True";
static const char* FALSE_STRING = "False";

static const char* BINNING_8X8_STRING = "Binning (8x8)";
static const char* BINNING_4X4_STRING = "Binning (4x4)";
static const char* BINNING_2X2_STRING = "Binning (2x2)";
static const char* NONE_1X1_STRING = "None (1x1)";


static const char* g_PropertyChannel = "PropertyNAme";
static const char* g_PixelType_8bit = "8bit mono";
static const char* g_PixelType_10bit = "10bit mono";
static const char* g_PixelType_12bit = "12bit mono";
static const char* g_PixelType_16bit = "16bit mono";
static const char* g_PixelType_10packedbit = "10bit mono";
static const char* g_PixelType_12packedbit = "12bit mono";


static const char* g_PixelType_8bitRGBA = "8bitBGRA";
static const  char* g_PixelType_8bitRGB = "8bitRGB";
static const  char* g_PixelType_8bitBGR = "8bitBGR";

#define RETURN_ON_ERROR(function) do { int result; if ((result = function) != DEVICE_OK) { return result; } } while (0)


///////////////////////////////////////////////////////////////////////////////
// Exported MMDevice API
///////////////////////////////////////////////////////////////////////////////




MODULE_API void InitializeModuleData()
{
	RegisterDevice(g_LumeneraCameraDeviceName, MM::CameraDevice, "Lumenera Camera");
}

MODULE_API MM::Device* CreateDevice(const char* deviceName)
{
	MM::Device* device = nullptr;

	if (deviceName)
	{
		// If the device is a Lumenera Camera
		if (std::string(deviceName) == g_LumeneraCameraDeviceName)
		{
			// Create the camera device
			device = new LumeneraCamera();
		}
	}

	return device;
}

MODULE_API void DeleteDevice(MM::Device* pDevice)
{
	if (pDevice)
	{
		delete pDevice;
		pDevice = nullptr;
	}
}

///////////////////////////////////////////////////////////////////////////////
// Lumenera Camera implementation
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/**
* Constructor.
*/
LumeneraCamera::LumeneraCamera() :
	CCameraBase<LumeneraCamera>(),
	components_(1),
	bitDepth_(Imaging::IMAGE_BIT_DEPTH::EIGHT_BIT),
	colorCamera_(true),
	cameraIndex_(1),
	initialized_(false),
	hasSwitchingExposure_(false),
	camera_(nullptr),
	format_(Imaging::IMAGE_FORMAT::MONO),
	switchingExposure_(0)
{
	CameraInterface::Camera::initializeWorkspace();

	// Call the base class method to set-up default error codes/messages
	InitializeDefaultErrorMessages();
	//SetErrorText(DEVICE_STREAM_START_FAILURE, g_Msg_STREAM_START_FAILED);
	//SetErrorText(DEVICE_CUSTOM_ERROR, g_Msg_ERR);

	SetErrorText(DEVICE_SNAP_IMAGE_FAILED, "Failed to capture image.");
	SetErrorText(DEVICE_NOT_CONNECTED, "Camera not connected.");

	/////////// Pre-init properties /////////
	CPropertyAction* pAction = new CPropertyAction(this, &LumeneraCamera::OnCameraIndex);
	CreateProperty(g_Camera_Index, std::to_string(cameraIndex_).c_str(), MM::Integer, false, pAction, true);

	CreateProperty(g_Camera_Hardware_Interface, g_Keyword_USB, MM::String, false, nullptr, true);
	SetAllowedValues(g_Camera_Hardware_Interface, HARDWARE_INTERFACES);


	sequenceThread_ = new VideoSequenceThread(this);
}

LumeneraCamera::~LumeneraCamera()
{
	
	if (sequenceThread_)
	{
		delete sequenceThread_;
		sequenceThread_ = nullptr;
	}

	CameraInterface::Camera::cleanupWorkspace();
}

/**
* Obtains device name.
*/
void LumeneraCamera::GetName(char* name) const
{
	CDeviceUtils::CopyLimitedString(name, g_LumeneraCameraDeviceName);
}

/**
* Initializes the hardware.
*/
int LumeneraCamera::Initialize()
{
	if (initialized_)
		return DEVICE_OK;

	//TODO: LuXApps needs a public  way to specify which Hardware Interfaces to scan for camera enumeration
	//		Only USB cameras are enumerated at this point.
	if (!initializeHardwareInterface()) { return DEVICE_NOT_CONNECTED; }

	std::map<std::string, std::string> cameraList = CameraInterface::Camera::getCameraList();

	if (cameraList.empty())
	{
		return DEVICE_NOT_CONNECTED;
	}

	long index = 0;

	if (GetProperty(g_Camera_Index, index) == DEVICE_OK)
	{
		cameraIndex_ = static_cast<int>(index);
		size_t cameraListIndex = static_cast<size_t>(cameraIndex_) - 1;

		if (cameraListIndex < cameraList.size())
		{
			LUXAPPS_TRY
			{
				std::vector<std::pair<std::string, std::string>> cameras{ std::begin(cameraList), std::end(cameraList) };

				std::string serialNumber = cameras.at(cameraListIndex).first;
				if (!serialNumber.empty())
				{
					camera_ = dynamic_cast<CameraInterface::LucamAdapter::LucamCamera*>(&CameraInterface::Camera::getCamera(serialNumber));
				}

				if (camera_)
				{
					camera_->connect();

					if (cameraRequiresClockThrottling(camera_))
					{
						throttleClockSpeed(camera_, LuXAppsProperties::LUXPROP_VIDEO_CLOCK_SPEED);
						throttleClockSpeed(camera_, LuXAppsProperties::LUXPROP_STILL_CLOCK_SPEED);
					}

					//NOTE: This is a hack to get the format to be applied after getting all of the properties
					camera_->startStream();
					std::this_thread::sleep_for(std::chrono::seconds(1));
					camera_->stopStream();

					CameraInterface::Property* videoExposure = nullptr;
					if (camera_->tryGetProperty(LuXAppsProperties::LUXPROP_VIDEO_EXPOSURE, videoExposure))
					{
						//NOTE: This is to refresh the range after any other format changes have been done.
						videoExposure->update();
					}

					CameraInterface::Property* stillExposure = nullptr;
					if (camera_->tryGetProperty(LuXAppsProperties::LUXPROP_STILL_EXPOSURE, stillExposure))
					{
						//NOTE: This is to refresh the range after any other format changes have been done.
						stillExposure->update();
					}

					RETURN_ON_ERROR(initializeImageBuffer());

					RETURN_ON_ERROR(createProperties(camera_));

					RETURN_ON_ERROR(UpdateStatus());

					initialized_ = true;

					return DEVICE_OK;
				}
				else
				{
					return DEVICE_NOT_CONNECTED;
				}
			}
			LUXAPPS_CATCH(const LuXAppsExceptions::LuXAppsException & e)
			{
				GetCoreCallback()->LogMessage(this, e.what(), false);
				
				return DEVICE_NOT_CONNECTED;
			}
		}
		else
		{
			return DEVICE_NOT_CONNECTED;
		}
	}
	else
	{
		return DEVICE_NOT_CONNECTED;
	}
}

int LumeneraCamera::createProperties(CameraInterface::Camera* camera)
{
	///////  Pixel type ///////
	Imaging::IMAGE_BIT_DEPTH bitDepth;
	RETURN_ON_ERROR(getBitDepthFromCamera(bitDepth));

	std::vector<std::string> pixelTypeValues;

	std::string pixelType;

	if (LuXApps::isMonoCamera(camera))
	{
		pixelTypeValues.push_back(g_PixelType_8bit_MONO);
		pixelTypeValues.push_back(g_PixelType_16bit_MONO);
		pixelType = ((bitDepth == Imaging::IMAGE_BIT_DEPTH::EIGHT_BIT) ? g_PixelType_8bit_MONO : g_PixelType_16bit_MONO);
	}
	else
	{
		pixelTypeValues.push_back(g_PixelType_32bit_COLOR);
		pixelTypeValues.push_back(g_PixelType_64bit_COLOR);
		pixelType = ((bitDepth == Imaging::IMAGE_BIT_DEPTH::EIGHT_BIT) ? g_PixelType_32bit_COLOR : g_PixelType_64bit_COLOR);
	}

	CPropertyAction* pAction = new CPropertyAction(this, &LumeneraCamera::OnPixelType);
	RETURN_ON_ERROR(CreateProperty(MM::g_Keyword_PixelType, pixelType.c_str(), MM::String, false, pAction));
	RETURN_ON_ERROR(SetAllowedValues(MM::g_Keyword_PixelType, pixelTypeValues));

	//// Exposure /////
	CameraInterface::Property* videoExposure = nullptr;
	if (camera->tryGetProperty(LuXAppsProperties::LUXPROP_VIDEO_EXPOSURE, videoExposure))
	{
		std::string value;
		double minimum = 0;
		double maximum = 0;

		std::string videoValue = videoExposure->getValue();
		double videoMinimum = std::stod(videoExposure->getMin());
		double videoMaximum = std::stod(videoExposure->getMax());

		CameraInterface::Property* stillExposure = nullptr;

		if (camera->tryGetProperty(LuXAppsProperties::LUXPROP_STILL_EXPOSURE, stillExposure))
		{
			std::string stillValue = stillExposure->getValue();
			double stillMaximum = std::stod(stillExposure->getMax());

			value = isVideoStreamingMode() ? videoValue : stillValue;

			minimum = videoMinimum;
			maximum = std::max(videoMaximum, stillMaximum);

			//NOTE: If still maximum is greater than video maximum, we will switch streaming
			//		modes when current exposure exceeds maximum video exposure.
			if (maximum > videoMaximum)
			{
				hasSwitchingExposure_ = true;
				switchingExposure_ = videoMaximum;
			}
		}
		else
		{
			value = videoValue;
			minimum = videoMinimum;
			maximum = videoMaximum;
		}

		pAction = new CPropertyAction(this, &LumeneraCamera::OnExposure);
		RETURN_ON_ERROR(CreateProperty(MM::g_Keyword_Exposure, value.c_str(), MM::Float, false, pAction, false));
		RETURN_ON_ERROR(SetPropertyLimits(MM::g_Keyword_Exposure, minimum, maximum));
	}

	//////// Binning /////
	CameraInterface::Property* prop = nullptr;

	std::string value = NONE_1X1_STRING;
	std::vector<std::string> samplingOptions;
	bool readOnly = true;

	if (camera->tryGetProperty(LuXAppsProperties::LUXPROP_SAMPLING_DESCRIPTION, prop))
	{
		value = prop->getValue();

		samplingOptions = prop->getValueList();

		readOnly = false;
	}

	std::string binSize = std::to_string(getBinValueFromSampling(value));

	std::vector<std::string> binSizes;

	if (samplingOptions.empty())
	{
		binSizes.push_back(binSize);
	}
	else
	{
		std::vector<std::string> binningOptions = getBinningOptions();

		for (const std::string& sampling : samplingOptions)
		{
			if (::Utilities::contains(binningOptions, sampling))
			{
				binSizes.push_back(std::to_string(getBinValueFromSampling(sampling)));
			}
		}
	}

	RETURN_ON_ERROR(CreateProperty(MM::g_Keyword_Binning, binSize.c_str(), MM::Integer, readOnly,
		readOnly ? nullptr : new CPropertyAction(this, &LumeneraCamera::OnBinning)));
	RETURN_ON_ERROR(SetAllowedValues(MM::g_Keyword_Binning, binSizes));

	//Device name
	RETURN_ON_ERROR(CreateProperty(MM::g_Keyword_Name, g_Device_Name, MM::String, true));

	//Device description
	RETURN_ON_ERROR(CreateProperty(MM::g_Keyword_Description, g_Camera_Description, MM::String, true));

	//Model name
	RETURN_ON_ERROR(CreateProperty(MM::g_Keyword_CameraName, camera->getModelName().c_str(), MM::String, true));

	//Camera ID
	if (camera->tryGetProperty(LuXAppsProperties::LUXPROP_CAMERA_ID, prop))
	{
		char cBuf[MM::MaxStrLength];
		unsigned long modelId = std::stoul(prop->getValue());
		sprintf(cBuf, "0x%04X", modelId);

		RETURN_ON_ERROR(CreateProperty(MM::g_Keyword_CameraID, cBuf, MM::String, true));
	}

	//Camera API
	if (camera->tryGetProperty(LuXAppsProperties::LUXPROP_VERSION_API, prop))
	{
		RETURN_ON_ERROR(CreateProperty(g_Camera_API, prop->getValue().c_str(), MM::String, true));
	}

	//Camera driver
	if (camera->tryGetProperty(LuXAppsProperties::LUXPROP_VERSION_DRIVER, prop))
	{
		RETURN_ON_ERROR(CreateProperty(g_Camera_Driver, prop->getValue().c_str(), MM::String, true));
	}

	//Camera firmware
	if (camera->tryGetProperty(LuXAppsProperties::LUXPROP_VERSION_FIRMWARE, prop))
	{
		RETURN_ON_ERROR(CreateProperty(g_Camera_Firmware, prop->getValue().c_str(), MM::String, true));
	}

	//Camera FPGA
	if (camera->tryGetProperty(LuXAppsProperties::LUXPROP_VERSION_FPGA, prop))
	{
		RETURN_ON_ERROR(CreateProperty(g_Camera_FPGA, prop->getValue().c_str(), MM::String, true));
	}

	//Hardware revision
	if (camera->tryGetProperty(LuXAppsProperties::LUXPROP_HARDWARE_REVISION, prop))
	{
		RETURN_ON_ERROR(CreateProperty(g_Camera_Hardware_Revision, prop->getValue().c_str(), MM::String, true));
	}

	//Sensor width
	if (camera->tryGetProperty(LuXAppsProperties::LUXPROP_MAXIMUM_WIDTH, prop))
	{
		RETURN_ON_ERROR(CreateProperty(g_Camera_Sensor_Width, prop->getValue().c_str(), MM::Integer, true));
	}

	//Sensor height
	if (camera->tryGetProperty(LuXAppsProperties::LUXPROP_MAXIMUM_HEIGHT, prop))
	{
		RETURN_ON_ERROR(CreateProperty(g_Camera_Sensor_Height, prop->getValue().c_str(), MM::Integer, true));
	}

	//Bit depth
	RETURN_ON_ERROR(getBitDepthFromCamera(bitDepth));
	if (camera->tryGetProperty(LuXAppsProperties::LUXPROP_PIXEL_FORMAT, prop))
	{
		std::vector<std::string> pixelFormats = prop->getValueList();

		std::vector<std::string> bitDepthOptions = getBitDepthOptions();
		std::vector<std::string> bitDepths;

		for (const std::string& pixelFormat : pixelFormats)
		{
			if (::Utilities::contains(bitDepthOptions, pixelFormat))
			{
				bitDepths.push_back(pixelFormat);
			}
		}

		pAction = new CPropertyAction(this, &LumeneraCamera::OnBitDepth);
		RETURN_ON_ERROR(CreateProperty(g_Camera_BitDepth, to_string(bitDepth).c_str(), MM::String, true, pAction));
		RETURN_ON_ERROR(SetAllowedValues(g_Camera_BitDepth, bitDepths));
	}

	//Tap configuration
	if (camera->tryGetProperty(LuXAppsProperties::LUXPROP_VIDEO_TAP_CONFIGURATION, prop))
	{
		value = prop->getValue();
		std::vector<std::string> values = prop->getValueList();

		pAction = new CPropertyAction(this, &LumeneraCamera::OnTapConfiguration);
		RETURN_ON_ERROR(CreateProperty(g_Camera_Tap_Configuration, value.c_str(), MM::String, false, pAction));
		RETURN_ON_ERROR(SetAllowedValues(g_Camera_Tap_Configuration, values));
	}

	//Gain
	RETURN_ON_ERROR(createLinkedGainProperty(camera, LuXAppsProperties::LUXPROP_VIDEO_GAIN, MM::g_Keyword_Gain));

	//red gain
	RETURN_ON_ERROR(createLinkedGainProperty(camera, LuXAppsProperties::LUXPROP_VIDEO_GAIN_RED, g_Camera_Gain_Red));

	//Green1 gain
	RETURN_ON_ERROR(createLinkedGainProperty(camera, LuXAppsProperties::LUXPROP_VIDEO_GAIN_GREEN_1, g_Camera_Gain_Green1));

	//Green2 gain
	RETURN_ON_ERROR(createLinkedGainProperty(camera, LuXAppsProperties::LUXPROP_VIDEO_GAIN_GREEN_2, g_Camera_Gain_Green2));

	//Blue gain
	RETURN_ON_ERROR(createLinkedGainProperty(camera, LuXAppsProperties::LUXPROP_VIDEO_GAIN_BLUE, g_Camera_Gain_Blue));

	//Demosaic method
	if (camera->tryGetProperty(LuXAppsProperties::LUXPROP_DEMOSAIC_METHOD, prop))
	{
		value = prop->getValue();
		std::vector<std::string> values = prop->getValueList();
		//TODO: Need to filter out HIGHER_QUALITY

		pAction = new CPropertyAction(this, &LumeneraCamera::OnDemosaicingMethod);
		RETURN_ON_ERROR(CreateProperty(g_Camera_Demosaic_Method, value.c_str(), MM::String, false, pAction));
		RETURN_ON_ERROR(SetAllowedValues(g_Camera_Demosaic_Method, values));
	}

	//Light source
	if (camera->tryGetProperty(LuXAppsProperties::LUXPROP_CORRECTION_MATRIX, prop))
	{
		value = prop->getValue();
		std::vector<std::string> values = prop->getValueList();

		::Utilities::remove_all(values, FILTERED_LIGHT_SOURCES);

		pAction = new CPropertyAction(this, &LumeneraCamera::OnLightSource);
		RETURN_ON_ERROR(CreateProperty(g_Camera_Light_Source, value.c_str(), MM::String, false, pAction));
		RETURN_ON_ERROR(SetAllowedValues(g_Camera_Light_Source, values));
	}

	//Flipping
	if (camera->tryGetProperty(LuXAppsProperties::LUXPROP_VERTICAL_FLIP, prop))
	{
		value = prop->getValue();
		std::vector<std::string> values = prop->getValueList();

		pAction = new CPropertyAction(this, &LumeneraCamera::OnFlip);
		RETURN_ON_ERROR(CreateProperty(g_Camera_Flip, value.c_str(), MM::String, false, pAction));
		RETURN_ON_ERROR(SetAllowedValues(g_Camera_Flip, values));
	}

	//Mirror
	if (camera->tryGetProperty(LuXAppsProperties::LUXPROP_HORIZONTAL_FLIP, prop))
	{
		value = prop->getValue();
		std::vector<std::string> values = prop->getValueList();

		pAction = new CPropertyAction(this, &LumeneraCamera::OnMirror);
		RETURN_ON_ERROR(CreateProperty(g_Camera_Mirror, value.c_str(), MM::String, false, pAction));
		RETURN_ON_ERROR(SetAllowedValues(g_Camera_Mirror, values));
	}

	//Hue
	if (camera->tryGetProperty(LuXAppsProperties::LUXPROP_HUE, prop))
	{
		value = prop->getValue();
		std::string minimum = prop->getMin();
		std::string maximum = prop->getMax();

		pAction = new CPropertyAction(this, &LumeneraCamera::OnHue);
		RETURN_ON_ERROR(CreateProperty(g_Camera_Hue, value.c_str(), MM::Float, false, pAction, false));
		RETURN_ON_ERROR(SetPropertyLimits(g_Camera_Hue, std::stod(minimum), std::stod(maximum)));
	}

	//Saturation
	if (camera->tryGetProperty(LuXAppsProperties::LUXPROP_SATURATION, prop))
	{
		value = prop->getValue();
		std::string minimum = prop->getMin();
		std::string maximum = prop->getMax();

		pAction = new CPropertyAction(this, &LumeneraCamera::OnSaturation);
		RETURN_ON_ERROR(CreateProperty(g_Camera_Saturation, value.c_str(), MM::Float, false, pAction, false));
		RETURN_ON_ERROR(SetPropertyLimits(g_Camera_Saturation, std::stod(minimum), std::stod(maximum)));
	}

	//Gamma
	if (camera->tryGetProperty(LuXAppsProperties::LUXPROP_GAMMA, prop))
	{
		value = prop->getValue();
		std::string minimum = prop->getMin();
		std::string maximum = prop->getMax();

		pAction = new CPropertyAction(this, &LumeneraCamera::OnGamma);
		RETURN_ON_ERROR(CreateProperty(g_Camera_Gamma, value.c_str(), MM::Float, false, pAction, false));
		RETURN_ON_ERROR(SetPropertyLimits(g_Camera_Gamma, std::stod(minimum), std::stod(maximum)));
	}

	//Contrast
	if (camera->tryGetProperty(LuXAppsProperties::LUXPROP_CONTRAST, prop))
	{
		value = prop->getValue();
		std::string minimum = prop->getMin();
		std::string maximum = prop->getMax();

		pAction = new CPropertyAction(this, &LumeneraCamera::OnContrast);
		RETURN_ON_ERROR(CreateProperty(g_Camera_Contrast, value.c_str(), MM::Float, false, pAction, false));
		RETURN_ON_ERROR(SetPropertyLimits(g_Camera_Contrast, std::stod(minimum), std::stod(maximum)));
	}

	//Brightness
	if (camera->tryGetProperty(LuXAppsProperties::LUXPROP_BRIGHTNESS, prop))
	{
		value = prop->getValue();
		std::string minimum = prop->getMin();
		std::string maximum = prop->getMax();

		pAction = new CPropertyAction(this, &LumeneraCamera::OnBrightness);
		RETURN_ON_ERROR(CreateProperty(g_Camera_Brigthness, value.c_str(), MM::Float, false, pAction, false));
		RETURN_ON_ERROR(SetPropertyLimits(g_Camera_Brigthness, std::stod(minimum), std::stod(maximum)));
	}

	//High Conversion Gain
	if (camera->tryGetProperty(LuXAppsProperties::LUXPROP_HIGH_CONVERSION_GAIN, prop))
	{
		value = prop->getValue();
		std::vector<std::string> values = prop->getValueList();

		pAction = new CPropertyAction(this, &LumeneraCamera::OnHighConversionGain);
		RETURN_ON_ERROR(CreateProperty(g_Camera_High_Conversion_Gain, value.c_str(), MM::String, false, pAction));
		RETURN_ON_ERROR(SetAllowedValues(g_Camera_High_Conversion_Gain, values));
	}

	//Cooling
	if (camera->isPropertySupported(LuXAppsProperties::LUXPROP_FAN) && camera->isPropertySupported(LuXAppsProperties::LUXPROP_COOLING))
	{
		CameraInterface::Property* fanProperty = nullptr;

		if (camera->tryGetProperty(LuXAppsProperties::LUXPROP_FAN, fanProperty))
		{
			value = fanProperty->getValue();
			std::vector<std::string> values = fanProperty->getValueList();

			pAction = new CPropertyAction(this, &LumeneraCamera::OnCooling);
			RETURN_ON_ERROR(CreateProperty(g_Camera_Cooling, value.c_str(), MM::String, false, pAction));
			RETURN_ON_ERROR(SetAllowedValues(g_Camera_Cooling, values));
		}
	}

	//TODO: Should create some kind of timer to refresh Temperature property
	//Temperature
	if (camera->tryGetProperty(LuXAppsProperties::LUXPROP_CAMERA_TEMPERATURE, prop))
	{
		pAction = new CPropertyAction(this, &LumeneraCamera::OnTemperature);
		RETURN_ON_ERROR(CreateProperty(g_Camera_Temperature, prop->getValue().c_str(), MM::String, true, pAction));
	}

	//Iris
	if (camera->tryGetProperty(LuXAppsProperties::LUXPROP_IRIS, prop))
	{
		value = prop->getValue();
		std::string minimum = prop->getMin();
		std::string maximum = prop->getMax();

		pAction = new CPropertyAction(this, &LumeneraCamera::OnIris);
		RETURN_ON_ERROR(CreateProperty(g_Camera_Iris, value.c_str(), MM::Float, false, pAction, false));
		RETURN_ON_ERROR(SetPropertyLimits(g_Camera_Iris, std::stod(minimum), std::stod(maximum)));
	}

	//Focus
	if (camera->tryGetProperty(LuXAppsProperties::LUXPROP_FOCUS, prop))
	{
		value = prop->getValue();
		std::string minimum = prop->getMin();
		std::string maximum = prop->getMax();

		pAction = new CPropertyAction(this, &LumeneraCamera::OnFocus);
		RETURN_ON_ERROR(CreateProperty(g_Camera_Focus, value.c_str(), MM::Integer, false, pAction, false));
		RETURN_ON_ERROR(SetPropertyLimits(g_Camera_Focus, std::stod(minimum), std::stod(maximum)));
	}

	//Absolute focus
	if (camera->tryGetProperty(LuXAppsProperties::LUXPROP_ABSOLUTE_FOCUS, prop))
	{
		value = prop->getValue();
		std::string minimum = prop->getMin();
		std::string maximum = prop->getMax();

		pAction = new CPropertyAction(this, &LumeneraCamera::OnAbsoluteFocus);
		RETURN_ON_ERROR(CreateProperty(g_Camera_Absolute_Focus, value.c_str(), MM::Integer, false, pAction, false));
		RETURN_ON_ERROR(SetPropertyLimits(g_Camera_Absolute_Focus, std::stod(minimum), std::stod(maximum)));
	}

	//White balance target red
	RETURN_ON_ERROR(createWhiteBalanceTargetProperty(camera_, g_Keyword_White_Balance_Target_Red));
	
	//White balance target green
	RETURN_ON_ERROR(createWhiteBalanceTargetProperty(camera_, g_Keyword_White_Balance_Target_Green));
	
	//White balance target blue
	RETURN_ON_ERROR(createWhiteBalanceTargetProperty(camera_, g_Keyword_White_Balance_Target_Blue));

	//White Balance
	if (LuXApps::isColorCamera(camera))
	{
		std::vector<std::string> values{ TRUE_STRING, FALSE_STRING };

		pAction = new CPropertyAction(this, &LumeneraCamera::OnWhiteBalance);
		RETURN_ON_ERROR(CreateProperty(g_Keyword_White_Balance, FALSE_STRING, MM::String, false, pAction));
		RETURN_ON_ERROR(SetAllowedValues(g_Keyword_White_Balance, values));
	}

	//Hardware Triggering
	if (camera->tryGetProperty(LuXAppsProperties::LUXPROP_HARDWARE_TRIGGER, prop))
	{
		value = prop->getValue();
		std::vector<std::string> values = prop->getValueList();
		pAction = new CPropertyAction(this, &LumeneraCamera::OnHardwareTrigger);
		RETURN_ON_ERROR(CreateProperty(LuXAppsProperties::LUXPROP_HARDWARE_TRIGGER, value.c_str(), MM::String, false, pAction));
		RETURN_ON_ERROR(SetAllowedValues(LuXAppsProperties::LUXPROP_HARDWARE_TRIGGER, values));
	}

	//Trigger Mode
	if (camera->tryGetProperty(LuXAppsProperties::LUXPROP_TRIGGER_MODE, prop))
	{
		value = prop->getValue();
		std::vector<std::string> values = prop->getValueList();
		pAction = new CPropertyAction(this, &LumeneraCamera::OnTriggerMode);
		RETURN_ON_ERROR(CreateProperty(LuXAppsProperties::LUXPROP_TRIGGER_MODE, value.c_str(), MM::String, false, pAction));
		RETURN_ON_ERROR(SetAllowedValues(LuXAppsProperties::LUXPROP_TRIGGER_MODE, values));
	}

	//Trigger Pin
	if (camera->tryGetProperty(LuXAppsProperties::LUXPROP_TRIGGER_PIN, prop))
	{
		value = prop->getValue(); 
		std::vector<std::string> values = prop->getValueList();
		pAction = new CPropertyAction(this, &LumeneraCamera::OnTriggerPin);
		RETURN_ON_ERROR(CreateProperty(LuXAppsProperties::LUXPROP_TRIGGER_PIN, value.c_str(), MM::String, false, pAction));
		RETURN_ON_ERROR(SetAllowedValues(LuXAppsProperties::LUXPROP_TRIGGER_PIN, values));
	}

	//Trigger Polarity
	if (camera->tryGetProperty(LuXAppsProperties::LUXPROP_TRIGGER_POLARITY, prop))
	{
		value = prop->getValue();
		std::vector<std::string> values = prop->getValueList();
		pAction = new CPropertyAction(this, &LumeneraCamera::OnTriggerPolarity);
		RETURN_ON_ERROR(CreateProperty(LuXAppsProperties::LUXPROP_TRIGGER_POLARITY, value.c_str(), MM::String, false, pAction));
		RETURN_ON_ERROR(SetAllowedValues(LuXAppsProperties::LUXPROP_TRIGGER_POLARITY, values));
	}

	//Timeout
	if (camera->tryGetProperty(LuXAppsProperties::LUXPROP_TIMEOUT, prop))
	{
		value = prop->getValue();
		std::string minimum = prop->getMin();
		std::string maximum = prop->getMax();

		double v = std::stod(value) / 1000.0;
		double min = std::stof(minimum) / 1000.0;
		double max = std::stof(maximum) / 1000.0;

		pAction = new CPropertyAction(this, &LumeneraCamera::OnTimeout);
		RETURN_ON_ERROR(CreateProperty(g_Camera_Timeout, LuXApps::Format::print_double(v, 3).c_str(), MM::Float, false, pAction, false));
		RETURN_ON_ERROR(SetPropertyLimits(g_Camera_Timeout, min, max));
	}

	return DEVICE_OK;
}

/**
* Shuts down (unloads) the device.
*/
int LumeneraCamera::Shutdown()
{
	StopSequenceAcquisition();

	if (camera_ != nullptr)
	{
		camera_->disconnect();
	}
	initialized_ = false;
	return DEVICE_OK;
}

unsigned  LumeneraCamera::GetNumberOfComponents() const
{
	return components_;
}

int LumeneraCamera::captureImage()
{
	if (hasCamera())
	{
		LUXAPPS_TRY
		{
			auto cam = camera();

			std::unique_ptr<Imaging::Image> rawImage = cam->getImage();

			// Demosaicing frame 

			LUCAM_IMAGE_FORMAT format = cam->getStreamImageFormat();
			LUCAM_CONVERSION_PARAMS params = LuXApps::Lucam::getConversionParametersFromCameraProperties(cam);

			std::unique_ptr<Imaging::Image> demosaicedImage;

			Imaging::IMAGE_FORMAT outputFormat = (rawImage->getFormat() == Imaging::IMAGE_FORMAT::MONO) ?
				Imaging::IMAGE_FORMAT::MONO : Imaging::IMAGE_FORMAT::BGRA;

			demosaicedImage.reset(LuXApps::Lucam::demosaic(rawImage.get(), cam
				->getHandle(), outputFormat, format, params).release());

			char pixelType[MM::MaxStrLength];
			RETURN_ON_ERROR(GetProperty(MM::g_Keyword_PixelType, pixelType));

			//NOTE: MicroManager expects BGRA format for color images so we need to convert from RGB
			outputFormat = getImageFormatFromPixelType(std::string(pixelType));
			if (demosaicedImage->getFormat() != outputFormat)
			{
				demosaicedImage.reset(Imaging::convertImage(demosaicedImage, outputFormat).release());
			}

			updateImageBuffer(std::move(demosaicedImage));

			return DEVICE_OK;
		}
		LUXAPPS_CATCH(...)
		{
			//TODO: Add a more meaningful error when timeout triggers
			return DEVICE_SNAP_IMAGE_FAILED;
		}
	}
	else
	{
		return DEVICE_NOT_CONNECTED;
	}
}

int LumeneraCamera::SnapImage()
{
	std::string t;

	CameraInterface::Property* timeout = nullptr;

	if (hasCamera())
	{
		if (camera()->tryGetProperty(LuXAppsProperties::LUXPROP_TIMEOUT, timeout))
		{
			t = timeout->getValue();

			double v = std::stod(t);

			if (v < 0)
			{
				timeout->setValue("5000.0");
			}
		}
	}

	int ret = captureImage();

	if (timeout)
	{
		timeout->setValue(t);
	}

	return ret;
	//return ret == DEVICE_OK ? ret : DEVICE_SNAP_IMAGE_FAILED;
}

/**
* Returns pixel data.
*/
const unsigned char* LumeneraCamera::GetImageBuffer()
{
	const unsigned char* imageBuffer = image_.GetPixels();
	return imageBuffer;
}

unsigned LumeneraCamera::GetImageWidth() const
{
	return image_.Width();
}

unsigned LumeneraCamera::GetImageHeight() const
{
	return image_.Height();
}

/**
* Returns image buffer pixel depth in bytes.
*/
unsigned LumeneraCamera::GetImageBytesPerPixel() const
{
	return image_.Depth();
}

/**
* Returns the bit depth (dynamic range) of the pixel.
*/
unsigned LumeneraCamera::GetBitDepth() const
{
	return  static_cast<unsigned>(bitDepth_);
}

/**
* Returns the size in bytes of the image buffer.
*/
long LumeneraCamera::GetImageBufferSize() const
{
	return GetImageWidth() * GetImageHeight() * GetImageBytesPerPixel();
}

/**
* Sets the camera Region Of Interest.
* @param x - top-left corner coordinate
* @param y - top-left corner coordinate
* @param xSize - width
* @param ySize - height
*/
int LumeneraCamera::SetROI(unsigned x, unsigned y, unsigned xSize, unsigned ySize)
{
	int ret = DEVICE_ERR;
	//char cBuf[MM::MaxStrLength];
	if (IsCapturing())
	{
		ret = DEVICE_CAMERA_BUSY_ACQUIRING;
	}
	else
	{
		if (xSize == 0 || ySize == 0)
		{
			ret = DEVICE_CAN_NOT_SET_PROPERTY;
		}
		else
		{
			LUXAPPS_TRY
			{
				unsigned lastX;
				unsigned lastY;
				unsigned lastXSize;
				unsigned lastYSize;
				bool validLastROI = GetROI(lastX, lastY, lastXSize, lastYSize) == DEVICE_OK;

				unsigned binningFactor = GetBinning();

				if (setCameraRoi(x, y, xSize, ySize, binningFactor) == DEVICE_OK)
				{
					ret = refreshStream();
					if (ret == DEVICE_SNAP_IMAGE_FAILED)
					{
						// Try again with last valid ROI
						if (validLastROI)
						{
							return setCameraRoi(lastX, lastY, lastXSize, lastYSize, binningFactor);
						}
					}
				}
				else
				{
					ret = DEVICE_CAN_NOT_SET_PROPERTY;
				}
			}
				LUXAPPS_CATCH(...)
			{
				ret = DEVICE_ERR;
			}
		}
	}

	return ret;
}

/**
* Returns the actual dimensions of the current ROI.
*/
int LumeneraCamera::GetROI(unsigned& x, unsigned& y, unsigned& xSize, unsigned& ySize)
{
	int ret = DEVICE_ERR;

	LUXAPPS_TRY
	{
		RETURN_ON_ERROR(getCameraRoi(x, y, xSize, ySize));
		ret = DEVICE_OK;
	}
	LUXAPPS_CATCH(...)
	{
		ret = DEVICE_ERR;
	}
	return ret;
}

/**
* Resets the Region of Interest to full frame.
*/
int LumeneraCamera::ClearROI()
{
	int ret = DEVICE_ERR;
	if (IsCapturing())
	{
		ret = DEVICE_CAMERA_BUSY_ACQUIRING;
	}
	else
	{
		LUXAPPS_TRY
		{
			if (hasCamera() && camera()->setRoi(getSensorWidth(), getSensorHeight(), 0, 0))
			{
				RETURN_ON_ERROR(refreshStream());

				ret = DEVICE_OK;
			}
			else
			{
				ret = DEVICE_CAN_NOT_SET_PROPERTY;
			}
		}
		LUXAPPS_CATCH(...)
		{
			ret = DEVICE_ERR;
		}
	}
	return ret;
}

/**
* Returns the current exposure setting in milliseconds.
* Required by the MM::Camera API.
*/
double LumeneraCamera::GetExposure() const
{
	//NOTE: Micromanager's core seems to use the value returned by this function
	//		to control the refresh rate of the Live View window.
	char buf[MM::MaxStrLength];
	int ret = GetProperty(MM::g_Keyword_Exposure, buf);
	if (ret != DEVICE_OK)
		return 0.0;
	return std::stof(buf);
}

/**
* Sets exposure in milliseconds.
* Required by the MM::Camera API.
*/
void LumeneraCamera::SetExposure(double exp)
{
	SetProperty(MM::g_Keyword_Exposure, CDeviceUtils::ConvertToString(exp));
}

/**
* Returns the current binning factor.
*/
int LumeneraCamera::GetBinning() const
{
	int binSize = BIN_SIZE_1;

	char buf[MM::MaxStrLength];
	if (GetProperty(MM::g_Keyword_Binning, buf) == DEVICE_OK)
	{
		binSize = std::stoi(std::string(buf));
	}
	return binSize;
}

int LumeneraCamera::SetBinning(int binSize)
{
	std::string binning = std::to_string(binSize);
	RETURN_ON_ERROR(SetProperty(MM::g_Keyword_Binning, binning.c_str()));

	return DEVICE_OK;
}


int LumeneraCamera::StartSequenceAcquisition(double interval)
{
	return StartSequenceAcquisition(-1, interval, false);
}

int LumeneraCamera::StartSequenceAcquisition(long numImages, double, bool)
{

	if (IsCapturing())
	{
		return DEVICE_CAMERA_BUSY_ACQUIRING;
	}
	else
	{
		LUXAPPS_TRY
		{
			if (hasCamera())
			{
				CameraInterface::Property* timeout = nullptr;
				if (camera()->tryGetProperty(LuXAppsProperties::LUXPROP_TIMEOUT, timeout))
				{
					timeout->setValue("-1");
				}
			}

			RETURN_ON_ERROR(GetCoreCallback()->PrepareForAcq(this));
			sequenceThread_->Start(numImages);

			waitForCameraStream();

			if (hasCamera())
			{
				if (hardwareTriggerEnabled())
				{
					//NOTE: If we're using a hardware trigger then we should be in snapshot mode for captures.
					double exposure = GetExposure();

					//TODO: Might need to put this minimum exposure before the sequenceThread_->Start() call since it 
					//		takes a few frames for the exposure to take effect. It didn't seem like it was going to the minimum
					//TODO: Figure out if this even matters, need to cut a few corners to get this out the door.

					//CameraInterface::Property* e = nullptr;
					//if (camera()->tryGetProperty(LuXAppsProperties::LUXPROP_STILL_EXPOSURE, e))
					//{
					//	e->setValue(e->getMin());
					//}

					//NOTE: For some unknown reason, when starting the stream with hardware trigger
					//		enabled, we need to trigger once in order for subsequent real triggers
					//		to generate image captures
					LucamTriggerFastFrame(camera()->getHandle());

					//if (e)
					//{
					//	e->setValue(std::to_string(exposure));
					//}
				}
			}

			return DEVICE_OK;
		}
		LUXAPPS_CATCH(...)
		{
			return DEVICE_ERR;
		}
	}
}

bool LumeneraCamera::IsCapturing()
{
	return not sequenceThread_->IsStopped();
}

void LumeneraCamera::enableHardwareTriggering()
{
	setHardwareTriggeringEnabledState(true);
}

void LumeneraCamera::disableHardwareTriggering()
{
	setHardwareTriggeringEnabledState(false);
}

void LumeneraCamera::setHardwareTriggeringEnabledState(bool state)
{
	if (hasCamera())
	{
		CameraInterface::Property* hwTrigger = nullptr;
		if (camera()->tryGetProperty(LuXAppsProperties::LUXPROP_HARDWARE_TRIGGER, hwTrigger))
		{
			std::string v = state ? "Enabled" : "Disabled";

			hwTrigger->setValue(v);
		}
	}
}

bool LumeneraCamera::hardwareTriggerEnabled()
{
	bool enabled = false;

	if (hasCamera())
	{
		CameraInterface::Property* hwTrigger = nullptr;
		if (camera()->tryGetProperty(LuXAppsProperties::LUXPROP_HARDWARE_TRIGGER, hwTrigger))
		{
			enabled = hwTrigger->getValue() == "Enabled";
		}
	}

	return enabled;
}

int LumeneraCamera::StopSequenceAcquisition()
{
	if (IsCapturing())
	{
		sequenceThread_->Stop();

		if (hasCamera())
		{
			if (hardwareTriggerEnabled())
			{
				//TODO: Probably need to exit stream if we get a LucamCancelled error, not setting m_ExitStreamRequested = true
				//LucamCancelTakeFastFrame(camera()->getHandle());

				LucamTriggerFastFrame(camera()->getHandle());
			}
		}

		sequenceThread_->wait();
	}
	return DEVICE_OK;
}

int LumeneraCamera::PrepareSequenceAcqusition()
{
	// nothing to prepare
	return DEVICE_OK;
}


//////////////////////////////////
///// Internal functions /////////
//////////////////////////////////

bool LumeneraCamera::hasCamera()
{
	return camera_ != nullptr;
}

CameraInterface::LucamAdapter::LucamCamera* LumeneraCamera::camera() const
{
	return camera_;
}

int LumeneraCamera::readCameraPropertyValue(const std::string& name, std::string& value) const
{
	int ret = DEVICE_ERR;

	CameraInterface::Property* prop = nullptr;
	if (camera()->tryGetProperty(name, prop))
	{
		LUXAPPS_TRY
		{
			value = prop->getValue();
			ret = DEVICE_OK;
		}
		LUXAPPS_CATCH(...)
		{
			ret = DEVICE_NO_PROPERTY_DATA;
		}
	}
	else
	{
		ret = DEVICE_INVALID_PROPERTY;
	}

	return ret;
}


int LumeneraCamera::createLinkedGainProperty(CameraInterface::Camera* camera, const std::string& propertyName, const char* uiName)
{
	CameraInterface::Property* prop = nullptr;

	if (camera->tryGetProperty(propertyName, prop))
	{
		std::string value = prop->getValue();
		std::string minimum = prop->getMin();
		std::string maximum = prop->getMax();

		CPropertyAction* pAction = new CPropertyAction(this, &LumeneraCamera::OnGain);
		RETURN_ON_ERROR(CreateProperty(uiName, value.c_str(), MM::Float, false, pAction, false));
		RETURN_ON_ERROR(SetPropertyLimits(uiName, std::stod(minimum), std::stod(maximum)));
	}

	return DEVICE_OK;
}

int LumeneraCamera::writeCameraPropertyValue(const std::string& name, const std::string& value)
{
	int ret = DEVICE_ERR;

	CameraInterface::Property* prop = nullptr;
	if (camera()->tryGetProperty(name, prop))
	{
		LUXAPPS_TRY
		{
			prop->setValue(value);
			ret = DEVICE_OK;
		}
		LUXAPPS_CATCH(...)
		{
			ret = DEVICE_CAN_NOT_SET_PROPERTY;
		}
	}
	else
	{
		ret = DEVICE_INVALID_PROPERTY;
	}

	return ret;
}


bool LumeneraCamera::cameraRequiresClockThrottling(CameraInterface::Camera* cam)
{
	const unsigned long INFINITY3S = 0x711;
	const unsigned long INFINITY1_2CB = 0x4F7;

	return (LuXApps::isCameraModel(cam, INFINITY3S) && LuXApps::isColorCamera(cam)) ||
		(LuXApps::isCameraModel(cam, INFINITY1_2CB));
}

void LumeneraCamera::throttleClockSpeed(CameraInterface::Camera* cam, const std::string& clock)
{
	CameraInterface::Property* prop = nullptr;
	if (cam->tryGetProperty(clock, prop))
	{
		std::vector<std::string> supportedClocks = prop->getValueList();
		if (supportedClocks.size() > 1)
		{
			prop->setValue(supportedClocks[1]);
		}
	}
}

int LumeneraCamera::createWhiteBalanceTargetProperty(CameraInterface::Camera* camera, const char* uiName)
{
	if (LuXApps::isColorCamera(camera))
	{
		RETURN_ON_ERROR(CreateIntegerProperty(uiName, 255, false));
		RETURN_ON_ERROR(SetPropertyLimits(uiName, 0, 255));
	}

	return DEVICE_OK;
}

bool LumeneraCamera::initializeHardwareInterface()
{
	bool success = false;

	char buf[MM::MaxStrLength];

	int nRet = GetProperty(g_Camera_Hardware_Interface, buf);
	if (nRet == DEVICE_OK)
	{
		std::string hardwareInterface(buf);

		if (hardwareInterface == g_Keyword_USB)
		{
			LucamSelectExternInterface(LUCAM_EXTERN_INTERFACE_USB2);
			//LOG_DEBUG("Hardware interface set to USB.");
			success = true;
		}
		else if (hardwareInterface == g_Keyword_GIGE)
		{
			LucamSelectExternInterface(LUCAM_EXTERN_INTERFACE_GIGEVISION);
			//LOG_DEBUG("Hardware interface set to GigE Vision");
			success = true;
		}
		else
		{
			//LOG_DEBUG("No harware interface selected.");
		}
	}

	return success;
}


int LumeneraCamera::initializeImageBuffer()
{
	Imaging::IMAGE_BIT_DEPTH bitDepth = Imaging::IMAGE_BIT_DEPTH::SIXTEEN_BIT;
	Imaging::IMAGE_FORMAT format = Imaging::IMAGE_FORMAT::MONO;

	if (hasCamera())
	{
		RETURN_ON_ERROR(getBitDepthFromCamera(bitDepth));

		format = LuXApps::isColorCamera(camera()) ? Imaging::IMAGE_FORMAT::BGRA : Imaging::IMAGE_FORMAT::MONO;
	}

	unsigned x = 0;
	unsigned y = 0;
	unsigned width = 0;
	unsigned height = 0;

	getCameraRoi(x, y, width, height);

	updateImageBuffer(Imaging::createImage(width, height, std::vector<uint8_t>{}, bitDepth, true, format));

	return DEVICE_OK;
}

int LumeneraCamera::getBitDepthFromCamera(Imaging::IMAGE_BIT_DEPTH& bitDepth) const
{
	int ret = DEVICE_ERR;

	std::string value;
	ret = readCameraPropertyValue(LuXAppsProperties::LUXPROP_PIXEL_FORMAT, value);
	if (ret == DEVICE_OK)
	{
		bitDepth = from_string(value);
	}

	return ret;
}

int LumeneraCamera::getCameraRoi(unsigned& x, unsigned& y, unsigned& xSize, unsigned& ySize)
{
	int ret = DEVICE_ERR;

	unsigned xOffset = 0;
	unsigned yOffset = 0;
	unsigned width = 0;
	unsigned height = 0;

	LUCAM_FRAME_FORMAT f = camera_->getStreamSettings().format;

	unsigned xBinningFactor = f.binningX;
	unsigned yBinningFactor = f.binningY;

	xOffset = f.xOffset;
	yOffset = f.yOffset;
	width = f.width;
	height = f.height;

	if (isMirrorEnabled())
	{
		xOffset = getSensorWidth() - width - xOffset;
	}

	if (isFlippingEnabled())
	{
		yOffset = getSensorHeight() - height - yOffset;
	}

	xOffset /= xBinningFactor;
	yOffset /= yBinningFactor;
	width /= xBinningFactor;
	height /= yBinningFactor;

	ret = DEVICE_OK;

	x = xOffset;
	y = yOffset;
	xSize = width;
	ySize = height;

	return ret;
}

int LumeneraCamera::setCameraRoi(unsigned x, unsigned y, unsigned xSize, unsigned ySize, unsigned binningFactor)
{
	int ret = DEVICE_ERR;

	x *= binningFactor;
	y *= binningFactor;
	xSize *= binningFactor;
	ySize *= binningFactor;


	if (isMirrorEnabled())
	{
		x = getSensorWidth() - xSize - x;
	}

	if (isFlippingEnabled())
	{
		y = getSensorHeight() - ySize - y;
	}

	ret = camera()->setRoi(xSize, ySize, x, y) ? DEVICE_OK : DEVICE_INVALID_PROPERTY_VALUE;
	return ret;
}

unsigned LumeneraCamera::getSensorWidth()
{
	return std::stoul(camera()->getProperty(LuXAppsProperties::LUXPROP_MAXIMUM_WIDTH).getValue());
}

unsigned LumeneraCamera::getSensorHeight()
{
	return std::stoul(camera()->getProperty(LuXAppsProperties::LUXPROP_MAXIMUM_HEIGHT).getValue());
}

void LumeneraCamera::updateImageBuffer(std::unique_ptr<Imaging::Image>&& image)
{
	MMThreadGuard g(imageLock_);

	image_ = ImgBuffer(image->getWidth(), image->getHeight(), image->getBytesPerPixel());
	image_.SetPixels(image->getDataAddress());
	format_ = image->getFormat();
	bitDepth_ = image->getBitDepth();
	components_ = (format_ == Imaging::IMAGE_FORMAT::BGRA) ? 4 : 1;
}

bool LumeneraCamera::isFlippingEnabled()
{
	bool enabled = false;

	CameraInterface::Property* flipping = nullptr;
	if (camera()->tryGetProperty(LuXAppsProperties::LUXPROP_VERTICAL_FLIP, flipping))
	{
		enabled = (flipping->getValue() == "Enabled");
	}

	return enabled;
}

bool LumeneraCamera::isMirrorEnabled()
{
	bool enabled = false;

	CameraInterface::Property* flipping = nullptr;
	if (camera()->tryGetProperty(LuXAppsProperties::LUXPROP_HORIZONTAL_FLIP, flipping))
	{
		enabled = (flipping->getValue() == "Enabled");
	}

	return enabled;
}

std::vector<std::string> LumeneraCamera::getBitDepthOptions()
{
	static std::vector<std::string> bitDepthOptions;
	{
		to_string(Imaging::IMAGE_BIT_DEPTH::EIGHT_BIT);
		to_string(Imaging::IMAGE_BIT_DEPTH::SIXTEEN_BIT);
	};

	return bitDepthOptions;
}

Imaging::IMAGE_BIT_DEPTH LumeneraCamera::getBitDepthFromPixelType(const std::string& pixelType)
{
	Imaging::IMAGE_BIT_DEPTH bitDepth = Imaging::IMAGE_BIT_DEPTH::EIGHT_BIT;

	if (pixelType == g_PixelType_8bit_MONO || pixelType == g_PixelType_32bit_COLOR)
	{
		bitDepth = Imaging::IMAGE_BIT_DEPTH::EIGHT_BIT;
	}
	else if (pixelType == g_PixelType_16bit_MONO || pixelType == g_PixelType_64bit_COLOR)
	{
		bitDepth = Imaging::IMAGE_BIT_DEPTH::SIXTEEN_BIT;
	}

	return bitDepth;
}

Imaging::IMAGE_FORMAT LumeneraCamera::getImageFormatFromPixelType(const std::string& pixelType)
{
	Imaging::IMAGE_FORMAT format = Imaging::IMAGE_FORMAT::MONO;

	if (pixelType == g_PixelType_8bit_MONO || pixelType == g_PixelType_16bit_MONO)
	{
		format = Imaging::IMAGE_FORMAT::MONO;
	}
	else if (pixelType == g_PixelType_32bit_COLOR || pixelType == g_PixelType_64bit_COLOR)
	{
		format = Imaging::IMAGE_FORMAT::BGRA;
	}

	return format;
}

std::string getPixelTypeFromFormatAndBitDepth(const Imaging::IMAGE_FORMAT& format, const Imaging::IMAGE_BIT_DEPTH bitDepth)
{
	std::string pixelType = g_PixelType_8bit_MONO;

	if (bitDepth == Imaging::IMAGE_BIT_DEPTH::EIGHT_BIT && format == Imaging::IMAGE_FORMAT::MONO)
	{
		pixelType = g_PixelType_8bit_MONO;
	}
	else if (bitDepth == Imaging::IMAGE_BIT_DEPTH::SIXTEEN_BIT && format == Imaging::IMAGE_FORMAT::MONO)
	{
		pixelType = g_PixelType_16bit_MONO;
	}
	else if (bitDepth == Imaging::IMAGE_BIT_DEPTH::EIGHT_BIT && format == Imaging::IMAGE_FORMAT::BGRA)
	{
		pixelType = g_PixelType_32bit_COLOR;
	}
	else if (bitDepth == Imaging::IMAGE_BIT_DEPTH::SIXTEEN_BIT && format == Imaging::IMAGE_FORMAT::BGRA)
	{
		pixelType = g_PixelType_64bit_COLOR;
	}

	return pixelType;
}

std::string LumeneraCamera::getVideoPropertyName(const std::string& name)
{
	static const std::map<std::string, std::string> VIDEO_PROPERTY_MAP
	{
		std::make_pair(std::string(MM::g_Keyword_Gain),				LuXAppsProperties::LUXPROP_VIDEO_GAIN),
		std::make_pair(std::string(g_Camera_Gain_Red),				LuXAppsProperties::LUXPROP_VIDEO_GAIN_RED),
		std::make_pair(std::string(g_Camera_Gain_Green1),			LuXAppsProperties::LUXPROP_VIDEO_GAIN_GREEN_1),
		std::make_pair(std::string(g_Camera_Gain_Green2),			LuXAppsProperties::LUXPROP_VIDEO_GAIN_GREEN_2),
		std::make_pair(std::string(g_Camera_Gain_Blue),				LuXAppsProperties::LUXPROP_VIDEO_GAIN_BLUE),
		std::make_pair(std::string(g_Camera_Tap_Configuration),		LuXAppsProperties::LUXPROP_VIDEO_TAP_CONFIGURATION),
	};

	return VIDEO_PROPERTY_MAP.at(name);
}

std::string LumeneraCamera::getStillPropertyName(const std::string& name)
{
	static const std::map<std::string, std::string> STILL_PROPERTY_MAP
	{
		std::make_pair(std::string(MM::g_Keyword_Gain),				LuXAppsProperties::LUXPROP_STILL_GAIN),
		std::make_pair(std::string(g_Camera_Gain_Red),				LuXAppsProperties::LUXPROP_STILL_GAIN_RED),
		std::make_pair(std::string(g_Camera_Gain_Green1),			LuXAppsProperties::LUXPROP_STILL_GAIN_GREEN_1),
		std::make_pair(std::string(g_Camera_Gain_Green2),			LuXAppsProperties::LUXPROP_STILL_GAIN_GREEN_2),
		std::make_pair(std::string(g_Camera_Gain_Blue),				LuXAppsProperties::LUXPROP_STILL_GAIN_BLUE),
		std::make_pair(std::string(g_Camera_Tap_Configuration),		LuXAppsProperties::LUXPROP_STILL_TAP_CONFIGURATION),
	};

	return STILL_PROPERTY_MAP.at(name);
}

int LumeneraCamera::refreshStream()
{
	bool streaming = IsCapturing();

	if (streaming)
	{
		StopSequenceAcquisition();
	}

	RETURN_ON_ERROR(resizeImageBuffer());

	if (streaming)
	{
		StartSequenceAcquisition(0);
	}

	return DEVICE_OK;
}

int LumeneraCamera::resizeImageBuffer()
{
	RETURN_ON_ERROR(captureImage());

	return DEVICE_OK;
}

int LumeneraCamera::setBitDepth(const Imaging::IMAGE_BIT_DEPTH& bitDepth)
{
	std::string bitDepthValue = to_string(bitDepth);
	return writeCameraPropertyValue(LuXAppsProperties::LUXPROP_PIXEL_FORMAT, bitDepthValue);
}

bool LumeneraCamera::cameraSupportsProperty(const std::string& name)
{
	bool supported = false;
	supported = camera()->isPropertySupported(name);

	return supported;
}

bool LumeneraCamera::requiresStillStream(double exposure, const std::string& hardwareTrigger)
{
	bool requiresStillStream = false;

	if (hasSwitchingExposure_)
	{
		if (exposure > switchingExposure_)
		{
			requiresStillStream = true;
		}
	}
	if (this->HasProperty(LuXAppsProperties::LUXPROP_HARDWARE_TRIGGER)) {
		if (strcmp(hardwareTrigger.c_str(), "Enabled") == 0) {
			requiresStillStream = true;
		}
	}

	return requiresStillStream;
}

bool LumeneraCamera::requiresVideoStream(double exposure, const std::string& hardwareTrigger)
{
	return !requiresStillStream(exposure,  hardwareTrigger);
}

bool LumeneraCamera::isVideoStreamingMode()
{
	bool isVideoMode = true;

	CameraInterface::Property* streamingModeProperty = nullptr;
	if (camera()->tryGetProperty(LuXAppsProperties::LUXPROP_STREAMING_MODE, streamingModeProperty))
	{
		isVideoMode = (streamingModeProperty->getValue() == "Video");
	}

	return isVideoMode;
}

bool LumeneraCamera::isStillStreamingMode()
{
	return !isVideoStreamingMode();
}

int LumeneraCamera::applyStreamMode(double exposure, const std::string& hardwareTrigger) {
	//Switch between video and still mode as appropriate

	bool requiresModeSwitch = false;
	std::string streamingMode;

	if (requiresVideoStream(exposure, hardwareTrigger) && isStillStreamingMode())
	{
		requiresModeSwitch = true;
		streamingMode = "Video";
	}
	else if (requiresStillStream(exposure, hardwareTrigger) && isVideoStreamingMode())
	{
		requiresModeSwitch = true;
		streamingMode = "Still";
	}

	//Always restart stream when hardware triggering enabled
	if (strcmp(hardwareTrigger.c_str(), "Enabled")) {
		requiresModeSwitch = true;
	}

	if (requiresModeSwitch)
	{
		bool streaming = IsCapturing();

		if (streaming)
		{
			StopSequenceAcquisition();
		}

		//TODO: Add error handling here in case of failure
		writeCameraPropertyValue(LuXAppsProperties::LUXPROP_STREAMING_MODE, streamingMode);

		if (streaming)
		{
			StartSequenceAcquisition(0);
		}
	}

	std::string propertyName = isVideoStreamingMode() ? LuXAppsProperties::LUXPROP_VIDEO_EXPOSURE : LuXAppsProperties::LUXPROP_STILL_EXPOSURE;
	return writeCameraPropertyValue(propertyName, std::to_string(exposure));
}

int LumeneraCamera::setCoolingState(const std::string& state)
{
	int ret = DEVICE_ERR;

	LUXAPPS_TRY
	{
		CameraInterface::Property * fanProperty = nullptr;
		CameraInterface::Property* coolingProperty = nullptr;
		if (camera()->tryGetProperty(LuXAppsProperties::LUXPROP_FAN, fanProperty) && camera()->tryGetProperty(LuXAppsProperties::LUXPROP_COOLING, coolingProperty))
		{
			fanProperty->setValue(state);
			coolingProperty->setValue(state);

			ret = DEVICE_OK;
		}
		else
		{
			ret = DEVICE_INVALID_PROPERTY;
		}
	}
		LUXAPPS_CATCH(...)
	{
		ret = DEVICE_CAN_NOT_SET_PROPERTY;
	}

	return ret;
}

int LumeneraCamera::getBinValueFromSampling(const std::string& sampling)
{
	if (sampling == NONE_1X1_STRING) { return BIN_SIZE_1; }
	else if (sampling == BINNING_2X2_STRING) { return BIN_SIZE_2; }
	else if (sampling == BINNING_4X4_STRING) { return BIN_SIZE_4; }
	else if (sampling == BINNING_8X8_STRING) { return BIN_SIZE_8; }
	else { return BIN_SIZE_1; }
}

std::vector<std::string> LumeneraCamera::getBinningOptions()
{
	static std::vector<std::string> binningOptions
	{
		NONE_1X1_STRING,
		BINNING_2X2_STRING,
		BINNING_4X4_STRING,
		BINNING_8X8_STRING,
	};

	return binningOptions;
}

std::string LumeneraCamera::getSamplingFromBinValue(int binValue)
{
	switch (binValue)
	{
	case BIN_SIZE_1:
		return NONE_1X1_STRING;
	case BIN_SIZE_2:
		return BINNING_2X2_STRING;
	case BIN_SIZE_4:
		return BINNING_4X4_STRING;
	case BIN_SIZE_8:
		return BINNING_8X8_STRING;

	default:
		return NONE_1X1_STRING;
	}
}

int LumeneraCamera::captureSequenceImage()
{
	RETURN_ON_ERROR(captureImage());
	RETURN_ON_ERROR(InsertImage());

	return DEVICE_OK;
}

void LumeneraCamera::sequenceEnded() noexcept
{
	LUXAPPS_TRY
	{
		GetCoreCallback() ? GetCoreCallback()->AcqFinished(this, 0) : DEVICE_OK;
	}
	LUXAPPS_CATCH(CMMError& e)
	{
		GetCoreCallback()->LogMessage(this, e.what(), false);
	}
}

void LumeneraCamera::waitForCameraStream(bool streaming)
{
	int runs = 0;
	do
	{
		if (camera()->isStreaming() == streaming)
			break;

		std::this_thread::sleep_for(std::chrono::milliseconds(50));
		runs++;

	} while (runs < 5);
}

int LumeneraCamera::OnLinkedVideoAndStillProperty(MM::PropertyBase* pProp, MM::ActionType eAct, bool requiresStreamRefresh)
{
	int ret = DEVICE_ERR;

	std::string propertyName = pProp->GetName();
	std::string videoPropertyName = getVideoPropertyName(propertyName);
	std::string stillPropertyName = getStillPropertyName(propertyName);

	switch (eAct)
	{
	case MM::BeforeGet:
	{

		if (hasCamera())
		{
			std::string value;
			ret = readCameraPropertyValue(videoPropertyName, value);
			if (ret == DEVICE_OK)
			{
				if (pProp->GetType() == MM::PropertyType::Float) {
					//Enforce property limits
					double dVal = atof(value.c_str());
					dVal = max(pProp->GetLowerLimit(), min(dVal, pProp->GetUpperLimit()));
					value = std::to_string(dVal);
				}
				if (pProp->Set(value.c_str()))
				{
					ret = DEVICE_OK;
				}
				else
				{
					ret = DEVICE_ERR;
				}
			}
		}
		else
		{
			ret = DEVICE_NOT_CONNECTED;
		}
	}
	break;

	case MM::AfterSet:
	{

		if (hasCamera())
		{
			std::string value;
			pProp->Get(value);

			bool streaming = IsCapturing();

			if (streaming && requiresStreamRefresh)
			{
				RETURN_ON_ERROR(StopSequenceAcquisition());
			}

			ret = writeCameraPropertyValue(videoPropertyName, value);

			if (ret == DEVICE_OK && cameraSupportsProperty(stillPropertyName))
			{
				ret = writeCameraPropertyValue(stillPropertyName, value);
			}

			if (ret == DEVICE_OK)
			{
				if (requiresStreamRefresh)
				{
					RETURN_ON_ERROR(resizeImageBuffer());
				}
			}

			if (streaming && requiresStreamRefresh)
			{
				RETURN_ON_ERROR(StartSequenceAcquisition(0));
			}
		}
		else
		{
			ret = DEVICE_NOT_CONNECTED;
		}
	}
	break;
	}

	return ret;
}



//////
// //Action handlers
///////////////////////////////////////////////////////////////////////////////

//TODO: Figure out if this ever gets called more than once at initialization
int LumeneraCamera::OnCameraIndex(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	switch (eAct)
	{
	case MM::BeforeGet:
	{
		pProp->Set(static_cast<long>(cameraIndex_));
	}
	break;

	case MM::AfterSet:
	{
		if (initialized_)
		{
			// revert
			pProp->Set(static_cast<long>(cameraIndex_));
			return DEVICE_ERR;
		}

		long lValue;
		if (pProp->Get(lValue))
		{
			cameraIndex_ = static_cast<int>(lValue);
		}
	}
	break;
	}

	return DEVICE_OK;
}


int LumeneraCamera::OnBitDepth(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	int ret = DEVICE_ERR;

	switch (eAct)
	{
	case MM::BeforeGet:
	{
		if (hasCamera())
		{
			long bitDepth = GetBitDepth();
			ret = DEVICE_OK;

			if (ret == DEVICE_OK)
			{
				pProp->Set(static_cast<long>(bitDepth));
			}
		}
		else
		{
			ret = DEVICE_NOT_CONNECTED;
		}
	}
	break;

	case MM::AfterSet:
	{
		//NOTE: Bit depth is read only and is controlled by setting the Pixel Type
	}
	break;
	}

	return ret;
}


int LumeneraCamera::OnPixelType(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	int ret = DEVICE_ERR;

	switch (eAct)
	{
	case MM::BeforeGet:
	{
		std::string pixelType = getPixelTypeFromFormatAndBitDepth(format_, bitDepth_);
		if (pProp->Set(pixelType.c_str()))
		{
			ret = DEVICE_OK;
		}
		else {
			ret = DEVICE_ERR;
		}
	}
	break;

	case MM::AfterSet:
	{
		if (IsCapturing())
		{
			ret = DEVICE_CAMERA_BUSY_ACQUIRING;
		}
		else
		{
			std::string pixelType;
			pProp->Get(pixelType);

			Imaging::IMAGE_BIT_DEPTH bitDepth = getBitDepthFromPixelType(pixelType);

			ret = setBitDepth(bitDepth);

			if (ret == DEVICE_OK)
			{
				RETURN_ON_ERROR(refreshStream());
			}
		}
	}
	break;

	default:
		ret = DEVICE_OK;
		break;
	}

	return ret;
}


int LumeneraCamera::OnExposure(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	int ret = DEVICE_ERR;

	switch (eAct)
	{
	case MM::BeforeGet:
	{
		std::string value;
		std::string propertyName = isVideoStreamingMode() ? LuXAppsProperties::LUXPROP_VIDEO_EXPOSURE : LuXAppsProperties::LUXPROP_STILL_EXPOSURE;
		ret = readCameraPropertyValue(propertyName, value);
		if (ret == DEVICE_OK)
		{
			if (pProp->Set(value.c_str()))
			{
				ret = DEVICE_OK;
			}
			else
			{
				ret = DEVICE_ERR;
			}
		}
	}
	break;

	case MM::AfterSet:
	{
		double value;
		pProp->Get(value);
		char hardwareTrigger [MM::MaxStrLength];
		this->GetProperty(LuXAppsProperties::LUXPROP_HARDWARE_TRIGGER, hardwareTrigger);
		return applyStreamMode(value, hardwareTrigger);
	}
	break;
	}

	return ret;
}


int LumeneraCamera::OnBinning(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	int ret = DEVICE_ERR;

	switch (eAct)
	{
	case MM::BeforeGet:
	{

		std::string value;
		ret = readCameraPropertyValue(LuXAppsProperties::LUXPROP_SAMPLING_DESCRIPTION, value);
		if (ret == DEVICE_OK)
		{
			int binSize = getBinValueFromSampling(value);
			if (pProp->Set(static_cast<long>(binSize)))
			{
				ret = DEVICE_OK;
			}
			else
			{
				ret = DEVICE_ERR;
			}
		}
	}
	break;

	case MM::AfterSet:
	{
		if (IsCapturing())
		{
			ret = DEVICE_CAMERA_BUSY_ACQUIRING;
		}
		else
		{

			std::string value;
			pProp->Get(value);

			std::string currentSampling;
			RETURN_ON_ERROR(readCameraPropertyValue(LuXAppsProperties::LUXPROP_SAMPLING_DESCRIPTION, currentSampling));
			int currentBinning = getBinValueFromSampling(currentSampling);

			int newBinning = std::stoi(value);
			std::string sampling = getSamplingFromBinValue(newBinning);
			ret = writeCameraPropertyValue(LuXAppsProperties::LUXPROP_SAMPLING_DESCRIPTION, sampling);

			if (ret == DEVICE_OK)
			{
				unsigned x, y, width, height;

				RETURN_ON_ERROR(getCameraRoi(x, y, width, height));

				double adjustmentFactor = static_cast<double>(newBinning) / currentBinning;

				x = (unsigned)std::floor(static_cast<double>(x) / adjustmentFactor);
				y = (unsigned)std::floor(static_cast<double>(y) / adjustmentFactor);
				width = (unsigned)std::floor(static_cast<double>(width) / adjustmentFactor);
				height = (unsigned)std::floor(static_cast<double>(height) / adjustmentFactor);

				RETURN_ON_ERROR(setCameraRoi(x, y, width, height, newBinning));
				RETURN_ON_ERROR(refreshStream());
			}
		}
	}
	break;
	}

	return ret;
}

int LumeneraCamera::OnTapConfiguration(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	RETURN_ON_ERROR(OnLinkedVideoAndStillProperty(pProp, eAct, true));

	return DEVICE_OK;
}

int LumeneraCamera::OnDemosaicingMethod(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	RETURN_ON_ERROR(OnSingleProperty(pProp, eAct));

	return DEVICE_OK;
}

int LumeneraCamera::OnHardwareTrigger(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	int ret = DEVICE_OK;

	std::string propertyName = pProp->GetName();
	std::string cameraPropertyName = getPropertyName(propertyName);

	switch (eAct)
	{
	case MM::BeforeGet:
	{
		if (hasCamera())
		{
			std::string value;
			ret = readCameraPropertyValue(cameraPropertyName, value);
			if (ret == DEVICE_OK)
			{
				if (pProp->Set(value.c_str()))
				{
					ret = DEVICE_OK;
				}
				else
				{
					ret = DEVICE_ERR;
				}
			}
		}
		else
		{
			ret = DEVICE_NOT_CONNECTED;
		}
	}
	break;

	case MM::AfterSet:
	{
		if (hasCamera())
		{
			std::string value;
			pProp->Get(value);

			writeCameraPropertyValue(LuXAppsProperties::LUXPROP_HARDWARE_TRIGGER, value);

			double exposure = GetExposure();

			// switch to correct mode based on hardware triggering and exposure
			applyStreamMode(exposure, value);
		}
	}
	break;
	}
	return ret;
}

int LumeneraCamera::OnTimeout(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	int ret = DEVICE_ERR;

	std::string propertyName = pProp->GetName();
	std::string cameraPropertyName = getPropertyName(propertyName);

	switch (eAct)
	{
		case MM::BeforeGet:
		{
			if (hasCamera())
			{
				std::string value;
				ret = readCameraPropertyValue(cameraPropertyName, value);
				if (ret == DEVICE_OK)
				{
					double v = std::stod(value);
					v /= 1000.0;

					if (pProp->Set(LuXApps::Format::print_double(v, 3).c_str()))
					{
						ret = DEVICE_OK;
					}
					else
					{
						ret = DEVICE_ERR;
					}
				}
			}
			else
			{
				ret = DEVICE_NOT_CONNECTED;
			}
		}
		break;

		case MM::AfterSet:
		{
			if (hasCamera())
			{
				std::string value;
				pProp->Get(value);

				double v = std::stod(value);
				v *= 1000;

				ret = writeCameraPropertyValue(cameraPropertyName, std::to_string(v));
			}
		}
		break;
	}

	return ret;
}

int LumeneraCamera::OnTriggerMode(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	RETURN_ON_ERROR(OnSingleProperty(pProp, eAct));

	return DEVICE_OK;
}

int LumeneraCamera::OnTriggerPolarity(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	RETURN_ON_ERROR(OnSingleProperty(pProp, eAct));

	return DEVICE_OK;
}

int LumeneraCamera::OnTriggerPin(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	RETURN_ON_ERROR(OnSingleProperty(pProp, eAct));

	return DEVICE_OK;
}

int LumeneraCamera::OnLightSource(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	RETURN_ON_ERROR(OnSingleProperty(pProp, eAct));

	return DEVICE_OK;
}

int LumeneraCamera::OnFlip(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	RETURN_ON_ERROR(OnSingleProperty(pProp, eAct));

	return DEVICE_OK;
}

int LumeneraCamera::OnMirror(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	RETURN_ON_ERROR(OnSingleProperty(pProp, eAct));

	return DEVICE_OK;
}

int LumeneraCamera::OnHue(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	RETURN_ON_ERROR(OnSingleProperty(pProp, eAct));

	return DEVICE_OK;
}

int LumeneraCamera::OnSaturation(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	RETURN_ON_ERROR(OnSingleProperty(pProp, eAct));

	return DEVICE_OK;
}

int LumeneraCamera::OnGamma(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	RETURN_ON_ERROR(OnSingleProperty(pProp, eAct));

	return DEVICE_OK;
}

int LumeneraCamera::OnContrast(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	RETURN_ON_ERROR(OnSingleProperty(pProp, eAct));

	return DEVICE_OK;
}

int LumeneraCamera::OnBrightness(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	RETURN_ON_ERROR(OnSingleProperty(pProp, eAct));

	return DEVICE_OK;
}

int LumeneraCamera::OnHighConversionGain(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	RETURN_ON_ERROR(OnSingleProperty(pProp, eAct));

	return DEVICE_OK;
}

int LumeneraCamera::OnCooling(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	int ret = DEVICE_ERR;

	switch (eAct)
	{
	case MM::BeforeGet:
	{
		if (hasCamera())
		{
			std::string value;
			ret = readCameraPropertyValue(LuXAppsProperties::LUXPROP_FAN, value);
			if (ret == DEVICE_OK)
			{
				if (pProp->Set(value.c_str()))
				{
					ret = DEVICE_OK;
				}
				else
				{
					ret = DEVICE_ERR;
				}
			}
		}
		else
		{
			ret = DEVICE_NOT_CONNECTED;
		}
	}
	break;

	case MM::AfterSet:
	{

		if (hasCamera())
		{
			std::string value;
			pProp->Get(value);

			ret = setCoolingState(value);

		}
		else
		{
			ret = DEVICE_NOT_CONNECTED;
		}
	}
	break;
	}

	return ret;
}

int LumeneraCamera::OnTemperature(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	char buf[MM::MaxStrLength];

	//NOTE: Camera temperature value is only valid if camera cooling is enabled.
	if (HasProperty(g_Camera_Cooling) && GetProperty(g_Camera_Cooling, buf) == DEVICE_OK && (std::string(buf) == to_string(LuXApps::PROPERTY_STATE::ENABLED)))
	{
		RETURN_ON_ERROR(OnReadOnlyProperty(pProp, eAct));
	}
	else
	{
		if (eAct == MM::BeforeGet)
		{
			pProp->Set("N/A");
		}
	}

	return DEVICE_OK;
}

int LumeneraCamera::OnIris(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	RETURN_ON_ERROR(OnSingleProperty(pProp, eAct));

	return DEVICE_OK;
}

int LumeneraCamera::OnFocus(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	RETURN_ON_ERROR(OnSingleProperty(pProp, eAct));

	return DEVICE_OK;
}

int LumeneraCamera::OnAbsoluteFocus(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	RETURN_ON_ERROR(OnSingleProperty(pProp, eAct));

	return DEVICE_OK;
}

int LumeneraCamera::OnReadOnlyProperty(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	int ret = DEVICE_ERR;

	std::string propertyName = pProp->GetName();
	std::string cameraPropertyName = getPropertyName(propertyName);

	switch (eAct)
	{
	case MM::BeforeGet:
	{
		if (hasCamera())
		{
			std::string value;
			ret = readCameraPropertyValue(cameraPropertyName, value);
			if (ret == DEVICE_OK)
			{
				if (pProp->Set(value.c_str()))
				{
					ret = DEVICE_OK;
				}
				else
				{
					ret = DEVICE_ERR;
				}
			}
		}
		else
		{
			ret = DEVICE_NOT_CONNECTED;
		}
	}
	break;

	case MM::AfterSet:
	{
		ret = DEVICE_CAN_NOT_SET_PROPERTY;
	}
	break;
	}

	return ret;
}

std::string LumeneraCamera::getPropertyName(const std::string& name)
{
	static const std::map<std::string, std::string> PROPERTY_MAP
	{
   std::make_pair(std::string(g_Camera_Flip), LuXAppsProperties::LUXPROP_VERTICAL_FLIP),
	   std::make_pair(std::string(g_Camera_Mirror), LuXAppsProperties::LUXPROP_HORIZONTAL_FLIP),
	   std::make_pair(std::string(g_Camera_Demosaic_Method), LuXAppsProperties::LUXPROP_DEMOSAIC_METHOD),
	   std::make_pair(std::string(g_Camera_Light_Source), LuXAppsProperties::LUXPROP_CORRECTION_MATRIX),
	   std::make_pair(std::string(g_Camera_Hue), LuXAppsProperties::LUXPROP_HUE),
	   std::make_pair(std::string(g_Camera_Saturation), LuXAppsProperties::LUXPROP_SATURATION),
	   std::make_pair(std::string(g_Camera_Gamma), LuXAppsProperties::LUXPROP_GAMMA),
	   std::make_pair(std::string(g_Camera_Contrast), LuXAppsProperties::LUXPROP_CONTRAST),
	   std::make_pair(std::string(g_Camera_Brigthness), LuXAppsProperties::LUXPROP_BRIGHTNESS),
	   std::make_pair(std::string(g_Camera_High_Conversion_Gain), LuXAppsProperties::LUXPROP_HIGH_CONVERSION_GAIN),
	   std::make_pair(std::string(g_Camera_Temperature), LuXAppsProperties::LUXPROP_CAMERA_TEMPERATURE),
	   std::make_pair(std::string(g_Camera_Iris), LuXAppsProperties::LUXPROP_IRIS),
	   std::make_pair(std::string(g_Camera_Focus), LuXAppsProperties::LUXPROP_FOCUS),
	   std::make_pair(std::string(g_Camera_Absolute_Focus), LuXAppsProperties::LUXPROP_ABSOLUTE_FOCUS),
	   std::make_pair(std::string(g_Camera_Timeout), LuXAppsProperties::LUXPROP_TIMEOUT),
	   std::make_pair(std::string(LuXAppsProperties::LUXPROP_HARDWARE_TRIGGER), LuXAppsProperties::LUXPROP_HARDWARE_TRIGGER),
	   std::make_pair(std::string(LuXAppsProperties::LUXPROP_TRIGGER_MODE), LuXAppsProperties::LUXPROP_TRIGGER_MODE),
	   std::make_pair(std::string(LuXAppsProperties::LUXPROP_TRIGGER_PIN), LuXAppsProperties::LUXPROP_TRIGGER_PIN),
	   std::make_pair(std::string(LuXAppsProperties::LUXPROP_TRIGGER_POLARITY), LuXAppsProperties::LUXPROP_TRIGGER_POLARITY),

	};

	return PROPERTY_MAP.at(name);
}

int LumeneraCamera::OnSingleProperty(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	int ret = DEVICE_ERR;

	std::string propertyName = pProp->GetName();
	std::string cameraPropertyName = getPropertyName(propertyName);

	switch (eAct)
	{
		case MM::BeforeGet:
		{
			if (hasCamera())
			{
				std::string value;
				ret = readCameraPropertyValue(cameraPropertyName, value);
				if (ret == DEVICE_OK)
				{
					if (pProp->Set(value.c_str()))
					{
						ret = DEVICE_OK;
					}
					else
					{
						ret = DEVICE_ERR;
					}
				}
			}
			else
			{
				ret = DEVICE_NOT_CONNECTED;
			}
		}
		break;

		case MM::AfterSet:
		{
			if (hasCamera())
			{
				std::string value;
				pProp->Get(value);

				ret = writeCameraPropertyValue(cameraPropertyName, value);
			}
		}
		break;
	}

	return ret;
}

int LumeneraCamera::OnWhiteBalance(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	int ret = DEVICE_ERR;

	switch (eAct)
	{
	case MM::BeforeGet:
	{
		pProp->Set(FALSE_STRING);
		ret = DEVICE_OK;
	}
	break;

	case MM::AfterSet:
	{
		if (hasCamera())
		{
			std::string value;
			pProp->Get(value);

			if (value == TRUE_STRING)
			{
				LUXAPPS_TRY
				{
					ret = DEVICE_OK;

					bool capturing = IsCapturing();
					
					if (capturing) 
					{
						ret = StopSequenceAcquisition(); 
					}

					bool trigger = hardwareTriggerEnabled();

					if (trigger)
					{
						disableHardwareTriggering();
					}

					LUXAPPS_TRY
					{ 
						if (ret == DEVICE_OK)
						{
							camera()->startStream();
							//NOTE: Need to give time for the video thread to enable the
							//		camera stream in order for white balance to work
							waitForCameraStream();

							double red = 255;
							double green = 255;
							double blue = 255;

							GetProperty(g_Keyword_White_Balance_Target_Red, red);
							GetProperty(g_Keyword_White_Balance_Target_Green, green);
							GetProperty(g_Keyword_White_Balance_Target_Blue, blue);
						
							LuXApps::whiteBalance(camera(), LuXApps::Rect{}, red, green, blue);

							camera()->stopStream();
						}
					}
					LUXAPPS_CATCH(...)
					{
						ret = DEVICE_ERR;//WHITE_BALANCE_OPERATION_FAILED;
					}

					if (trigger)
					{
						enableHardwareTriggering();
					}

					if (capturing) 
					{ 
						ret = StartSequenceAcquisition(0); 
					}
				}
				LUXAPPS_CATCH(LuXAppsExceptions::LuXAppsException & e)
				{
					GetCoreCallback()->LogMessage(this, e.what(), false);
					ret = DEVICE_ERR;
				}
				catch (...)
				{
					ret = DEVICE_ERR;
				}
			}
			else
			{
				ret = DEVICE_OK;
			}
		}
		else
		{
			ret = DEVICE_NOT_CONNECTED;
		}
	}
	break;
	}

	return ret;
}

int LumeneraCamera::OnGain(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	RETURN_ON_ERROR(OnLinkedVideoAndStillProperty(pProp, eAct));

	return DEVICE_OK;
}