///////////////////////////////////////////////////////////////////////////////
// FILE:          LumeneraAce.h
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Adapter for Lumenera  Cameras
//
// Copyright 2022 Photomics, Inc.
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
#include "Utilities.hxx"
//#include "Tools/Debugger.h"
#include "Properties/PropertyValueEnumerations.h"
#include "Images/ImageConversion.h" 

#include "VideoSequenceThread.h"

using namespace std;

const char* g_LumeneraCameraDeviceName = "LumeneraCamera";

static const char* g_Keyword_USB = "USB";
static const char* g_Keyword_GIGE = "GigE Vision";
std::vector<std::string> HARDWARE_INTERFACES
{
	g_Keyword_USB,
	g_Keyword_GIGE,
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
	colorCamera_(true),
	cameraIndex_(1),
	initialized_(false),
	hasSwitchingExposure_(false),
	switchingExposure_(0)
{
	CameraInterface::Camera::initializeWorkspace();

	// Call the base class method to set-up default error codes/messages
	InitializeDefaultErrorMessages();
	//SetErrorText(DEVICE_STREAM_START_FAILURE, g_Msg_STREAM_START_FAILED);
	//SetErrorText(DEVICE_CUSTOM_ERROR, g_Msg_ERR);



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
		size_t cameraListIndex = cameraIndex_ - 1;

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
				LUXAPPS_CATCH(const LuXAppsExceptions::LuXAppsException & e)
			{
				//LOG_DEBUG(e.what());
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

	initialized_ = true;

	return DEVICE_OK;
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

		CPropertyAction* pAction = new CPropertyAction(this, &LumeneraCamera::OnExposure);
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




	//RETURN_ON_ERROR(createDeviceNameProperty());
	//RETURN_ON_ERROR(createDeviceDescriptionProperty());


	//RETURN_ON_ERROR(createModelNameProperty(camera));
	//RETURN_ON_ERROR(createCameraIdProperty(camera));
	//RETURN_ON_ERROR(createCameraApiProperty(camera));
	//RETURN_ON_ERROR(createCameraDriverProperty(camera));
	//RETURN_ON_ERROR(createCameraFirmwareProperty(camera));
	//RETURN_ON_ERROR(createCameraFpgaProperty(camera));
	//RETURN_ON_ERROR(createSensorWidthProperty(camera));
	//RETURN_ON_ERROR(createSensorHeightProperty(camera));
	//RETURN_ON_ERROR(createBitDepthProperty(camera));
	//RETURN_ON_ERROR(createTapConfigurationProperty(camera));
	//RETURN_ON_ERROR(createGainProperty(camera));
	//RETURN_ON_ERROR(createRedGainProperty(camera));
	//RETURN_ON_ERROR(createGreen1GainProperty(camera));
	//RETURN_ON_ERROR(createGreen2GainProperty(camera));
	//RETURN_ON_ERROR(createBlueGainProperty(camera));
	//RETURN_ON_ERROR(createDemosaicMethodProperty(camera));
	//RETURN_ON_ERROR(createLightSourceProperty(camera));
	//RETURN_ON_ERROR(createFlippingProperty(camera));
	//RETURN_ON_ERROR(createMirrorProperty(camera));
	//RETURN_ON_ERROR(createHueProperty(camera));
	//RETURN_ON_ERROR(createSaturationProperty(camera));
	//RETURN_ON_ERROR(createGammaProperty(camera));
	//RETURN_ON_ERROR(createContrastProperty(camera));
	//RETURN_ON_ERROR(createBrightnessProperty(camera));
	//RETURN_ON_ERROR(createHighConversionGainProperty(camera));
	//RETURN_ON_ERROR(createCoolingProperty(camera));
	//RETURN_ON_ERROR(createTemperatureProperty(camera));
	//RETURN_ON_ERROR(createIrisProperty(camera));
	//RETURN_ON_ERROR(createFocusProperty(camera));
	//RETURN_ON_ERROR(createAbsoluteFocusProperty(camera));
	//RETURN_ON_ERROR(createWhiteBalanceTargetRedProperty(camera));
	//RETURN_ON_ERROR(createWhiteBalanceTargetGreenProperty(camera));
	//RETURN_ON_ERROR(createWhiteBalanceTargetBlueProperty(camera));
	//RETURN_ON_ERROR(createWhiteBalanceProperty(camera));

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


int LumeneraCamera::SnapImage()
{
	if (camera_)
	{
		LUXAPPS_TRY
		{
			std::unique_ptr<Imaging::Image> rawImage = camera_->getImage();

		// Demosaicing frame 

		LUCAM_IMAGE_FORMAT format = camera_->getStreamImageFormat();
		LUCAM_CONVERSION_PARAMS params = LuXApps::Lucam::getConversionParametersFromCameraProperties(camera_);

		std::unique_ptr<Imaging::Image> demosaicedImage;

		Imaging::IMAGE_FORMAT outputFormat = (rawImage->getFormat() == Imaging::IMAGE_FORMAT::MONO) ?
			Imaging::IMAGE_FORMAT::MONO : Imaging::IMAGE_FORMAT::BGRA;

		demosaicedImage.reset(LuXApps::Lucam::demosaic(rawImage.get(), camera_
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
			return DEVICE_ERR;
		}
	}
	else
	{
		return DEVICE_NOT_CONNECTED;
	}
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
	char cBuf[MM::MaxStrLength];
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
							return setCameraRoi(lastX, lastY, lastXSize, lastYSize, binningFactor) == DEVICE_OK;
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
			if (camera_->setRoi(getSensorWidth(), getSensorHeight(), 0, 0))
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
	return StartSequenceAcquisition(LONG_MAX, interval, false);
}

int LumeneraCamera::StartSequenceAcquisition(long numImages, double interval_ms, bool stopOnOverflow)
{
	//TODO: num is not respected here


	if (IsCapturing())
	{
		return DEVICE_CAMERA_BUSY_ACQUIRING;
	}
	else
	{
			LUXAPPS_TRY
			{
				RETURN_ON_ERROR(GetCoreCallback()->PrepareForAcq(this));
				sequenceThread_->Start();

				return DEVICE_OK;
			}
				LUXAPPS_CATCH(...)
			{
				return DEVICE_ERR;
			}
	}
	return DEVICE_OK;
}

bool LumeneraCamera::IsCapturing()
{
	return not sequenceThread_->IsStopped();
}

int LumeneraCamera::StopSequenceAcquisition()
{
	if (IsCapturing())
	{
		sequenceThread_->Stop();
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

CameraInterface::LucamAdapter::LucamCamera* LumeneraCamera::camera() 
{
	return camera_;
}

int LumeneraCamera::readCameraPropertyValue(const std::string& name, std::string& value) const
{
	int ret = DEVICE_ERR;

	CameraInterface::Property* prop = nullptr;
	if (camera_->tryGetProperty(name, prop))
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

int LumeneraCamera::writeCameraPropertyValue(const std::string& name, const std::string& value)
{
	int ret = DEVICE_ERR;

	CameraInterface::Property* prop = nullptr;
	if (camera_->tryGetProperty(name, prop))
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

	if (camera_ != nullptr)
	{
		RETURN_ON_ERROR(getBitDepthFromCamera(bitDepth));

		format = LuXApps::isColorCamera(camera_) ? Imaging::IMAGE_FORMAT::BGRA : Imaging::IMAGE_FORMAT::MONO;
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

	ret = camera_->setRoi(xSize, ySize, x, y) ? DEVICE_OK : DEVICE_INVALID_PROPERTY_VALUE;
	return ret;
}

unsigned LumeneraCamera::getSensorWidth()
{
	return std::stoul(camera_->getProperty(LuXAppsProperties::LUXPROP_MAXIMUM_WIDTH).getValue());
}

unsigned LumeneraCamera::getSensorHeight()
{
	return std::stoul(camera_->getProperty(LuXAppsProperties::LUXPROP_MAXIMUM_HEIGHT).getValue());
}

void LumeneraCamera::updateImageBuffer(std::unique_ptr<Imaging::Image>&& image)
{
	MMThreadGuard{ imageLock_ };

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
	if (camera_->tryGetProperty(LuXAppsProperties::LUXPROP_VERTICAL_FLIP, flipping))
	{
		enabled = (flipping->getValue() == "Enabled");
	}

	return enabled;
}

bool LumeneraCamera::isMirrorEnabled()
{
	bool enabled = false;

	CameraInterface::Property* flipping = nullptr;
	if (camera_->tryGetProperty(LuXAppsProperties::LUXPROP_HORIZONTAL_FLIP, flipping))
	{
		enabled = (flipping->getValue() == "Enabled");
	}


	return enabled;
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
	RETURN_ON_ERROR(SnapImage());
	return DEVICE_OK;
}

int LumeneraCamera::setBitDepth(const Imaging::IMAGE_BIT_DEPTH& bitDepth)
{
	std::string bitDepthValue = to_string(bitDepth);
	return writeCameraPropertyValue(LuXAppsProperties::LUXPROP_PIXEL_FORMAT, bitDepthValue);
}

bool LumeneraCamera::exposureRequiresStillStream(double value)
{
	bool requiresStillStream = false;

	if (hasSwitchingExposure_)
	{
		if (value > switchingExposure_)
		{
			requiresStillStream = true;
		}
	}

	return requiresStillStream;
}

bool LumeneraCamera::exposureRequiresVideoStream(double value)
{
	return !exposureRequiresStillStream(value);
}

bool LumeneraCamera::isVideoStreamingMode()
{
	bool isVideoMode = true;

	CameraInterface::Property* streamingModeProperty = nullptr;
	if (camera_->tryGetProperty(LuXAppsProperties::LUXPROP_STREAMING_MODE, streamingModeProperty))
	{
		isVideoMode = (streamingModeProperty->getValue() == "Video");
	}

	return isVideoMode;
}

bool LumeneraCamera::isStillStreamingMode()
{
	return !isVideoStreamingMode();
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
	RETURN_ON_ERROR(SnapImage());
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
		
	}
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

		bool requiresModeSwitch = false;
		std::string streamingMode;

		if (exposureRequiresVideoStream(value) && isStillStreamingMode())
		{
			requiresModeSwitch = true;
			streamingMode = "Video";
		}
		else if (exposureRequiresStillStream(value) && isVideoStreamingMode())
		{
			requiresModeSwitch = true;
			streamingMode = "Still";
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
		return writeCameraPropertyValue(propertyName, std::to_string(value));
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

					x = std::floor(static_cast<double>(x) / adjustmentFactor);
					y = std::floor(static_cast<double>(y) / adjustmentFactor);
					width = std::floor(static_cast<double>(width) / adjustmentFactor);
					height = std::floor(static_cast<double>(height) / adjustmentFactor);

					RETURN_ON_ERROR(setCameraRoi(x, y, width, height, newBinning));
					RETURN_ON_ERROR(refreshStream());
				}
		}
	}
	break;
	}

	return ret;
}

