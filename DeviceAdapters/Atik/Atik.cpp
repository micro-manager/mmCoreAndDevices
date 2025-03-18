///////////////////////////////////////////////////////////////////////////////
// FILE:          MMCamera.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Skeleton code for the micro-manager camera adapter. Use it as
//                starting point for writing custom device adapters
//                
// AUTHOR:        Nenad Amodaj, http://nenad.amodaj.com
//                
// COPYRIGHT:     University of California, San Francisco, 2011
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

#include "Atik.h"
#include "ModuleInterface.h"

using namespace std;

#ifdef _WIN32
#pragma warning(disable : 26812)
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#else
#include <unistd.h>
#define Sleep(a) usleep(a * 1000)
#endif

const char* g_CameraName = "Universal Atik Cameras Device Adapter";

const char* g_PixelType_8bit = "8bit";
const char* g_PixelType_16bit = "16bit";

const char* g_CameraModelProperty = "Model";
const char* g_CameraModel_A = "A";
const char* g_CameraModel_B = "B";

template<typename ... Args>
void Atik::log(const char* format, Args ... args) const
{
	int sizeNeeded = snprintf(nullptr, 0, format, args ...) + 1;
	if (sizeNeeded > 0)
	{
		auto size = static_cast<size_t>(sizeNeeded); // val is def above 0 so safe to cast
		auto buf = make_unique<char[]>(size);
		snprintf(buf.get(), size, format, args ...);
		LogMessage(buf.get());
	}
}

#define CHECK(code) do { \
	int ret = (code); \
	log("API call ret: %s, %d, %s", #code, ret, sdkEnumToString(ret)); \
} while (0)

#define CHECK_STRICT(code, successCode, err) do { \
	int ret = (code); \
	log("API call strict ret: %s, %d, %s", #code, ret, sdkEnumToString(ret)); \
	if (ret != (int)(successCode)) { \
		log("Failed strict API call: return %s", #err); \
		return err; } \
} while (0)

#define CHECK_STRICT_ART(code) do { \
	int ret = (code); \
	log("API call strict ret: %s, %d, %s", #code, ret, sdkEnumToString(ret)); \
	if (ret != (int)(ARTEMIS_OK)) { \
		log("Failed strict API call: return %s", DEVICE_ERR); \
		return DEVICE_ERR; } \
} while (0)

static const char* sdkEnumToString(int code)
{
	switch (code)
	{
	case ARTEMIS_OK:
		return "ARTEMIS_OK";
	case ARTEMIS_INVALID_PARAMETER:
		return "ARTEMIS_INVALID_PARAMETER";
	case ARTEMIS_NOT_CONNECTED:
		return "ARTEMIS_NOT_CONNECTED";
	case ARTEMIS_NOT_IMPLEMENTED:
		return "ARTEMIS_NOT_IMPLEMENTED";
	case ARTEMIS_NO_RESPONSE:
		return "ARTEMIS_NO_RESPONSE";
	case ARTEMIS_INVALID_FUNCTION:
		return "ARTEMIS_INVALID_FUNCTION";
	case ARTEMIS_NOT_INITIALIZED:
		return "ARTEMIS_NOT_INITIALIZED";
	case ARTEMIS_OPERATION_FAILED:
		return "ARTEMIS_OPERATION_FAILED";
	default:
		return "Out of range of ARTEMISERROR enum";
	}
}

///////////////////////////////////////////////////////////////////////////////
// Exported MMDevice API
///////////////////////////////////////////////////////////////////////////////

/**
 * List all supported hardware devices here
 */
MODULE_API void InitializeModuleData()
{
	RegisterDevice(g_CameraName, MM::CameraDevice, "Atik");
}

MODULE_API MM::Device* CreateDevice(const char* deviceName)
{
	if (deviceName == 0)
		return 0;

	// decide which device class to create based on the deviceName parameter
	if (strcmp(deviceName, g_CameraName) == 0)
	{
		// create camera
		return new Atik();
	}

	// ...supplied name not recognized
	return 0;
}

MODULE_API void DeleteDevice(MM::Device* pDevice)
{
	delete pDevice;
}

///////////////////////////////////////////////////////////////////////////////
// MMCamera implementation
// ~~~~~~~~~~~~~~~~~~~~~~~

/**
* MMCamera constructor.
* Setup default all variables and create device properties required to exist
* before intialization. In this case, no such properties were required. All
* properties will be created in the Initialize() method.
*
* As a general guideline Micro-Manager devices do not access hardware in the
* the constructor. We should do as little as possible in the constructor and
* perform most of the initialization in the Initialize() method.
*/
Atik::Atik() :
	handle(0),
	setGOType(0),
	height(0),
	width(0),
	modelName(""),
	binning_(1),
	gain_(0),
	offset_(0),
	bytesPerPixel_(2),
	initialized_(false),
	exposureMs_(100.0),
	roiX_(0),
	roiY_(0),
	thd_(0),
	coolingEnabled(0),
	coolingPower(0),
	coolingTargetTemp(0),
	enableLogging(false)
{
	log("constructor");

	// call the base class method to set-up default error codes/messages
	InitializeDefaultErrorMessages();

	// Description property
	int ret = CreateProperty(MM::g_Keyword_Description, "Universal Atik Cameras device adapter", MM::String, true);

	// camera type pre-initialization property
	ret = CreateProperty(g_CameraModelProperty, g_CameraModel_A, MM::String, false, 0, true);

	vector<string> modelValues;
	modelValues.push_back(g_CameraModel_A);
	modelValues.push_back(g_CameraModel_A);

	ret = SetAllowedValues(g_CameraModelProperty, modelValues);

	// create live video thread
	thd_ = new SequenceThread(this);
}

/**
* MMCamera destructor.
* If this device used as intended within the Micro-Manager system,
* Shutdown() will be always called before the destructor. But in any case
* we need to make sure that all resources are properly released even if
* Shutdown() was not called.
*/
Atik::~Atik()
{
	log("destructor");

	if (initialized_)
		Shutdown();

	delete thd_;
}

/**
* Obtains device name.
* Required by the MM::Device API.
*/
void Atik::GetName(char* name) const
{
	log("");

	// We just return the name we use for referring to this
	// device adapter.
	CDeviceUtils::CopyLimitedString(name, g_CameraName);
}

int Atik::initialiseAtikCamera()
{
	bool present = false;

	// Note ArtemisLoadDLL returns true on success, not ARTEMIS_OK;
	CHECK_STRICT(ArtemisLoadDLL("AtikCameras.dll"), true, DEVICE_NATIVE_MODULE_FAILED);

	// Connect to camera now, try for 10 secs in total
	for (int i = 0; i < 10; i++)
	{
		present = ArtemisDeviceIsPresent(0);

		if (present)
		{
			break;
		}
		else
		{
			Sleep(100);
		}
	}

	log("Found device? %d", present);

	if (!present)
	{
		log("Exit as device not found");
		return DEVICE_NOT_CONNECTED;
	}

	Sleep(100);

	log("Try connect");
	handle = ArtemisConnect(0);

	if (!handle)
	{
		log("No handle, return error");
		return DEVICE_NOT_CONNECTED;
	}
	else
	{
		log("Got handle");
	}

	Sleep(100);

	CHECK(ArtemisDeviceName(0, this->modelName));
	

	log("Got device name");

	Sleep(100);

	ARTEMISPROPERTIES props;
	CHECK_STRICT_ART(ArtemisProperties(handle, &props));

	this->width = props.nPixelsX;
	this->height = props.nPixelsY;
	log("Got props");

	Sleep(100);

	return DEVICE_OK;
}

/**
* Intializes the hardware.
* Typically we access and initialize hardware at this point.
* Device properties are typically created here as well.
* Required by the MM::Device API.
*/
int Atik::Initialize()
{
	if (initialized_)
		return DEVICE_OK;

	log("");

	int ret = initialiseAtikCamera();

	if (ret != DEVICE_OK)
		return ret;

	ARTEMISPROPERTIES prop;
	ArtemisProperties(handle, &prop);

	auto hasTrigger = ((prop.cameraflags & 2) == 2); // Trigger mode 
	auto hasPreview = ((prop.cameraflags & 4) == 4); // Preview mode
	auto hasShutter = ((prop.cameraflags & 16) == 16); // Dark mode

	// Binning
	{
		CPropertyAction* pAct = new CPropertyAction(this, &Atik::OnBinning);
		ret = CreateProperty(MM::g_Keyword_Binning, "1", MM::Integer, false, pAct);
		assert(ret == DEVICE_OK);

		int binX, binY;
		ArtemisGetMaxBin(handle, &binX, &binY);

		int curX, curY;
		ArtemisGetBin(handle, &curX, &curY);
		binning_ = curX; // symmetrical based on x bin

		int maxBin = min(binX, binY);
		log("Max bin reported by camera: %d", maxBin);

		maxBin = maxBin > 16 ? 16 : (maxBin < 1 ? 1 : maxBin);
		log("Max bin set to %d", maxBin);

		vector<string> binningValues;
		for (int bin = 1; bin <= maxBin; bin++)
			binningValues.push_back(std::to_string(bin));

		ret = SetAllowedValues(MM::g_Keyword_Binning, binningValues);
		assert(ret == DEVICE_OK);
	}


	int flags, minlevel, maxlevel;
	ArtemisCoolingInfo(handle, &flags, (int*)&coolingPower, &minlevel, &maxlevel, (int*)&coolingTargetTemp);
	bool hasCooling = ((ARTEMIS_COOLING_INFO_HASCOOLING & flags) == 1);
	bool isControllable = ((ARTEMIS_COOLING_INFO_CONTROLLABLE & flags) == 2);
	hasPowerlvl = ((ARTEMIS_COOLING_INFO_POWERLEVELCONTROL & flags) == 8);
	coolingEnabled = ((ARTEMIS_COOLING_INFO_COOLINGON & flags) == 64);
	hasSetPoint = ((ARTEMIS_COOLING_INFO_SETPOINTCONTROL & flags) == 16);

	// Sensor Temperature
	{
		int numSensors = 0;
		ArtemisTemperatureSensorInfo(handle, 0, &numSensors);

		if (numSensors > 0)
		{
			int temp = 0;
			ArtemisTemperatureSensorInfo(handle, 1, &temp);
			float tempF = static_cast<float>(temp);
			tempF /= 100;

			// Camera sensor temperature
			auto pAct = new CPropertyAction(this, &Atik::OnSensorTemp);
			ret = CreateFloatProperty("Camera Sensor Temperature", tempF, true, pAct);
			assert(ret == DEVICE_OK);
		}
	}

	// Cooling Enable
	{
		if (hasCooling && isControllable)
		{
			CPropertyAction* pAct = new CPropertyAction(this, &Atik::OnCoolingEnable);
			ret = CreateIntegerProperty("Cooling Enable", coolingEnabled, false, pAct);
			assert(ret == DEVICE_OK);

			ret = SetPropertyLimits("Cooling Enable", 0, 1);
			assert(ret == DEVICE_OK);
		}
		else if (hasCooling && !isControllable)
		{
			ret = CreateProperty("Cooling Auto On", "1", MM::Integer, true);
			assert(ret == DEVICE_OK);
		}
	}

	// Cooling Target Temp
	{
		if (hasCooling && hasSetPoint)
		{

			CPropertyAction* pAct = new CPropertyAction(this, &Atik::OnCoolingTargetTemp);
			ret = CreateIntegerProperty("Cooling Target Temp", coolingTargetTemp, false, pAct);
			assert(ret == DEVICE_OK);

			ret = SetPropertyLimits("Cooling Target Temp", -40, 20);
			assert(ret == DEVICE_OK);
		}
	}

	// Cooling Power
	{
		if (hasCooling && hasPowerlvl && !hasSetPoint)
		{
			CPropertyAction* pAct = new CPropertyAction(this, &Atik::OnCoolingPower);
			ret = CreateIntegerProperty("Cooling Power", coolingPower, false, pAct);
			assert(ret == DEVICE_OK);

			ret = SetPropertyLimits("Cooling Power", minlevel, maxlevel);
			assert(ret == DEVICE_OK);
		}
	}

	// Dark Mode
	{
		if (hasShutter) 
		{
			darkModeEnabled = ArtemisGetDarkMode(handle);

			CPropertyAction* pAct = new CPropertyAction(this, &Atik::OnDarkMode);
			ret = CreateIntegerProperty("Dark Mode Enabled", darkModeEnabled, false, pAct);
			assert(ret == DEVICE_OK);

			ret = SetPropertyLimits("Dark Mode Enabled", 0, 1);
			assert(ret == DEVICE_OK);
		}
	}

	// Preview Mode
	{
		if (hasPreview)
		{
			CPropertyAction* pAct = new CPropertyAction(this, &Atik::OnPreview);
			ret = CreateProperty("Preview Enabled", "0", MM::Integer, false, pAct);
			assert(ret == DEVICE_OK);

			ret = SetPropertyLimits("Preview Enabled", 0, 1);
			assert(ret == DEVICE_OK);
		}
	}

	// Gain
	{
		if (ArtemisHasCameraSpecificOption(handle, ID_GOCustomGain))
		{
			setGOType = FX3;

			unsigned short mode = 0; // Put the camera into custom Gain/Offset mode
			ArtemisCameraSpecificOptionSetData(handle, ID_GOPresetMode, (unsigned char*)&mode, 2);

			int actualLength;
			unsigned short minG, maxG;
			unsigned char gData[6];
			ArtemisCameraSpecificOptionGetData(handle, ID_GOCustomGain, gData, 6, &actualLength);

			unsigned short* ptr = (unsigned short*)gData;
			minG = *ptr;
			maxG = ptr[1];
			gain_ = ptr[2];

			CPropertyAction* pAct = new CPropertyAction(this, &Atik::OnGain);
			ret = CreateIntegerProperty(MM::g_Keyword_Gain, gain_, false, pAct);
			assert(ret == DEVICE_OK);

			SetPropertyLimits(MM::g_Keyword_Gain, minG, maxG);
			assert(ret == DEVICE_OK);
		}
		else
		{
			auto res = ArtemisGetGain(handle, false, &gain_, &offset_);
			if (res != ARTEMIS_INVALID_FUNCTION)
			{
				setGOType = FX2;
				CPropertyAction* pAct = new CPropertyAction(this, &Atik::OnGain);
				ret = CreateIntegerProperty(MM::g_Keyword_Gain, gain_, false, pAct);
				assert(ret == DEVICE_OK);

				SetPropertyLimits(MM::g_Keyword_Gain, 0, 63);
				assert(ret == DEVICE_OK);
			}
		}
	}

	// Pixel Type
	{
		CPropertyAction* pAct = new CPropertyAction(this, &Atik::OnPixelType);
		ret = CreateProperty(MM::g_Keyword_PixelType, g_PixelType_16bit, MM::String, false, pAct);
		assert(ret == DEVICE_OK);

		vector<string> pixelTypeValues;
		pixelTypeValues.push_back(g_PixelType_16bit);

		ret = SetAllowedValues(MM::g_Keyword_PixelType, pixelTypeValues);
		assert(ret == DEVICE_OK);
	}

	// Offset
	{
		if (ArtemisHasCameraSpecificOption(handle, ID_GOCustomOffset))
		{
			setGOType = FX3;

			unsigned short mode = 0; // Put the camera into custom Gain/Offset mode
			ArtemisCameraSpecificOptionSetData(handle, ID_GOPresetMode, (unsigned char*)&mode, 2);

			int actualLength;
			unsigned short minO, maxO;
			unsigned char oData[6];
			ArtemisCameraSpecificOptionGetData(handle, ID_GOCustomOffset, oData, 6, &actualLength);

			unsigned short* ptr = (unsigned short*)oData;
			minO = *ptr;
			maxO = ptr[1];
			offset_ = ptr[2];

			CPropertyAction* pAct = new CPropertyAction(this, &Atik::OnOffset);
			ret = CreateIntegerProperty(MM::g_Keyword_Offset, offset_, false, pAct);
			assert(ret == DEVICE_OK);

			SetPropertyLimits(MM::g_Keyword_Offset, minO, maxO);
			assert(ret == DEVICE_OK);
		}
		else
		{
			auto res = ArtemisGetGain(handle, false, &gain_, &offset_);
			if (res != ARTEMIS_INVALID_FUNCTION)
			{
				setGOType = FX2;
				CPropertyAction* pAct = new CPropertyAction(this, &Atik::OnOffset);
				ret = CreateIntegerProperty(MM::g_Keyword_Offset, offset_, false, pAct);
				assert(ret == DEVICE_OK);

				SetPropertyLimits(MM::g_Keyword_Offset, 0, 512);
				assert(ret == DEVICE_OK);
			}
		}
	}

	//Exposure Mode
	{
		if (ArtemisHasCameraSpecificOption(handle, ID_ExposureSpeed))
		{
			unsigned short expMode;
			int actual = 0;
			ArtemisCameraSpecificOptionGetData(handle, ID_ExposureSpeed, (unsigned char*)&expMode, 2, &actual);
			
			std::vector<std::string> expModeVals = { "Long Exposure",  "Short Exposure", "Auto" };
			switch (expMode)
			{
			case 0:
				exposureMode_ = expModeVals[0];
				break;
			case 1:
				exposureMode_ = expModeVals[1];
				break;
			case 2: // Fast Mode not currently implemented in MM, force the camera into Auto(3) mode.
			{
				exposureMode_ = expModeVals[2];
				unsigned short expS = 3;
				ArtemisCameraSpecificOptionSetData(handle, ID_ExposureSpeed, (unsigned char*)&expS, 2);
				break;
			}
			default:
				exposureMode_ = expModeVals[2];
				break;
			}

			auto pAct = new CPropertyAction(this, &Atik::OnExposureMode);
			ret = CreateProperty("Exposure Mode", exposureMode_.c_str(), MM::String, false, pAct);
			SetAllowedValues("Exposure Mode", expModeVals);
		}
	}

	// Trigger
	{
		if (hasTrigger)
		{
			CPropertyAction* pAct = new CPropertyAction(this, &Atik::OnTrigger);
			ret = CreateProperty("Use Trigger", "0", MM::Integer, false, pAct);
			assert(ret == DEVICE_OK);

			ret = SetPropertyLimits("Use Trigger", 0, 1);
			assert(ret == DEVICE_OK);
		}
	}

	// synchronize all properties
	// --------------------------
	ret = UpdateStatus();
	if (ret != DEVICE_OK)
		return ret;

	// setup the buffer
	// ----------------
	ret = ResizeImageBuffer();
	if (ret != DEVICE_OK)
		return ret;

	initialized_ = true;
	return DEVICE_OK;
}

/**
* Shuts down (unloads) the device.
* Ideally this method will completely unload the device and release all resources.
* Shutdown() may be called multiple times in a row.
* Required by the MM::Device API.
*/
int Atik::Shutdown()
{
	if (initialized_)
	{
		log("Shutdown of SDK");
		CHECK(ArtemisDisconnect(handle));
		handle = nullptr;

		initialized_ = false;
	}

	return DEVICE_OK;
}

/**
* Performs exposure and grabs a single image.
* This function should block during the actual exposure and return immediately afterwards
* (i.e., before readout).  This behavior is needed for proper synchronization with the shutter.
* Required by the MM::Camera API.
*/
int Atik::SnapImage()
{
	log("");
	CHECK_STRICT_ART(ArtemisStartExposureMS(handle, (int)exposureMs_));

	while (!ArtemisImageReady(handle))
	{
		Sleep(10);
	}

	const char* pixels = (const char*)ArtemisImageBuffer(handle);

	int x, y, w, h, bx, by;
	CHECK_STRICT_ART(ArtemisGetImageData(handle, &x, &y, &w, &h, &bx, &by));

	memcpy(img_.GetPixelsRW(), pixels, static_cast<size_t>(img_.Width()) * img_.Height() * img_.Depth());

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
const unsigned char* Atik::GetImageBuffer()
{
	//log("");
	return img_.GetPixels();
}

/**
* Returns image buffer X-size in pixels.
* Required by the MM::Camera API.
*/
unsigned Atik::GetImageWidth() const
{
	//log("");
	return img_.Width();
}

/**
* Returns image buffer Y-size in pixels.
* Required by the MM::Camera API.
*/
unsigned Atik::GetImageHeight() const
{
	//log("");
	return img_.Height();
}

/**
* Returns image buffer pixel depth in bytes.
* Required by the MM::Camera API.
*/
unsigned Atik::GetImageBytesPerPixel() const
{
	//log("");
	return img_.Depth();
}

/**
* Returns the bit depth (dynamic range) of the pixel.
* This does not affect the buffer size, it just gives the client application
* a guideline on how to interpret pixel values.
* Required by the MM::Camera API.
*/
unsigned Atik::GetBitDepth() const
{
	//log("");
	return img_.Depth() == 1 ? 8 : 16;
}

/**
* Returns the size in bytes of the image buffer.
* Required by the MM::Camera API.
*/
long Atik::GetImageBufferSize() const
{
	//log("");
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
* @param x - top-left corner coordinate
* @param y - top-left corner coordinate
* @param xSize - width
* @param ySize - height
*/
int Atik::SetROI(unsigned x, unsigned y, unsigned xSize, unsigned ySize)
{
	//log("");

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
* Required by the MM::Camera API.
*/
int Atik::GetROI(unsigned& x, unsigned& y, unsigned& xSize, unsigned& ySize)
{
	//log("");
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
int Atik::ClearROI()
{
	//log("");

	ResizeImageBuffer();
	roiX_ = 0;
	roiY_ = 0;

	return DEVICE_OK;
}

/**
* Returns the current exposure setting in milliseconds.
* Required by the MM::Camera API.
*/
double Atik::GetExposure() const
{
	//log("");
	return exposureMs_;
}

/**
* Sets exposure in milliseconds.
* Required by the MM::Camera API.
*/
void Atik::SetExposure(double exp)
{
	//log("");
	exposureMs_ = exp;
}

/**
* Returns the current binning factor.
* Required by the MM::Camera API.
*/
int Atik::GetBinning() const
{
	//log("");
	return binning_;
}

/**
* Sets binning factor.
* Required by the MM::Camera API.
*/
int Atik::SetBinning(int binF)
{
	//log("");
	return SetProperty(MM::g_Keyword_Binning, CDeviceUtils::ConvertToString(binF));
}

int Atik::PrepareSequenceAcqusition()
{
	//log("");
	if (IsCapturing())
		return DEVICE_CAMERA_BUSY_ACQUIRING;

	int ret = GetCoreCallback()->PrepareForAcq(this);
	if (ret != DEVICE_OK)
		return ret;

	return DEVICE_OK;
}

/**
 * Required by the MM::Camera API
 * Please implement this yourself and do not rely on the base class implementation
 * The Base class implementation is deprecated and will be removed shortly
 */
int Atik::StartSequenceAcquisition(double interval) {

	//log("");
	return CLegacyCameraBase<Atik>::StartSequenceAcquisition(interval);
}

/**
* Stop and wait for the Sequence thread finished
*/
int Atik::StopSequenceAcquisition()
{
	//log("");
	return CLegacyCameraBase<Atik>::StopSequenceAcquisition();
}

/**
* Simple implementation of Sequence Acquisition
* A sequence acquisition should run on its own thread and transport new images
* coming of the camera into the MMCore circular buffer.
*/
int Atik::StartSequenceAcquisition(long numImages, double interval_ms, bool stopOnOverflow)
{
	//log("");
	return CLegacyCameraBase<Atik>::StartSequenceAcquisition(numImages, interval_ms, stopOnOverflow);
}

/*
 * Inserts Image and MetaData into MMCore circular Buffer
 */
int Atik::InsertImage()
{
	//log("");
	MM::MMTime timeStamp = this->GetCurrentMMTime();
	char label[MM::MaxStrLength];
	this->GetLabel(label);

	// Important:  metadata about the image are generated here:
	Metadata md;

	md.put(MM::g_Keyword_Metadata_CameraLabel, "Atik SDK Camera");

	string serialised = md.Serialize();

	auto imgBuf = GetImageBuffer();
	auto w = GetImageWidth();
	auto h = GetImageHeight();
	auto bpp = GetImageBytesPerPixel();

	int ret = GetCoreCallback()->InsertImage(this, imgBuf, w, h, bpp, serialised.c_str());

	if (!isStopOnOverflow() && ret == DEVICE_BUFFER_OVERFLOW)
	{
		GetCoreCallback()->ClearImageBuffer(this);
		return GetCoreCallback()->InsertImage(this, imgBuf, w, h, bpp, serialised.c_str(), false);
	}
	else
	{
		return ret;
	}
}


bool Atik::IsCapturing() {
	log("");
	return CLegacyCameraBase<Atik>::IsCapturing();
}


///////////////////////////////////////////////////////////////////////////////
// MMCamera Action handlers
///////////////////////////////////////////////////////////////////////////////

/**
* Handles "Binning" property.
*/
int Atik::OnBinning(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	//log("");
	if (eAct == MM::AfterSet)
	{
		long binSize;
		pProp->Get(binSize);
		binning_ = (int)binSize;

		CHECK_STRICT_ART(ArtemisBin(handle, binning_, binning_));

		return ResizeImageBuffer();
	}
	else if (eAct == MM::BeforeGet)
	{
		pProp->Set((long)binning_);
	}

	return DEVICE_OK;
}

int Atik::OnSensorTemp(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		int temp = 0;
		CHECK_STRICT_ART(ArtemisTemperatureSensorInfo(handle, 1, &temp));
		float tempF = static_cast<float>(temp);
		tempF /= 100;

		pProp->Set(tempF);
	}
	return DEVICE_OK;
}

int Atik::OnCoolingEnable(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::AfterSet)
	{
		pProp->Get(coolingEnabled);

		log("CoolingEnabled set to %d", coolingEnabled);

		if (coolingEnabled > 0)
		{
			if (hasSetPoint)
			{
				CHECK_STRICT_ART(ArtemisSetCooling(handle, (int)coolingTargetTemp * 100));
				log("Set target temp to %d", coolingTargetTemp);
			}
			else if (hasPowerlvl)
			{
				CHECK_STRICT_ART(ArtemisSetCoolingPower(handle, (int)coolingPower));
				log("Set Cooling power to %d", coolingPower);
			}
		}
		else
		{
			CHECK(ArtemisCoolerWarmUp(handle));
		}
	}
	else if (eAct == MM::BeforeGet)
	{
		pProp->Set(coolingEnabled);
	}

	return DEVICE_OK;
}

int Atik::OnCoolingTargetTemp(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::AfterSet)
	{
		pProp->Get(coolingTargetTemp);
		log("Set CoolingTargetTemp to %d", coolingTargetTemp);
	}
	else if (eAct == MM::BeforeGet)
	{
		pProp->Set(coolingTargetTemp);
	}

	return DEVICE_OK;
}

int Atik::OnCoolingPower(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::AfterSet)
	{
		pProp->Get(coolingPower);
		log("Set CoolingPower to %d", coolingPower);
	}
	else if (eAct == MM::BeforeGet)
	{
		pProp->Set(coolingPower);
	}

	return DEVICE_OK;
}

int Atik::OnPixelType(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	//log("");

	if (eAct == MM::AfterSet)
	{
		string val;
		pProp->Get(val);
		if (val.compare(g_PixelType_8bit) == 0)
			bytesPerPixel_ = 1;
		else if (val.compare(g_PixelType_16bit) == 0)
			bytesPerPixel_ = 2;

		ResizeImageBuffer();
	}
	else if (eAct == MM::BeforeGet)
	{
		if (bytesPerPixel_ == 1)
			pProp->Set(g_PixelType_8bit);
		else if (bytesPerPixel_ == 2)
			pProp->Set(g_PixelType_16bit);
	}

	return DEVICE_OK;
}


/**
* Handles "Gain" property.
*/
int Atik::OnGain(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	//log("");
	if (eAct == MM::AfterSet)
	{
		long gain;
		pProp->Get(gain);
		gain_ = gain;

		switch (setGOType)
		{
		case FX2:
			CHECK_STRICT_ART(ArtemisSetGain(handle, previewEnabled, gain_, offset_));
			break;
		case FX3:
			CHECK_STRICT_ART(ArtemisCameraSpecificOptionSetData(handle, ID_GOCustomGain, (unsigned char*)&gain_, 2));
			break;
		default:
			break;
		}
	}
	else if (eAct == MM::BeforeGet)
	{
		pProp->Set((long)gain_);
	}

	return DEVICE_OK;
}

/**
* Handles "Offset" property.
*/
int Atik::OnOffset(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	//log("");
	if (eAct == MM::AfterSet)
	{
		long offset;
		pProp->Get(offset);
		offset_ = (int)offset;

		switch (setGOType)
		{
		case FX2:
			CHECK_STRICT_ART(ArtemisSetGain(handle, previewEnabled, gain_, offset_));
			break;
		case FX3:
			CHECK_STRICT_ART(ArtemisCameraSpecificOptionSetData(handle, ID_GOCustomOffset, (unsigned char*)&offset_, 2));
			break;
		default:
			break;
		}
	}
	else if (eAct == MM::BeforeGet)
	{
		pProp->Set((long)offset_);
	}

	return DEVICE_OK;
}

int Atik::OnExposureMode(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::AfterSet)
	{
		std::string expMode;
		pProp->Get(expMode);

		unsigned short expModeI = USHRT_MAX;
		if (expMode.find("Long Exposure") != string::npos)
		{
			expModeI = 0;
		}
		else if (expMode.find("Short Exposure") != string::npos)
		{
			expModeI = 1;
		}
		else if (expMode.find("Auto") != string::npos)
		{
			expModeI = 3;
		}

		if (expModeI == USHRT_MAX)
			return DEVICE_INVALID_PROPERTY_VALUE;

		exposureMode_ = expMode;

		CHECK_STRICT_ART(ArtemisCameraSpecificOptionSetData(handle, ID_ExposureSpeed, (unsigned char*)&expModeI, 2));
	}
	else if (eAct == MM::BeforeGet)
	{
		pProp->Set(exposureMode_.c_str());
	}

	return DEVICE_OK;
}

int Atik::OnDarkMode(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::AfterSet)
	{
		pProp->Get(darkModeEnabled);

		log("Dark Mode set to %d", darkModeEnabled);

		CHECK_STRICT_ART(ArtemisSetDarkMode(handle, darkModeEnabled));
	}
	else if (eAct == MM::BeforeGet)
	{
		pProp->Set(darkModeEnabled);
	}

	return DEVICE_OK;
}

int Atik::OnTrigger(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::AfterSet)
	{
		pProp->Get(triggerEnabled);

		log("Dark Mode set to %d", triggerEnabled);

		CHECK_STRICT_ART(ArtemisTriggeredExposure(handle, triggerEnabled));
	}
	else if (eAct == MM::BeforeGet)
	{
		pProp->Set(triggerEnabled);
	}

	return DEVICE_OK;
}

int Atik::OnPreview(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::AfterSet)
	{
		pProp->Get(previewEnabled);

		log("Preview mode set to %d", previewEnabled);

		CHECK_STRICT_ART(ArtemisSetPreview(handle, previewEnabled));
	}
	else if (eAct == MM::BeforeGet)
	{
		pProp->Set(previewEnabled);
	}

	return DEVICE_OK;
}

///////////////////////////////////////////////////////////////////////////////
// Private MMCamera methods
///////////////////////////////////////////////////////////////////////////////

/**
* Sync internal image buffer size to the chosen property values.
*/
int Atik::ResizeImageBuffer()
{
	log("");
	img_.Resize(width / binning_, height / binning_, bytesPerPixel_);

	return DEVICE_OK;
}
