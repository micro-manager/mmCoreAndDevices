#include <list>
#include <string>
#include <map>
#include <iterator>
#include <algorithm>
#include <string>
#include <sstream> 
//#include <atltypes.h>

#include <PixeLINKAPI.h>
#include "Pixelink.h"
#include "../../MMDevice/ModuleInterface.h"


/*
NOTE FROM SIMON:

Make sure application is compiled for most recent build

current build version is for MMSetup_64bit_1.4.23_20190115.exe  (15 January 2019)
*/


using namespace std;
 
// instead of translating all PG Errors into MM error codes
// we use one error code and set the description whenever
// an error occurs
const int ALLERRORS = 10001;



/////////////////////////////////////////////////////

const char* g_Interface = "Interface";
const char* g_VendorName = "Vendor Name";
const char* g_SensorResolution = "Sensor Resolution";
const char* g_SensorInfo = "Sensor Info";
const char* g_DriverName = "Driver Name";
const char* g_FirmwareVersion = "Firmware Version";
const char* g_FirmwareBuildTime = "Firmware Build Time";
const char* g_MaxBusSpeed = "Maximum Bus Speed";
const char* g_InterfaceType = "Interface Type";
const char* g_ColorMonoChrome = "Color or Monochrome";
const char* g_IIDCVersion = "IIDC version";
const char* g_CameraId = "CameraID";
const char* g_VideoModeAndFrameRate = "Video Mode and Frame Rate";
const char* g_NotSet = "Not set";
const char* g_Format7Mode = "Format-7 Mode";
const char* g_InternalTrigger = "Internal";
const char* g_ExternalTrigger = "External";
const char* g_SoftwareTrigger = "Software";

/////////////////////////////////////////////////////

const char* g_Data_Format = "Data format";
const char* g_PixelType_Mono8 = "Mono 8";
const char* g_PixelType_Raw8 = "Raw 8";
const char* g_PixelType_Mono10 = "Mono 10";
const char* g_PixelType_Raw10 = "Raw 10";
const char* g_PixelType_Mono12 = "Mono 12";
const char* g_PixelType_Raw12 = "Raw 12";
const char* g_PixelType_Mono14 = "Mono 14";
const char* g_PixelType_Raw14 = "Raw 14";
const char* g_PixelType_RGB32 = "RGB 32";

/////////////////////////////////////////////////////
// constants for naming pixel types (allowed values of the "PixelType" property)
const char* g_PixelType_32bitRGB = "32bitRGB";
const char* g_PixelType_64bitRGB = "64bitRGB";


bool isColour;

int supportedBayer8 = -1;
int supporterBayer12 = -1;
int supportedBayer16 = -1;
int suppportedYUV = -1;



const int g_NumProps = 4;
const U32 g_PropertyTypes[g_NumProps] = { FEATURE_EXPOSURE, FEATURE_GAIN, FEATURE_SATURATION, FEATURE_FRAME_RATE};
const std::string g_PropertyNames[g_NumProps] = { "Exposure", "Gain", "Saturation","FrameRate"};

const int g_NumOffProps = 3;
const U32 g_OffPropertyTypes[g_NumOffProps] = {  FEATURE_BRIGHTNESS, FEATURE_AUTO_ROI, FEATURE_AUTO_ROI };

const int g_NumFrameRates = 9;
const std::string g_FrameRates[g_NumFrameRates] = { "1.875 fps", "3.75 fps",
"7.5 fps", "15 fps", "30 fps", "60 fps", "120 fps", "240 fps", "FORMAT7" };

const int g_NumVideoModes = 24;
const std::string g_VideoModes[g_NumVideoModes] = { "160x120 YUV444", "320x240 YUV422",
"640x480 YUV411", "640x480 YUV422", "640x480 24-bit", "640x480 8-bit", "640x480 16-bit",
" 800x600 YUV422", "800x600 RGB",  "800x600 8-bit", "800x600 16-bit", "1024x768YUV422",
"1024x768 RGB", "1024x768 8-bit", "1024x768 16-bit", "1280x960 YUV422", "1280x960 RGB",
"1280x960 8-bit", "1280x960 16-bit", "1600x1200 YUV422", "1600x1200 RGB", "1600x1200 8-bit",
"1600x1200 16-bit", "FORMAT7" };





///////////////////////////////////////////////////////////////////////////////
// Exported MMDevice API
///////////////////////////////////////////////////////////////////////////////


/***********************************************************************
* List all supported hardware devices here
*/
MODULE_API void InitializeModuleData()
{

	PXL_RETURN_CODE rc = 0;
	U32 numCameras = 10;


	std::vector<CAMERA_ID_INFO> tempCameraIdInfo(numCameras);
	memset(&tempCameraIdInfo[0], 0, sizeof(tempCameraIdInfo[0]) * tempCameraIdInfo.size());
	tempCameraIdInfo[0].StructSize = sizeof(CAMERA_ID_INFO);
	PxLGetNumberCamerasEx(&tempCameraIdInfo[0], &numCameras);




	if (!API_SUCCESS(rc))
	{
		// TODO work out how to return/report errors, 
		return;
	}
	for (unsigned int i = 0; i < numCameras; i++)
	{

		U32 serialNumber = tempCameraIdInfo[i].CameraSerialNum;

		string Result;
		stringstream convert;
		convert << serialNumber;
		Result = convert.str();

		RegisterDevice(Result.c_str(), MM::CameraDevice, "Pixelink Camera");
	}
}


//***********************************************************************

MODULE_API MM::Device* CreateDevice(const char* deviceName)
{
	return new Pixelink(deviceName);
}

//***********************************************************************

MODULE_API void DeleteDevice(MM::Device* pDevice)
{
	delete pDevice;
}

///////////////////////////////////////////////////////////////////////////////
// Pixelink implementation
/***********************************************************************
* Pixelink constructor.
* Setup default variables and create device properties required to exist
* before intialization. Most properties will be created in the
* Initialize() method.
*
* As a general guideline Micro-Manager devices do not access hardware in the
* the constructor. We should do as little as possible in the constructor and
* perform most of the initialization in the Initialize() method.
*/
Pixelink::Pixelink(const char* deviceName) :
	nComponents_(1),
	initialized_(false),
	deviceName_(deviceName),
	sequenceStartTime_(0),
	imageCounter_(0),
	stopOnOverflow_(false),
	isCapturing_(false),
	f7InUse_(false),
	externalTriggerGrabTimeout_(60000),
	bytesPerPixel_(1),
	imgBuf_(0),
	bufSize_(0),
	tempCounter(0)
{
	// call the base class method to set-up default error codes/messages
	InitializeDefaultErrorMessages();

	thd_ = new MySequenceThread(this);
	//SetErrorText(ERR_NOT_READY_FOR_SOFTWARE_TRIGGER,
	//	"Camera not ready for software trigger");

}

/***********************************************************************
* Pixelink destructor.
* If this device is used as intended within the Micro-Manager system,
* Shutdown() will be always called before the destructor. But in any case
* we need to make sure that all resources are properly released even if
* Shutdown() was not called.
*/
Pixelink::~Pixelink()
{
	StopSequenceAcquisition();
	delete thd_;

	if (initialized_)
		Shutdown();
}

/***********************************************************************
* Obtains device name.
* Required by the MM::Device API.
*/
void Pixelink::GetName(char* name) const
{
	CDeviceUtils::CopyLimitedString(name, deviceName_.c_str());
}

/***********************************************************************
* Intializes the hardware.
* Gets the PGRGuid based on the CameraId set in the Module initializer.
* Uses this to retrieve information about the camera and expose all
* possible properties.
* Required by the MM::Device API.
*/
int Pixelink::Initialize()
{
	isStreaming = false;


	if (initialized_)
		return DEVICE_OK;



	U32 serial = std::stoi(deviceName_);

	PXL_RETURN_CODE rc = PxLInitialize(serial, &m_handle);

	LoadCameraFeatures();


	if (!(API_SUCCESS(rc))) 
	{
		return rc;
	}


	
	
	isColour = IsColourCamera();


	if (isColour == true) 
	{
		nComponents_ = 4;
	}
	else
	{
		nComponents_ = 1;
	}


	// Get the camera information
	CAMERA_INFO cameraInfo;
	rc = PxLGetCameraInfo(m_handle, &cameraInfo);
	if (!(API_SUCCESS(rc)))
	{
		return rc;
	}

	// -------------------------------------------------------------------------
	// Set property list
	// -------------------------------------------------------------------------

	// camera identification and other read-only information

	// camera identification and other read-only information
	char buf[512] = "";

	sprintf(buf, "%d", cameraInfo.SerialNumber);
	int ret = CreateProperty(MM::g_Keyword_CameraID, buf, MM::String, true);
	assert(ret == DEVICE_OK);

	sprintf(buf, "%s", cameraInfo.ModelName);
	ret = CreateProperty(MM::g_Keyword_CameraName, buf, MM::String, true);
	assert(ret == DEVICE_OK);

	sprintf(buf, "%s", cameraInfo.CameraName);
	ret = CreateProperty(MM::g_Keyword_Description, buf, MM::String, true);
	assert(ret == DEVICE_OK);

	sprintf(buf, "%s", cameraInfo.VendorName);
	ret = CreateProperty(g_VendorName, buf, MM::String, true);
	assert(ret == DEVICE_OK);



	// For exposure, use the shutter property.  Make sure it is present 
	// and set it to manual

	exposureTimeMs_ = ReturnFeature(m_handle, FEATURE_EXPOSURE,1000);



	LoadSupportedPaValues();

	

	CPropertyAction* pAct = new CPropertyAction(this, &Pixelink::OnBinning);


	if (m_supportedDecimationModePairs.size() > 0) 
	{
		SetBinning(0);

		CreateProperty(MM::g_Keyword_Binning, (m_supportedDecimationModePairs[0].binName).c_str(), MM::String, false, pAct, false);
	}

	for (std::size_t i = 0; i < m_supportedDecimationModePairs.size(); ++i)
	{
		AddAllowedValue(MM::g_Keyword_Binning, (m_supportedDecimationModePairs[i].binName).c_str());
	}




	PixelAddressing currentPa = GetPixelAddressing();
	currentPa.Mode = PIXEL_ADDRESSING_MODE_BIN;
	SetPixelAddressing(currentPa);

	// pixel addressing mode
	pAct = new CPropertyAction(this, &Pixelink::OnPaMode);
	CreateStringProperty("PixelAddressingMode", "Bin", false, pAct);

	vector<string> pixelAddressingMode;
	pixelAddressingMode.push_back("Bin");
	pixelAddressingMode.push_back("Decimate");

	SetAllowedValues("PixelAddressingMode", pixelAddressingMode);




	//CreateExposureSlider();
	CreateGainSlider();


	LoadSupportedPixelFormats();


	//if ((m_supportedPixelFormats[0]) == PIXEL_FORMAT_)

	//bitDepth_ = 8;

	// pixel type
	pAct = new CPropertyAction(this, &Pixelink::OnPixelType);
	/*CreateStringProperty(MM::g_Keyword_PixelType, g_PixelType_8bit, false, pAct);*/

	vector<string> pixelTypeValues;




	for (std::size_t i = 0; i < m_supportedPixelFormats.size(); ++i)
	{

		if (SupportedPixelType(m_supportedPixelFormats[i]))
		{
			pixelTypeValues.push_back(PixelTypeAsString(m_supportedPixelFormats[i], false).c_str());
		}
	}

	CreateStringProperty(MM::g_Keyword_PixelType, (PixelFormatAsString(m_supportedPixelFormats[2]).c_str()), false, pAct);
	SetAllowedValues(MM::g_Keyword_PixelType, pixelTypeValues);












	std::string colorType = "MonoChrome";
	if (IsColourCamera()) {
		colorType = "Color";
	}
	ret = CreateProperty(g_ColorMonoChrome, colorType.c_str(), MM::String, true);




	// Start Camera Stream so that Snaps can succeed
	rc = PxLSetStreamState(m_handle, START_STREAM);
	if (!(API_SUCCESS(rc)))
	{
		return rc;
	}

	isStreaming = true;


	GenerateEmptyImage(img_);


	// Make sure that we have an image so that 
	// things like bitdepth are set correctly
	//SetExposure(30.0);
	unsigned short nrTries = 0;
	do {
		ret = SnapImage(); // first snap often times out, ignore error
		nrTries++;
	} while (ret != DEVICE_OK && nrTries < 10);
	if (ret != DEVICE_OK) {
		return ret;
	}

	unsigned int size = frameDesc.uSize;
	unsigned int width = frameDesc.Roi.fWidth / frameDesc.PixelAddressingValue.fHorizontal;
	unsigned int height = frameDesc.Roi.fHeight / frameDesc.PixelAddressingValue.fVertical;
	unsigned int pixels = width * height;
	bytesPerPixel_ = unsigned short(size / pixels);

	if (ret != DEVICE_OK)
	{
		return ret;
	}


	//-------------------------------------------------------------------------
	// synchronize all properties
	ret = UpdateStatus();




	return ret;
}

bool  Pixelink::SupportedPixelType(int i) {

	switch (i)
	{
	case PIXEL_FORMAT_MONO8:
	case PIXEL_FORMAT_BAYER8_BGGR:
	case PIXEL_FORMAT_BAYER8_GBRG:
	case PIXEL_FORMAT_BAYER8_GRBG:
	case PIXEL_FORMAT_BAYER8_RGGB:
	case PIXEL_FORMAT_MONO16:
	case PIXEL_FORMAT_MONO12_PACKED:
	case PIXEL_FORMAT_BAYER16_BGGR:
	case PIXEL_FORMAT_BAYER16_GBRG:
	case PIXEL_FORMAT_BAYER16_GRBG:
	case PIXEL_FORMAT_BAYER16_RGGB:
	case PIXEL_FORMAT_YUV422:
		return true;
	}



	return false;
}


void Pixelink::GenerateEmptyImage(ImgBuffer& img)
{
	std::vector<U8> frameBuffer(3000 * 3000 * 2);
	frameDesc.uSize = sizeof(frameDesc);
	PXL_RETURN_CODE rc = GetNextFrame(m_handle, (U32)frameBuffer.size(), &frameBuffer[0], &frameDesc, 5);

	img.Resize(frameDesc.Roi.fWidth/frameDesc.PixelAddressingValue.fHorizontal, frameDesc.Roi.fHeight / frameDesc.PixelAddressingValue.fVertical,1);

	MMThreadGuard g(imgPixelsLock_);
	if (img.Height() == 0 || img.Width() == 0 || img.Depth() == 0)
		return;
	unsigned char* pBuf = const_cast<unsigned char*>(img.GetPixels());
	memset(pBuf, 0, img.Height()*img.Width()*img.Depth());
}

float
Pixelink::ReturnFeature(HANDLE hCamera, U32 featureId, int multiplier)
{
	U32 flags;
	U32 numParams = 1;
	float value;

	if (!(API_SUCCESS(PxLGetFeature(hCamera, featureId, &flags, &numParams, &value))))
	{
		return -1;
	}

	return value * multiplier;
}


/***********************************************************************
* Shuts down (unloads) the device.
* Ideally this method will completely unload the device and release
* all resources.
* Shutdown() may be called multiple times in a row.
* Required by the MM::Device API.
*/
int Pixelink::Shutdown()
{
	PXL_RETURN_CODE rc = PxLSetStreamState(m_handle, STOP_STREAM);
	if (!(API_SUCCESS(rc)))
	{
		return rc;
	}


	 rc = PxLUninitialize(m_handle);
	if (!(API_SUCCESS(rc)))
	{
		return rc;
	}


	if (initialized_) {
		//cam_;
	}
	initialized_ = false;
	return DEVICE_OK;
}

float
GetPixelSize(U32 pixelFormat)
{
	float retVal = 0.0;

	switch (pixelFormat) {

	case PIXEL_FORMAT_MONO8:
	case PIXEL_FORMAT_BAYER8_GRBG:
	case PIXEL_FORMAT_BAYER8_RGGB:
	case PIXEL_FORMAT_BAYER8_GBRG:
	case PIXEL_FORMAT_BAYER8_BGGR:
		retVal = 1.0;
		break;

		// the following areused by USB2 & GigE
	case PIXEL_FORMAT_MONO12_PACKED:
	case PIXEL_FORMAT_BAYER12_GRBG_PACKED:
	case PIXEL_FORMAT_BAYER12_RGGB_PACKED:
	case PIXEL_FORMAT_BAYER12_GBRG_PACKED:
	case PIXEL_FORMAT_BAYER12_BGGR_PACKED:
		// the follwojg areused by USB3 & 10 GigE
	case PIXEL_FORMAT_MONO12_PACKED_MSFIRST:
	case PIXEL_FORMAT_BAYER12_GRBG_PACKED_MSFIRST:
	case PIXEL_FORMAT_BAYER12_RGGB_PACKED_MSFIRST:
	case PIXEL_FORMAT_BAYER12_GBRG_PACKED_MSFIRST:
	case PIXEL_FORMAT_BAYER12_BGGR_PACKED_MSFIRST:
		retVal = 1.5;
		break;

	case PIXEL_FORMAT_MONO10_PACKED_MSFIRST:
	case PIXEL_FORMAT_BAYER10_GRBG_PACKED_MSFIRST:
	case PIXEL_FORMAT_BAYER10_RGGB_PACKED_MSFIRST:
	case PIXEL_FORMAT_BAYER10_GBRG_PACKED_MSFIRST:
	case PIXEL_FORMAT_BAYER10_BGGR_PACKED_MSFIRST:
		retVal = 1.25;
		break;


	case PIXEL_FORMAT_YUV422:
	case PIXEL_FORMAT_MONO16:
	case PIXEL_FORMAT_BAYER16_GRBG:
	case PIXEL_FORMAT_BAYER16_RGGB:
	case PIXEL_FORMAT_BAYER16_GBRG:
	case PIXEL_FORMAT_BAYER16_BGGR:
		retVal = 2.0;
		break;

	case PIXEL_FORMAT_RGB24:
	case PIXEL_FORMAT_BGR24:
		retVal = 3.0;
		break;

	case PIXEL_FORMAT_RGB48:
		retVal = 6.0;
		break;

	default:
		assert(0);
		break;
	}
	return retVal;
}

//
// Query the camera for region of interest (ROI), decimation, and pixel format
// Using this information, we can calculate the size of a raw image
//
// Returns 0 on failure
//
U32
DetermineRawImageSize(HANDLE hCamera)
{
	float parms[4];		// reused for each feature query
	float parmsPA[4];		// reused for each feature query
	U32 roiWidth;
	U32 roiHeight;
	U32 pixelAddressingValue;		// integral factor by which the image is reduced
	U32 pixelFormat;
	U32 numPixels;
	float pixelSize;
	U32 flags = FEATURE_FLAG_MANUAL;
	U32 numParams;

	assert(0 != hCamera);

	// Get region of interest (ROI)
	numParams = 4; // left, top, width, height
	PxLGetFeature(hCamera, FEATURE_ROI, &flags, &numParams, &parms[0]);
	roiWidth = (U32)parms[FEATURE_ROI_PARAM_WIDTH];
	roiHeight = (U32)parms[FEATURE_ROI_PARAM_HEIGHT];

	// Query pixel addressing 
	numParams = 4; // pixel addressing value, pixel addressing type (e.g. bin, average, ...)
	PxLGetFeature(hCamera, FEATURE_PIXEL_ADDRESSING, &flags, &numParams, &parmsPA[0]);
	pixelAddressingValue = (U32)parmsPA[FEATURE_PIXEL_ADDRESSING_PARAM_VALUE];

	// We can calulate the number of pixels now.
	numPixels = (roiWidth / (U32)parmsPA[FEATURE_PIXEL_ADDRESSING_PARAM_X_VALUE]) * (roiHeight / (U32)parmsPA[FEATURE_PIXEL_ADDRESSING_PARAM_Y_VALUE]) ;

	// Knowing pixel format means we can determine how many bytes per pixel.
	numParams = 1;
	PxLGetFeature(hCamera, FEATURE_PIXEL_FORMAT, &flags, &numParams, &parms[0]);
	pixelFormat = (U32)parms[0];

	// And now the size of the frame
	pixelSize = GetPixelSize(pixelFormat);

	return (U32)((float)numPixels * pixelSize);
}

/***********************************************************************
* Performs exposure and grabs a single image.
* This function should block during the actual exposure and return immediately afterwards
* (i.e., before readout).  This behavior is needed for proper synchronization with the shutter.
* Required by the MM::Camera API.
*
* Camera is continuously capturing.  In software trigger mode, we wait for the camera to be
* ready for a trigger, then send the software trigger, then wait to retrieve the image.
* it is potentially possible to skip the wait for the image itself, but some heuristics would
* be needed (at least, I could not find anything in the API about this).
* Readout times have gotten pretty short, so this issues is only important for very short
* exposure times.
* In external trigger mode, the code simply waits for the image.  The grabtimeout that was previously
* set determines whether or not the camera gives up.
* In internal trigger mode, the first received image is discarded and we wait for a second one.
* Exposure of the first one most likely was started before the shutter opened.
*/
int Pixelink::SnapImage()
{


	U32 rawImageSize = DetermineRawImageSize(m_handle);

	char* frameBuffer;

	frameBuffer = (char*)malloc(rawImageSize);

	PXL_RETURN_CODE rc = GetNextFrame(m_handle, rawImageSize, &frameBuffer[0], &frameDesc, 5);



	if (!isColour)
	{
		if (bitDepth_ == 8)
		{
			img_.Resize(frameDesc.Roi.fWidth / frameDesc.PixelAddressingValue.fHorizontal, frameDesc.Roi.fHeight / frameDesc.PixelAddressingValue.fVertical);

			long nrPixels = img_.Width() * img_.Height();

			unsigned char* pBuf = (unsigned char*) const_cast<unsigned char*>(img_.GetPixels());

			for (long i = 0; i < nrPixels; i++)
			{
				*(pBuf + i) = (unsigned char)frameBuffer[i];
			}
		}
		else if (bitDepth_ == 12)
		{
			int h = frameDesc.Roi.fHeight / frameDesc.PixelAddressingValue.fVertical;
			int w = frameDesc.Roi.fWidth / frameDesc.PixelAddressingValue.fHorizontal;

			unsigned char* pBuf = (unsigned char*) const_cast<unsigned char*>(img_.GetPixels());

			pPixel = Mono12ToBuffer((unsigned char*)&frameBuffer[0], w, h);

			for (long i = 0; i < w * h * 2; i++)
			{
				memcpy((void*)(pBuf + i), &pPixel[i], 1);
			}

		}
		else if (bitDepth_ == 16)
		{
			int h = frameDesc.Roi.fHeight / frameDesc.PixelAddressingValue.fVertical;
			int w = frameDesc.Roi.fWidth / frameDesc.PixelAddressingValue.fHorizontal;

			unsigned char* pBuf = (unsigned char*) const_cast<unsigned char*>(img_.GetPixels());

			pPixel = Mono16ToBuffer((unsigned char*)&frameBuffer[0], w, h);

			for (long i = 0; i < w * h * 2; i++)
			{
				memcpy((void*)(pBuf + i), &pPixel[i], 1);
			}

		}

	}
	else
	{
		if (bitDepth_ == 8)
		{
			img_.Resize(frameDesc.Roi.fWidth / frameDesc.PixelAddressingValue.fHorizontal, frameDesc.Roi.fHeight / frameDesc.PixelAddressingValue.fVertical, 4);

			U32 destSize = (frameDesc.Roi.fWidth / frameDesc.PixelAddressingValue.fHorizontal) * (frameDesc.Roi.fHeight / frameDesc.PixelAddressingValue.fVertical) * 3;
			std::vector<byte> dest;

			dest.resize(destSize);
			// Allocate memory and format the image.
			rc = PxLFormatImage(&frameBuffer[0], &frameDesc, IMAGE_FORMAT_RAW_BGR24_NON_DIB, &dest[0], &destSize);

			long nrPixels = img_.Width() * img_.Height();

			unsigned char* pBuf = (unsigned char*) const_cast<unsigned char*>(img_.GetPixels());

			unsigned char* outPutData = &dest[0];
			for (long i = 0; i < nrPixels; i++)
			{
				memcpy((void*)(pBuf + (4 * i) + 0), &dest[(i * 3) + 0], 1);
				memcpy((void*)(pBuf + (4 * i) + 1), &dest[(i * 3) + 1], 1);
				memcpy((void*)(pBuf + (4 * i) + 2), &dest[(i * 3) + 2], 1);
			}
		}
		else if (bitDepth_ == 16)
		{
			std::vector<byte> dest;
			U32 destSize = 0;
			// Find out how much space we need to allocate to hold the formatted image.
			PXL_RETURN_CODE rc = PxLFormatImage(&frameBuffer[0], &frameDesc, IMAGE_FORMAT_RAW_RGB48, NULL, &destSize);

			// Allocate memory and format the image.
			dest.resize(destSize);

			rc = PxLFormatImage(&frameBuffer[0], &frameDesc, IMAGE_FORMAT_RAW_RGB48, &dest[0], &destSize);

			ERROR_REPORT errorReport;
			if (!API_SUCCESS(rc))
			{
				PxLGetErrorReport(m_handle, &errorReport);
				int j = 1;
			}

			int h = frameDesc.Roi.fHeight / frameDesc.PixelAddressingValue.fVertical;
			int w = frameDesc.Roi.fWidth / frameDesc.PixelAddressingValue.fHorizontal;

			pPixel = RGB48ToRGBA((unsigned char*)&dest[0], w, h);

			unsigned char* pBuf = (unsigned char*) const_cast<unsigned char*>(img_.GetPixels());

			for (long i = 0; i < w * h * 8; i++)
			{
				memcpy((void*)(pBuf + i), &pPixel[i], 1);
			}

			dest.clear();
		}
	}



	return DEVICE_OK;
}

std::vector<U8> frameBuffer(5000 * 5000 * 2);


/***********************************************************************
* Inserts Image and MetaData into MMCore circular Buffer
*/
int Pixelink::InsertImage() 
{

	int ret = DEVICE_OK;




	imageCounter_++;

	unsigned int w ;
	unsigned int h;
	unsigned int b;


	

	FRAME_DESC frameDesc1;

	frameDesc1.uSize = sizeof(frameDesc1);
	//PXL_RETURN_CODE rc = GetNextFrame(m_handle, (U32)frameBuffer.size(), &frameBuffer[0], &frameDesc1, 5);

	PXL_RETURN_CODE rc = PxLGetNextFrame(m_handle, (U32)frameBuffer.size(), &frameBuffer[0], &frameDesc1);

	if (rc != 0)
	{
		return 1;
	}

	if (!API_SUCCESS(rc)) 
	{
		ERROR_REPORT errorReport;
		PxLGetErrorReport(m_handle, &errorReport);
		return ret;
	}



	w = frameDesc1.Roi.fWidth / frameDesc.PixelAddressingValue.fHorizontal;
	h = frameDesc1.Roi.fHeight / frameDesc.PixelAddressingValue.fVertical;
	b = GetImageBytesPerPixel();


	unsigned char* pPixel;
	//////////////////////

	if (!isColour)
	{
		if (bitDepth_ == 16)
		{
			pPixel = Mono16ToBuffer((unsigned char*)&frameBuffer[0], w, h);
		}
		else if (bitDepth_ == 12)
		{
			pPixel = Mono12ToBuffer((unsigned char*)&frameBuffer[0], w, h);
		}
		else
		{
			pPixel = (unsigned char*)&frameBuffer[0];
		}
	}
	else
	{

		if (bitDepth_ == 8)
		{
			std::vector<byte> dest;
			U32 destSize = 0;
			// Find out how much space we need to allocate to hold the formatted image.
			rc = PxLFormatImage(&frameBuffer[0], &frameDesc1, IMAGE_FORMAT_RAW_BGR24, NULL, &destSize);

			// Allocate memory and format the image.
			dest.resize(destSize);

			PxLFormatImage(&frameBuffer[0], &frameDesc1, IMAGE_FORMAT_RAW_BGR24, &dest[0], &destSize);

			pPixel = RGB24ToRGBA((unsigned char*)&dest[0], w, h);

			dest.clear();
		}
		else if (bitDepth_ == 16)
		{
			std::vector<byte> dest;
			U32 destSize = 0;
			// Find out how much space we need to allocate to hold the formatted image.
			PXL_RETURN_CODE rc = PxLFormatImage(&frameBuffer[0], &frameDesc1, IMAGE_FORMAT_RAW_RGB48, NULL, &destSize);

			// Allocate memory and format the image.
			dest.resize(destSize);

			rc = PxLFormatImage(&frameBuffer[0], &frameDesc1, IMAGE_FORMAT_RAW_RGB48, &dest[0], &destSize);

			ERROR_REPORT errorReport;
			if (!API_SUCCESS(rc))
			{
				PxLGetErrorReport(m_handle, &errorReport);
				int j = 1;
			}

			pPixel = RGB48ToRGBA((unsigned char*)&dest[0], w, h);

			dest.clear();
		}
		

	}

	int frameCounter = frameDesc1.uFrameNumber;


	

	Metadata md;
	//md.put(MM::g_Keyword_Metadata_StartTime, CDeviceUtils::ConvertToString(sequenceStartTime_.getMsec()));
	md.put(MM::g_Keyword_Metadata_ImageNumber, CDeviceUtils::ConvertToString(imageCounter_));
	md.put("FrameCounter", frameCounter);




	//GetCoreCallback()->ClearImageBuffer(this);
	ret = GetCoreCallback()->InsertImage(this, pPixel, w, h, b, md.Serialize().c_str(), false);

	if (ret == DEVICE_BUFFER_OVERFLOW)
	{
		// do not stop on overflow - just reset the buffer
		GetCoreCallback()->ClearImageBuffer(this);
		GetCoreCallback()->InsertImage(this, pPixel, w, h, b, md.Serialize().c_str(), false);
		return DEVICE_OK;
	}


	
	//frameBuffer.clear();


	return ret;
}


unsigned char* Pixelink::RGB24ToRGBA(const unsigned char* img, int width, int height) 
{

	const unsigned long newImageSize = width *
		height *
		GetImageBytesPerPixel();

	if (newImageSize > bufSize_)
	{
		if (imgBuf_ != 0)
		{
			delete[](imgBuf_);
		}
		unsigned long* b;
		b = (unsigned long*)&bufSize_;
		*b = width * height * 4;
		unsigned char** c;
		c = (unsigned char**)&imgBuf_;
		*c = new unsigned char[bufSize_];
	}
	// go from RGB to ABGR, there may be a more efficient way
	for (unsigned long i = 0; i < width * height; i++)
	{
		memcpy((void*)(imgBuf_ + (4 * i) + 0), img + (i * 3) + 0, 1);
		memcpy((void*)(imgBuf_ + (4 * i) + 1), img + (i * 3) + 1, 1);
		memcpy((void*)(imgBuf_ + (4 * i) + 2), img + (i * 3) + 2, 1);
	}
	return imgBuf_;
}


unsigned char* Pixelink::RGB48ToRGBA(const unsigned char* img, int width, int height)
{

	const unsigned long newImageSize = width *
		height *
		GetImageBytesPerPixel();
	if (newImageSize > bufSize_)
	{
		if (imgBuf_ != 0)
		{
			delete[](imgBuf_);
		}
		unsigned long* b;
		b = (unsigned long*)&bufSize_;
		*b = width * height * GetImageBytesPerPixel();
		unsigned char** c;
		c = (unsigned char**)&imgBuf_;
		*c = new unsigned char[bufSize_];
	}
	// go from RGB to ABGR, there may be a more efficient way
	for (unsigned long i = 0; i < width * height; i++)
	{
		/*byte zero = *(img + (i * 3) + 0);
		byte one = *(img + (i * 3) + 1);
		byte two = *(img + (i * 3) + 2);*/

		byte zero = 0;
		byte one = 255;
	


		memcpy((void*)(imgBuf_ + (8 * i) + 0), img + (i * 6) + 4, 1);//BLUE CHANNEL
		memcpy((void*)(imgBuf_ + (8 * i) + 1), img + (i * 6) + 5, 1);

		memcpy((void*)(imgBuf_ + (8 * i) + 2), img + (i * 6) + 2, 1);//GREEN CHANNEL
		memcpy((void*)(imgBuf_ + (8 * i) + 3), img + (i * 6) + 3, 1);

		memcpy((void*)(imgBuf_ + (8 * i) + 4), img + (i * 6) + 0, 1);//RED CHANNEL
		memcpy((void*)(imgBuf_ + (8 * i) + 5), img + (i * 6) + 1, 1);

	}
	return imgBuf_;
}



unsigned char* Pixelink::Mono16ToBuffer(const unsigned char* img, int width, int height)
{

	const unsigned long newImageSize = width *
		height *
		GetImageBytesPerPixel();
	if (newImageSize > bufSize_)
	{
		if (imgBuf_ != 0)
		{
			delete[](imgBuf_);
		}
		unsigned long* b;
		b = (unsigned long*)&bufSize_;
		*b = width * height * GetImageBytesPerPixel();
		unsigned char** c;
		c = (unsigned char**)&imgBuf_;
		*c = new unsigned char[bufSize_];
	}
	// swap the bits
	for (unsigned long i = 0; i < width * height; i++)
	{
		byte zero = *(img + (i * 2) + 0);
		byte one  = *(img + (i * 2) + 1);

		memcpy((void*)(imgBuf_ + (2 * i) + 0), &one , 1);
		memcpy((void*)(imgBuf_ + (2 * i) + 1), &zero, 1);
	}
	return imgBuf_;
}


unsigned char* Pixelink::Mono12ToBuffer(const unsigned char* img, int width, int height)
{

	const unsigned long newImageSize = width *
		height *
		GetImageBytesPerPixel();
	if (newImageSize > bufSize_)
	{
		if (imgBuf_ != 0)
		{
			delete[](imgBuf_);
		}
		unsigned long* b;
		b = (unsigned long*)&bufSize_;
		*b = width * height * GetImageBytesPerPixel();
		unsigned char** c;
		c = (unsigned char**)&imgBuf_;
		*c = new unsigned char[bufSize_];
	}
	
	int frameRows = height;
	int bytesPerRow = width + (width / 2);

	for (int y = 0; y < frameRows - 1; y += 1)
	{
		for (int x = 0; x < bytesPerRow - 1; x += 3)
		{

			int i = (x * 2 / 3) + (width * y);

			byte zero = (*(img + y*bytesPerRow + x) >> 4) & 0x000F;
			byte one = (*(img + y*bytesPerRow + x) << 4)  |  (*(img + y*bytesPerRow + x + 2) & 0x000F);

			memcpy((void*)(imgBuf_ + (2 * i) + 0), &one, 1);
			memcpy((void*)(imgBuf_ + (2 * i) + 1), &zero, 1);



			i = (x * 2 / 3) + (width * y) + 1;

			zero = (*(img + y*bytesPerRow + x + 1) >> 4) & 0x000F;
			one = (*(img + y*bytesPerRow + x + 1) << 4)    |    (*(img + y*bytesPerRow + x + 2) & 0x000F);


			memcpy((void*)(imgBuf_ + (2 * i) + 0), &one, 1);
			memcpy((void*)(imgBuf_ + (2 * i) + 1), &zero, 1);


		}
	}




	return imgBuf_;
}

//
//inline U16
//Unpack12Bitg(const U8* const data, const bool even, int pixelSize)
//{
//	// even is refernce to an even (first) or odd (second) byte in the packed sequence
//	if (even) {
//		//    MS 8 bits              LS 4 bits
//		//  ----------------      -------------------
//		if (pixelSize == 10)
//			return (((U16)data[0]) << 2 | (((U16)data[2]) & 0x000F) >> 2);
//		else
//			return (((U16)data[0]) << 4 | ((U16)data[2]) & 0x000F);
//	}
//	else {
//		if (pixelSize == 10)
//			return (((U16)data[1]) << 2 | ((U16)data[2]) >> 6);
//		else
//			return (((U16)data[1]) << 4 | ((U16)data[2]) >> 4);
//	}
//}
//


//
// A robust wrapper around PxLGetNextFrame.
// This will handle the occasional error that can be returned by the API
// because of timeouts. 
//
// Note that this should only be called when grabbing images from 
// a camera NOT currently configured for triggering. 
//
PXL_RETURN_CODE
Pixelink::GetNextFrame(const HANDLE hCamera, const U32 frameBufferSize, void* const pFrameBuffer, FRAME_DESC* const pFrameDesc, const U32 maximumNumberOfTries) const
{


	// Record the frame desc size in case we need it later
	const U32 frameDescSize = pFrameDesc->uSize;

	PXL_RETURN_CODE rc = ApiUnknownError;

	for (U32 i = 0; i < maximumNumberOfTries; i++) {
		rc = PxLGetNextFrame(hCamera, frameBufferSize, pFrameBuffer, pFrameDesc);
		if (API_SUCCESS(rc)) {
			return rc;
		}
		else {
			// If the streaming is turned off, no sense in continuing.
			if (ApiStreamStopped == rc) {
				return rc;
			}
			else {
				// Is the camera still connected? Try reading the exposure.
				// If the user cannot disconnect the camera, you can skip this.
				U32 flags;
				float exposure;
				U32 numParams = 1;
				PXL_RETURN_CODE rcExposure = PxLGetFeature(hCamera, FEATURE_SHUTTER, &flags, &numParams, &exposure);
				if (!API_SUCCESS(rcExposure)) {
					return rcExposure;
				}
			}
		}
		// Camera's still there, so maybe we just hit a bubble in the frame pipeline.
		// Reset the frame descriptor uSize field (in case the API is newer than what we were compiled with) and try PxLGetNextFrame again.
		pFrameDesc->uSize = frameDescSize;
	}

	// Ran out of tries, so return whatever the last error was.
	return rc;
}



/***********************************************************************
* Returns pixel data.
* Required by the MM::Camera API.
* The calling program will assume the size of the buffer based on the values
* obtained from GetImageBufferSize(), which in turn should be consistent with
* values returned by GetImageWidth(), GetImageHight() and GetImageBytesPerPixel().
* The calling program allso assumes that camera never changes the size of
* the pixel buffer on its own. In other words, the buffer can change only if
* appropriate properties are set (such as binning, pixel type, etc.)
*/
const unsigned char* Pixelink::GetImageBuffer()
{
	if (nComponents_ == 1) {
		// This seems to work without a DeepCopy first
		return img_.GetPixels();
	}
	else if (nComponents_ == 4) {
		return img_.GetPixels();
	}
	return img_.GetPixels();



}

/***********************************************************************
* Returns image buffer X-size in pixels.
* Required by the MM::Camera API.
*/
unsigned int Pixelink::GetImageWidth() const
{
	return frameDesc.Roi.fWidth / frameDesc.PixelAddressingValue.fHorizontal;
}

/***********************************************************************
* Returns image buffer Y-size in pixels.
* Required by the MM::Camera API.
*/
unsigned int Pixelink::GetImageHeight() const
{
	return frameDesc.Roi.fHeight / frameDesc.PixelAddressingValue.fVertical;
}

/***********************************************************************
* Returns image buffer pixel depth in bytes.
* Required by the MM::Camera API.
*/
unsigned int Pixelink::GetImageBytesPerPixel() const
{
	//return GetPixelFormatByteSize(frameDesc.PixelFormat.fValue);
	return img_.Depth();
}

/**
* Function: GetPixelFormatSize
* Purpose:  Return the number of bytes per pixel for a pixel format.
*/
unsigned int Pixelink::GetPixelFormatByteSize(int pixelFormat) const
{
	switch (pixelFormat)
	{
	case PIXEL_FORMAT_MONO8:
	case PIXEL_FORMAT_BAYER8_BGGR:
	case PIXEL_FORMAT_BAYER8_GBRG:
	case PIXEL_FORMAT_BAYER8_GRBG:
	case PIXEL_FORMAT_BAYER8_RGGB:
		return 1;
	case PIXEL_FORMAT_MONO16:
	case PIXEL_FORMAT_MONO12_PACKED:
	case PIXEL_FORMAT_BAYER16_BGGR:
	case PIXEL_FORMAT_BAYER16_GBRG:
	case PIXEL_FORMAT_BAYER16_GRBG:
	case PIXEL_FORMAT_BAYER16_RGGB:
	case PIXEL_FORMAT_YUV422:
		return 2;
	case PIXEL_FORMAT_RGB24:
		return 3;
	case PIXEL_FORMAT_RGB48:
		return 6;

	}
	return 1;
}



int Pixelink::GetPixelFormatSize(int pixelFormat) const
{
	switch (pixelFormat)
	{
	case PIXEL_FORMAT_MONO8:
	case PIXEL_FORMAT_BAYER8_BGGR:
	case PIXEL_FORMAT_BAYER8_GBRG:
	case PIXEL_FORMAT_BAYER8_GRBG:
	case PIXEL_FORMAT_BAYER8_RGGB:
		return 8;
	case PIXEL_FORMAT_MONO16:
	case PIXEL_FORMAT_BAYER16_BGGR:
	case PIXEL_FORMAT_BAYER16_GBRG:
	case PIXEL_FORMAT_BAYER16_GRBG:
	case PIXEL_FORMAT_BAYER16_RGGB:
	case PIXEL_FORMAT_YUV422:
		return 16;
	case PIXEL_FORMAT_RGB24:
		return 24;
	case PIXEL_FORMAT_RGB48:
		return 48;
	case PIXEL_FORMAT_MONO12_PACKED:
	case PIXEL_FORMAT_BAYER12_BGGR_PACKED:
	case PIXEL_FORMAT_BAYER12_GBRG_PACKED:
	case PIXEL_FORMAT_BAYER12_GRBG_PACKED:
	case PIXEL_FORMAT_BAYER12_RGGB_PACKED:
	case PIXEL_FORMAT_MONO12_PACKED_MSFIRST:
	case PIXEL_FORMAT_BAYER12_BGGR_PACKED_MSFIRST:
	case PIXEL_FORMAT_BAYER12_GBRG_PACKED_MSFIRST:
	case PIXEL_FORMAT_BAYER12_GRBG_PACKED_MSFIRST:
	case PIXEL_FORMAT_BAYER12_RGGB_PACKED_MSFIRST:
		return 12;

	}
	return -1;
}

/***********************************************************************
* Returns the size in bytes of the image buffer.
* Required by the MM::Camera API.
*/
long Pixelink::GetImageBufferSize() const
{
	return frameDesc.uSize;
}

/***********************************************************************
* Sets the camera Region Of Interest.
* Required by the MM::Camera API.
* This command will change the dimensions of the image.
* @param x - top-left corner coordinate
* @param y - top-left corner coordinate
* @param xSize - width
* @param ySize - height
*/
int Pixelink::SetROI(unsigned x, unsigned y, unsigned xSize, unsigned ySize)
{

	SetFeatureRoi(x, y, xSize, ySize, true,false);


	//SetRoi(x, y, xSize, ySize,false);

	// Make sure that we have an image so that 
	// things like bitdepth are set correctly
	int ret = SnapImage();
	if (ret != DEVICE_OK) {
		return ret;
	}

	return DEVICE_OK;;
}

/***********************************************************************
* Returns the actual dimensions of the current ROI.
* Required by the MM::Camera API.
*/
int Pixelink::GetROI(unsigned& x, unsigned& y, unsigned& xSize, unsigned& ySize)
{

		x = frameDesc.Roi.fLeft;
		y = frameDesc.Roi.fTop;
		xSize = frameDesc.Roi.fWidth;
		ySize = frameDesc.Roi.fHeight;

		return DEVICE_OK;
	


}

int Pixelink::SetGain(float gain) 
{
	U32 flags = 0;
	U32 numParms = 1;
	std::vector<float> parms(numParms, 0);

	PXL_RETURN_CODE rc = PxLGetFeature(m_handle, FEATURE_GAIN, &flags, &numParms, &parms[0]);

	parms[0] = gain;

	bool cameraStreaming = false;

	rc = PxLSetFeature(m_handle, FEATURE_GAIN, flags, numParms, &parms[0]);

	return 0;
}

float Pixelink::GetGain() 
{
	U32 flags = 0;
	U32 numParms = 1;
	std::vector<float> parms(numParms, 0);

	PXL_RETURN_CODE rc = PxLGetFeature(m_handle, FEATURE_GAIN, &flags, &numParms, &parms[0]);


	return parms[0];
}




float Pixelink::GetPixelFormat()
{
	U32 flags = 0;
	U32 numParms = 1;
	std::vector<float> parms(numParms, 0);

	PXL_RETURN_CODE rc = PxLGetFeature(m_handle, FEATURE_PIXEL_FORMAT, &flags, &numParms, &parms[0]);


	return parms[0];
}



/***********************************************************************
* Resets the Region of Interest to full frame.
* Required by the MM::Camera API.
*/
int Pixelink::ClearROI()
{
	PCAMERA_FEATURES camFeature;
	U32 size = 0;
	PxLGetCameraFeatures(m_handle, FEATURE_ROI, NULL, &size);
	// Allocate the amount of memory that we were told is required.
	camFeature = reinterpret_cast<CAMERA_FEATURES*>(new byte[size]);

	PXL_RETURN_CODE rc = PxLGetCameraFeatures(m_handle, FEATURE_ROI, camFeature, &size);


	ERROR_REPORT errorReport;
	if (!API_SUCCESS(rc))
	{
		PxLGetErrorReport(m_handle, &errorReport);
		int j = 1;
	}


	float parms[4] = { 0,0,0,0 };
	parms[0] = camFeature[0].pFeatures[0].pParams[0].fMinValue;
	parms[1] = camFeature[0].pFeatures[0].pParams[1].fMinValue;
	parms[2] = camFeature[0].pFeatures[0].pParams[2].fMaxValue;
	parms[3] = camFeature[0].pFeatures[0].pParams[3].fMaxValue;

	SetFeatureRoi(parms[0], parms[1], parms[2], parms[3], true,true);



	// Make sure that we have an image so that 
	// things like bitdepth are set correctly
	int ret = SnapImage();
	if (ret != DEVICE_OK) {
		return ret;
	}

	return DEVICE_OK;
}


void Pixelink::SetFeatureRoi(float left, float top, float width, float height, bool cameraStreaming, bool isClearingRoi)
{
	U32 flags = 0;
	U32 numParms = 4;
	std::vector<float> parms(numParms, 0);

	PXL_RETURN_CODE rc = PxLGetFeature(m_handle, FEATURE_ROI, &flags, &numParms, &parms[0]);

	if (isClearingRoi == false)
	{
		parms[0] = left * frameDesc.PixelAddressingValue.fHorizontal;
		parms[1] = top * frameDesc.PixelAddressingValue.fVertical;
		parms[2] = width * frameDesc.PixelAddressingValue.fHorizontal;
		parms[3] = height * frameDesc.PixelAddressingValue.fVertical;
	}
	else
	{
		parms[0] = left;
		parms[1] = top;
		parms[2] = width;
		parms[3] = height;
	}

	if (isStreaming == true)
	{
		PxLSetStreamState(m_handle, STOP_STREAM);
		rc = PxLSetFeature(m_handle, FEATURE_ROI, flags, numParms, &parms[0]);
		PxLSetStreamState(m_handle, START_STREAM);
	}
	else 
	{
		rc = PxLSetFeature(m_handle, FEATURE_ROI, flags, numParms, &parms[0]);
	}



	ERROR_REPORT errorReport;
	if (!API_SUCCESS(rc))
	{
		PxLGetErrorReport(m_handle, &errorReport);
	}



}


/***********************************************************************
* Returns the current exposure setting in milliseconds.
* Required by the MM::Camera API.
*/
double Pixelink::GetExposure() const
{
	// Since this function is const, we can not use the cam_ object.
	// Hence, the only way to report exposure time is to cache it when we 
	// we set it
	U32 flags = 0;
	U32 numParms = 1;
	std::vector<float> parms(numParms, 0);

	PXL_RETURN_CODE rc = PxLGetFeature(m_handle, FEATURE_EXPOSURE, &flags, &numParms, &parms[0]);

	return (double)parms[0] * 1000;
}

/***********************************************************************
* Sets exposure in milliseconds.
* Required by the MM::Camera API.
*/
void Pixelink::SetExposure(double exp)
{
	if (exp != 0) 
	{
		U32 flags = 0;
		U32 numParms = 3;

		// Allocate memory. All parameters are returned as floats.
		std::vector<float> parms(numParms, 0.0f);

		PXL_RETURN_CODE rc = PxLGetFeature(m_handle, FEATURE_EXPOSURE, &flags, &numParms, &parms[0]);

		float parms2[1] = { exp / 1000.0f };

		rc = PxLSetFeature(m_handle, FEATURE_EXPOSURE, flags, 1, parms2);

		exposureTimeMs_ = exp;

		SetProperty(MM::g_Keyword_Exposure, CDeviceUtils::ConvertToString(exp));
		GetCoreCallback()->OnExposureChanged(this, exp);;

	}
	return;
}


/***********************************************************************
* Required by the MM::Camera API.
*/
void Pixelink::SetCameraDevice(const char* cameraLabel)
{

	return;
}

/***********************************************************************
* Returns the current exposure setting in milliseconds.
* Required by the MM::Camera API.
*/
double Pixelink::GetCameraDevice() const
{

	return 1000;
}







/***********************************************************************
* Returns the current binning factor.
* Required by the MM::Camera API.
*/
int Pixelink::GetBinning() const
{
	if (HasProperty(g_Format7Mode))
	{
		char mode[MM::MaxStrLength];
		int ret = GetProperty(g_Format7Mode, mode);
		if (ret != DEVICE_OK)
		{
			return ret;
		}

		try
		{
			return mode2Bin_.at(mode);
		}
		catch (const std::out_of_range& /*oor*/) {
			// very ugly to use try/catch here, but I somehow can not 
			// get an iterator to compile
		}

	}

	return 1;
}


/***********************************************************************
* Sets binning factor.
* Required by the MM::Camera API.
*/
int Pixelink::SetBinning(int binF)
{
	U32 flags = 0;
	U32 numParms = 4;
	std::vector<float> parms(numParms, 0);

	PXL_RETURN_CODE rc = PxLGetFeature(m_handle, FEATURE_PIXEL_ADDRESSING, &flags, &numParms, &parms[0]);

	parms[2] = m_supportedDecimationModePairs[binF].x;
	parms[3] = m_supportedDecimationModePairs[binF].y;



	bool cameraStreaming = true;

	if (isStreaming == true)
	{
		PxLSetStreamState(m_handle, STOP_STREAM);
		rc = PxLSetFeature(m_handle, FEATURE_PIXEL_ADDRESSING, flags, numParms, &parms[0]);
		PxLSetStreamState(m_handle, START_STREAM);

		UpdateFrameBufferSize(parms[2],parms[3]);

	}
	else
	{
		rc = PxLSetFeature(m_handle, FEATURE_PIXEL_ADDRESSING, flags, numParms, &parms[0]);
		UpdateFrameBufferSize(parms[2], parms[3]);
	}


	// not sure if we should return an error code here
	return DEVICE_OK;
}

/***********************************************************************
* Required by the MM::Camera API
*/
int Pixelink::StartSequenceAcquisition(double interval)
{
	return StartSequenceAcquisition(LONG_MAX, interval, false);
}

/**
* Stop and wait for the Sequence thread finished
*/
int Pixelink::StopSequenceAcquisition()
{
	if (!thd_->IsStopped()) {
		thd_->Stop();
		thd_->wait();
	}


	isCapturing_ = false;


	//Stop Pixelink Callback
	//PxLSetCallback(m_handle, CALLBACK_FRAME, NULL, NULL);

	int ret = DEVICE_OK;


	return GetCoreCallback()->AcqFinished(this, ret);
}

/***********************************************************************

* API.  The global function PGCallback matches the ImageEvent typedef.
* All it does (in a complicated way) is to call our InsertImage function
* that inserts the newly acquired image into the circular buffer.
* Because of the syntax, InsertImage needs to be const, which poses a few
* problems maintaining state.
*/
int Pixelink::StartSequenceAcquisition(long numImages, double interval_ms,
	bool stopOnOverflow)
{
	capturedOneFrame = false;

	if (IsCapturing())
		return DEVICE_CAMERA_BUSY_ACQUIRING;

	//Start Pixelink Callback
	//PXL_RETURN_CODE rc = PxLSetCallback(m_handle, CALLBACK_FRAME, NULL, CallbackFrameFromCamera);

	
	//int counter = 0;
	//while (capturedOneFrame == false && counter < 500) 
	//{
	//	CDeviceUtils::SleepMs(1);
	//	counter++;
	//}


	//CDeviceUtils::SleepMs(500);



	int ret = GetCoreCallback()->PrepareForAcq(this);
	if (ret != DEVICE_OK)
		return ret;
	sequenceStartTime_ = GetCurrentMMTime();
	imageCounter_ = 0;
	thd_->Start(numImages, interval_ms);
	stopOnOverflow_ = stopOnOverflow;
	return DEVICE_OK;


}


//***********************************************************************




/***********************************************************************
* Handles Binning property.
*/
int Pixelink::OnBinning(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set((long)GetBinning());
	}
	else if (eAct == MM::AfterSet)
	{
		string binning;
		pProp->Get(binning);

		int selectedBinIndex = 0;

		for (std::size_t i = 0; i < m_supportedDecimationModePairs.size(); ++i)
		{
			if (binning == m_supportedDecimationModePairs[i].binName)
			{
				selectedBinIndex = i;
			}
		}

		return SetBinning(selectedBinIndex);
	}

	return DEVICE_OK;
}

/***********************************************************************
* Handles Binning property.
*/
int Pixelink::OnPaMode(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		//pProp->Set((long)GetBinning());
	}
	else if (eAct == MM::AfterSet)
	{
		string binning;
		pProp->Get(binning);

		
		PixelAddressing currentPa = GetPixelAddressing();


		if (binning == "Bin")
		{
			currentPa.Mode = PIXEL_ADDRESSING_MODE_BIN;
		}
		else
		{
			currentPa.Mode = PIXEL_ADDRESSING_MODE_DECIMATE;
		}

		SetPixelAddressing(currentPa);
		 
	}

	return DEVICE_OK;
}

int Pixelink::OnGain(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		//pProp->Set((double)GetGain())

		string j;

	}
	else if (eAct == MM::AfterSet)
	{
		double gain;
		pProp->Get(gain);
		SetGain(gain);
		
		string j;

	}

	return DEVICE_OK;
}

int Pixelink::OnExposure(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		//pProp->Set((double)GetGain())

		string j;

	}
	else if (eAct == MM::AfterSet)
	{
		double exp;
		pProp->Get(exp);

		SetExposure(exp);

		SetProperty(MM::g_Keyword_Exposure, CDeviceUtils::ConvertToString(exp));
		GetCoreCallback()->OnExposureChanged(this, exp);;


		//double gain;
		//pProp->Get(gain);
		//SetGain(gain);

		//string j;

	}

	return DEVICE_OK;
}





static void * callbackFrameData;
static FRAME_DESC const *	callbackFrameDesc;

U32 __stdcall
Pixelink::CallbackFrameFromCamera(
	HANDLE		hCamera,
	LPVOID		pData,
	U32			dataFormat,
	FRAME_DESC const *	pFrameDesc,
	LPVOID		userData)
{

	capturedOneFrame = true;
	callbackFrameData = pData;
	callbackFrameDesc = pFrameDesc;

	return ApiSuccess;
}

bool Pixelink::capturedOneFrame;

/*
* Do actual capturing
* Called from inside the thread
*/
int Pixelink::RunSequenceOnThread(MM::MMTime startTime)
{
	int ret = DEVICE_ERR;



	ret = InsertImage();



	if (ret != DEVICE_OK)
	{
		return ret;
	}
	return ret;
};

bool Pixelink::IsCapturing() {
	return !thd_->IsStopped();
}

/*
* called from the thread function before exit
*/
void Pixelink::OnThreadExiting() throw()
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


MySequenceThread::MySequenceThread(Pixelink* pCam)
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


void
Pixelink::CreateGainSlider() 
{
	PCAMERA_FEATURES camFeature;
	U32 size = 0;
	PxLGetCameraFeatures(m_handle, FEATURE_GAIN, NULL, &size);
	// Allocate the amount of memory that we were told is required.
	camFeature = reinterpret_cast<CAMERA_FEATURES*>(new byte[size]);

	PXL_RETURN_CODE rc = PxLGetCameraFeatures(m_handle, FEATURE_GAIN, camFeature, &size);


	ERROR_REPORT errorReport;
	if (!API_SUCCESS(rc))
	{
		PxLGetErrorReport(m_handle, &errorReport);
		int j = 1;
	}


	float gainMin = camFeature[0].pFeatures[0].pParams[0].fMinValue;
	float gainMax = camFeature[0].pFeatures[0].pParams[0].fMaxValue;

	float currentGain = GetGain();

	// camera gain
	CPropertyAction* pActGain = new CPropertyAction(this, &Pixelink::OnGain);
	CreateFloatProperty("Gain", currentGain, false, pActGain, false);
	SetPropertyLimits("Gain", gainMin, gainMax);

}

bool
Pixelink::IsColourCamera() 
{
	float pixelFormat = ReturnFeature(m_handle, FEATURE_PIXEL_FORMAT, 1);

		switch ((int)pixelFormat)
		{
		case PIXEL_FORMAT_MONO8:
		case PIXEL_FORMAT_MONO16:
		case PIXEL_FORMAT_MONO12_PACKED:
		case PIXEL_FORMAT_MONO12_PACKED_MSFIRST:
			return false;

		case PIXEL_FORMAT_BAYER8_BGGR:
		case PIXEL_FORMAT_BAYER8_GBRG:
		case PIXEL_FORMAT_BAYER8_GRBG:
		case PIXEL_FORMAT_BAYER8_RGGB:
		case PIXEL_FORMAT_BAYER16_BGGR:
		case PIXEL_FORMAT_BAYER16_GBRG:
		case PIXEL_FORMAT_BAYER16_GRBG:
		case PIXEL_FORMAT_BAYER16_RGGB:
		case PIXEL_FORMAT_YUV422:
		case PIXEL_FORMAT_RGB24:
		case PIXEL_FORMAT_RGB48:
		case PIXEL_FORMAT_BAYER12_BGGR_PACKED:
		case PIXEL_FORMAT_BAYER12_GBRG_PACKED:
		case PIXEL_FORMAT_BAYER12_GRBG_PACKED:
		case PIXEL_FORMAT_BAYER12_RGGB_PACKED:
		case PIXEL_FORMAT_BAYER12_BGGR_PACKED_MSFIRST:
		case PIXEL_FORMAT_BAYER12_GBRG_PACKED_MSFIRST:
		case PIXEL_FORMAT_BAYER12_GRBG_PACKED_MSFIRST:
		case PIXEL_FORMAT_BAYER12_RGGB_PACKED_MSFIRST:
			return true;
		}

	return false;

}

/**
* Handles "PixelType" property.
*/
int Pixelink::OnPixelType(MM::PropertyBase* pProp, MM::ActionType eAct)
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


		SetPixelType(StringAsPixelType(pixelType));

		
		if (pixelType.compare(   PixelTypeAsString(PIXEL_FORMAT_MONO8,false)  ) == 0)
		{
			nComponents_ = 1;
			img_.Resize(img_.Width(), img_.Height(), 1);
			bitDepth_ = 8;
			ret = DEVICE_OK;
		}
		else if (pixelType.compare(PixelTypeAsString(PIXEL_FORMAT_MONO12_PACKED_MSFIRST, false)) == 0 || pixelType.compare(PixelTypeAsString(PIXEL_FORMAT_MONO12_PACKED, false)) == 0)
		{
			nComponents_ = 1;
			img_.Resize(img_.Width(), img_.Height(), 2);
			bitDepth_ = 12;
			ret = DEVICE_OK;
		}
		else if (pixelType.compare(PixelTypeAsString(PIXEL_FORMAT_MONO16,false) ) == 0)
		{
			nComponents_ = 1;
			img_.Resize(img_.Width(), img_.Height(), 2);
			bitDepth_ = 16;
			ret = DEVICE_OK;
		}
		else if (pixelType.compare(PixelTypeAsString(PIXEL_FORMAT_BAYER8, false)) == 0)
		{
			nComponents_ = 4;
			img_.Resize(img_.Width(), img_.Height(), 4);
			bitDepth_ = 8;
			ret = DEVICE_OK;
		}
		else if (pixelType.compare(PixelTypeAsString(PIXEL_FORMAT_BAYER16, false)) == 0 
			|| pixelType.compare(PixelTypeAsString(PIXEL_FORMAT_YUV422, false)) == 0 )
		{
			nComponents_ = 4;
			img_.Resize(img_.Width(), img_.Height(), 8);
			bitDepth_ = 16;
			ret = DEVICE_OK;
		}
		else if (pixelType.compare(PixelTypeAsString(PIXEL_FORMAT_BAYER12_PACKED, false)) == 0
			|| pixelType.compare(PixelTypeAsString(PIXEL_FORMAT_BAYER12_PACKED_MSFIRST, false)) == 0)
		{
			nComponents_ = 4;
			img_.Resize(img_.Width(), img_.Height(), 4);
			bitDepth_ = 8;
			ret = DEVICE_OK;
		}
		else
		{
			// on error switch to default pixel type
			nComponents_ = 1;
			img_.Resize(img_.Width(), img_.Height(), 1);
			pProp->Set(PixelTypeAsString(PIXEL_FORMAT_MONO8, false).c_str());
			bitDepth_ = 8;
			ret = DEVICE_ERR;
		}
	}
	break;
	case MM::BeforeGet:
	{
		long bytesPerPixel = GetImageBytesPerPixel();

		PXL_RETURN_CODE rc;
		U32 flags = FEATURE_FLAG_MANUAL;
		U32 nParms = 1;
		float parms[1] = { 0 };

		PxLGetFeature(m_handle, FEATURE_PIXEL_FORMAT, &flags, &nParms, &parms[0]);



		if ((PixelTypeAsString(parms[0], false)) == (PixelTypeAsString(PIXEL_FORMAT_MONO8, false)))
		{
			pProp->Set(PixelTypeAsString(PIXEL_FORMAT_MONO8, false).c_str());
			nComponents_ = 1;
			img_.Resize(img_.Width(), img_.Height(), 1);
			bitDepth_ = 8;
		}
		else if ((PixelTypeAsString(parms[0], false)) == (PixelTypeAsString(PIXEL_FORMAT_MONO16, false)))
		{
			pProp->Set(PixelTypeAsString(PIXEL_FORMAT_MONO16, false).c_str());
			nComponents_ = 1;
			img_.Resize(img_.Width(), img_.Height(), 2);
			bitDepth_ = 16;
		}
		else if ((PixelTypeAsString(parms[0], false)) == (PixelTypeAsString(PIXEL_FORMAT_MONO12_PACKED_MSFIRST, false)))
		{
			pProp->Set(PixelTypeAsString(PIXEL_FORMAT_MONO12_PACKED_MSFIRST, false).c_str());
			nComponents_ = 1;
			img_.Resize(img_.Width(), img_.Height(), 2);
			bitDepth_ = 12;
		}
		else if ((PixelTypeAsString(parms[0], false)) == (PixelTypeAsString(PIXEL_FORMAT_MONO12_PACKED, false)))
		{
			pProp->Set(PixelTypeAsString(PIXEL_FORMAT_MONO12_PACKED, false).c_str());
			nComponents_ = 1;
			img_.Resize(img_.Width(), img_.Height(), 2);
			bitDepth_ = 12;
		}
		else if ((PixelTypeAsString(parms[0], false)) == (PixelTypeAsString(PIXEL_FORMAT_BAYER8, false)))
		{
			pProp->Set(PixelTypeAsString(PIXEL_FORMAT_BAYER8, false).c_str());
			bitDepth_ = 8;

			nComponents_ = 4;
			img_.Resize(img_.Width(), img_.Height(), 4);

		}
		else if ((PixelTypeAsString(parms[0], false)) == (PixelTypeAsString(PIXEL_FORMAT_BAYER16, false)))
		{
			pProp->Set(PixelTypeAsString(PIXEL_FORMAT_BAYER16, false).c_str());
			nComponents_ = 4;
			img_.Resize(img_.Width(), img_.Height(), 8);
			bitDepth_ = 16;
		}
		else if ((PixelTypeAsString(parms[0], false)) == (PixelTypeAsString(PIXEL_FORMAT_YUV422, false)))
		{
			pProp->Set(PixelTypeAsString(PIXEL_FORMAT_YUV422, false).c_str());
			nComponents_ = 4;
			img_.Resize(img_.Width(), img_.Height(), 8);
			bitDepth_ = 16;
		}
		else if ((PixelTypeAsString(parms[0], false)) == (PixelTypeAsString(PIXEL_FORMAT_BAYER12_PACKED, false)))
		{
			pProp->Set(PixelTypeAsString(PIXEL_FORMAT_BAYER12_PACKED, false).c_str());
			bitDepth_ = 8;
		}
		else if ((PixelTypeAsString(parms[0], false)) == (PixelTypeAsString(PIXEL_FORMAT_BAYER12_PACKED_MSFIRST, false)))
		{
			pProp->Set(PixelTypeAsString(PIXEL_FORMAT_BAYER12_PACKED_MSFIRST, false).c_str());
			bitDepth_ = 8;
		}
		else
		{
			pProp->Set(PixelTypeAsString(PIXEL_FORMAT_MONO8, false).c_str());
		}
		ret = DEVICE_OK;
	} break;
	default:
		break;
	}
	return ret;
}


std::string Pixelink::PixelTypeAsString(int pixelFormat, bool asCodeConst) const
{
	switch (pixelFormat)
	{

	

	case PIXEL_FORMAT_MONO8:
		return asCodeConst ? ("PIXEL_FORMAT_MONO8") : ("MONO8");
	case PIXEL_FORMAT_MONO16:
		return asCodeConst ? ("PIXEL_FORMAT_MONO16") : ("MONO16");
	case PIXEL_FORMAT_YUV422:
		return asCodeConst ? ("PIXEL_FORMAT_YUV422") : ("YUV422");
	case PIXEL_FORMAT_BAYER8_GRBG:
		return asCodeConst ? ("PIXEL_FORMAT_BAYER8_GRBG") : ("BAYER8");
	case PIXEL_FORMAT_BAYER8_RGGB:
		return asCodeConst ? ("PIXEL_FORMAT_BAYER8_RGGB") : ("BAYER8");
	case PIXEL_FORMAT_BAYER8_GBRG:
		return asCodeConst ? ("PIXEL_FORMAT_BAYER8_GBRG") : ("BAYER8");
	case PIXEL_FORMAT_BAYER8_BGGR:
		return asCodeConst ? ("PIXEL_FORMAT_BAYER8_BGGR") : ("BAYER8");
	case PIXEL_FORMAT_BAYER16_GRBG:
		return asCodeConst ? ("PIXEL_FORMAT_BAYER16_GRBG") : ("BAYER16");
	case PIXEL_FORMAT_BAYER16_RGGB:
		return asCodeConst ? ("PIXEL_FORMAT_BAYER16_RGGB") : ("BAYER16");
	case PIXEL_FORMAT_BAYER16_GBRG:
		return asCodeConst ? ("PIXEL_FORMAT_BAYER16_GBRG") : ("BAYER16");
	case PIXEL_FORMAT_BAYER16_BGGR:
		return asCodeConst ? ("PIXEL_FORMAT_BAYER16_BGGR") : ("BAYER16");
	case PIXEL_FORMAT_RGB24:
		return asCodeConst ? ("PIXEL_FORMAT_RGB24") : ("RGB24");
	case PIXEL_FORMAT_RGB48:
		return asCodeConst ? ("PIXEL_FORMAT_RGB48") : ("RGB48");
	case PIXEL_FORMAT_MONO12_PACKED:
		return asCodeConst ? ("PIXEL_FORMAT_MONO12_PACKED") : ("MONO12_PACKED");
	case PIXEL_FORMAT_BAYER12_GRBG_PACKED:
		return asCodeConst ? ("PIXEL_FORMAT_BAYER12_GRBG_PACKED") : ("BAYER12_PACKED");
	case PIXEL_FORMAT_BAYER12_RGGB_PACKED:
		return asCodeConst ? ("PIXEL_FORMAT_BAYER12_RGGB_PACKED") : ("BAYER12_PACKED");
	case PIXEL_FORMAT_BAYER12_GBRG_PACKED:
		return asCodeConst ? ("PIXEL_FORMAT_BAYER12_GBRG_PACKED") : ("BAYER12_PACKED");
	case PIXEL_FORMAT_BAYER12_BGGR_PACKED:
		return asCodeConst ? ("PIXEL_FORMAT_BAYER12_BGGR_PACKED") : ("BAYER12_PACKED");
	case PIXEL_FORMAT_MONO12_PACKED_MSFIRST:
		return asCodeConst ? ("PIXEL_FORMAT_MONO12_PACKED_MSFIRST") : ("MONO12_PACKED_MSFIRST");
	case PIXEL_FORMAT_BAYER12_GRBG_PACKED_MSFIRST:
		return asCodeConst ? ("PIXEL_FORMAT_BAYER12_GRBG_PACKED_MSFIRST") : ("BAYER12_PACKED_MSFIRST");
	case PIXEL_FORMAT_BAYER12_RGGB_PACKED_MSFIRST:
		return asCodeConst ? ("PIXEL_FORMAT_BAYER12_RGGB_PACKED_MSFIRST") : ("BAYER12_PACKED_MSFIRST");
	case PIXEL_FORMAT_BAYER12_GBRG_PACKED_MSFIRST:
		return asCodeConst ? ("PIXEL_FORMAT_BAYER12_GBRG_PACKED_MSFIRST") : ("BAYER12_PACKED_MSFIRST");
	case PIXEL_FORMAT_BAYER12_BGGR_PACKED_MSFIRST:
		return asCodeConst ? ("PIXEL_FORMAT_BAYER12_BGGR_PACKED_MSFIRST") : ("BAYER12_PACKED_MSFIRST");
	case PIXEL_FORMAT_MONO10_PACKED_MSFIRST:
		return asCodeConst ? ("PIXEL_FORMAT_MONO10_PACKED_MSFIRST") : ("MONO10_PACKED_MSFIRST");
	case PIXEL_FORMAT_BAYER10_GRBG_PACKED_MSFIRST:
		return asCodeConst ? ("PIXEL_FORMAT_BAYER10_GRBG_PACKED_MSFIRST") : ("BAYER10_PACKED_MSFIRST");
	case PIXEL_FORMAT_BAYER10_RGGB_PACKED_MSFIRST:
		return asCodeConst ? ("PIXEL_FORMAT_BAYER10_RGGB_PACKED_MSFIRST") : ("BAYER10_PACKED_MSFIRST");
	case PIXEL_FORMAT_BAYER10_GBRG_PACKED_MSFIRST:
		return asCodeConst ? ("PIXEL_FORMAT_BAYER10_GBRG_PACKED_MSFIRST") : ("BAYER10_PACKED_MSFIRST");
	case PIXEL_FORMAT_BAYER10_BGGR_PACKED_MSFIRST:
		return asCodeConst ? ("PIXEL_FORMAT_BAYER10_BGGR_PACKED_MSFIRST") : ("BAYER10_PACKED_MSFIRST");
	case PIXEL_FORMAT_STOKES4_12:
		return asCodeConst ? ("PIXEL_FORMAT_STOKES4_12") : ("STOKES4_12");
	case PIXEL_FORMAT_POLAR4_12:
		return asCodeConst ? ("PIXEL_FORMAT_POLAR4_12") : ("POLAR4_12");
	case PIXEL_FORMAT_POLAR_RAW4_12:
		return asCodeConst ? ("PIXEL_FORMAT_POLAR_RAW4_12") : ("POLAR_RAW4_12");
	case PIXEL_FORMAT_HSV4_12:
		return asCodeConst ? ("PIXEL_FORMAT_HSV4_12") : ("HSV4_12"); 
	case PIXEL_FORMAT_RGB24_NON_DIB:
			return asCodeConst ? ("PIXEL_FORMAT_RGB_24") : ("RGB_24");
	}
}




void Pixelink::LoadSupportedPaValues() 
{
	// NOTE: See the comment at the start of LoadSupportedPixelFormats(), above.


	std::vector<int>& xDecs = m_supportedXDecimations;
	std::vector<int>& yDecs = m_supportedYDecimations;
	std::vector<int>& modes = m_supportedDecimationModes;
	std::vector<PixelAddressingPair>& pairs = m_supportedDecimationModePairs;


	xDecs.clear();
	yDecs.clear();
	modes.clear();

						// Keep track of what the decimation is to start with.
	PixelAddressing currentPa = GetPixelAddressing();


	if (currentPa.supportAsymmetry)
	{

		//try all x's first
		PixelAddressing pa;
		pa.Mode = currentPa.Mode;
		pa.y = 1;

		CAMERA_FEATURE* addressingFeature = GetFeaturePtr(FEATURE_PIXEL_ADDRESSING);
		int minX = addressingFeature->pParams[FEATURE_PIXEL_ADDRESSING_PARAM_X_VALUE].fMinValue;
		int maxX = addressingFeature->pParams[FEATURE_PIXEL_ADDRESSING_PARAM_X_VALUE].fMaxValue;

		for (int x = minX; x <= maxX; x++)
		{
			pa.x = x;

			PXL_RETURN_CODE rc = SetPixelAddressing(pa);

			if (API_SUCCESS(rc))
			{
				xDecs.push_back(pa.x);
			}
		}

		pa.x = 1;


 		int minY = addressingFeature->pParams[FEATURE_PIXEL_ADDRESSING_PARAM_Y_VALUE].fMinValue;
		int maxY = addressingFeature->pParams[FEATURE_PIXEL_ADDRESSING_PARAM_Y_VALUE].fMaxValue;

		for (int y = minY; y <= maxY; y++)
		{
			pa.y = y;

			PXL_RETURN_CODE rc = SetPixelAddressing(pa);

			if (API_SUCCESS(rc))
			{
				yDecs.push_back(pa.y);
			}
		}


		for (std::size_t x = 0; x<xDecs.size(); ++x) {
			for (std::size_t y = 0; y<yDecs.size(); ++y) {


				//std::string xStr = std::to_string((long long)(xDecs[x])) + 'x' + std::to_string((long long)(yDecs[y]));
				std::string xStr = std::to_string((long long)(xDecs[x])) + 'x' + std::to_string((long long)(yDecs[y]));
				//const char * display = &xStr[0u];

				PixelAddressingPair pair =
				{
					xStr,
					xDecs[x],
					yDecs[y]
				};

				pairs.push_back(pair);
			}
		}




		int i = 0;

	}
	else
	{
		//try all x's first
		PixelAddressing pa;
		pa.Mode = currentPa.Mode;


		int isSupported = 0;
		U32 bufferSize = 0;
		PxLGetCameraFeatures(m_handle, FEATURE_PIXEL_ADDRESSING, NULL, &bufferSize);
		CAMERA_FEATURES* pFeatureInfo = (CAMERA_FEATURES*)malloc(bufferSize);


		PxLGetCameraFeatures(m_handle, FEATURE_PIXEL_ADDRESSING, pFeatureInfo, &bufferSize);

		int minX = pFeatureInfo->pFeatures->pParams->fMinValue;
		int maxX = pFeatureInfo->pFeatures->pParams->fMaxValue;

		PxLGetCameraFeatures(m_handle, FEATURE_PIXEL_ADDRESSING, pFeatureInfo, &bufferSize);

		free(pFeatureInfo);
		



		for (int x = minX; x <= maxX; x++)
		{
			pa.x = x;
			pa.y = x;

			PXL_RETURN_CODE rc = SetPixelAddressing(pa);

			if (API_SUCCESS(rc))
			{
				xDecs.push_back(pa.x);
				yDecs.push_back(pa.y);
			}

		}

		for (std::size_t x = 0; x<xDecs.size(); ++x) {

				//std::string xStr = std::to_string((long long)(xDecs[x])) + 'x' + std::to_string((long long)(yDecs[y]));
				std::string xStr = std::to_string((long long)(xDecs[x])) + 'x' + std::to_string((long long)(yDecs[x]));
				//const char * display = &xStr[0u];

				PixelAddressingPair pair =
				{
					xStr,
					xDecs[x],
					xDecs[x]
				};

				pairs.push_back(pair);
			
		}






	}



	// Set the decimation back to what it was.
	SetPixelAddressing(currentPa);
}

/**
* Function: GetPixelAddressing
* Purpose:  Return a Pixel Addressing object that defines the value of the active
*           camera's FEATURE_PIXEL_ADDRESSING.
*/
PixelAddressing Pixelink::GetPixelAddressing(U32* const pFlags /*=NULL*/)
{
	PixelAddressing pa = { 0, 1, 1, false };


	U32 flags = FEATURE_FLAG_MANUAL;
	U32 nParms = 4;


	// Allocate memory. All parameters are returned as floats.
	std::vector<float> parms(nParms, 0.0f);

	PxLGetFeature(m_handle, FEATURE_PIXEL_ADDRESSING, &flags, &nParms, &parms[0]);

	if (nParms == 1)
	{
		// This is a really old camera/API, and it only supports symmetric decimation
		pa.Mode = 0;
		pa.x = static_cast<int>(parms[FEATURE_PIXEL_ADDRESSING_PARAM_VALUE]);
		pa.y = static_cast<int>(parms[FEATURE_PIXEL_ADDRESSING_PARAM_VALUE]);
		pa.supportAsymmetry = false;
	}
	else if (nParms < 4) {
		// Newer camera that does not support asymmetric pixel addressing.
		pa.Mode = static_cast<int>(parms[FEATURE_PIXEL_ADDRESSING_PARAM_MODE]);
		pa.x = static_cast<int>(parms[FEATURE_PIXEL_ADDRESSING_PARAM_VALUE]);
		pa.y = static_cast<int>(parms[FEATURE_PIXEL_ADDRESSING_PARAM_VALUE]);
		pa.supportAsymmetry = false;
	}
	else {
		// New camera that does support asymmetric pixel addressing.
		pa.Mode = static_cast<int>(parms[FEATURE_PIXEL_ADDRESSING_PARAM_MODE]);
		pa.x = static_cast<int>(parms[FEATURE_PIXEL_ADDRESSING_PARAM_X_VALUE]);
		pa.y = static_cast<int>(parms[FEATURE_PIXEL_ADDRESSING_PARAM_Y_VALUE]);
		pa.supportAsymmetry = true;
	}
	// Return the flags, if they were requested:
	if (pFlags != NULL)
		*pFlags = flags;

	return pa;
}



/**
* Function: SetPixelAddressing
* Purpose:  Set the value of the current camera's FEATURE_PIXEL_ADDRESSING.
*/
PXL_RETURN_CODE
Pixelink::SetPixelAddressing(const PixelAddressing pixelAddressing)
{

	U32 flags = FEATURE_FLAG_MANUAL;

	U32 nParms = 4;



	std::vector<float> parms(nParms, 0.0f);

	PxLGetFeature(m_handle, FEATURE_PIXEL_ADDRESSING, &flags, &nParms, &parms[0]);

	if (nParms == 1)
	{
		// This is a really old camera/API, and it only supports symmetric decimation
		parms[FEATURE_PIXEL_ADDRESSING_PARAM_VALUE] = static_cast<float>(pixelAddressing.x);
	}
	else if (nParms >= 2 && nParms < 4) {
		// Newer camera that does not support asymmetric pixel addressing.
		parms[FEATURE_PIXEL_ADDRESSING_PARAM_VALUE] = static_cast<float>(pixelAddressing.x);
		parms[FEATURE_PIXEL_ADDRESSING_PARAM_MODE] = static_cast<float>(pixelAddressing.Mode);
	}
	else {
		// New camera that does support asymmetric pixel addressing.
		//parms[FEATURE_PIXEL_ADDRESSING_PARAM_VALUE] = static_cast<float>(pixelAddressing.x);
		parms[FEATURE_PIXEL_ADDRESSING_PARAM_MODE] = static_cast<float>(pixelAddressing.Mode);
		parms[FEATURE_PIXEL_ADDRESSING_PARAM_X_VALUE] = static_cast<float>(pixelAddressing.x);
		parms[FEATURE_PIXEL_ADDRESSING_PARAM_Y_VALUE] = static_cast<float>(pixelAddressing.y);
	}

	// The video stream must be stopped to change some features.

	bool stopStream = false;

	PXL_RETURN_CODE rc;

	if (isStreaming)
	{
		PxLSetStreamState(m_handle, STOP_STREAM);
		rc = (PxLSetFeature(m_handle, FEATURE_PIXEL_ADDRESSING, flags, nParms, &parms[0]));
		PxLSetStreamState(m_handle, START_STREAM);
	}
	else
	{
		rc = (PxLSetFeature(m_handle, FEATURE_PIXEL_ADDRESSING, flags, nParms, &parms[0]));
	}

	return rc;
}

/**
* Function: LoadCameraFeatures
* Purpose:  Query the camera to find out which features it supports, and store
*           all the information that it returns in the CAMERA_FEATURES struct
*           pointed to by m_pFeatures.
*/
void
Pixelink::LoadCameraFeatures(void)
{
	ClearFeatures();

	// Determine how much memory to allocate for the CAMERA_FEATURES struct.
	// API Note: By passing NULL in the second parameter, we are telling the
	//   API that we don't want it to populate our CAMERA_FEATURES structure
	//   yet - instead we just want it to tell us how much memory it will
	//   need when it does populate it.
	U32 size = 0;
	PxLGetCameraFeatures(m_handle, FEATURE_ALL, NULL, &size);


					  // Allocate the amount of memory that we were told is required.
	m_pFeatures = reinterpret_cast<CAMERA_FEATURES*>(new byte[size]);

	(PxLGetCameraFeatures(m_handle, FEATURE_ALL, m_pFeatures, &size));

	// The CAMERA_FEATURES structure is loaded only when a camera is 
	// initialized, but is read from frequently to determine the camera's
	// capabilities. To make reading this data easier, we populate a vector
	// that maps feature IDs to their corresponding CAMERA_FEATURE structure.
	m_features.resize(m_pFeatures->uNumberOfFeatures, NULL);


	CAMERA_FEATURE* pCurr = &m_pFeatures->pFeatures[0];
	for (int i = 0; i < m_pFeatures->uNumberOfFeatures; i++, pCurr++)
	{
		pCurr->uFeatureId < FEATURES_TOTAL;
		m_features[pCurr->uFeatureId] = pCurr;
	}
}

void
Pixelink::ClearFeatures(void)
{
	if (m_pFeatures != NULL)
	{
		m_features.clear();
		//delete[] reinterpret_cast<byte*>(m_pFeatures);
		m_pFeatures = NULL;
	}
}


/**
* Function: GetFeaturePtr
* Purpose:
*/
CAMERA_FEATURE*
Pixelink::GetFeaturePtr(const U32 featureId)
{
	static CAMERA_FEATURE dummy = { 0, 0, 0, NULL };
	// If we're running on an older version that doesn't know about 
	// this feature, just say "Unsupported" via a dummy feature.
	if ((featureId >= m_features.size()) || (m_pFeatures == NULL) || (m_features[featureId] == NULL)) {
		return &dummy;
	}
	return m_features[featureId];
}


void
Pixelink::UpdateFrameBufferSize(int fHorizontal, int fVertical)
{
	// Make sure that we have an image so that 
	// things like bitdepth are set correctly
	//SetExposure(30.0);
	int ret = 0;
	unsigned short nrTries = 0;
	do {
		ret = SnapImage(); // first snap often times out, ignore error
		nrTries++;
	} while (ret != DEVICE_OK && nrTries < 10);



	unsigned int width = frameDesc.Roi.fWidth;
	if (fHorizontal != 0) {
		width = frameDesc.Roi.fWidth / fHorizontal;
	}

	unsigned int height = frameDesc.Roi.fHeight;
	if (fHorizontal != 0) {
		height = frameDesc.Roi.fHeight / fVertical;
	}

	unsigned int size = frameDesc.uSize;
	unsigned int pixels = width * height;

	if (pixels == 0) {
		return;
	}

	bytesPerPixel_ = unsigned short(size / pixels);
}


/**
* Function: LoadSupportedPixelFormats
* Purpose:  Fill a vector with all the values of FEATURE_PIXEL_FORMAT that are
*           supported by the current camera.
*/
void
Pixelink::LoadSupportedPixelFormats()
{
	// NOTE: Pixel Formats do not constitute a logically consecutive sequence
	// of values. In other words, just because the camera says that the Min
	// and Max Pixel Formats that it supports are X and Y respectively, it
	// does not mean that *all* the values between X and Y will be supported
	// too. We have to explicitly test whether the camera supports the
	// values between X and Y by trying to actually set the PIXEL_FORMAT
	// feature, and checking for an error return code.



	std::vector<int>& formats = m_supportedPixelFormats;
	formats.clear();

	// Keep track of what the pixel format is to start with.
	float currentFormat = ReturnFeature(m_handle, FEATURE_PIXEL_FORMAT);




	CAMERA_FEATURE* addressingFeature = GetFeaturePtr(FEATURE_PIXEL_FORMAT);
	int min = addressingFeature->pParams[0].fMinValue;
	int max = addressingFeature->pParams[0].fMaxValue;





	// Try to set the pixel format to each value in the legal range, and add
	// to the vector only those values that we can set without error.
	for (int fmt = min; fmt <= max; fmt++)
	{
		// Skip over the specific bayer formats - eg: PIXEL_FOMAT_BAYER8_RGGB, etc.
		// These specific formats only exist for reporting purposes - the user is
		// only supposed to be able to set the camera into BAYER format, and the 
		// camera/API determines which specific variant of the BAYER pattern is
		// actually output.
		// In other words, we only want to put BAYER8 in the combo box, and not
		// any of BAYER8_GRGB, BAYER8_RGGB, etc.
		if (fmt == PIXEL_FORMAT_BAYER8_BGGR
			|| fmt == PIXEL_FORMAT_BAYER8_GBRG
			|| fmt == PIXEL_FORMAT_BAYER8_RGGB
			|| fmt == PIXEL_FORMAT_BAYER16_BGGR
			|| fmt == PIXEL_FORMAT_BAYER16_GBRG
			|| fmt == PIXEL_FORMAT_BAYER16_RGGB
			|| fmt == PIXEL_FORMAT_BAYER12_BGGR_PACKED
			|| fmt == PIXEL_FORMAT_BAYER12_GBRG_PACKED
			|| fmt == PIXEL_FORMAT_BAYER12_RGGB_PACKED
			|| fmt == PIXEL_FORMAT_BAYER12_BGGR_PACKED_MSFIRST
			|| fmt == PIXEL_FORMAT_BAYER12_GBRG_PACKED_MSFIRST
			|| fmt == PIXEL_FORMAT_BAYER12_RGGB_PACKED_MSFIRST)
		{
			continue;
		}


		U32 flags = FEATURE_FLAG_MANUAL;
		int nParms = 1;
		float parms[1] = { fmt };


		PXL_RETURN_CODE rc = (PxLSetFeature(m_handle, FEATURE_PIXEL_FORMAT, flags, nParms, &parms[0]));


		if (API_SUCCESS(rc))
		{
			formats.push_back(fmt);







		}

	}

	// Set the pixel format back to what it was.

	U32 flags = FEATURE_FLAG_MANUAL;
	int nParms = 1;
	float parms[1] = { currentFormat };
	PXL_RETURN_CODE rc = (PxLSetFeature(m_handle, FEATURE_PIXEL_FORMAT, flags, nParms, &parms[0]));

}

void
Pixelink::SetPixelType(int pixelType)
{
	PXL_RETURN_CODE rc;
	U32 flags = FEATURE_FLAG_MANUAL;
	int nParms = 1;
	float parms[1] = { pixelType };

	if (isStreaming)
	{
		PxLSetStreamState(m_handle, STOP_STREAM);
		rc = (PxLSetFeature(m_handle, FEATURE_PIXEL_FORMAT, flags, nParms, &parms[0]));
		PxLSetStreamState(m_handle, START_STREAM);
	}
	else
	{
		rc = (PxLSetFeature(m_handle, FEATURE_PIXEL_FORMAT, flags, nParms, &parms[0]));
	}


	ERROR_REPORT errorReport;
	if (!API_SUCCESS(rc))
	{
		PxLGetErrorReport(m_handle, &errorReport);
	}



}

int 
Pixelink::StringAsPixelType(string pixelFormat) const
{

	for (std::size_t i = 0; i < m_supportedPixelFormats.size(); ++i)
	{
		if (pixelFormat == PixelTypeAsString(m_supportedPixelFormats[i], false))
		{
			return (int)m_supportedPixelFormats[i];
		}
	}

	return 0;
}


unsigned int 
Pixelink::GetBitDepth() const
{
	if (bitDepth_ == NULL)
	{
		return 8;
	}

	if (bitDepth_ == 0)
	{
		return 8;
	}

	return bitDepth_;
}

std::string
Pixelink::PixelFormatAsString(int pixelFormat)
{
	switch (pixelFormat)
	{
	case PIXEL_FORMAT_MONO8:
		return "PIXEL_FORMAT_MONO8";
	case PIXEL_FORMAT_MONO16:
		return "PIXEL_FORMAT_MONO16";
	case PIXEL_FORMAT_YUV422:
		return "PIXEL_FORMAT_YUV422";
	case PIXEL_FORMAT_BAYER8_GRBG:
		return "PIXEL_FORMAT_BAYER8_GRBG";
	case PIXEL_FORMAT_BAYER8_RGGB:
		return "PIXEL_FORMAT_BAYER8_RGGB";
	case PIXEL_FORMAT_BAYER8_GBRG:
		return "PIXEL_FORMAT_BAYER8_GBRG";
	case PIXEL_FORMAT_BAYER8_BGGR:
		return "PIXEL_FORMAT_BAYER8_BGGR";
	case PIXEL_FORMAT_BAYER16_GRBG:
		return "PIXEL_FORMAT_BAYER16_GRBG";
	case PIXEL_FORMAT_BAYER16_RGGB:
		return "PIXEL_FORMAT_BAYER16_RGGB";
	case PIXEL_FORMAT_BAYER16_GBRG:
		return "PIXEL_FORMAT_BAYER16_GBRG";
	case PIXEL_FORMAT_BAYER16_BGGR:
		return "PIXEL_FORMAT_BAYER16_BGGR";
	case PIXEL_FORMAT_RGB24:
		return "PIXEL_FORMAT_RGB24";
	case PIXEL_FORMAT_RGB48:
		return "PIXEL_FORMAT_RGB48";
	case PIXEL_FORMAT_MONO12_PACKED:
		return "PIXEL_FORMAT_MONO12_PACKED";
	case PIXEL_FORMAT_BAYER12_GRBG_PACKED:
		return "PIXEL_FORMAT_BAYER12_GRBG_PACKED";
	case PIXEL_FORMAT_BAYER12_RGGB_PACKED:
		return "PIXEL_FORMAT_BAYER12_RGGB_PACKED";
	case PIXEL_FORMAT_BAYER12_GBRG_PACKED:
		return "PIXEL_FORMAT_BAYER12_GBRG_PACKED";
	case PIXEL_FORMAT_BAYER12_BGGR_PACKED:
		return "PIXEL_FORMAT_BAYER12_BGGR_PACKED";
	case PIXEL_FORMAT_MONO12_PACKED_MSFIRST:
		return "PIXEL_FORMAT_MONO12_PACKED_MSFIRST";
	case PIXEL_FORMAT_BAYER12_GRBG_PACKED_MSFIRST:
		return "PIXEL_FORMAT_BAYER12_GRBG_PACKED_MSFIRST";
	case PIXEL_FORMAT_BAYER12_RGGB_PACKED_MSFIRST:
		return "PIXEL_FORMAT_BAYER12_RGGB_PACKED_MSFIRST";
	case PIXEL_FORMAT_BAYER12_GBRG_PACKED_MSFIRST:
		return "PIXEL_FORMAT_BAYER12_GBRG_PACKED_MSFIRST";
	case PIXEL_FORMAT_BAYER12_BGGR_PACKED_MSFIRST:
		return "PIXEL_FORMAT_BAYER12_BGGR_PACKED_MSFIRST";
	case PIXEL_FORMAT_MONO10_PACKED_MSFIRST:
		return "PIXEL_FORMAT_MONO10_PACKED_MSFIRST";
	case PIXEL_FORMAT_BAYER10_GRBG_PACKED_MSFIRST:
		return "PIXEL_FORMAT_BAYER10_GRBG_PACKED_MSFIRST";
	case PIXEL_FORMAT_BAYER10_RGGB_PACKED_MSFIRST:
		return "PIXEL_FORMAT_BAYER10_RGGB_PACKED_MSFIRST";
	case PIXEL_FORMAT_BAYER10_GBRG_PACKED_MSFIRST:
		return "PIXEL_FORMAT_BAYER10_GBRG_PACKED_MSFIRST";
	case PIXEL_FORMAT_BAYER10_BGGR_PACKED_MSFIRST:
		return "PIXEL_FORMAT_BAYER10_BGGR_PACKED_MSFIRST";
	case PIXEL_FORMAT_STOKES4_12:
		return "PIXEL_FORMAT_STOKES4_12";
	case PIXEL_FORMAT_POLAR4_12:
		return "PIXEL_FORMAT_POLAR4_12";
	case PIXEL_FORMAT_POLAR_RAW4_12:
		return "PIXEL_FORMAT_POLAR_RAW4_12";
	case PIXEL_FORMAT_HSV4_12:
		return "PIXEL_FORMAT_HSV4_12";
	}
}



int
Pixelink::StringasPixelFormat(char* pixelFormat)
{


	if (pixelFormat == "PIXEL_FORMAT_MONO8")
	{
		return PIXEL_FORMAT_MONO8;
	}
	else if (pixelFormat == "PIXEL_FORMAT_MONO16")
	{
		return PIXEL_FORMAT_MONO16 ;
	}
	else if (pixelFormat == "PIXEL_FORMAT_YUV422")
	{
		return PIXEL_FORMAT_YUV422;
	}
	else if (pixelFormat == "PIXEL_FORMAT_BAYER8_GRBG")
	{
		return PIXEL_FORMAT_BAYER8_GRBG;
	}
	else if (pixelFormat == "PIXEL_FORMAT_BAYER8_RGGB")
	{
		return PIXEL_FORMAT_BAYER8_RGGB;
	}
	else if (pixelFormat == "PIXEL_FORMAT_BAYER8_GBRG")
	{
		return PIXEL_FORMAT_BAYER8_GBRG;
	}
	else if (pixelFormat == "PIXEL_FORMAT_BAYER8_BGGR")
	{
		return PIXEL_FORMAT_BAYER8_BGGR;
	}
	else if (pixelFormat == "PIXEL_FORMAT_BAYER16_GRBG")
	{
		return PIXEL_FORMAT_BAYER16_GRBG;
	}
	else if (pixelFormat == "PIXEL_FORMAT_BAYER16_RGGB")
	{
		return PIXEL_FORMAT_BAYER16_RGGB;
	}
	else if (pixelFormat == "PIXEL_FORMAT_BAYER16_GBRG")
	{
		return PIXEL_FORMAT_BAYER16_GBRG;
	}
	else if (pixelFormat == "PIXEL_FORMAT_BAYER16_BGGR")
	{
		return PIXEL_FORMAT_BAYER16_BGGR;
	}
	else if (pixelFormat == "PIXEL_FORMAT_RGB24")
	{
		return PIXEL_FORMAT_RGB24;
	}
	else if (pixelFormat == "PIXEL_FORMAT_RGB48")
	{
		return PIXEL_FORMAT_RGB48;
	}
	else if (pixelFormat == "PIXEL_FORMAT_MONO12_PACKED")
	{
		return PIXEL_FORMAT_MONO12_PACKED;
	}
	else if (pixelFormat == "PIXEL_FORMAT_BAYER12_GRBG_PACKED")
	{
		return PIXEL_FORMAT_BAYER12_GRBG_PACKED;
	}
	else if (pixelFormat == "PIXEL_FORMAT_BAYER12_RGGB_PACKED")
	{
		return PIXEL_FORMAT_BAYER12_RGGB_PACKED;
	}
	else if (pixelFormat == "PIXEL_FORMAT_BAYER12_GBRG_PACKED")
	{
		return PIXEL_FORMAT_BAYER12_GBRG_PACKED;
	}
	else if (pixelFormat == "PIXEL_FORMAT_BAYER12_BGGR_PACKED")
	{
		return PIXEL_FORMAT_BAYER12_BGGR_PACKED;
	}
	else if (pixelFormat == "PIXEL_FORMAT_MONO12_PACKED_MSFIRST")
	{
		return PIXEL_FORMAT_MONO12_PACKED_MSFIRST;
	}
	else if (pixelFormat == "PIXEL_FORMAT_BAYER12_GRBG_PACKED_MSFIRST")
	{
		return PIXEL_FORMAT_BAYER12_GRBG_PACKED_MSFIRST;
	}
	else if (pixelFormat == "PIXEL_FORMAT_BAYER12_RGGB_PACKED_MSFIRST")
	{
		return PIXEL_FORMAT_BAYER12_RGGB_PACKED_MSFIRST;
	}
	else if (pixelFormat == "PIXEL_FORMAT_BAYER12_GBRG_PACKED_MSFIRST")
	{
		return PIXEL_FORMAT_BAYER12_GBRG_PACKED_MSFIRST;
	}
	else if (pixelFormat == "PIXEL_FORMAT_BAYER12_BGGR_PACKED_MSFIRST")
	{
		return PIXEL_FORMAT_BAYER12_BGGR_PACKED_MSFIRST;
	}
	else if (pixelFormat == "PIXEL_FORMAT_MONO10_PACKED_MSFIRST")
	{
		return PIXEL_FORMAT_MONO10_PACKED_MSFIRST;
	}
	else if (pixelFormat == "PIXEL_FORMAT_BAYER10_GRBG_PACKED_MSFIRST")
	{
		return PIXEL_FORMAT_BAYER10_GRBG_PACKED_MSFIRST;
	}
	else if (pixelFormat == "PIXEL_FORMAT_BAYER10_RGGB_PACKED_MSFIRST")
	{
		return  PIXEL_FORMAT_BAYER10_RGGB_PACKED_MSFIRST;
	}
	else if (pixelFormat == "PIXEL_FORMAT_BAYER10_GBRG_PACKED_MSFIRST")
	{
		return PIXEL_FORMAT_BAYER10_GBRG_PACKED_MSFIRST;
	}
	else if (pixelFormat == "PIXEL_FORMAT_BAYER10_BGGR_PACKED_MSFIRST")
	{
		return PIXEL_FORMAT_BAYER10_BGGR_PACKED_MSFIRST;
	}

	return 0;
}