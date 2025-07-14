///////////////////////////////////////////////////////////////////////////////
// FILE:          JAI.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Adapter for JAI eBus compatible cameras
//                
// AUTHOR:        Nenad Amodaj, 2018
// COPYRIGHT:     JAI
//
// LICENSE:			LGPL v3
//						https://www.gnu.org/licenses/lgpl-3.0.en.html
//
// DISCLAIMER:    This file is provided WITHOUT ANY WARRANTY;
//                without even the implied warranty of MERCHANTABILITY or
//                FITNESS FOR A PARTICULAR PURPOSE.
//
//                IN NO EVENT SHALL THE COPYRIGHT OWNER OR
//                CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
//                INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES.
//

#include <ModuleInterface.h>
#include "JAI.h"
#include <PvInterface.h>
#include <PvDevice.h>
#include <PvSystem.h>
#include <PvAcquisitionStateManager.h>
#include <PvPipeline.h>

#ifdef WIN32
#endif

#ifdef __APPLE__
#endif

#ifdef __linux__
#endif

#include <string>
#include <sstream>
#include <iomanip>

using namespace std;

///////////////////////////////////////////////////////////////////////////////
// Exported MMDevice API
///////////////////////////////////////////////////////////////////////////////
MODULE_API void InitializeModuleData()
{
   RegisterDevice(g_DeviceJAICam, MM::CameraDevice, "JAI camera");
}

MODULE_API void DeleteDevice(MM::Device* pDevice)
{
   delete pDevice;
}

MODULE_API MM::Device* CreateDevice(const char* deviceName)
{
   if (deviceName == 0)
      return 0;
   
   if (strcmp(deviceName, g_DeviceJAICam) == 0)
      return new JAICamera();
   
   return 0;
}

JAICamera::JAICamera() :
   initialized(0),
	stopOnOverflow(false),
   acquiring(0),
	camera(0),
	genParams(0),
	bitDepth(8),
	pixelSize(4),
	whiteBalancePending(0)
{
   // set default error messages
   InitializeDefaultErrorMessages();

	// set device specific error messages
	SetErrorText(ERR_CAMERA_OPEN_FAILED, "Camera detected, but unable to read any information. See log file.");
	SetErrorText(ERR_CAMERA_NOT_FOUND, "No JAI compatible cameras were detected.");
	SetErrorText(ERR_INTERNAL_ERROR, "Internal, unspecified error occurred. See log file for clues.");
	SetErrorText(ERR_CAMERA_UNKNOWN_PIXEL_FORMAT, "Camera returned pixel format that this driver can't process.");
	SetErrorText(ERR_STREAM_OPEN_FAILED, "eBUS stream open failed.");
	SetErrorText(ERR_UNSUPPORTED_IMAGE_FORMAT, "Unsupported image format received from the camera. See log file for more info.");
	SetErrorText(ERR_NOT_ALLOWED_DURING_CAPTURE, "This operation is not allowed during live streaming");

   // this identifies which camera we want to access
   CreateProperty(MM::g_Keyword_CameraID, "0", MM::Integer, false, 0, true);
   liveAcqThd_ = new AcqSequenceThread(this);

}


JAICamera::~JAICamera()
{
   Shutdown();
   delete liveAcqThd_;
}

///////////////////////////////////////////////////////////////////////////////
// MMDevice API
//
void JAICamera::GetName(char* name) const
{
   CDeviceUtils::CopyLimitedString(name, g_DeviceJAICam);
}

int JAICamera::Initialize()
{
	PvResult lResult;
	const PvDeviceInfo* lLastDeviceInfo(NULL);

	// Find all devices on the network.
	PvSystem lSystem;
	lResult = lSystem.Find();
	if (!lResult.IsOK())
	{
		LogMessage(string("PvSystem::Find Error: ") + lResult.GetCodeString().GetAscii());
		return ERR_CAMERA_NOT_FOUND;
	}

	// Go through all interfaces 
	uint32_t lInterfaceCount = lSystem.GetInterfaceCount();
	for (uint32_t x = 0; x < lInterfaceCount; x++)
	{
		//cout << "Interface " << x << endl;

		// Get pointer to the interface.
		const PvInterface* lInterface = lSystem.GetInterface(x);

		// Is it a PvNetworkAdapter?
		// We are not interested in network adapters
		const PvNetworkAdapter* lNIC = dynamic_cast<const PvNetworkAdapter*>(lInterface);
		if (lNIC != NULL)
			continue;

		// Is it a PvUSBHostController?
		const PvUSBHostController* lUSB = dynamic_cast<const PvUSBHostController*>(lInterface);
		if (lUSB != NULL)
		{
			// cout << "  Name: " << lUSB->GetName().GetAscii() << endl << endl;
		}

		// Go through all the devices attached to the interface
		uint32_t lDeviceCount = lInterface->GetDeviceCount();
		for (uint32_t y = 0; y < lDeviceCount; y++)
		{
			const PvDeviceInfo *lDeviceInfo = lInterface->GetDeviceInfo(y);

			//cout << "  Device " << y << endl;
			//cout << "    Display ID: " << lDeviceInfo->GetDisplayID().GetAscii() << endl;

			const PvDeviceInfoGEV* lDeviceInfoGEV = dynamic_cast<const PvDeviceInfoGEV*>(lDeviceInfo);
			const PvDeviceInfoU3V *lDeviceInfoU3V = dynamic_cast<const PvDeviceInfoU3V *>(lDeviceInfo);
			const PvDeviceInfoUSB *lDeviceInfoUSB = dynamic_cast<const PvDeviceInfoUSB *>(lDeviceInfo);
			const PvDeviceInfoPleoraProtocol* lDeviceInfoPleora = dynamic_cast<const PvDeviceInfoPleoraProtocol*>(lDeviceInfo);

			if (lDeviceInfoGEV != NULL) // Is it a GigE Vision device?
			{
				continue; // we don't care about network interfaces
			}

			else if (lDeviceInfoU3V != NULL) // Is it a USB3 Vision device?
			{
				ostringstream os;
				os << "    GUID: " << lDeviceInfoU3V->GetDeviceGUID().GetAscii() << endl;
				os << "    S/N: " << lDeviceInfoU3V->GetSerialNumber().GetAscii() << endl;
				os << "    Speed: " << lUSB->GetSpeed() << endl << endl;
				LogMessage(os.str());
				lLastDeviceInfo = lDeviceInfo;
			}
			else if (lDeviceInfoUSB != NULL) // Is it an unidentified USB device?
			{
				continue;
			}
			else if (lDeviceInfoPleora != NULL) // Is it a Pleora Protocol device?
			{
				continue;
			}
		}
	}

	// Connect to the last device found
	if (lLastDeviceInfo != NULL)
	{
		ostringstream os;
		os << "Connecting to " << lLastDeviceInfo->GetDisplayID().GetAscii() << endl;
		LogMessage(os.str());

		// Creates and connects the device controller based on the selected device.
		camera = PvDevice::CreateAndConnect(lLastDeviceInfo, &lResult);
		if (!lResult.IsOK())
		{
			ostringstream osErr;
			osErr << "Unable to connect to " << lLastDeviceInfo->GetDisplayID().GetAscii() << endl;
			LogMessage(osErr.str());
			return ERR_CAMERA_OPEN_FAILED;
		}
		connectionId = lLastDeviceInfo->GetConnectionID().GetAscii();
	}
	else
	{
		return ERR_CAMERA_NOT_FOUND;
	}

	// get handle to camera parameters
	genParams = camera->GetParameters();

	// collect information about the camera
	PvString modelName;
	PvResult pvr = genParams->GetStringValue("DeviceModelName", modelName);
	if (!pvr.IsOK())
		return processPvError(pvr);
	CreateProperty("Model", modelName.GetAscii(), MM::String, true);

	PvString sn;
	pvr = genParams->GetStringValue("DeviceSerialNumber", sn);
	if (!pvr.IsOK())
		return processPvError(pvr);
	CreateProperty("SerialNumber", sn.GetAscii(), MM::String, true);

	PvString fpgaVer;
	pvr = genParams->GetStringValue("DeviceFpgaVersion", fpgaVer);
	if (!pvr.IsOK())
		return processPvError(pvr);

	PvString fwVer;
	pvr = genParams->GetStringValue("DeviceFirmwareVersion", fwVer);
	if (!pvr.IsOK())
		return processPvError(pvr);

	PvString devVer;
	pvr = genParams->GetStringValue("DeviceVersion", devVer);
	if (!pvr.IsOK())
		return processPvError(pvr);

	ostringstream os;
	os << "Version=" << devVer.GetAscii() << ", Firmware=" << fwVer.GetAscii() << ", Fpga=" << fpgaVer.GetAscii();
	LogMessage(os.str());

	// reset to factory defaults
	pvr = genParams->SetEnumValue("UserSetSelector", 0);
	if (!pvr.IsOK())
		return processPvError(pvr);

	pvr = genParams->ExecuteCommand("UserSetLoad");
		if (!pvr.IsOK())
		return processPvError(pvr);

	// set timed exposure mode
	pvr = genParams->SetEnumValue("ExposureMode", 1);
	if (!pvr.IsOK())
		return processPvError(pvr);

	// FRAME RATE
	CPropertyAction *pAct = new CPropertyAction(this, &JAICamera::OnFrameRate);
	CreateProperty(g_FrameRate, "0", MM::Float, false, pAct);

	double frMinHz, frMaxHz;
	pvr = genParams->GetFloatRange("AcquisitionFrameRate", frMinHz, frMaxHz);
	if (!pvr.IsOK())
		return processPvError(pvr);

	SetPropertyLimits(g_FrameRate, frMinHz, frMaxHz);

	// EXPOSURE
	double expMinUs, expMaxUs;
	pvr = genParams->GetFloatRange("ExposureTime", expMinUs, expMaxUs);
	if (!pvr.IsOK())
		return processPvError(pvr);

	double expUs;
	pvr = genParams->GetFloatValue("ExposureTime", expUs);
	if (!pvr.IsOK())
		return processPvError(pvr);

	pAct = new CPropertyAction(this, &JAICamera::OnExposure);
	int ret = CreateProperty(MM::g_Keyword_Exposure, CDeviceUtils::ConvertToString(expUs), MM::Float, false, pAct);
	SetPropertyLimits(MM::g_Keyword_Exposure, expMinUs / 1000, expMaxUs / 1000);
	assert(ret == DEVICE_OK);

	// BINNING
	pAct = new CPropertyAction(this, &JAICamera::OnBinning);
	ret = CreateProperty(MM::g_Keyword_Binning, "1", MM::Integer, false, pAct);
	assert(ret == DEVICE_OK);
   
   int64_t binMin, binMax;
   pvr = genParams->GetIntegerRange(g_pv_BinH, binMin, binMax);
   if (!pvr.IsOK())
      return processPvError(pvr);

	vector<string> binValues;
   for (int64_t i=binMin; i<=binMax; i++)
      binValues.push_back(CDeviceUtils::ConvertToString((int)i));
	SetAllowedValues(MM::g_Keyword_Binning, binValues);

	// TEMPERATURE
	pAct = new CPropertyAction(this, &JAICamera::OnTemperature);
	CreateProperty(g_Temperature, "0", MM::Float, true, pAct);

	// AUTO WHITE BALANCE
	pAct = new CPropertyAction(this, &JAICamera::OnWhiteBalance);
	ret = CreateProperty(g_WhiteBalance, g_Off, MM::String, false, pAct);
	assert(ret == DEVICE_OK);

	// list white balance options
	PvGenEnum *wbMode = genParams->GetEnum("BalanceWhiteAuto");
	int64_t numModes = 0;
	wbMode->GetEntriesCount(numModes);

	for (uint32_t i = 0; i < numModes; i++)
	{
		const PvGenEnumEntry *entry = NULL;
		wbMode->GetEntryByIndex(i, &entry);

		if (entry->IsAvailable())
		{
			PvString name;
			entry->GetName(name);

			int64_t val;
			entry->GetValue(val);
			AddAllowedValue(g_WhiteBalance, name.GetAscii(), (long)val);

			ostringstream osWB;
			osWB << g_WhiteBalance << " : " << name.GetAscii() << "=" << val;
			LogMessage(osWB.str());
		}
	}

	// GAIN
	double gain, gainMin, gainMax;
	pvr = genParams->GetFloatValue("Gain", gain);
	if (!pvr.IsOK())
		return processPvError(pvr);

	pvr = genParams->GetFloatRange("Gain", gainMin, gainMax);
	if (!pvr.IsOK())
		return processPvError(pvr);

	pAct = new CPropertyAction(this, &JAICamera::OnGain);
	ret = CreateProperty(g_Gain, CDeviceUtils::ConvertToString(gain), MM::Float, false, pAct);
	SetPropertyLimits(g_Gain, gainMin, gainMax);
	assert(ret == DEVICE_OK);

	// GAMMA
	double gamma, gammaMin, gammaMax;
	pvr = genParams->GetFloatValue("Gamma", gamma);
	if (!pvr.IsOK())
		return processPvError(pvr);

	pvr = genParams->GetFloatRange("Gamma", gammaMin, gammaMax);
	if (!pvr.IsOK())
		return processPvError(pvr);

	pAct = new CPropertyAction(this, &JAICamera::OnGamma);
	ret = CreateProperty(g_Gamma, CDeviceUtils::ConvertToString(gamma), MM::Float, false, pAct);
	SetPropertyLimits(g_Gamma, gammaMin, gammaMax);
	assert(ret == DEVICE_OK);

	// TEST PATTERN
	// list test pattern optiopns
	pAct = new CPropertyAction(this, &JAICamera::OnTestPattern);
	ret = CreateProperty(g_TestPattern, g_Off, MM::String, false, pAct);

	PvGenEnum *testPattern = genParams->GetEnum("TestPattern");
	int64_t numPatterns = 0;
	testPattern->GetEntriesCount(numPatterns);

	for (uint32_t i = 0; i < numPatterns; i++)
	{
		const PvGenEnumEntry *entry = NULL;
		testPattern->GetEntryByIndex(i, &entry);

		if (entry->IsAvailable())
		{
			PvString name;
			entry->GetName(name);

			int64_t val;
			entry->GetValue(val);
			AddAllowedValue(g_TestPattern, name.GetAscii(), (long)val);
		}
	}

	// PIXEL TYPE
	pvr = genParams->SetEnumValue(g_pv_PixelFormat, g_pv_PixelFormat_BGR8);
	if (!pvr.IsOK())
		return processPvError(pvr);

   pAct = new CPropertyAction (this, &JAICamera::OnPixelType);
	pixelSize = 4; // 32bitRGB
   ret = CreateStringProperty(MM::g_Keyword_PixelType, g_PixelType_32bitRGB, false, pAct);
   assert(ret == DEVICE_OK);

   vector<string> pixelTypeValues;
	pixelTypeValues.push_back(g_PixelType_32bitRGB);
	pixelTypeValues.push_back(g_PixelType_64bitRGB);

   ret = SetAllowedValues(MM::g_Keyword_PixelType, pixelTypeValues);
   if (ret != DEVICE_OK)
      return ret;

	// disable multi roi mode
	pvr = genParams->SetEnumValue("MultiRoiMode", "Off");
	if (!pvr.IsOK())
		return processPvError(pvr);

	//* Root\Connection:DeviceGUID, String : 14FB0164E37A
	//	>> Root\ImageFormatControl : SensorDigitizationBits, Enum : Twelve
	//	Root\ImageFormatControl : TestPattern, Enum : Off
	//	>> Root\AcquisitionControl : TriggerSelector, Enum : AcquisitionStart
	//	>> Root\AcquisitionControl : TriggerMode, Enum : Off
	//	>> Root\AcquisitionControl : TriggerSource, Enum : Low
	//	>> Root\AcquisitionControl : TriggerActivation, Enum : RisingEdge
	//	* Root\AcquisitionControl : ExposureMode, Enum : Timed
	//	>> Root\AnalogControl : Gain, Float : 1
	//	>> Root\AnalogControl : GainAuto, Enum : Off
	// Gamma?

   ret = ResizeImageBuffer();
   if (ret != DEVICE_OK)
      return ret;

	UpdateStatus();

   initialized = true;
   return DEVICE_OK;
}

int JAICamera::Shutdown()
{
   if (!initialized)
      return DEVICE_OK;

   if (IsCapturing())
      StopSequenceAcquisition();

	// release camera
	if (camera)
	{
		PvDevice::Free(camera);
		camera = 0;
	}

   initialized = false;
   return DEVICE_OK;
}

bool JAICamera::Busy()
{
   return false;
}

long JAICamera::GetImageBufferSize() const
{
	return img.Width() * img.Height() * GetImageBytesPerPixel();
}

/**
 * Access single image buffer 
 */
const unsigned char* JAICamera::GetImageBuffer()
{
   return img.GetPixels();
}

const unsigned char* JAICamera::GetImageBuffer(unsigned /* chNum */)
{
	// this camera is not multi-channel, so ignore channel
   return img.GetPixels();
}

const unsigned int* JAICamera::GetImageBufferAsRGB32()
{
   return (unsigned int*) img.GetPixels();
}

unsigned JAICamera::GetNumberOfComponents() const
{
	return 4;
}

unsigned JAICamera::GetNumberOfChannels() const
{
   return 1;
}

int JAICamera::GetChannelName(unsigned channel, char* name)
{
   // TODO: multichannel

   if (channel != 0)
      return ERR_INVALID_CHANNEL_INDEX;
   
   strncpy(name, "Channel-0", MM::MaxStrLength);
   return DEVICE_OK;
}

/**
 * Snaps a single image, blocks at least until exposure is finished 
 */
int JAICamera::SnapImage()
{
	if (whiteBalancePending)
	{
		LogMessage("WB Once Pending in SnapImage() will be ignored, WB will be applied on the next Live acqisition frame");
	}

	// set single frame mode
	PvResult pvr = genParams->SetEnumValue("AcquisitionMode", "SingleFrame");
	if (!pvr.IsOK())
		return pvr.GetCode();

	// Create a stream
	PvString cid(connectionId.c_str());

	PvStream* pvStream = PvStream::CreateAndOpen(cid, &pvr);
	if (!pvStream)
		return ERR_STREAM_OPEN_FAILED;

	// create smart pointer to clean up stream when function exits
	std::shared_ptr<PvStream> camStream(pvStream, [](PvStream *s) { PvStream::Free(s); }); // deleter 

	uint32_t payloadSize = camera->GetPayloadSize();

	// setup camera buffers
	const int numBufs = 1;
	ClearPvBuffers();
	for (int i = 0; i < numBufs; i++)
	{
		// Create new buffer object
		PvBuffer *lBuffer = new PvBuffer;
		pvBuffers.push_back(lBuffer);

		// Have the new buffer object allocate payload memory
		lBuffer->Alloc(payloadSize);

		camStream->QueueBuffer(lBuffer);
	}

	// Reset stream statistics
	pvr = camStream->GetParameters()->ExecuteCommand("Reset");
	if (pvr.IsFailure())
	{
		return processPvError(pvr);
	}

	pvr = camera->StreamEnable();
	if (pvr.IsFailure())
	{
		return processPvError(pvr);
	}

	pvr = camera->GetParameters()->ExecuteCommand("AcquisitionStart");
	if (pvr.IsFailure())
	{
		return processPvError(pvr);
	}

	PvBuffer* pvBuf(0);
	PvResult pvrOp;
	pvr = camStream->RetrieveBuffer(&pvBuf, &pvrOp, 4000);
	if (pvr.IsOK())
	{
		if (pvrOp.IsFailure())
		{
			return processPvError(pvr);
		}

		PvImage* pvImg = pvBuf->GetImage();
		int height = pvImg->GetHeight();
		int width = pvImg->GetWidth();

		if (!verifyPvFormat(pvImg))
			return ERR_UNSUPPORTED_IMAGE_FORMAT;

		// transfer pv image to camera buffer
		img.Resize((unsigned)width, (unsigned)height, pixelSize);
		uint8_t* pSrcImg = pvImg->GetDataPointer();
		uint8_t* pDestImg = img.GetPixelsRW();
		if (pixelSize == 4)
			convert_BGR8_RGBA32(pSrcImg, pDestImg, img.Width(), img.Height());
		else if (pixelSize == 8)
			convert_BGR12P_RGBA64(pSrcImg, pDestImg, img.Width(), img.Height());
		else
			assert(!"Wrong pixel size");
	}
	else
	{
		return processPvError(pvr);
	}

	pvr = camera->GetParameters()->ExecuteCommand("AcquisitionStop");
	if (pvr.IsFailure())
	{
		return processPvError(pvr);
	}

	pvr = camera->StreamDisable();
	if (pvr.IsFailure())
	{
		return processPvError(pvr);
	}
	camStream->Close();

	ClearPvBuffers();
	return DEVICE_OK;
}

unsigned JAICamera::GetBitDepth() const
{
	return bitDepth;
}

int JAICamera::GetBinning() const
{
   char bin[MM::MaxStrLength];
	int ret = GetProperty(MM::g_Keyword_Binning, bin);
   if (ret != DEVICE_OK)
      return ret;
	return atoi(bin);
}

int JAICamera::SetBinning(int binSize)
{
   ostringstream os;                                                         
   os << binSize;
   return SetProperty(MM::g_Keyword_Binning, os.str().c_str());                                                                                     
}

double JAICamera::GetExposure() const
{
	double expUs;
	PvResult pvr = genParams->GetFloatValue("ExposureTime", expUs);
	if (!pvr.IsOK())
		return 0.0;

	return expUs / 1000;
}

void JAICamera::SetExposure(double expMs)
{
	PvResult pvr = genParams->SetFloatValue("ExposureTime", expMs * 1000);
}

int JAICamera::SetROI(unsigned x, unsigned y, unsigned xSize, unsigned ySize)
{
	// obtain constraints on ROI parameters
	PvGenInteger* pvX = genParams->GetInteger(g_pv_OffsetX);
	int64_t xIncr;
	pvX->GetIncrement(xIncr);

	PvGenInteger* pvY = genParams->GetInteger(g_pv_OffsetY);
	int64_t yIncr;
	pvY->GetIncrement(yIncr);

	PvGenInteger* pvW = genParams->GetInteger(g_pv_Width);
	int64_t wIncr;
	pvW->GetIncrement(wIncr);

	PvGenInteger* pvH = genParams->GetInteger(g_pv_Height);
	int64_t hIncr;
	pvH->GetIncrement(hIncr);

	// apply Roi with modifications based on the constraints
	int64_t newXSize = xSize / wIncr * wIncr;
   PvResult pvr = genParams->SetIntegerValue(g_pv_Width, newXSize);
   if (pvr.IsFailure())
      return processPvError(pvr);

	int64_t newYSize = ySize / hIncr * hIncr;
   pvr = genParams->SetIntegerValue(g_pv_Height, newYSize);
   if (pvr.IsFailure())
      return processPvError(pvr);

	int64_t newX =  x / xIncr * xIncr;
   pvr = genParams->SetIntegerValue(g_pv_OffsetX, newX);
   if (pvr.IsFailure())
      return processPvError(pvr);

	int64_t newY = y / yIncr * yIncr;
   pvr = genParams->SetIntegerValue(g_pv_OffsetY, newY);
   if (pvr.IsFailure())
      return processPvError(pvr);

   return ResizeImageBuffer();
}

int JAICamera::GetROI(unsigned& x, unsigned& y, unsigned& xSize, unsigned& ySize)
{
   int64_t offsetX, offsetY, w, h;
   PvResult pvr = genParams->GetIntegerValue(g_pv_OffsetX, offsetX);
   if (pvr.IsFailure())
      return processPvError(pvr);
   pvr = genParams->GetIntegerValue(g_pv_OffsetY, offsetY);
   if (pvr.IsFailure())
      return processPvError(pvr);
   pvr = genParams->GetIntegerValue(g_pv_Width, w);
   if (pvr.IsFailure())
      return processPvError(pvr);
   pvr = genParams->GetIntegerValue(g_pv_Height, h);
   if (pvr.IsFailure())
      return processPvError(pvr);

	x = (unsigned)offsetX;
	y = (unsigned)offsetY;
	xSize = img.Width();
	ySize = img.Height();
   assert(xSize == w);
   assert(ySize == h);

	return DEVICE_OK;
}

int JAICamera::ClearROI()
{
	int64_t wmax, hmax;
   PvResult pvr = genParams->GetIntegerValue("WidthMax", wmax);
   if (pvr.IsFailure())
      return processPvError(pvr);

   pvr = genParams->GetIntegerValue("HeightMax", hmax);
   if (pvr.IsFailure())
      return processPvError(pvr);

	// apply max roi
   pvr = genParams->SetIntegerValue(g_pv_OffsetX, 0);
   if (pvr.IsFailure())
      return processPvError(pvr);

   pvr = genParams->SetIntegerValue(g_pv_OffsetY, 0);
   if (pvr.IsFailure())
      return processPvError(pvr);

   pvr = genParams->SetIntegerValue(g_pv_Width, wmax);
   if (pvr.IsFailure())
      return processPvError(pvr);

   pvr = genParams->SetIntegerValue(g_pv_Height, hmax);
   if (pvr.IsFailure())
      return processPvError(pvr);

	return ResizeImageBuffer();
}

int JAICamera::PrepareSequenceAcqusition()
{
   if (IsCapturing())
   {
      return DEVICE_CAMERA_BUSY_ACQUIRING;
   }

   int ret = GetCoreCallback()->PrepareForAcq(this);
   if (ret != DEVICE_OK)
      return ret;

   return DEVICE_OK;
}

int JAICamera::StartSequenceAcquisition(long numImages, double /*interval_ms*/, bool stopOnOvl)
{
   if (IsCapturing())
   {
      return DEVICE_CAMERA_BUSY_ACQUIRING;
   }

   // the camera ignores interval, running at the rate dictated by the exposure
   stopOnOverflow = stopOnOvl;
   liveAcqThd_->SetNumFrames(numImages); // continuous
   liveAcqThd_->Start();

   return DEVICE_OK;
}

int JAICamera::StartSequenceAcquisition(double /*interval_ms*/)
{
   if (IsCapturing())
      return DEVICE_CAMERA_BUSY_ACQUIRING;

   // the camera ignores interval, running at the rate dictated by the exposure
   stopOnOverflow = false;

   liveAcqThd_->SetNumFrames(0); // continuous
   liveAcqThd_->Start();

   return DEVICE_OK;
}

int JAICamera::StopSequenceAcquisition()
{
   liveAcqThd_->Stop();
   liveAcqThd_->wait();
   return DEVICE_OK;
}

bool JAICamera::IsCapturing()
{
   return acquiring == 1;
}

///////////////////////////////////////////////////////////////////////////////
// Private utility functions

int JAICamera::ResizeImageBuffer()
{
	int64_t width, height;
	PvResult pvr = camera->GetParameters()->GetIntegerValue("Width", width);
	assert(pvr.IsOK());
	pvr = camera->GetParameters()->GetIntegerValue("Height", height);
	assert(pvr.IsOK());

	img.Resize((unsigned)width, (unsigned)height, pixelSize);
	return DEVICE_OK;
}

int JAICamera::PushImage(unsigned char* imgBuf)
{
	int retCode = GetCoreCallback()->InsertImage(this,
		imgBuf,
		img.Width(),
		img.Height(),
		img.Depth());

	if (!stopOnOverflow && retCode == DEVICE_BUFFER_OVERFLOW)
	{
		// do not stop on overflow - just reset the buffer
		GetCoreCallback()->ClearImageBuffer(this);
		retCode = GetCoreCallback()->InsertImage(this,
			imgBuf,
			img.Width(),
			img.Height(),
			img.Depth());
	}

	return DEVICE_OK;
}

int JAICamera::processPvError(const PvResult& pvr)
{
	ClearPvBuffers();
	SetErrorText(pvr.GetCode(), pvr.GetDescription().GetAscii());
	return pvr.GetCode();
}

/**
 * Converts BGR 24-bit image to RGBA 32-bit image. Alpha channel is set to 0.
 *
 * @return void
 * @param src - source buffer, should be pixSize * 3 bytes
 * @param dest - destination buffer, should be pixSize * 4 bytes
 * @param pixSize - image buffer size in pixels
 */
void JAICamera::convert_BGR8_RGBA32(const uint8_t * src, uint8_t * dest, unsigned w, unsigned h)
{
	const int byteDepth = 4;
	int srcCounter = 0;
	unsigned sizeInPixels = w * h;
	for (unsigned i = 0; i < sizeInPixels; i++)
	{
		dest[i*byteDepth] = src[srcCounter++]; // R
		dest[i*byteDepth + 1] = src[srcCounter++]; // G
		dest[i*byteDepth + 2] = src[srcCounter++]; // B
		dest[i*byteDepth + 3] = 0; // alpha
	}
}

/**
 * Converts BGR 36-bit image to RGBA 64-bit image. Alpha channel is set to 0.
 *
 * @return void
 * @param src - source buffer, should be pixSize * 3 bytes
 * @param dest - destination buffer, should be pixSize * 4 bytes
 * @param pixSize - image buffer size in pixels
 */
void JAICamera::convert_BGR12P_RGBA64(const uint8_t * src, uint8_t * dest, unsigned w, unsigned h)
{
	const int byteDepth = 8;
	unsigned sizeInPixels = w * h;
	for (unsigned i = 0; i < sizeInPixels; i++)
	{
		int pixPtrR = i * 36 / 8;
		int bitPtrR = i * 36 % 8;
		uint16_t* buf = (uint16_t*)(src + pixPtrR);
		uint16_t r = *buf << bitPtrR;
		r = r >> 4;
		*((uint16_t*)(dest + i*byteDepth)) = r; // R

		int pixPtrG = (i * 36 + 12) / 8;
		int bitPtrG = (i * 36 + 12) % 8;
		buf = (uint16_t*)(src + pixPtrG);
		uint16_t g = *buf << bitPtrG;
		g = g >> 4;
		*((uint16_t*)(dest + i*byteDepth + 2)) = g; // G

		int pixPtrB = (i * 36 + 24) / 8;
		int bitPtrB = (i * 36 + 24) % 8;
		buf = (uint16_t*)(src + pixPtrB);
		uint16_t b = *buf << bitPtrB;
		b = b >> 4;
		*((uint16_t*)(dest + i*byteDepth + 4)) = b; // B

		*((uint16_t*)(dest + i*byteDepth + 6)) = 0; // alpha
	}
}

/**
 * Verify that camera is returning image in the format we expect
 *
 * @return bool - true if format is OK
 * @param pvImg - image
 */
bool JAICamera::verifyPvFormat(const PvImage * pvImg)
{
	uint32_t height = pvImg->GetHeight();
	uint32_t width = pvImg->GetWidth();
	PvPixelType pt = pvImg->GetPixelType();
	// we are looking for PvPixelBGR8 pixel type

	bool color = pvImg->IsPixelColor(pt);

	if (!color)
	{
		LogMessage("Only color images are supported.");
		return false;
	}

	uint32_t bpc = pvImg->GetBitsPerComponent(pt);
	if (bpc != 8 && bpc != 12)
	{
		ostringstream os;
		os << "Only 8-bits and 12-bits per color plane are supported: BPC=" << bpc;
		LogMessage(os.str());
		return false;
	}

	uint32_t pvPixSize = pvImg->GetPixelSize(pt);
	uint32_t pvBpp = pvImg->GetBitsPerPixel();
	if (pvPixSize != 24 && pvPixSize != 36)
	{
		ostringstream os;
		os << "Only 3-byte and 4-byte RGB camera pixels are supported: PixSize=" << pvPixSize << ", BPP=" << pvBpp;
		LogMessage(os.str());
		return false;
	}

	uint32_t imgSize = pvImg->GetImageSize();
	uint32_t effImgSize = pvImg->GetEffectiveImageSize();
	if (imgSize != width * height * pvPixSize / 8 + ((width * height * pvPixSize) % 8 ? 1 : 0))
	{
		ostringstream os;
		os << "Image size does not match width and height, W=" << width << ", H=" << height << ", Size=" << imgSize << ", EffSize=" << effImgSize;
		LogMessage(os.str());
		return false;
	}
	return true;
}

void JAICamera::ClearPvBuffers()
{
	for (unsigned int i=0; i<pvBuffers.size(); i++)
	{
		if (pvBuffers[i]->IsAttached())
			pvBuffers[i]->Detach();

		if (pvBuffers[i]->IsAllocated())
			pvBuffers[i]->Free();

		delete pvBuffers[i];
	}
	pvBuffers.clear();
}

int JAICamera::InsertImage()
{
   int retCode = GetCoreCallback()->InsertImage(this,
         img.GetPixels(),
         img.Width(),
         img.Height(),
         img.Depth());

   if (!stopOnOverflow)
   {
      if (retCode == DEVICE_BUFFER_OVERFLOW)
      {
         // do not stop on overflow - just reset the buffer
         GetCoreCallback()->ClearImageBuffer(this);
         retCode = GetCoreCallback()->InsertImage(this,
            img.GetPixels(),
            img.Width(),
            img.Height(),
            img.Depth());
         return DEVICE_OK;
      }
      else
         return retCode;
   }

   return retCode;
}

/**
 * Live stream thread function
 *
 * @return int - 0 if success, 1 otherwise (return code is not used)
 */
int AcqSequenceThread::svc (void)
{
   InterlockedExchange(&moduleInstance->acquiring, 1);

	// setup continuous acquisition
	// set single frame mode
	PvResult pvr = moduleInstance->genParams->SetEnumValue("AcquisitionMode", "Continuous");
	if (!pvr.IsOK())
		return pvr.GetCode();

   // Create a stream
	PvString cid(moduleInstance->connectionId.c_str());
	PvStream* pvStream = PvStream::CreateAndOpen(cid, &pvr);
	if (!pvStream)
	{
		moduleInstance->LogMessage("Failed opening the camera stream.");
		InterlockedExchange(&moduleInstance->acquiring, 0);
		return 1;
	}

	// create smart pointer to clean up stream when function exits
	std::shared_ptr<PvStream> camStream(pvStream, [](PvStream *s) { s->Close(); PvStream::Free(s); }); // deleter 

	uint32_t payloadSize = moduleInstance->camera->GetPayloadSize();

	// setup camera buffers
	const int numBufs = 8;
	moduleInstance->ClearPvBuffers();
	for (int i = 0; i < numBufs; i++)
	{
		// Create new buffer object
		PvBuffer *lBuffer = new PvBuffer;
		moduleInstance->pvBuffers.push_back(lBuffer);

		// Have the new buffer object allocate payload memory
		lBuffer->Alloc(payloadSize);

		camStream->QueueBuffer(lBuffer);
	}

	// Reset stream statistics
	pvr = camStream->GetParameters()->ExecuteCommand("Reset");
	if (pvr.IsFailure())
	{
		return processPvError(pvr, camStream);
	}

	pvr = moduleInstance->camera->StreamEnable();
	if (pvr.IsFailure())
	{
		return processPvError(pvr, camStream);
	}

	pvr = moduleInstance->camera->GetParameters()->ExecuteCommand("AcquisitionStart");
	if (pvr.IsFailure())
	{
		return processPvError(pvr, camStream);
	}

	if (moduleInstance->whiteBalancePending)
	{
		// Acquire first image, which will be used only for WB and then discarded
		PvBuffer* pvBuf(0);
		PvResult pvrOp;
		pvr = camStream->RetrieveBuffer(&pvBuf, &pvrOp, 4000);
		if (pvr.IsOK())
		{
			if (pvrOp.IsFailure())
			{
				ostringstream os;
				os << "WB Once Pending: PvError=" << pvr.GetCode() << ", Description: " << pvr.GetDescription().GetAscii();
			}

			PvImage* pvImg = pvBuf->GetImage();

			if (!moduleInstance->verifyPvFormat(pvImg))
			{
				ostringstream os;
				os << "WB Once Pending: image format verification failed";
				return 1;
			}
		}
		else
		{
			// attempt to grab WB image unsuccessful
			ostringstream os;
			os << "WB Once Pending: PvError=" << pvr.GetCode() << ", Description: " << pvr.GetDescription().GetAscii();
		}

		PvResult pvr = moduleInstance->genParams->SetEnumValue("BalanceWhiteAuto", 1);
		if (!pvr.IsOK())
		{
			ostringstream os;
			os << "WB Once Pending: PvError=" << pvr.GetCode() << ", Description: " << pvr.GetDescription().GetAscii();
			moduleInstance->LogMessage(os.str());
		}
		else
		{
			ostringstream os;
			os << "WB Once Pending: White Balance set based on the current image";
			moduleInstance->LogMessage(os.str());
		}

		InterlockedExchange(&moduleInstance->whiteBalancePending, 0L); // clear wb pending flag
		ostringstream os;
		os << "WB Once Pending flag cleared";
	}

   unsigned count = 0;
   while(stopFlag != 1)
   {         
		PvBuffer* pvBuf(0);
		PvResult pvrOp;
		pvr = camStream->RetrieveBuffer(&pvBuf, &pvrOp, 4000);
		if (pvr.IsOK())
		{
			if (pvrOp.IsFailure())
			{
				return processPvError(pvr, camStream);
			}

			PvImage* pvImg = pvBuf->GetImage();
			int height = pvImg->GetHeight();
			int width = pvImg->GetWidth();

			if (!moduleInstance->verifyPvFormat(pvImg))
			{
				InterlockedExchange(&moduleInstance->acquiring, 0);
				return 1;
			}

			// transfer pv image to camera buffer
			uint8_t* pSrcImg = pvImg->GetDataPointer();
			moduleInstance->img.Resize(width, height, 4); // RGBA format
			uint8_t* pDestImg = moduleInstance->img.GetPixelsRW();
			JAICamera::convert_BGR8_RGBA32(pSrcImg, pDestImg, moduleInstance->img.Width(), moduleInstance->img.Height());

			// push image to queue
			moduleInstance->InsertImage();
		}
		else
		{
			return processPvError(pvr, camStream);
		}

		count++;
		ostringstream os;
		os << "Acquired image: " << count;
		moduleInstance->LogMessage(os.str());

		if (numFrames > 0 && count >= numFrames)
		{
			moduleInstance->LogMessage("Number of frames reached.");
			break;
		}

		// reuse the buffer
		pvr = camStream->QueueBuffer(pvBuf);
		if (pvrOp.IsFailure())
		{
			return processPvError(pvr, camStream);
		}
	} // while

	if (stopFlag)
		moduleInstance->LogMessage("User pressed stop.");

	// stop acquisition
	pvr = moduleInstance->camera->GetParameters()->ExecuteCommand("AcquisitionStop");
	if (pvr.IsFailure())
	{
		return processPvError(pvr, camStream);
	}

	pvr = camStream->AbortQueuedBuffers();
	if (pvr.IsFailure())
	{
		return processPvError(pvr, camStream);
	}

	pvr = moduleInstance->camera->StreamDisable();
	if (pvr.IsFailure())
	{
		return processPvError(pvr, camStream);
	}

	if (pvr.IsFailure())
	{
		return processPvError(pvr, camStream);
	}

	camStream->Close();
	moduleInstance->ClearPvBuffers();
	InterlockedExchange(&moduleInstance->acquiring, 0);
   return 0;
}

void AcqSequenceThread::Stop()
{
	InterlockedExchange(&stopFlag, 1);
}

int AcqSequenceThread::processPvError(PvResult pvr, std::shared_ptr<PvStream>& stream)
{
	if (stream)
		stream->Close();

	moduleInstance->ClearPvBuffers();

	ostringstream os;
	os << "PvError=" << pvr.GetCode() << ", Description: " << pvr.GetDescription().GetAscii();
	moduleInstance->LogMessage(os.str());
	InterlockedExchange(&moduleInstance->acquiring, 0);
	return 1;
}



