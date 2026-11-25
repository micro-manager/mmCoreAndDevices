///////////////////////////////////////////////////////////////////////////////
// FILE:          DemoCamera.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   The example implementation of the demo camera.
//                Simulates generic digital camera and associated automated
//                microscope devices and enables testing of the rest of the
//                system without the need to connect to the actual hardware. 
//                
// AUTHOR:        Nenad Amodaj, nenad@amodaj.com, 06/08/2005
//
// COPYRIGHT:     University of California, San Francisco, 2006
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

#include "MMTUCam.h"
#include "CameraImageMetadata.h"
#include <cstdio>
#include <string>
#include <math.h>
#include "ModuleInterface.h"
#include <sstream>
#include <algorithm>
#include "WriteCompactTiffRGB.h"
#include <iostream>
#include <process.h>
#include <unordered_set>
#include "EnumProperty.h"
#include "IntegerProperty.h"
#include "FloatProperty.h"
#include "BooleanProperty.h"
#include "CommandProperty.h"
#include "CstringProperty.h"

#ifndef _WIN64
#pragma comment(lib, "./sdk3/lib/x86/TUCam.lib")
#else
#pragma comment(lib, "./sdk3/lib/x64/TUCam.lib")
#endif

#include <shlwapi.h>
#pragma comment(lib, "shlwapi.lib")

#define SEQUENCE_ACQUISITION_INTERTUPTED 10000

using namespace std;

namespace {

int64_t ClampFeatureInt64(TUC_HDEVICE hDevice, const char* name, int64_t requested)
{
    TUC_FEATURE_INT64_INFO ii = {};
    if (!TUC_STATUS_IS_SUCCESS(TucGetFeatureInt64Info(hDevice, name, &ii)))
        return requested;

    int64_t step = ii.increment > 0 ? ii.increment : 1;
    int64_t val = (requested / step) * step;
    if (val < ii.minimum)
        val = ii.minimum;
    if (val > ii.maximum)
        val = ii.maximum;
    return val;
}

bool GetFeatureEnumSymbolic(TUC_HDEVICE hDevice, const char* featureName, int64_t value, std::string& symbolic)
{
    uint32_t count = 0;
    if (!TUC_STATUS_IS_SUCCESS(TucGetFeatureEnumCount(hDevice, featureName, &count)))
        return false;

    for (uint32_t i = 0; i < count; ++i)
    {
        TUC_FEATURE_ENUM_ENTRY entry = {};
        if (!TUC_STATUS_IS_SUCCESS(TucGetFeatureEnumEntryByIndex(hDevice, featureName, i, &entry)))
            continue;
        if (entry.value == value)
        {
            symbolic = entry.symbolic;
            return true;
        }
    }
    return false;
}

void InferPixelLayoutFromFormat(const std::string& pixelFormat, uint8_t& channels, uint8_t& elemBytes)
{
    channels = 1;
    elemBytes = 2;

    if (pixelFormat.find("Mono8") != std::string::npos)
    {
        channels = 1;
        elemBytes = 1;
    }
    else if (pixelFormat.find("Mono16") != std::string::npos ||
             pixelFormat.find("Mono12") != std::string::npos)
    {
        channels = 1;
        elemBytes = 2;
    }
    else if (pixelFormat.find("RGB") != std::string::npos ||
             pixelFormat.find("Bayer") != std::string::npos ||
             pixelFormat.find("BGR") != std::string::npos)
    {
        channels = 3;
        elemBytes = 1;
    }
}

} // namespace

const double CMMTUCam::nominalPixelSizeUm_ = 1.0;
double g_IntensityFactor_ = 1.0;
std::unordered_set<int> callbackCamSet = { ARIES_6510, ARIES_6506, ARIES_6504, ARIES_6504LT, LEO_3243, LEO_3243_PRO, LIBRA_3412M, LIBRA_3405M, LEO_5514, LIBRA_3412C };

// External names used used by the rest of the system
// to load particular device from the "DemoCamera.dll" library
const char* g_TUDeviceName  = "TUCamGen";

const char* g_Format_PNG = "PNG";
const char* g_Format_TIF = "TIF";
const char* g_Format_JPG = "JPG";
const char* g_Format_BMP = "BMP";
const char* g_Format_RAW = "RAW";
const char* g_FileName   = "\\Image";

// constants for naming pixel types (allowed values of the "PixelType" property)
const char* g_PixelType_8bit     = "8bit";
const char* g_PixelType_16bit    = "16bit";
const char* g_PixelType_32bitRGB = "32bitRGB";
const char* g_PixelType_64bitRGB = "64bitRGB";
const char* g_PixelType_32bit    = "32bit";  // floating point greyscale

///////////////////////////////////////////////////////////////////////////////
// Exported MMDevice API
///////////////////////////////////////////////////////////////////////////////

MODULE_API void InitializeModuleData()
{
   RegisterDevice(g_TUDeviceName, MM::CameraDevice, "TUCSEN Gen Camera");
}

MODULE_API MM::Device* CreateDevice(const char* deviceName)
{
   if (deviceName == 0)
      return 0;

   // decide which device class to create based on the deviceName parameter
   if (strcmp(deviceName, g_TUDeviceName) == 0)
   {
      // create camera
      return new CMMTUCam();
   }

   // ...supplied name not recognized
   return 0;
}

MODULE_API void DeleteDevice(MM::Device* pDevice)
{
   delete pDevice;
}


int CMMTUCam::s_nNumCam  = 0;
int CMMTUCam::s_nCntCam  = 0;
///////////////////////////////////////////////////////////////////////////////
// CMMTUCam implementation
// ~~~~~~~~~~~~~~~~~~~~~~~~~~

/**
* CMMTUCam constructor.
* Setup default all variables and create device properties required to exist
* before intialization. In this case, no such properties were required. All
* properties will be created in the Initialize() method.
*
* As a general guideline Micro-Manager devices do not access hardware in the
* the constructor. We should do as little as possible in the constructor and
* perform most of the initialization in the Initialize() method.
*/
CMMTUCam::CMMTUCam() :
    CCameraBase<CMMTUCam> (),
    exposureMaximum_(2147483.647),     
    exposureMinimum_(0.0),
    dPhase_(0),
    initialized_(false),
    readoutUs_(0.0),
    scanMode_(1),
    bitDepth_(8),
    roiX_(0),
    roiY_(0),
    sequenceStartTime_(0),
    isSequenceable_(false),
    sequenceMaxLength_(100),
    sequenceRunning_(false),
    sequenceIndex_(0),
    binSize_(1),
    cameraCCDXSize_(512),
    cameraCCDYSize_(512),
    ccdT_ (0.0),
    triggerDevice_(""),
    stopOnOverflow_(false),
    dropPixels_(false),
    fastImage_(false),
    saturatePixels_(false),
    fractionOfPixelsToDropOrSaturate_(0.002),
    shouldRotateImages_(false),
    shouldDisplayImageNumber_(false),
    stripeWidth_(1.0),
    nComponents_(1),
    returnToSoftwareTriggers_(false),
    frameProcessingThreadRunning_(false),
    convertWorkersRunning_(false),
    convertReady_(0),
    convertFailed_(false)
#ifdef ENABLE_FPS_STATISTICS
    ,callbackCount_(0),
	processFrameCount_(0),
	waitForFrameCount_(0)
#endif
{
    memset(testProperty_,0,sizeof(testProperty_));

    // call the base class method to set-up default error codes/messages
    InitializeDefaultErrorMessages();
    readoutStartTime_ = GetCurrentMMTime();
    thd_ = new CTUCamThread(this);

    // parent ID display
//   CreateHubIDProperty();

    m_fCurTemp    = 0.0f;
    m_fValTemp    = 0.0f;

    m_bROI        = false;
    m_bSaving     = false;
    m_bLiving     = false;
    m_bTemping    = false;
    m_hThdWaitEvt = NULL;
    m_hThdTempEvt = NULL;

    m_hDevice = NULL;
    m_hBuffers = NULL;
    m_bufferCount = 8;
    converterCreated_ = false;
    hostFrameBytes_ = 0;
    m_frame = TuCamImageFrame();

	m_nPID     = 0;
	m_nBCD     = 0;
    m_nIdxGain = 0;

	m_vecEnum.clear();
    m_vecInteger.clear();
    m_vecFloat.clear();
    m_vecBoolean.clear();
    m_vecCommand.clear();
	m_vecString.clear();
}

/**
* CMMTUCam destructor.
* If this device used as intended within the Micro-Manager system,
* Shutdown() will be always called before the destructor. But in any case
* we need to make sure that all resources are properly released even if
* Shutdown() was not called.
*/
CMMTUCam::~CMMTUCam()
{
	if (m_hThdTempEvt != NULL)
	{
        m_bTemping = false;
		WaitForSingleObject(m_hThdTempEvt, INFINITE);	
		CloseHandle(m_hThdTempEvt);
		m_hThdTempEvt = NULL;
	}

    StopSequenceAcquisition();
	StopConvertWorkers();
	StopCapture();
	StopFrameProcessingThread();  // 停止帧处理线程
	FreeHostFramePool();
	s_nCntCam--;

	if (s_nCntCam <= 0)
	{
		s_nCntCam = 0;
		UninitTUCamApi();
	}

    delete thd_;   
}

/**
* Obtains device name.
* Required by the MM::Device API.
*/
void CMMTUCam::GetName(char* name) const
{
    // Return the name used to referr to this device adapte
    CDeviceUtils::CopyLimitedString(name, g_TUDeviceName);
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
int CMMTUCam::Initialize()
{
    OutputDebugString("[Initialize]:Enter!\n");

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
        LogMessage("Parent Hub not defined.");

    // init camera api
    // -----------------
    int nRet = InitTUCamApi();
    if (DEVICE_OK != nRet)
    {
        return nRet;
    }

	m_pid = GetPID();

	if (!TrySetFeatureBool("ReverseY", false))
		TrySetFeatureBool("VerticalFlip", false);

	TuEnumProperty    *pEnumProp    = NULL;
    TuIntegerProperty *pIntProp     = NULL;
    TuFloatProperty   *pFloatProp   = NULL;
    TuBooleanProperty *pBoolProp    = NULL;
    TuCommandProperty *pCommandProp = NULL;
    TuCstringProperty *pStringProp  = NULL;

	uint32_t featCount = 0;
	TucGetFeatureCount(m_hDevice, &featCount);
	for (uint32_t i = 0; i < featCount; ++i)
	{
		TUC_FEATURE_INFO info = {};
		if (!TUC_STATUS_IS_SUCCESS(TucGetFeatureInfoByIndex(m_hDevice, i, &info)))
			continue;
		if (!info.isFeature)
			continue;
		if (info.visibility > TUC_VISIBILITY_GURU)
			continue;

		bool bReadOnly = !(info.accessMode == TUC_ACCESS_WRITE_ONLY || info.accessMode == TUC_ACCESS_READ_WRITE);
		std::string mmName = info.displayName[0] ? info.displayName : info.name;
		std::string featName = info.name;

		switch (info.featureType)
		{
		case TUC_FEATURE_TYPE_BOOL:
			pBoolProp = new TuBooleanProperty(mmName, m_hDevice, featName, this, bReadOnly);
			m_vecBoolean.push_back(pBoolProp);
			break;

		case TUC_FEATURE_TYPE_INTEGER:
			if (info.representation == TUC_REPRESENTATION_MACADDRESS ||
				info.representation == TUC_REPRESENTATION_HEXNUMBER ||
				info.representation == TUC_REPRESENTATION_IPV4ADDRESS)
			{
				pStringProp = new TuCstringProperty(mmName, m_hDevice, featName,
					info.featureType, info.representation, "", this, bReadOnly);
				m_vecString.push_back(pStringProp);
			}
			else
			{
				TUC_FEATURE_INT64_INFO ii = {};
				if (TUC_STATUS_IS_SUCCESS(TucGetFeatureInt64Info(m_hDevice, featName.c_str(), &ii)))
				{
					pIntProp = new TuIntegerProperty(mmName, m_hDevice, featName,
						ii.minimum, ii.maximum, this, bReadOnly);
					m_vecInteger.push_back(pIntProp);
				}
			}
			break;

		case TUC_FEATURE_TYPE_FLOAT:
		{
			TUC_FEATURE_FLOAT_INFO fi = {};
			if (TUC_STATUS_IS_SUCCESS(TucGetFeatureFloatInfo(m_hDevice, featName.c_str(), &fi)))
			{
				pFloatProp = new TuFloatProperty(mmName, m_hDevice, featName,
					fi.minimum, fi.maximum, this, bReadOnly);
				m_vecFloat.push_back(pFloatProp);
			}
			break;
		}

		case TUC_FEATURE_TYPE_STRING:
		{
			char strBuf[256] = {};
			uint32_t strSize = sizeof(strBuf);
			std::string initialValue;
			if (TUC_STATUS_IS_SUCCESS(TucGetFeatureString(m_hDevice, featName.c_str(), strBuf, &strSize)))
				initialValue = strBuf;
			pStringProp = new TuCstringProperty(mmName, m_hDevice, featName,
				info.featureType, info.representation, initialValue, this, bReadOnly);
			m_vecString.push_back(pStringProp);
			break;
		}

		case TUC_FEATURE_TYPE_ENUM:
			pEnumProp = new TuEnumProperty(mmName, m_hDevice, featName, this, bReadOnly);
			m_vecEnum.push_back(pEnumProp);
			break;

		case TUC_FEATURE_TYPE_COMMAND:
			pCommandProp = new TuCommandProperty(mmName, m_hDevice, featName, this, bReadOnly);
			m_vecCommand.push_back(pCommandProp);
			break;

		default:
			break;
		}
	}

    // initialize image buffer
    nRet = StartCapture();
    if (nRet != DEVICE_OK)
        return nRet;

	// ExposureTime limits (microseconds, matching legacy SetExposure scaling)
	{
		TUC_FEATURE_FLOAT_INFO fi = {};
		if (TUC_STATUS_IS_SUCCESS(TucGetFeatureFloatInfo(m_hDevice, "ExposureTime", &fi)))
		{
			exposureMaximum_ = fi.maximum;
			exposureMinimum_ = fi.minimum;
		}
		else
		{
			TUC_FEATURE_INT64_INFO ii = {};
			if (TUC_STATUS_IS_SUCCESS(TucGetFeatureInt64Info(m_hDevice, "ExposureTime", &ii)))
			{
				exposureMaximum_ = static_cast<double>(ii.maximum);
				exposureMinimum_ = static_cast<double>(ii.minimum);
			}
		}
	}

	// pixel type
	vector<string> pixelTypeValues;
	CPropertyAction *pAct = new CPropertyAction(this, &CMMTUCam::OnPixelType);

	if (3 == m_frame.channels)
	{
#ifdef _WIN64        
		if (2 == m_frame.elemBytes)
		{
			nRet = CreateProperty(MM::g_Keyword_PixelType, g_PixelType_64bitRGB, MM::String, false, pAct);
			assert(nRet == DEVICE_OK);

			pixelTypeValues.push_back(g_PixelType_64bitRGB);
		}  
		else
		{
			nRet = CreateProperty(MM::g_Keyword_PixelType, g_PixelType_32bitRGB, MM::String, false, pAct);
			assert(nRet == DEVICE_OK);

			pixelTypeValues.push_back(g_PixelType_32bitRGB);
		} 
#else
		nRet = CreateProperty(MM::g_Keyword_PixelType, g_PixelType_32bitRGB, MM::String, false, pAct);
		assert(nRet == DEVICE_OK);
		pixelTypeValues.push_back(g_PixelType_32bitRGB);
#endif

	}
	else
	{
		if (2 == m_frame.elemBytes)
		{
			nRet = CreateProperty(MM::g_Keyword_PixelType, g_PixelType_16bit, MM::String, false, pAct);
			assert(nRet == DEVICE_OK);
			pixelTypeValues.push_back(g_PixelType_16bit);
		}  
		else 
		{
			nRet = CreateProperty(MM::g_Keyword_PixelType, g_PixelType_8bit, MM::String, false, pAct);
			assert(nRet == DEVICE_OK);
			pixelTypeValues.push_back(g_PixelType_8bit);
		} 
	}  

	nRet = SetAllowedValues(MM::g_Keyword_PixelType, pixelTypeValues);
	if (nRet != DEVICE_OK)
		return nRet;

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
    LogMessage("TestResourceLocking OK",true);
#endif

    initialized_ = true;

    // initialize image buffer
    GenerateEmptyImage(img_);

	// initialize image buffer
	nRet = StopCapture();
	if (nRet != DEVICE_OK)
		return nRet;

    OutputDebugString("[Initialize]:Success!\n");

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
int CMMTUCam::Shutdown()
{
    // Close the get value of temperature thread

    OutputDebugString("[Shutdown]:enter");

    if (NULL != m_hThdTempEvt)
    {
        m_bTemping = false;
        WaitForSingleObject(m_hThdTempEvt, INFINITE);	
        CloseHandle(m_hThdTempEvt);
        m_hThdTempEvt = NULL;
    }

    StopSequenceAcquisition();

    UninitTUCamApi();
    initialized_ = false;

    int i = 0;
    int nSize = (int)m_vecEnum.size();

    for (i=0; i<nSize; ++i)
    {
        TU_DELOBJ(m_vecEnum[i]);
    }
    m_vecEnum.clear();

    nSize = (int)m_vecInteger.size();

    for (i=0; i<nSize; ++i)
    {
        TU_DELOBJ(m_vecInteger[i]);
    }
    m_vecInteger.clear();

	nSize = (int)m_vecFloat.size();

    for (i=0; i<nSize; ++i)
    {
        TU_DELOBJ(m_vecFloat[i]);
    }
    m_vecFloat.clear();

	nSize = (int)m_vecBoolean.size();

    for (i=0; i<nSize; ++i)
    {
        TU_DELOBJ(m_vecBoolean[i]);
    }
    m_vecBoolean.clear();

	nSize = (int)m_vecCommand.size();

    for (i=0; i<nSize; ++i)
    {
        TU_DELOBJ(m_vecCommand[i]);
    }
    m_vecCommand.clear();

	nSize = (int)m_vecString.size();

    for (i=0; i<nSize; ++i)
    {
        TU_DELOBJ(m_vecString[i]);
    }
    m_vecString.clear();

    return DEVICE_OK;
}

/**
* Performs exposure and grabs a single image.
* This function should block during the actual exposure and return immediately afterwards 
* (i.e., before readout).  This behavior is needed for proper synchronization with the shutter.
* Required by the MM::Camera API.
*/
int CMMTUCam::SnapImage()
{
    static int callCounter = 0;
    ++callCounter;

	bool bSoftTrigger = false;
	MM::MMTime startTime = GetCurrentMMTime();

	if (m_hDevice)
	{
		int64_t triggerVal = 0;
		if (TUC_STATUS_IS_SUCCESS(TucGetFeatureEnum(m_hDevice, "TriggerMode", &triggerVal)))
		{
			std::string modeStr;
			TUC_FEATURE_ENUM_ENTRY entry = {};
			if (TUC_STATUS_IS_SUCCESS(TucGetFeatureEnumEntryByIndex(m_hDevice, "TriggerMode", triggerVal, &entry)))
			{
				modeStr = entry.symbolic;
				if (modeStr == "FreeRunning" || modeStr == "Standard")
					bSoftTrigger = false;
				else if (modeStr == "Software")
					bSoftTrigger = true;
				else
					return DEVICE_NOT_SUPPORTED;
			}
		}
	}
 
	double exp = GetExposure();
    if (sequenceRunning_) 
    {
		// Change the exposure time
        exp = GetSequenceExposure();
    }

	if (!m_bLiving)  // should never happen, but just in case...
	{
		StartCapture();
		ResizeImageBuffer();
		UpdateLevelsRange();
	}

	if (bSoftTrigger)
	{
		int nCnt = 0;
		int nRet = DEVICE_ERR;

		do 
		{
			TucExecuteFeatureCommand(m_hDevice, "TriggerSoftwarePulse");
			nRet = WaitForFrameInSnap();
			nCnt++;
		} while (DEVICE_OK != nRet && nCnt < 2);
	}
	else
	{
		if (!fastImage_)
		{	
			MM::MMTime s0(0,0);
			if( s0 < startTime )
			{
				CDeviceUtils::SleepMs((long) exp);
			}
			TUDBG_PRINTF("SanpImage fastImage_ = %d",fastImage_);
			WaitForFrameInSnap();
		}
	}

   readoutStartTime_ = GetCurrentMMTime();

   if (m_bLiving)
   {
	   StopCapture();
   }

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
const unsigned char* CMMTUCam::GetImageBuffer()
{
    MMThreadGuard g(imgPixelsLock_);
    MM::MMTime readoutTime(readoutUs_);
    while (readoutTime > (GetCurrentMMTime() - readoutStartTime_)) {}		
    unsigned char *pB = (unsigned char*)(img_.GetPixels());
    return pB;  //NULL
}

/**
* Returns image buffer X-size in pixels.
* Required by the MM::Camera API.
*/
unsigned CMMTUCam::GetImageWidth() const
{
    return img_.Width();
}

/**
* Returns image buffer Y-size in pixels.
* Required by the MM::Camera API.
*/
unsigned CMMTUCam::GetImageHeight() const
{
    return img_.Height();
}

/**
* Returns image buffer pixel depth in bytes.
* Required by the MM::Camera API.
*/
unsigned CMMTUCam::GetImageBytesPerPixel() const
{
    return img_.Depth();
} 

/**
* Returns the bit depth (dynamic range) of the pixel.
* This does not affect the buffer size, it just gives the client application
* a guideline on how to interpret pixel values.
* Required by the MM::Camera API.
*/
unsigned CMMTUCam::GetBitDepth() const
{
    return bitDepth_;
}

/**
* Returns the size in bytes of the image buffer.
* Required by the MM::Camera API.
*/
long CMMTUCam::GetImageBufferSize() const
{
/*
    if (NULL != m_frame.pBuffer)
    {
        return (m_frame.width * m_frame.height * GetImageBytesPerPixel());
    }

    return 0;
*/    
    
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

int CMMTUCam::SetROI(unsigned x, unsigned y, unsigned xSize, unsigned ySize)
{
    if (NULL == m_hDevice)
        return DEVICE_NOT_CONNECTED;

    if (xSize == 0 && ySize == 0)
    {
        // effectively clear ROI
        ResizeImageBuffer();
        roiX_ = 0;
        roiY_ = 0;
        m_bROI = false;
    }
    else
    {
        if (NULL == m_hDevice)
            return DEVICE_NOT_CONNECTED;

        m_bLiving = false;
        TucAcquisitionStop(m_hDevice, TUC_ACQ_STOP_FLAGS_DEFAULT);
        ReleaseBuffer();

		TrySetFeatureInt64("Width", ClampFeatureInt64(m_hDevice, "Width", xSize));
		TrySetFeatureInt64("Height", ClampFeatureInt64(m_hDevice, "Height", ySize));
		TrySetFeatureInt64("OffsetX", ClampFeatureInt64(m_hDevice, "OffsetX", x));
		TrySetFeatureInt64("OffsetY", ClampFeatureInt64(m_hDevice, "OffsetY", y));

        char sz[256] = {0};
        sprintf(sz, "x:%d, y:%d, xsize:%d, ysize:%d", x, y, xSize, ySize);
        OutputDebugString(sz);

        roiX_ = x;
        roiY_ = y;
        m_bROI = true;

        StartCapture();
        ResizeImageBuffer();
    }

    return DEVICE_OK;
}

/**
* Returns the actual dimensions of the current ROI.
* Required by the MM::Camera API.
*/
int CMMTUCam::GetROI(unsigned& x, unsigned& y, unsigned& xSize, unsigned& ySize)
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
int CMMTUCam::ClearROI()
{
    if (NULL == m_hDevice)
        return DEVICE_NOT_CONNECTED;

	int64_t nSensorHei = 6144;
	int64_t nSensorWid = 6144;
    m_bLiving = false;
    TucAcquisitionStop(m_hDevice, TUC_ACQ_STOP_FLAGS_DEFAULT);
    ReleaseBuffer();

    TrySetFeatureInt64("OffsetY", 0);
    TrySetFeatureInt64("OffsetX", 0);

    if (!TryGetFeatureInt64("SensorWidth", &nSensorWid))
        TryGetFeatureInt64("WidthMax", &nSensorWid);

    TrySetFeatureInt64("Width", nSensorWid);

    if (!TryGetFeatureInt64("SensorHeight", &nSensorHei))
        TryGetFeatureInt64("HeightMax", &nSensorHei);

    TrySetFeatureInt64("Height", nSensorHei);
    
    roiX_ = 0;
    roiY_ = 0;
    m_bROI = false;

    StartCapture();
    ResizeImageBuffer();

    return DEVICE_OK;
}

/**
* Returns the current exposure setting in milliseconds.
* Required by the MM::Camera API.
*/
double CMMTUCam::GetExposure() const
{
	int64_t expUs = 0;
	if (TryGetFeatureInt64("ExposureTime", &expUs))
		return expUs / 1000.0;

	double expVal = 0;
	if (TryGetFeatureFloat("ExposureTime", &expVal))
		return expVal / 1000.0;

	return 0;
}

/**
 * Returns the current exposure from a sequence and increases the sequence counter
 * Used for exposure sequences
 */
double CMMTUCam::GetSequenceExposure() 
{
	return this->GetExposure();
}

/**
* Sets exposure in milliseconds.
* Required by the MM::Camera API.
*/
void CMMTUCam::SetExposure(double exp)
{
	double exposure = exp * 1000;
	exposure = min(max(exposure, exposureMinimum_), exposureMaximum_);

	if (!TrySetFeatureInt64("ExposureTime", static_cast<int64_t>(exposure)))
		TrySetFeatureFloat("ExposureTime", exposure);

	int64_t expUs = 0;
	if (TryGetFeatureInt64("ExposureTime", &expUs))
		exp = expUs / 1000.0;
	else
	{
		double expVal = 0;
		if (TryGetFeatureFloat("ExposureTime", &expVal))
			exp = expVal / 1000.0;
	}

    SetProperty(MM::g_Keyword_Exposure, CDeviceUtils::ConvertToString(exp));
    GetCoreCallback()->OnExposureChanged(this, exp);
}

/**
* Returns the current binning factor.
* Required by the MM::Camera API.
*/
int CMMTUCam::GetBinning() const
{
    return 1;  // fdy20190212
}

/**
* Sets binning factor.
* Required by the MM::Camera API.
*/
int CMMTUCam::SetBinning(int binF)
{
    return SetProperty(MM::g_Keyword_Binning, CDeviceUtils::ConvertToString(binF));
}

int CMMTUCam::IsExposureSequenceable(bool& isSequenceable) const
{
    isSequenceable = isSequenceable_;
    return DEVICE_OK;
}

int CMMTUCam::GetExposureSequenceMaxLength(long& nrEvents) const
{
    if (!isSequenceable_) 
    {
        return DEVICE_UNSUPPORTED_COMMAND;
    }

    nrEvents = sequenceMaxLength_;
    return DEVICE_OK;
}

int CMMTUCam::StartExposureSequence()
{
    if (!isSequenceable_) {
        return DEVICE_UNSUPPORTED_COMMAND;
    }

    // may need thread lock
    sequenceRunning_ = true;
    return DEVICE_OK;
}

int CMMTUCam::StopExposureSequence()
{
    if (!isSequenceable_) 
    {
        return DEVICE_UNSUPPORTED_COMMAND;
    }

    // may need thread lock
    sequenceRunning_ = false;
    sequenceIndex_ = 0;
    return DEVICE_OK;
}

void CMMTUCam::OnCaptureStart()
{
	StartCapture();
	ResizeImageBuffer();

	roiX_ = 0;
	roiY_ = 0;
}

void CMMTUCam::OnCaptureStop()
{
	m_bLiving = false;
	TucAcquisitionStop(m_hDevice, TUC_ACQ_STOP_FLAGS_DEFAULT);
	ReleaseBuffer();
	TUDBG_PRINTF("[OnCaptureStop] -TucAcquisitionStop");
}

void CMMTUCam::UpdateLevelsRange()
{
	TUC_FEATURE_FLOAT_INFO leftInfo = {};
	if (TUC_STATUS_IS_SUCCESS(TucGetFeatureFloatInfo(m_hDevice, "LeftLevels", &leftInfo)))
		TrySetFeatureFloat("LeftLevels", leftInfo.minimum);
	else if (TUC_STATUS_IS_SUCCESS(TucGetFeatureFloatInfo(m_hDevice, "BlackLevel", &leftInfo)))
		TrySetFeatureFloat("BlackLevel", leftInfo.minimum);

	TUC_FEATURE_FLOAT_INFO rightInfo = {};
	if (TUC_STATUS_IS_SUCCESS(TucGetFeatureFloatInfo(m_hDevice, "RightLevels", &rightInfo)))
		TrySetFeatureFloat("RightLevels", rightInfo.maximum);
}

void  CMMTUCam::PropertyReadOnly(MM::PropertyBase* pProp, const char* featureName, TUC_FEATURE_ACCESS access)
{
	MM::Property *prop = (MM::Property *)pProp;
	prop->SetReadOnly(!(access == TUC_ACCESS_WRITE_ONLY || access == TUC_ACCESS_READ_WRITE));

	if (0 == strcmp(featureName, "BinningSelector") ||
		0 == strcmp(featureName, MM::g_Keyword_Binning))
	{
		prop->SetReadOnly(false);
	}

	if (0 == strcmp(featureName, "PixelFormat") || 0 == strcmp(featureName, "PixelSize"))
	{
		UpdateLevelsRange();
	}
}

int CMMTUCam::PropertyChanged(const char* propName, const char* propValue)
{
	return OnPropertyChanged(propName, propValue);
}
/**
 * Clears the list of exposures used in sequences
 */
int CMMTUCam::ClearExposureSequence()
{
    if (!isSequenceable_) 
    {
        return DEVICE_UNSUPPORTED_COMMAND;
    }

    exposureSequence_.clear();
    return DEVICE_OK;
}

/**
 * Adds an exposure to a list of exposures used in sequences
 */
int CMMTUCam::AddToExposureSequence(double exposureTime_ms) 
{
    if (!isSequenceable_)
    {
        return DEVICE_UNSUPPORTED_COMMAND;
    }

    exposureSequence_.push_back(exposureTime_ms);
    return DEVICE_OK;
}

int CMMTUCam::SendExposureSequence() const 
{
    if (!isSequenceable_) 
    {
        return DEVICE_UNSUPPORTED_COMMAND;
    }

    return DEVICE_OK;
}

/**
 * Required by the MM::Camera API
 * Please implement this yourself and do not rely on the base class implementation
 * The Base class implementation is deprecated and will be removed shortly
 */
int CMMTUCam::StartSequenceAcquisition(double interval)
{
    return StartSequenceAcquisition(LONG_MAX, interval, false);            
}

/**                                                                       
* Stop and wait for the Sequence thread finished                                   
*/                                                                        
int CMMTUCam::StopSequenceAcquisition()                                     
{
	OutputDebugString("[StopSequenceAcquisition]:Enter\n");

	// Thread-already-stopped is not "idle": a convert failure exits the
	// sequence thread while m_bLiving and the workers are still up.
	// Only skip cleanup when the device side is stopped as well.
	if (thd_->IsStopped() && !m_bLiving && !convertWorkersRunning_.load()
		&& !returnToSoftwareTriggers_)
		return DEVICE_OK;

	m_bLiving = false;

	if (NULL == m_hDevice)
	{
		return DEVICE_NOT_CONNECTED;
	}
    thd_->Stop(); 
    TucAcquisitionAbort(m_hDevice);
    thd_->wait();

    // Join Convert workers before AcquisitionStop (Convert after Stop fails).
    StopConvertWorkers();
    StopFrameProcessingThread();  // 停止帧处理线程
    FreeHostFramePool();

#ifdef ENABLE_FPS_STATISTICS
	// 输出最终统计信息
	{
		std::lock_guard<std::mutex> statsLock(statsLock_);
		MM::MMTime endTime = GetCurrentMMTime();
		unsigned long long callbackCnt = callbackCount_.load();
		unsigned long long processCnt = processFrameCount_.load();
		unsigned long long waitCnt = waitForFrameCount_.load();

		double totalElapsedMs = (endTime - statsStartTime_).getMsec();
		if (totalElapsedMs > 0)
		{
			double callbackFps = callbackCnt * 1000.0 / totalElapsedMs;
			double processFps = processCnt * 1000.0 / totalElapsedMs;
			double waitFps = waitCnt * 1000.0 / totalElapsedMs;

			char dbgMsg[512] = { 0 };
			sprintf_s(dbgMsg, "[StopSequenceAcquisition] Final Stats - TotalTime=%.3f ms, OnCallbackBuffer: %llu calls (%.2f Hz), ProcessFrame: %llu calls (%.2f Hz), WaitForFrame: %llu calls (%.2f Hz)\n",
				totalElapsedMs, callbackCnt, callbackFps, processCnt, processFps, waitCnt, waitFps);
			OutputDebugString(dbgMsg);
		}
	}
#endif

    TucAcquisitionStop(m_hDevice, TUC_ACQ_STOP_FLAGS_DEFAULT);
    ReleaseBuffer();
		
	ResetCallback();

	int ret = GetCoreCallback()->OnPropertiesChanged(this);
	if (ret != DEVICE_OK)
		return ret;

	OutputDebugString("[StopSequenceAcquisition]:TucAcquisitionStop()\n");

    return DEVICE_OK;
} 

/**
* Simple implementation of Sequence Acquisition
* A sequence acquisition should run on its own thread and transport new images
* coming of the camera into the MMCore circular buffer.
*/
int CMMTUCam::StartSequenceAcquisition(long numImages, double interval_ms, bool stopOnOverflow)
{
    OutputDebugString("[StartSequenceAcquisition]:Enter\n");

    if (IsCapturing())
        return DEVICE_CAMERA_BUSY_ACQUIRING;
    if (m_bLiving)
    {
       StopCapture();
    }

	// Listed cameras: callback queues handles; convert workers copy then requeue.
	int nRet = DEVICE_OK;
	if (callbackCamSet.find(m_pid) != callbackCamSet.end())
		nRet = StartCaptureWithCallback();
	else
		nRet = StartCapture();
	if (nRet != DEVICE_OK)
		return nRet;
	
	ResizeImageBuffer();
	UpdateLevelsRange();
    int ret = GetCoreCallback()->PrepareForAcq(this);
    if (ret != DEVICE_OK)
        return ret;

	ret = GetCoreCallback()->OnPropertiesChanged(this);
	if (ret != DEVICE_OK)
		return ret;

    sequenceStartTime_ = GetCurrentMMTime();
    imageCounter_ = 0;

#ifdef ENABLE_FPS_STATISTICS
	// 重置统计计数器
	{
		std::lock_guard<std::mutex> statsLock(statsLock_);
		callbackCount_ = 0;
		processFrameCount_ = 0;
		waitForFrameCount_ = 0;
		statsStartTime_ = GetCurrentMMTime();
		lastStatsOutputTime_ = GetCurrentMMTime();
	}
#endif

    StartFrameProcessingThread();  // 启动帧处理线程
    thd_->Start(numImages,interval_ms);
    stopOnOverflow_ = stopOnOverflow;
    return DEVICE_OK;
}

/*
 * Inserts Image and MetaData into MMCore circular Buffer
 */
int CMMTUCam::InsertImage()
{
    char label[MM::MaxStrLength];
    this->GetLabel(label);

    // Important:  metadata about the image are generated here:
    MM::CameraImageMetadata md;
    md.AddTag("Camera", label);
    ///md.AddTag(MM::g_Keyword_Metadata_StartTime, CDeviceUtils::ConvertToString(sequenceStartTime_.getMsec()));
    // Leave ElapsedTime-ms unset. Core writes it beside ImageNumber. A plugin
    // timestamp that only increases makes Live Camera fps about 2x after the
    // sequence buffer resets.
    md.AddTag(MM::g_Keyword_Metadata_ROI_X, CDeviceUtils::ConvertToString( (long) roiX_)); 
    md.AddTag(MM::g_Keyword_Metadata_ROI_Y, CDeviceUtils::ConvertToString( (long) roiY_)); 

    imageCounter_++;

    char szTemp[256] = {0};
    sprintf(szTemp, "%.3f", m_fCurTemp);
    md.AddTag("Temperature", szTemp); 

    MMThreadGuard g(imgPixelsLock_);

    const unsigned char* pI;
    pI = GetImageBuffer();

    unsigned int w = GetImageWidth();
    unsigned int h = GetImageHeight();
    unsigned int b = GetImageBytesPerPixel();

    int ret = GetCoreCallback()->InsertImage(this, pI, w, h, b, md.Serialize());

    // MMCore 74: InsertImage() no longer returns DEVICE_BUFFER_OVERFLOW when
    // stopOnOverflow == false, and ClearImageBuffer() has been removed.
    // Any error from InsertImage() should stop the acquisition.
    return ret;
}

/*
 * Do actual capturing
 * Called from inside the thread  
 */
int CMMTUCam::RunSequenceOnThread(MM::MMTime startTime)
{
    int ret = DEVICE_ERR;

    // Trigger
    if (triggerDevice_.length() > 0) 
    {
        MM::Device* triggerDev = GetDevice(triggerDevice_.c_str());
        if (triggerDev != 0) 
        {
            LogMessage("trigger requested");
            triggerDev->SetProperty("Trigger","+");
        }
    }

    ret = WaitForFrame(img_);

#ifdef ENABLE_FPS_STATISTICS
	// 统计：增加WaitForFrame计数
	waitForFrameCount_.fetch_add(1);
#endif

	if (ret != DEVICE_OK)
	{
		return ret;
	}

    // InsertImage现在在帧处理线程中被调用，这里不需要再调用

    /*while (((double) (this->GetCurrentMMTime() - startTime).getMsec() / imageCounter_) < this->GetSequenceExposure())
    {
        CDeviceUtils::SleepMs(1);
    }*/

    return DEVICE_OK;
}

bool CMMTUCam::IsCapturing() 
{
     return !thd_->IsStopped();
}

/*
 * called from the thread function before exit 
 */
void CMMTUCam::OnThreadExiting() throw()
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

CTUCamThread::CTUCamThread(CMMTUCam* pCam)
   :intervalMs_(default_intervalMS)
   ,numImages_(default_numImages)
   ,imageCounter_(0)
   ,stop_(true)
   ,suspend_(false)
   ,camera_(pCam)
   ,startTime_(0)
   ,actualDuration_(0)
   ,lastFrameTime_(0)
{};

CTUCamThread::~CTUCamThread() {};

void CTUCamThread::Stop() 
{
    MMThreadGuard g(this->stopLock_);
    stop_=true;
	camera_->queueCV_.notify_all();
	camera_->convertWaitCV_.notify_all();
}

void CTUCamThread::Start(long numImages, double intervalMs)
{
    OutputDebugString("[CTUCamThread]:Start");
    MMThreadGuard g1(this->stopLock_);
    MMThreadGuard g2(this->suspendLock_);
    numImages_=numImages;
    intervalMs_=intervalMs;
    imageCounter_=0;
    stop_ = false;
    suspend_=false;
    activate();
	actualDuration_ = MM::MMTime{};
    startTime_= camera_->GetCurrentMMTime();
	lastFrameTime_ = MM::MMTime{};
}

bool CTUCamThread::IsStopped()
{
    MMThreadGuard g(this->stopLock_);
    return stop_;
}

void CTUCamThread::Suspend() 
{
    MMThreadGuard g(this->suspendLock_);
    suspend_ = true;
}

bool CTUCamThread::IsSuspended()
{
    MMThreadGuard g(this->suspendLock_);
    return suspend_;
}

void CTUCamThread::Resume()
{
    MMThreadGuard g(this->suspendLock_);
    suspend_ = false;
}

bool CTUCamThread::HasTakenSpecifiedPhotos()
{
	return imageCounter_ == numImages_;
}

int CTUCamThread::svc(void) throw()
{
    int ret=DEVICE_ERR;
    try 
    {
        do
        {  
            ret = camera_->RunSequenceOnThread(startTime_);
        } while (DEVICE_OK == ret && !IsStopped() && imageCounter_++ < numImages_-1);
        if (IsStopped())
            camera_->LogMessage("SeqAcquisition interrupted by the user\n");
    }catch(...){
        camera_->LogMessage(g_Msg_EXCEPTION_IN_THREAD, false);
    }
    stop_=true;

    actualDuration_ = camera_->GetCurrentMMTime() - startTime_;
    camera_->OnThreadExiting();
    return ret;
}


///////////////////////////////////////////////////////////////////////////////
// CMMTUCam Action handlers
///////////////////////////////////////////////////////////////////////////////

int CMMTUCam::OnPixelType(MM::PropertyBase* pProp, MM::ActionType eAct)
{      
	int ret = DEVICE_ERR;
	switch(eAct)
	{
	case MM::AfterSet:
		{
			if(IsCapturing())
				return DEVICE_CAMERA_BUSY_ACQUIRING;

			string pixelType;
			pProp->Get(pixelType);

			if (pixelType.compare(g_PixelType_8bit) == 0)
			{
				nComponents_ = 1;
				img_.Resize(img_.Width(), img_.Height(), 1);
				bitDepth_ = 8;
				ret=DEVICE_OK;
			}
			else if (pixelType.compare(g_PixelType_16bit) == 0)
			{
				nComponents_ = 1;
				img_.Resize(img_.Width(), img_.Height(), 2);
				bitDepth_ = 16;
				ret=DEVICE_OK;
			}
			else if ( pixelType.compare(g_PixelType_32bitRGB) == 0)
			{
				nComponents_ = 4;
				img_.Resize(img_.Width(), img_.Height(), 4);
				bitDepth_ = 8;
				ret=DEVICE_OK;
			}
			else if ( pixelType.compare(g_PixelType_64bitRGB) == 0)
			{
				nComponents_ = 4;
				img_.Resize(img_.Width(), img_.Height(), 8);
				bitDepth_ = 16;
				ret=DEVICE_OK;
			}
			else if ( pixelType.compare(g_PixelType_32bit) == 0)
			{
				nComponents_ = 1;
				img_.Resize(img_.Width(), img_.Height(), 4);
				bitDepth_ = 32;
				ret=DEVICE_OK;
			}
			else
			{
				// on error switch to default pixel type
				nComponents_ = 1;
				img_.Resize(img_.Width(), img_.Height(), 1);
				pProp->Set(g_PixelType_8bit);
				bitDepth_ = 8;
				ret = ERR_UNKNOWN_MODE;
			}
		}
		break;
	case MM::BeforeGet:
		{
			long bytesPerPixel = GetImageBytesPerPixel();
			if (bytesPerPixel == 1)
			{
				pProp->Set(g_PixelType_8bit);
			}
			else if (bytesPerPixel == 2)
			{
				pProp->Set(g_PixelType_16bit);
			}
			else if (bytesPerPixel == 4)
			{              
				if (nComponents_ == 4)
				{
					pProp->Set(g_PixelType_32bitRGB);
				}
				else if (nComponents_ == 1)
				{
					pProp->Set(::g_PixelType_32bit);
				}
			}
			else if (bytesPerPixel == 8)
			{
				pProp->Set(g_PixelType_64bitRGB);
			}
			else
			{
				pProp->Set(g_PixelType_8bit);
			}
			ret = DEVICE_OK;
		} break;
	default:
		break;
	}

	return ret; 
}

/**
* Handles "ReadoutTime" property.
*/
int CMMTUCam::OnReadoutTime(MM::PropertyBase* pProp, MM::ActionType eAct)
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

int CMMTUCam::OnDropPixels(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::AfterSet)
   {
      long tvalue = 0;
      pProp->Get(tvalue);
		dropPixels_ = (0==tvalue)?false:true;
   }
   else if (eAct == MM::BeforeGet)
   {
      pProp->Set(dropPixels_?1L:0L);
   }

   return DEVICE_OK;
}

int CMMTUCam::OnFastImage(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::AfterSet)
   {
      long tvalue = 0;
      pProp->Get(tvalue);
		fastImage_ = (0==tvalue)?false:true;
   }
   else if (eAct == MM::BeforeGet)
   {
      pProp->Set(fastImage_?1L:0L);
   }

   return DEVICE_OK;
}

int CMMTUCam::OnSaturatePixels(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::AfterSet)
   {
      long tvalue = 0;
      pProp->Get(tvalue);
		saturatePixels_ = (0==tvalue)?false:true;
   }
   else if (eAct == MM::BeforeGet)
   {
      pProp->Set(saturatePixels_?1L:0L);
   }

   return DEVICE_OK;
}

int CMMTUCam::OnFractionOfPixelsToDropOrSaturate(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::AfterSet)
   {
      double tvalue = 0;
      pProp->Get(tvalue);
		fractionOfPixelsToDropOrSaturate_ = tvalue;
   }
   else if (eAct == MM::BeforeGet)
   {
      pProp->Set(fractionOfPixelsToDropOrSaturate_);
   }

   return DEVICE_OK;
}

int CMMTUCam::OnShouldRotateImages(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::AfterSet)
   {
      long tvalue = 0;
      pProp->Get(tvalue);
      shouldRotateImages_ = (tvalue != 0);
   }
   else if (eAct == MM::BeforeGet)
   {
      pProp->Set((long) shouldRotateImages_);
   }

   return DEVICE_OK;
}

int CMMTUCam::OnShouldDisplayImageNumber(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::AfterSet)
   {
      long tvalue = 0;
      pProp->Get(tvalue);
      shouldDisplayImageNumber_ = (tvalue != 0);
   }
   else if (eAct == MM::BeforeGet)
   {
      pProp->Set((long) shouldDisplayImageNumber_);
   }

   return DEVICE_OK;
}

int CMMTUCam::OnStripeWidth(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::AfterSet)
   {
      pProp->Get(stripeWidth_);
   }
   else if (eAct == MM::BeforeGet)
   {
      pProp->Set(stripeWidth_);
   }

   return DEVICE_OK;
}
/*
* Handles "ScanMode" property.
* Changes allowed Binning values to test whether the UI updates properly
*/
int CMMTUCam::OnScanMode(MM::PropertyBase* pProp, MM::ActionType eAct)
{ 
   if (eAct == MM::AfterSet) {
      pProp->Get(scanMode_);
      if (initialized_) {
         int ret = OnPropertiesChanged();
         if (ret != DEVICE_OK)
            return ret;
      }
   } else if (eAct == MM::BeforeGet) {
      LogMessage("Reading property ScanMode", true);
      pProp->Set(scanMode_);
   }
   return DEVICE_OK;
}


int CMMTUCam::OnCameraCCDXSize(MM::PropertyBase* pProp , MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
		pProp->Set(cameraCCDXSize_);
   }
   else if (eAct == MM::AfterSet)
   {
      long value;
      pProp->Get(value);
		if ( (value < 16) || (33000 < value))
			return DEVICE_ERR;  // invalid image size
		if( value != cameraCCDXSize_)
		{
			cameraCCDXSize_ = value;
			img_.Resize(cameraCCDXSize_/binSize_, cameraCCDYSize_/binSize_);
		}
   }
	return DEVICE_OK;

}

int CMMTUCam::OnCameraCCDYSize(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
		pProp->Set(cameraCCDYSize_);
   }
   else if (eAct == MM::AfterSet)
   {
      long value;
      pProp->Get(value);
		if ( (value < 16) || (33000 < value))
			return DEVICE_ERR;  // invalid image size
		if( value != cameraCCDYSize_)
		{
			cameraCCDYSize_ = value;
			img_.Resize(cameraCCDXSize_/binSize_, cameraCCDYSize_/binSize_);
		}
   }
	return DEVICE_OK;

}

int CMMTUCam::OnTriggerDevice(MM::PropertyBase* pProp, MM::ActionType eAct)
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


int CMMTUCam::OnCCDTemp(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(ccdT_);
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Get(ccdT_);
   }
   return DEVICE_OK;
}

int CMMTUCam::OnIsSequenceable(MM::PropertyBase* pProp, MM::ActionType eAct)
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
// Private CMMTUCam methods
///////////////////////////////////////////////////////////////////////////////

/**
* Sync internal image buffer size to the chosen property values.
*/
int CMMTUCam::ResizeImageBuffer()
{
    if (NULL == m_hDevice)
        return DEVICE_NOT_CONNECTED;

    char buf[MM::MaxStrLength];

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
    else if ( pixelType.compare(g_PixelType_32bitRGB) == 0)
    {
        byteDepth = 4;
    }
    else if ( pixelType.compare(g_PixelType_32bit) == 0)
    {
        byteDepth = 4;
    }
    else if ( pixelType.compare(g_PixelType_64bitRGB) == 0)
    {
        byteDepth = 8;
    }

	int nHei = 6144;
	int nWid = 6144;

    int64_t w = 0;
    int64_t h = 0;
    if (TryGetFeatureInt64("Width", &w))
        nWid = static_cast<int>(w);
    if (TryGetFeatureInt64("Height", &h))
        nHei = static_cast<int>(h);

#ifdef _WIN64
    img_.Resize(nWid, nHei, byteDepth);
#else
	img_.Resize(nWid, nHei, (4 == m_frame.channels ? 4 : (m_frame.elemBytes * m_frame.channels)));
#endif

    TUDBG_PRINTF("[ResizeImageBuffer] width = %d, height = %d",nWid,  nHei);
    
    return DEVICE_OK;
}

void CMMTUCam::GenerateEmptyImage(ImgBuffer& img)
{
   MMThreadGuard g(imgPixelsLock_);
   if (img.Height() == 0 || img.Width() == 0 || img.Depth() == 0)
      return;

   unsigned char* pBuf = const_cast<unsigned char*>(img.GetPixels());
   memset(pBuf, 0, img.Height()*img.Width()*img.Depth());
}

void CMMTUCam::TestImage(ImgBuffer& img, double exp)
{
	//std::string pixelType;
	char buf[MM::MaxStrLength];
	GetProperty(MM::g_Keyword_PixelType, buf);
	std::string pixelType(buf);

	if (img.Height() == 0 || img.Width() == 0 || img.Depth() == 0)
		return;

	double lSinePeriod = 3.14159265358979 * stripeWidth_;
	unsigned imgWidth = img.Width();
	unsigned int* rawBuf = (unsigned int*) img.GetPixelsRW();
	double maxDrawnVal = 0;
	long lPeriod = (long) imgWidth / 2;
	double dLinePhase = 0.0;
	const double dAmp = exp;
	double cLinePhaseInc = 2.0 * lSinePeriod / 4.0 / img.Height();
	if (shouldRotateImages_) {
		// Adjust the angle of the sin wave pattern based on how many images
		// we've taken, to increase the period (i.e. time between repeat images).
		cLinePhaseInc *= (((int) dPhase_ / 6) % 24) - 12;
	}

	static bool debugRGB = false;
#ifdef TIFFDEMO
	debugRGB = true;
#endif
	static  unsigned char* pDebug  = NULL;
	static unsigned long dbgBufferSize = 0;
	static long iseq = 1;

	// for integer images: bitDepth_ is 8, 10, 12, 16 i.e. it is depth per component
	long maxValue = (1L << bitDepth_)-1;

	long pixelsToDrop = 0;
	if( dropPixels_)
		pixelsToDrop = (long)(0.5 + fractionOfPixelsToDropOrSaturate_*img.Height()*imgWidth);
	long pixelsToSaturate = 0;
	if( saturatePixels_)
		pixelsToSaturate = (long)(0.5 + fractionOfPixelsToDropOrSaturate_*img.Height()*imgWidth);

	unsigned j, k;
	if (pixelType.compare(g_PixelType_8bit) == 0)
	{
		double pedestal = 127 * exp / 100.0 * GetBinning() * GetBinning();
		unsigned char* pBuf = const_cast<unsigned char*>(img.GetPixels());
		for (j=0; j<img.Height(); j++)
		{
			for (k=0; k<imgWidth; k++)
			{
				long lIndex = imgWidth*j + k;
				unsigned char val = (unsigned char) (g_IntensityFactor_ * min(255.0, (pedestal + dAmp * sin(dPhase_ + dLinePhase + (2.0 * lSinePeriod * k) / lPeriod))));
				if (val > maxDrawnVal) {
					maxDrawnVal = val;
				}
				*(pBuf + lIndex) = val;
			}
			dLinePhase += cLinePhaseInc;
		}
		for(int snoise = 0; snoise < pixelsToSaturate; ++snoise)
		{
			j = (unsigned)( (double)(img.Height()-1)*(double)rand()/(double)RAND_MAX);
			k = (unsigned)( (double)(imgWidth-1)*(double)rand()/(double)RAND_MAX);
			*(pBuf + imgWidth*j + k) = (unsigned char)maxValue;
		}
		int pnoise;
		for(pnoise = 0; pnoise < pixelsToDrop; ++pnoise)
		{
			j = (unsigned)( (double)(img.Height()-1)*(double)rand()/(double)RAND_MAX);
			k = (unsigned)( (double)(imgWidth-1)*(double)rand()/(double)RAND_MAX);
			*(pBuf + imgWidth*j + k) = 0;
		}

	}
	else if (pixelType.compare(g_PixelType_16bit) == 0)
	{
		double pedestal = maxValue/2 * exp / 100.0 * GetBinning() * GetBinning();
		double dAmp16 = dAmp * maxValue/255.0; // scale to behave like 8-bit
		unsigned short* pBuf = (unsigned short*) const_cast<unsigned char*>(img.GetPixels());
		for (j=0; j<img.Height(); j++)
		{
			for (k=0; k<imgWidth; k++)
			{
				long lIndex = imgWidth*j + k;
				unsigned short val = (unsigned short) (g_IntensityFactor_ * min((double)maxValue, pedestal + dAmp16 * sin(dPhase_ + dLinePhase + (2.0 * lSinePeriod * k) / lPeriod)));
				if (val > maxDrawnVal) {
					maxDrawnVal = val;
				}
				*(pBuf + lIndex) = val;
			}
			dLinePhase += cLinePhaseInc;
		}         
		for(int snoise = 0; snoise < pixelsToSaturate; ++snoise)
		{
			j = (unsigned)(0.5 + (double)img.Height()*(double)rand()/(double)RAND_MAX);
			k = (unsigned)(0.5 + (double)imgWidth*(double)rand()/(double)RAND_MAX);
			*(pBuf + imgWidth*j + k) = (unsigned short)maxValue;
		}
		int pnoise;
		for(pnoise = 0; pnoise < pixelsToDrop; ++pnoise)
		{
			j = (unsigned)(0.5 + (double)img.Height()*(double)rand()/(double)RAND_MAX);
			k = (unsigned)(0.5 + (double)imgWidth*(double)rand()/(double)RAND_MAX);
			*(pBuf + imgWidth*j + k) = 0;
		}

	}
	else if (pixelType.compare(g_PixelType_32bit) == 0)
	{
		double pedestal = 127 * exp / 100.0 * GetBinning() * GetBinning();
		float* pBuf = (float*) const_cast<unsigned char*>(img.GetPixels());
		float saturatedValue = 255.;
		memset(pBuf, 0, img.Height()*imgWidth*4);
		// static unsigned int j2;
		for (j=0; j<img.Height(); j++)
		{
			for (k=0; k<imgWidth; k++)
			{
				long lIndex = imgWidth*j + k;
				double value =  (g_IntensityFactor_ * min(255.0, (pedestal + dAmp * sin(dPhase_ + dLinePhase + (2.0 * lSinePeriod * k) / lPeriod))));
				if (value > maxDrawnVal) {
					maxDrawnVal = value;
				}
				*(pBuf + lIndex) = (float) value;
				if( 0 == lIndex)
				{
					std::ostringstream os;
					os << " first pixel is " << (float)value;
					LogMessage(os.str().c_str(), true);

				}
			}
			dLinePhase += cLinePhaseInc;
		}

		for(int snoise = 0; snoise < pixelsToSaturate; ++snoise)
		{
			j = (unsigned)(0.5 + (double)img.Height()*(double)rand()/(double)RAND_MAX);
			k = (unsigned)(0.5 + (double)imgWidth*(double)rand()/(double)RAND_MAX);
			*(pBuf + imgWidth*j + k) = saturatedValue;
		}
		int pnoise;
		for(pnoise = 0; pnoise < pixelsToDrop; ++pnoise)
		{
			j = (unsigned)(0.5 + (double)img.Height()*(double)rand()/(double)RAND_MAX);
			k = (unsigned)(0.5 + (double)imgWidth*(double)rand()/(double)RAND_MAX);
			*(pBuf + imgWidth*j + k) = 0;
		}

	}
	else if (pixelType.compare(g_PixelType_32bitRGB) == 0)
	{
		double pedestal = 127 * exp / 100.0;
		unsigned int * pBuf = (unsigned int*) rawBuf;

		unsigned char* pTmpBuffer = NULL;

		if(debugRGB)
		{
			const unsigned long bfsize = img.Height() * imgWidth * 3;
			if(  bfsize != dbgBufferSize)
			{
				if (NULL != pDebug)
				{
					free(pDebug);
					pDebug = NULL;
				}
				pDebug = (unsigned char*)malloc( bfsize);
				if( NULL != pDebug)
				{
					dbgBufferSize = bfsize;
				}
			}
		}

		// only perform the debug operations if pTmpbuffer is not 0
		pTmpBuffer = pDebug;
		unsigned char* pTmp2 = pTmpBuffer;
		if( NULL!= pTmpBuffer)
			memset( pTmpBuffer, 0, img.Height() * imgWidth * 3);

		for (j=0; j<img.Height(); j++)
		{
			unsigned char theBytes[4];
			for (k=0; k<imgWidth; k++)
			{
				long lIndex = imgWidth*j + k;
				unsigned char value0 =   (unsigned char) min(255.0, (pedestal + dAmp * sin(dPhase_ + dLinePhase + (2.0 * lSinePeriod * k) / lPeriod)));
				theBytes[0] = value0;
				if( NULL != pTmpBuffer)
					pTmp2[2] = value0;
				unsigned char value1 =   (unsigned char) min(255.0, (pedestal + dAmp * sin(dPhase_ + dLinePhase*2 + (2.0 * lSinePeriod * k) / lPeriod)));
				theBytes[1] = value1;
				if( NULL != pTmpBuffer)
					pTmp2[1] = value1;
				unsigned char value2 = (unsigned char) min(255.0, (pedestal + dAmp * sin(dPhase_ + dLinePhase*4 + (2.0 * lSinePeriod * k) / lPeriod)));
				theBytes[2] = value2;

				if( NULL != pTmpBuffer){
					pTmp2[0] = value2;
					pTmp2+=3;
				}
				theBytes[3] = 0;
				unsigned long tvalue = *(unsigned long*)(&theBytes[0]);
				if (tvalue > maxDrawnVal) {
					maxDrawnVal = tvalue;
				}
				*(pBuf + lIndex) =  tvalue ;  //value0+(value1<<8)+(value2<<16);
			}
			dLinePhase += cLinePhaseInc;
		}


		// ImageJ's AWT images are loaded with a Direct Color processor which expects BGRA, that's why we swapped the Blue and Red components in the generator above.
		if(NULL != pTmpBuffer)
		{
			// write the compact debug image...
			char ctmp[12];
			snprintf(ctmp,12,"%ld",iseq++);
			writeCompactTiffRGB(imgWidth, img.Height(), pTmpBuffer, ("democamera" + std::string(ctmp)).c_str());
		}

	}

	// generate an RGB image with bitDepth_ bits in each color
	else if (pixelType.compare(g_PixelType_64bitRGB) == 0)
	{
		double pedestal = maxValue/2 * exp / 100.0 * GetBinning() * GetBinning();
		double dAmp16 = dAmp * maxValue/255.0; // scale to behave like 8-bit

		double maxPixelValue = (1<<(bitDepth_))-1;
		unsigned long long * pBuf = (unsigned long long*) rawBuf;
		for (j=0; j<img.Height(); j++)
		{
			for (k=0; k<imgWidth; k++)
			{
				long lIndex = imgWidth*j + k;
				unsigned long long value0 = (unsigned short) min(maxPixelValue, (pedestal + dAmp16 * sin(dPhase_ + dLinePhase + (2.0 * lSinePeriod * k) / lPeriod)));
				unsigned long long value1 = (unsigned short) min(maxPixelValue, (pedestal + dAmp16 * sin(dPhase_ + dLinePhase*2 + (2.0 * lSinePeriod * k) / lPeriod)));
				unsigned long long value2 = (unsigned short) min(maxPixelValue, (pedestal + dAmp16 * sin(dPhase_ + dLinePhase*4 + (2.0 * lSinePeriod * k) / lPeriod)));
				unsigned long long tval = value0+(value1<<16)+(value2<<32);
				if (tval > maxDrawnVal) {
					maxDrawnVal = static_cast<double>(tval);
				}
				*(pBuf + lIndex) = tval;
			}
			dLinePhase += cLinePhaseInc;
		}
	}

	if (shouldDisplayImageNumber_) {
		// Draw a seven-segment display in the upper-left corner of the image,
		// indicating the image number.
		int divisor = 1;
		int numDigits = 0;
		while (imageCounter_ / divisor > 0) {
			divisor *= 10;
			numDigits += 1;
		}
		int remainder = imageCounter_;
		for (int i = 0; i < numDigits; ++i) {
			// Black out the background for this digit.
			// TODO: for now, hardcoded sizes, which will cause buffer
			// overflows if the image size is too small -- but that seems
			// unlikely.
			int xBase = (numDigits - i - 1) * 20 + 2;
			int yBase = 2;
			for (int x = xBase; x < xBase + 20; ++x) {
				for (int y = yBase; y < yBase + 20; ++y) {
					long lIndex = imgWidth*y + x;

					if (pixelType.compare(g_PixelType_8bit) == 0) {
						*((unsigned char*) rawBuf + lIndex) = 0;
					}
					else if (pixelType.compare(g_PixelType_16bit) == 0) {
						*((unsigned short*) rawBuf + lIndex) = 0;
					}
					else if (pixelType.compare(g_PixelType_32bit) == 0 ||
						pixelType.compare(g_PixelType_32bitRGB) == 0) {
							*((unsigned int*) rawBuf + lIndex) = 0;
					}
				}
			}
			// Draw each segment, if appropriate.
			int digit = remainder % 10;
			for (int segment = 0; segment < 7; ++segment) {
				if (!((1 << segment) & SEVEN_SEGMENT_RULES[digit])) {
					// This segment is not drawn.
					continue;
				}
				// Determine if the segment is horizontal or vertical.
				int xStep = SEVEN_SEGMENT_HORIZONTALITY[segment];
				int yStep = (xStep + 1) % 2;
				// Calculate starting point for drawing the segment.
				int xStart = xBase + SEVEN_SEGMENT_X_OFFSET[segment] * 16;
				int yStart = yBase + SEVEN_SEGMENT_Y_OFFSET[segment] * 8 + 1;
				// Draw one pixel at a time of the segment.
				for (int pixNum = 0; pixNum < 8 * (xStep + 1); ++pixNum) {
					long lIndex = imgWidth * (yStart + pixNum * yStep) + (xStart + pixNum * xStep);
					if (pixelType.compare(g_PixelType_8bit) == 0) {
						*((unsigned char*) rawBuf + lIndex) = static_cast<unsigned char>(maxDrawnVal);
					}
					else if (pixelType.compare(g_PixelType_16bit) == 0) {
						*((unsigned short*) rawBuf + lIndex) = static_cast<unsigned short>(maxDrawnVal);
					}
					else if (pixelType.compare(g_PixelType_32bit) == 0 ||
						pixelType.compare(g_PixelType_32bitRGB) == 0) {
							*((unsigned int*) rawBuf + lIndex) = static_cast<unsigned int>(maxDrawnVal);
					}
				}
			}
			remainder /= 10;
		}
	}
	dPhase_ += lSinePeriod / 4.;
}

/**
* Generate a spatial sine wave.
*/
void CMMTUCam::GenerateSyntheticImage(ImgBuffer& img, double exp)
{ 
    MMThreadGuard g(imgPixelsLock_);

    TestImage(img, exp);
    OutputDebugString("[GenerateSyntheticImage]\n");
}


void CMMTUCam::TestResourceLocking(const bool recurse)
{
    if(recurse)
        TestResourceLocking(false);
}

void CMMTUCam::RequeueBuffer(TUC_HBUFFER hBuffer)
{
	if (NULL == m_hDevice || NULL == hBuffer)
		return;
	TucBufferQueue(m_hDevice, hBuffer);
}

size_t CMMTUCam::HandleRingCapacity() const
{
	// Leave the rest of the SDK pool in INPUT. A deep ring holds buffers
	// out of the camera and the frame rate falls when Convert is slow.
	const size_t kRingDepth = 3;
	if (m_bufferCount <= 1)
		return 1;
	size_t cap = static_cast<size_t>(m_bufferCount - 1);
	if (cap > kRingDepth)
		cap = kRingDepth;
	return cap;
}

bool CMMTUCam::HostPoolHasFreeSlot()
{
	std::lock_guard<std::mutex> lock(hostPoolLock_);
	if (hostFrameBytes_ == 0)
		return true;
	return !hostFrameFree_.empty();
}

uint8_t* CMMTUCam::AcquireHostFrame(uint32_t bytes)
{
	if (bytes == 0)
		return NULL;

	std::lock_guard<std::mutex> lock(hostPoolLock_);
	if (hostFrameBytes_ != bytes)
	{
		if (hostFrameFree_.size() != hostFramePool_.size())
			return NULL;
		FreeHostFramePoolLocked();
		hostFrameBytes_ = bytes;
		const int kSlots = 8;
		hostFramePool_.reserve(kSlots);
		hostFrameFree_.reserve(kSlots);
		for (int i = 0; i < kSlots; ++i)
		{
			uint8_t* slot = new uint8_t[bytes];
			hostFramePool_.push_back(slot);
			hostFrameFree_.push_back(slot);
		}
	}
	if (hostFrameFree_.empty())
		return NULL;
	uint8_t* slot = hostFrameFree_.back();
	hostFrameFree_.pop_back();
	return slot;
}

void CMMTUCam::ReleaseHostFrame(uint8_t* buffer)
{
	if (NULL == buffer)
		return;
	std::lock_guard<std::mutex> lock(hostPoolLock_);
	for (size_t i = 0; i < hostFramePool_.size(); ++i)
	{
		if (hostFramePool_[i] == buffer)
		{
			hostFrameFree_.push_back(buffer);
			return;
		}
	}
	delete[] buffer;
}

void CMMTUCam::FreeHostFramePool()
{
	std::lock_guard<std::mutex> lock(hostPoolLock_);
	FreeHostFramePoolLocked();
}

void CMMTUCam::FreeHostFramePoolLocked()
{
	for (size_t i = 0; i < hostFramePool_.size(); ++i)
		delete[] hostFramePool_[i];
	hostFramePool_.clear();
	hostFrameFree_.clear();
	hostFrameBytes_ = 0;
}

int CMMTUCam::QueueAllAnnouncedBuffers()
{
	if (NULL == m_hDevice)
		return DEVICE_NOT_CONNECTED;

	for (uint32_t i = 0; i < m_bufferCount; ++i)
	{
		TUC_HBUFFER hBuffer = NULL;
		if (!TUC_STATUS_IS_SUCCESS(TucBufferGetByID(m_hDevice, i, &hBuffer)) || NULL == hBuffer)
			return DEVICE_NATIVE_MODULE_FAILED;
		if (!TUC_STATUS_IS_SUCCESS(TucBufferQueue(m_hDevice, hBuffer)))
			return DEVICE_NATIVE_MODULE_FAILED;
	}
	return DEVICE_OK;
}

int CMMTUCam::StartCaptureWithCallback()
{
	if (m_bLiving)
		return DEVICE_OK;

	m_bLiving = true;

	if (!TUC_STATUS_IS_SUCCESS(TucBufferSetQueuePolicy(m_hDevice, TUC_BUFFER_QUEUE_POLICY_MANUAL)))
	{
		m_bLiving = false;
		return DEVICE_NATIVE_MODULE_FAILED;
	}

	int nRet = AllocBuffer();
	if (nRet != DEVICE_OK)
	{
		m_bLiving = false;
		return nRet;
	}

	nRet = QueueAllAnnouncedBuffers();
	if (nRet != DEVICE_OK)
	{
		ReleaseBuffer();
		m_bLiving = false;
		return nRet;
	}

	if (!TUC_STATUS_IS_SUCCESS(TucBufferConverterCreate(m_hDevice)))
	{
		ReleaseBuffer();
		TucBufferSetQueuePolicy(m_hDevice, TUC_BUFFER_QUEUE_POLICY_AUTO);
		m_bLiving = false;
		return DEVICE_ERR;
	}
	converterCreated_ = true;

	if (!TUC_STATUS_IS_SUCCESS(TucRegisterBufferCallback(m_hDevice, OnCallbackBuffer, this)))
	{
		ResetCallback();
		ReleaseBuffer();
		m_bLiving = false;
		return DEVICE_NATIVE_MODULE_FAILED;
	}
	callbackFlag_ = true;

	// Start Convert workers before Start so the handle ring can drain (WF-C-02).
	StartConvertWorkers();

	if (!TUC_STATUS_IS_SUCCESS(TucAcquisitionStart(m_hDevice, TUC_ACQ_START_FLAGS_DEFAULT, 0)))
	{
		StopConvertWorkers();
		ResetCallback();
		ReleaseBuffer();
		m_bLiving = false;
		return DEVICE_ERR;
	}

	TUDBG_PRINTF("[StartCaptureWithCallback] -TucAcquisitionStart");
	return DEVICE_OK;
}

void CMMTUCam::OnCallbackBuffer(TUC_HDEVICE hDevice, TUC_HBUFFER hBuffer, void *userContext)
{
	CMMTUCam *pIn = (CMMTUCam *)userContext;
	if (!pIn)
		return;

#ifdef ENABLE_FPS_STATISTICS
	// 统计：增加回调计数
	unsigned long long count = pIn->callbackCount_.fetch_add(1) + 1;
	MM::MMTime currentTime = pIn->GetCurrentMMTime();

	// 每秒输出一次统计信息
	{
		std::lock_guard<std::mutex> statsLock(pIn->statsLock_);
		if (count == 1)
		{
			pIn->statsStartTime_ = currentTime;
			pIn->lastStatsOutputTime_ = currentTime;
		}
		else
		{
			double elapsedMs = (currentTime - pIn->lastStatsOutputTime_).getMsec();
			if (elapsedMs >= 1000.0)
			{
				unsigned long long callbackCnt = pIn->callbackCount_.load();
				unsigned long long processCnt = pIn->processFrameCount_.load();
				unsigned long long waitCnt = pIn->waitForFrameCount_.load();

				double totalElapsedMs = (currentTime - pIn->statsStartTime_).getMsec();
				double callbackFps = (callbackCnt > 0 && totalElapsedMs > 0) ? (callbackCnt * 1000.0 / totalElapsedMs) : 0.0;
				double processFps = (processCnt > 0 && totalElapsedMs > 0) ? (processCnt * 1000.0 / totalElapsedMs) : 0.0;
				double waitFps = (waitCnt > 0 && totalElapsedMs > 0) ? (waitCnt * 1000.0 / totalElapsedMs) : 0.0;

				char dbgMsg[512] = { 0 };
				sprintf_s(dbgMsg, "[Stats] OnCallbackBuffer: %llu calls (%.2f Hz), ProcessFrame: %llu calls (%.2f Hz), WaitForFrame: %llu calls (%.2f Hz)\n",
					callbackCnt, callbackFps, processCnt, processFps, waitCnt, waitFps);
				OutputDebugString(dbgMsg);

				pIn->lastStatsOutputTime_ = currentTime;
			}
		}
	}
#endif

	if (!pIn->callbackFlag_ || NULL == hBuffer)
	{
		if (NULL != hBuffer)
			TucBufferQueue(hDevice, hBuffer);
		return;
	}

	// Ring full: Queue immediately (no Convert). Keeps INPUT from draining.
	bool queued = false;
	{
		std::lock_guard<std::mutex> lck(pIn->queueLock_);
		if (pIn->queueBuffers_.size() < pIn->HandleRingCapacity())
		{
			pIn->queueBuffers_.push(hBuffer);
			queued = true;
		}
	}
	if (!queued)
		TucBufferQueue(hDevice, hBuffer);
	else
		pIn->queueCV_.notify_all();
}

void CMMTUCam::ResetCallback()
{
	if (callbackCamSet.find(m_pid) != callbackCamSet.end())
	{
		callbackFlag_ = false;
		if (NULL != m_hDevice)
			TucUnregisterBufferCallback(m_hDevice);
		{
			std::lock_guard<std::mutex> lck(queueLock_);
			while (queueBuffers_.size())
			{
				TUC_HBUFFER hBuffer = queueBuffers_.front();
				queueBuffers_.pop();
				RequeueBuffer(hBuffer);
			}
		}
		queueCV_.notify_all();

		if (NULL != m_hDevice)
			TucBufferSetQueuePolicy(m_hDevice, TUC_BUFFER_QUEUE_POLICY_AUTO);
	}
}

int CMMTUCam::InitTUCamApi()
{
    TUC_STATUS nRet = TucInitialize(TUC_LOG_LEVEL_INFO);

	if (TUC_STATUS_IS_SUCCESS(nRet) || TUC_STATUS_ALREADY_INITIALIZED == nRet)
	{
		if (TUC_STATUS_IS_SUCCESS(nRet))
		{
			uint32_t count = 0;
			if (TUC_STATUS_IS_SUCCESS(TucDeviceEnumerate(&count)))
				s_nNumCam = static_cast<int>(count);
		}
	}
	else
		return DEVICE_NOT_CONNECTED;

	uint32_t uiCamCount = static_cast<uint32_t>(s_nNumCam);
	if (0 == uiCamCount)
	{
		if (!TUC_STATUS_IS_SUCCESS(TucDeviceEnumerate(&uiCamCount)) || 0 == uiCamCount)
			return DEVICE_NOT_CONNECTED;
		s_nNumCam = static_cast<int>(uiCamCount);
	}

	if (0 == uiCamCount)
		return DEVICE_NOT_CONNECTED;

	if (!TUC_STATUS_IS_SUCCESS(TucDeviceOpen(s_nCntCam, TUC_OPEN_POLICY_EXCLUSIVE, &m_hDevice)))
		return DEVICE_NOT_CONNECTED;

	s_nCntCam++;

    if (NULL == m_hDevice)
        return DEVICE_NOT_CONNECTED;

	if (s_nCntCam > 1)
	{
		LogMessage("Multi-camera mode: CAM_MULTIPLE not configured in SDK3", true);
	}

    return DEVICE_OK;
}

int CMMTUCam::UninitTUCamApi()
{
    ReleaseBuffer();

    if (NULL != m_hDevice)
    {
        OutputDebugString("[TucDeviceClose]\n");
        TucDeviceClose(m_hDevice);
        m_hDevice = NULL;
    }

    TucTerminate();

    return DEVICE_OK;
}

int CMMTUCam::AllocBuffer()
{
    if (NULL == m_hDevice)
        return DEVICE_NOT_CONNECTED;

    if (m_hBuffers)
    {
        TucBuffersRevokeAndDestroy(m_hDevice, m_hBuffers);
        m_hBuffers = NULL;
    }

    if (!TUC_STATUS_IS_SUCCESS(TucBuffersAllocAndAnnounce(m_hDevice, m_bufferCount, &m_hBuffers)))
        return DEVICE_OUT_OF_MEMORY;

    UpdateFrameGeometryFromFeatures();

    std::string pixelFormat;
    int64_t pfVal = 0;
    if (TUC_STATUS_IS_SUCCESS(TucGetFeatureEnum(m_hDevice, "PixelFormat", &pfVal)))
        GetFeatureEnumSymbolic(m_hDevice, "PixelFormat", pfVal, pixelFormat);

    InferPixelLayoutFromFormat(pixelFormat, m_frame.channels, m_frame.elemBytes);

    if (3 == m_frame.channels)
    {
        nComponents_ = 4;

#ifdef _WIN64
        if (2 == m_frame.elemBytes)
        {
            bitDepth_ = 16;
            SetProperty(MM::g_Keyword_PixelType, g_PixelType_64bitRGB);
        }  
        else
        {
            bitDepth_ = 8;
            SetProperty(MM::g_Keyword_PixelType, g_PixelType_32bitRGB);
        } 
#else
        bitDepth_ = 8;
        SetProperty(MM::g_Keyword_PixelType, g_PixelType_32bitRGB);
#endif
    }
    else
    {
        nComponents_ = 1;

        if (2 == m_frame.elemBytes)
        {
            bitDepth_ = 16;
            SetProperty(MM::g_Keyword_PixelType, g_PixelType_16bit);
        }  
        else
        {
            bitDepth_ = 8;
            SetProperty(MM::g_Keyword_PixelType, g_PixelType_8bit);
        }
    }

	img_.Resize(m_frame.width, m_frame.height, m_frame.elemBytes * nComponents_);

    return DEVICE_OK;
}

int CMMTUCam::UpdateImageBuffer()
{
    if (NULL == m_hDevice)
        return DEVICE_NOT_CONNECTED;

    const bool wasLiving = m_bLiving;
    m_bLiving = false;
    TucAcquisitionAbort(m_hDevice);
    TucAcquisitionStop(m_hDevice, TUC_ACQ_STOP_FLAGS_DEFAULT);
    ReleaseBuffer();

    int nRet = AllocBuffer();
    if (nRet != DEVICE_OK)
        return nRet;

    ResizeImageBuffer();

    if (wasLiving)
    {
        if (!TUC_STATUS_IS_SUCCESS(TucAcquisitionStart(m_hDevice, TUC_ACQ_START_FLAGS_DEFAULT, 0)))
            return DEVICE_ERR;
        m_bLiving = true;
    }

    return DEVICE_OK;
}

int CMMTUCam::ReleaseBuffer()
{
    if (NULL == m_hDevice)
        return DEVICE_NOT_CONNECTED;

    if (converterCreated_)
    {
        TucBufferConverterDestroy(m_hDevice);
        converterCreated_ = false;
    }

    if (m_hBuffers)
    {
        TucBuffersRevokeAndDestroy(m_hDevice, m_hBuffers);
        m_hBuffers = NULL;
    }

    return DEVICE_OK;
}

int CMMTUCam::StopCapture()
{
    m_bLiving = false;

	StopConvertWorkers();
	ResetCallback();

    if (NULL != m_hDevice)
    {
        TucAcquisitionAbort(m_hDevice);
        TucAcquisitionStop(m_hDevice, TUC_ACQ_STOP_FLAGS_DEFAULT);
        ReleaseBuffer();
    }
    TUDBG_PRINTF("[StopCapture] ---");
    return DEVICE_OK;
}

int CMMTUCam::StartCapture()
{
    if (m_bLiving)
        return DEVICE_OK;

    m_bLiving = true;

    int nRet = AllocBuffer();
    if (nRet != DEVICE_OK)
    {
        return nRet;
    }

    if (TUC_STATUS_IS_SUCCESS(TucAcquisitionStart(m_hDevice, TUC_ACQ_START_FLAGS_DEFAULT, 0)))
    {
		TUDBG_PRINTF("[StartCapture] -TucAcquisitionStart");
        return nRet;
    }

    TUDBG_PRINTF("[StartCapture] --");

    return DEVICE_ERR;
}

int CMMTUCam::WaitForFrame(ImgBuffer& img)
{
	if (callbackFlag_)
	{
		// Convert workers own the handle ring. Wait here for one completed frame.
		std::unique_lock<std::mutex> lock(convertWaitLock_);
		convertWaitCV_.wait(lock, [this] {
			return thd_->IsStopped() || convertFailed_.load() || convertReady_ > 0;
		});
		if (thd_->IsStopped())
			return SEQUENCE_ACQUISITION_INTERTUPTED;
		if (convertFailed_.load())
			return DEVICE_NATIVE_MODULE_FAILED;
		--convertReady_;
		return DEVICE_OK;
	}

	{
		std::unique_lock<std::mutex> lck(frameCopyLock_);
		TUC_IMAGE_INFO info = {};
		if (!TUC_STATUS_IS_SUCCESS(TucAcquisitionWait(m_hDevice, &info, INFINITE)))
		{
			return DEVICE_NATIVE_MODULE_FAILED;
		}
		FillFrameFromImageInfo(info, m_frame);
	}

	{
		std::lock_guard<std::mutex> lock(frameQueueLock_);
		if (frameQueue_.size() < cycBufSize_)
		{
			frameQueue_.push(m_frame);
			m_frame.owned = false;
			m_frame.pBuffer = NULL;
		}
	}
	ReleaseOwnedFrame(m_frame);
	frameQueueCV_.notify_one();

	return DEVICE_OK;
}

int CMMTUCam::WaitForFrameInSnap()
{
	MMThreadGuard g(imgPixelsLock_);

	TUC_IMAGE_INFO info = {};
	if (TUC_STATUS_IS_SUCCESS(TucAcquisitionWait(m_hDevice, &info, INFINITE)))
	{
		if (img_.Height() == 0 || img_.Width() == 0 || img_.Depth() == 0)
			return DEVICE_OUT_OF_MEMORY;

		FillFrameFromImageInfo(info, m_frame);
		CopyImg(m_frame);

		return DEVICE_OK;
	}

	return DEVICE_NATIVE_MODULE_FAILED;
}

void CMMTUCam::CopyImg(const TuCamImageFrame& frame)
{
	if (NULL == frame.pBuffer || 0 == frame.width || 0 == frame.height)
		return;

	int nWid = frame.width;
	int nHei = frame.height;
	int nPix = nWid * nHei;

	if (2 == frame.elemBytes)
	{
		if (3 == frame.channels)
		{
#ifdef _WIN64
			unsigned short* pSrc = (unsigned short*)(frame.pBuffer);
			unsigned short* pDst = (unsigned short*)(img_.GetPixelsRW());

			for (int i = 0; i < nPix; ++i)
			{
				*pDst++ = *pSrc++;
				*pDst++ = *pSrc++;
				*pDst++ = *pSrc++;
				*pDst++ = 0;
			}
#else
			unsigned short* pSrc = (unsigned short*)(frame.pBuffer);
			unsigned char* pDst = (unsigned char*)(img_.GetPixelsRW());

			for (int i = 0; i < nPix; ++i)
			{
				*pDst++ = (*pSrc++) >> 8;
				*pDst++ = (*pSrc++) >> 8;
				*pDst++ = (*pSrc++) >> 8;
				*pDst++ = 0;
			}
#endif
		}
		else
		{
			unsigned short* pSrc = (unsigned short*)(frame.pBuffer);
			unsigned short* pDst = (unsigned short*)(img_.GetPixelsRW());

			memcpy(pDst, pSrc, frame.imgSize);
		}
	}
	else
	{
		unsigned char* pSrc = (unsigned char*)(frame.pBuffer);
		unsigned char* pDst = (unsigned char*)(img_.GetPixelsRW());

		if (3 == frame.channels)
		{
			for (int i = 0; i < nPix; ++i)
			{
				*pDst++ = *pSrc++;
				*pDst++ = *pSrc++;
				*pDst++ = *pSrc++;
				*pDst++ = 0;
			}
		}
		else
		{
			memcpy(pDst, pSrc, frame.imgSize);
		}
	}
}

bool CMMTUCam::SaveRaw(char *pfileName, unsigned char *pData, unsigned long ulSize)
{
    FILE *pfile = NULL;
    string szPath = pfileName;
    szPath += ".raw";

    OutputDebugString(szPath.c_str());

    pfile = fopen(szPath.c_str(), "wb");

    if(NULL != pfile) 
    {
        fwrite(pData, 1, ulSize, pfile);
        fclose(pfile);

//        delete pfile;

        pfile = NULL;
        OutputDebugString("[SaveRaw]:NULL!\n");

        return true;
    }

    return false;
}

int CMMTUCam::GetPID()
{
	char buf[4] = { 0 };
	TucVendorAddressRead(m_hDevice, buf, GENICAM_PID_ADDRESS, 4);
	int value = ((INT)(buf[0] & 0xFF)) << 24 | ((INT)(buf[1] & 0xFF)) << 16 | ((INT)(buf[2] & 0xFF)) << 8 | (buf[3] & 0xFF);
	return value & 0xFFFF;
}

void CMMTUCam::ReleaseOwnedFrame(TuCamImageFrame& frame)
{
	if (frame.owned && NULL != frame.pBuffer)
	{
		ReleaseHostFrame(frame.pBuffer);
		frame.pBuffer = NULL;
		frame.owned = false;
	}
}

int CMMTUCam::ProcessFrameAndInsertImage(TuCamImageFrame frame)
{
	{
		std::unique_lock<std::mutex> lock(frameCopyLock_);
		CopyImg(frame);
	}
	ReleaseOwnedFrame(frame);

    return InsertImage();
}

void CMMTUCam::StartConvertWorkers()
{
	if (convertWorkersRunning_.load())
		return;

	// Join workers that already left after a convert failure.
	StopConvertWorkers();

	convertFailed_.store(false);
	{
		std::lock_guard<std::mutex> lock(convertWaitLock_);
		convertReady_ = 0;
	}
	convertWorkersRunning_.store(true);
	convertWorkers_.reserve(kConvertWorkerCount);
	for (int i = 0; i < kConvertWorkerCount; ++i)
		convertWorkers_.emplace_back(&CMMTUCam::ConvertWorkerFunc, this);
}

void CMMTUCam::StopConvertWorkers()
{
	convertWorkersRunning_.store(false);
	queueCV_.notify_all();
	convertWaitCV_.notify_all();
	for (size_t i = 0; i < convertWorkers_.size(); ++i)
	{
		if (convertWorkers_[i].joinable())
			convertWorkers_[i].join();
	}
	convertWorkers_.clear();
	{
		std::lock_guard<std::mutex> lock(convertWaitLock_);
		convertReady_ = 0;
	}
}

void CMMTUCam::ConvertWorkerFunc()
{
	while (convertWorkersRunning_.load())
	{
		TUC_HBUFFER hBuffer = NULL;
		{
			std::unique_lock<std::mutex> lck(queueLock_);
			queueCV_.wait(lck, [this] {
				return !convertWorkersRunning_.load() || !queueBuffers_.empty();
			});
			if (!convertWorkersRunning_.load())
				break;
			if (queueBuffers_.empty())
				continue;
			hBuffer = queueBuffers_.front();
			queueBuffers_.pop();
		}

		// No free host slot: return the SDK buffer before Convert.
		if (!HostPoolHasFreeSlot())
		{
			RequeueBuffer(hBuffer);
			continue;
		}

		TUC_IMAGE_INFO info = {};
		TuCamImageFrame frame;
		TUC_STATUS convertStatus = TUC_STATUS_OK;
		{
			// One Convert at a time per device; other workers can memcpy/Queue meanwhile.
			std::lock_guard<std::mutex> convertGuard(convertLock_);
			convertStatus = TucBufferConvertToImage(m_hDevice, hBuffer, TUC_CONVERSION_POLICY_BASIC, &info);
		}
		if (!TUC_STATUS_IS_SUCCESS(convertStatus))
		{
			RequeueBuffer(hBuffer);
			convertFailed_.store(true);
			// Ask both workers to leave. Do not join here (that deadlocks).
			convertWorkersRunning_.store(false);
			queueCV_.notify_all();
			convertWaitCV_.notify_all();
			break;
		}

		FillFrameFromImageInfo(info, frame);
		uint8_t* copy = (NULL != frame.pBuffer && 0 != frame.imgSize)
			? AcquireHostFrame(frame.imgSize) : NULL;
		if (NULL == copy)
		{
			RequeueBuffer(hBuffer);
			continue;
		}

		memcpy(copy, frame.pBuffer, frame.imgSize);
		frame.pBuffer = copy;
		frame.owned = true;
		RequeueBuffer(hBuffer);

		{
			std::lock_guard<std::mutex> lock(frameQueueLock_);
			if (frameQueue_.size() < cycBufSize_)
			{
				frameQueue_.push(frame);
				frame.owned = false;
				frame.pBuffer = NULL;
			}
		}
		ReleaseOwnedFrame(frame);
		frameQueueCV_.notify_one();

		{
			std::lock_guard<std::mutex> lock(convertWaitLock_);
			++convertReady_;
		}
		convertWaitCV_.notify_one();
	}
}

void CMMTUCam::StartFrameProcessingThread()
{
    if (!frameProcessingThreadRunning_.load())
    {
        frameProcessingThreadRunning_.store(true);
        frameProcessingThread_ = std::thread(&CMMTUCam::FrameProcessingThreadFunc, this);
    }
}

void CMMTUCam::StopFrameProcessingThread()
{
    if (frameProcessingThreadRunning_.load())
    {
        frameProcessingThreadRunning_.store(false);
        frameQueueCV_.notify_all();
        
        if (frameProcessingThread_.joinable())
        {
            frameProcessingThread_.join();
        }

		while (!frameQueue_.empty())
		{
			TuCamImageFrame dropped = frameQueue_.front();
			frameQueue_.pop();
			ReleaseOwnedFrame(dropped);
		}
    }
}

void CMMTUCam::FrameProcessingThreadFunc()
{
    while (frameProcessingThreadRunning_.load())
    {
        TuCamImageFrame frame;
        bool hasFrame = false;
        
        {
            std::unique_lock<std::mutex> lock(frameQueueLock_);
            frameQueueCV_.wait(lock, [this] { 
                return !frameProcessingThreadRunning_.load() || !frameQueue_.empty(); 
            });
            
			if (!frameProcessingThreadRunning_.load())
			{
				break;
			}
                
            if (!frameQueue_.empty())
            {
                frame = frameQueue_.front();
                frameQueue_.pop();
                hasFrame = true;
            }
        }
        
        if (hasFrame)
        {
            ProcessFrameAndInsertImage(frame);
#ifdef ENABLE_FPS_STATISTICS
			// 统计：增加ProcessFrameAndInsertImage计数
			processFrameCount_.fetch_add(1);
#endif
        }
    }
}

void CMMTUCam::FillFrameFromImageInfo(const TUC_IMAGE_INFO& info, TuCamImageFrame& frame)
{
    frame.width = info.width;
    frame.height = info.height;
    frame.widthStep = info.widthStep;
    frame.imgSize = info.bytesCount;
    frame.channels = info.channels;
    frame.elemBytes = info.bytesPerElement;
    frame.pBuffer = info.data;
}

bool CMMTUCam::TryGetFeatureInt64(const char* name, int64_t* v) const
{
    return m_hDevice && TUC_STATUS_IS_SUCCESS(TucGetFeatureInt64(m_hDevice, name, v));
}

bool CMMTUCam::TrySetFeatureInt64(const char* name, int64_t v)
{
    return m_hDevice && TUC_STATUS_IS_SUCCESS(TucSetFeatureInt64(m_hDevice, name, v));
}

bool CMMTUCam::TryGetFeatureFloat(const char* name, double* v) const
{
    return m_hDevice && TUC_STATUS_IS_SUCCESS(TucGetFeatureFloat(m_hDevice, name, v));
}

bool CMMTUCam::TrySetFeatureFloat(const char* name, double v)
{
    return m_hDevice && TUC_STATUS_IS_SUCCESS(TucSetFeatureFloat(m_hDevice, name, v));
}

bool CMMTUCam::TrySetFeatureBool(const char* name, bool v)
{
    return m_hDevice && TUC_STATUS_IS_SUCCESS(TucSetFeatureBool(m_hDevice, name, v));
}

void CMMTUCam::UpdateFrameGeometryFromFeatures()
{
    int64_t w = 0;
    int64_t h = 0;
    TryGetFeatureInt64("Width", &w);
    TryGetFeatureInt64("Height", &h);
    if (w > 0)
        m_frame.width = static_cast<uint32_t>(w);
    if (h > 0)
        m_frame.height = static_cast<uint32_t>(h);
}
