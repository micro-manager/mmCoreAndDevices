///////////////////////////////////////////////////////////////////////////////
// FILE:          Mightex_SB_Camera.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   The example implementation of the Mightex Super Speed USB camera(SB-Series).
//                Simulates generic digital camera and associated automated
//                microscope devices and enables testing of the rest of the
//                system without the need to connect to the actual hardware.
//
// AUTHOR:        Yihui, mightexsystem.com, 02/21/2025
//
// COPYRIGHT:     University of California, San Francisco, 2006
//                100X Imaging Inc, 2008
//                Mightex Systems, 2025
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

#include "Mightex_SB_Camera.h"
#include <cstdio>
#include <string>
#include <math.h>
#include "ModuleInterface.h"
#include <sstream>
#include <algorithm>
#include <iostream>

#include "SSBuffer_USBCamera_SDK_Stdcall.h"


using namespace std;
const double CMightex_SB_Camera::nominalPixelSizeUm_ = 1.0;
double g_IntensityFactor_ = 1.0;
int OnExposureCnt = 0;

// External names used used by the rest of the system
// to load particular device from the "SSBuffer_USBCamera_SDK_Stdcall.dll" library
const char* g_Camera_SB_DeviceName = "Mightex_SB_Camera";
const char* g_Keyword_Resolution = "Resolution";
const char* g_Keyword_Resolution_Ry = "Resolution_Ry";
const char* g_Keyword_YStart = "Y_Offset";
const char* g_Keyword_BinMode = "BinMode";

// constants for naming pixel types (allowed values of the "PixelType" property)
const char* g_PixelType_8bit = "8bit";
const char* g_PixelType_16bit = "16bit";
const char* g_PixelType_32bitRGB = "32bitRGB";


///////////////////////////////////////////////////////////////////////////////
// Exported MMDevice API
///////////////////////////////////////////////////////////////////////////////

MODULE_API void InitializeModuleData()
{
   RegisterDevice(g_Camera_SB_DeviceName, MM::CameraDevice, "Mightex Super Speed USB camera(SB-Series)");
}

MODULE_API MM::Device* CreateDevice(const char* deviceName)
{
   if (deviceName == 0)
      return 0;

   // decide which device class to create based on the deviceName parameter
   if (strcmp(deviceName, g_Camera_SB_DeviceName) == 0)
   {
      // create camera
      return new CMightex_SB_Camera();
   }

   // ...supplied name not recognized
   return 0;
}

MODULE_API void DeleteDevice(MM::Device* pDevice)
{
   delete pDevice;
}


//////////////////////////////////////////////////////////////////////////
enum eDEVICETYPE_SB { IMX273M, IMX273C };
enum eWORKMODE { CONTINUE_MODE , EXT_TRIGGER_MODE };

const int NORMAL_FRAMES = 4;
const int TRIGGER_FRAMES= 24;
const int CAMERA_SB_BUFFER_BW[] = { 128, 128, 128, 128 };
const int CAMERA_SB_BUFFER_COLOR[] = { 128, 128,128, 128 };

const int  SB_MAX_RESOLUTIONS = 3; //15;
const int  SB_MAX_RESOLUTIONS_V015 = 3;

const int SB_DEFAULT_WIDTH = 1408;
const int SB_DEFAULT_HEIGHT = 1088;
const int SB_DEFAULT_SIZE = SB_DEFAULT_WIDTH*SB_DEFAULT_HEIGHT;

/********************************************** 
 * 
 *  struct declarations
 * 
 **********************************************/

static struct { int width; int height; int frameSize; }
s_vidFrameSize[] =
{
	{ 1408,  272, 1408 * 272},
	{ 1408,  362, 1408 * 362},
	{ 1408,  544, 1408 * 544},
	{ 1408, 1088, 1408 * 1088},
};

const char g_Res[SB_MAX_RESOLUTIONS+1][10] =
{
	{ "1408*272"},
	{ "1408*362"},
	{ "1408*544"},
	{ "1408*1088"},
};

static struct { int width; int height; }
frameSize_IMX273[] =
{
	{  1408,  1088},
	{  704,  544},
	{  468,  362},
	{  352,  272},
};

//////////////////////////////////////////////////////////////////////////
//Camera API function pointer variables

SSBufferUSB_InitDevicePtr SSBufferUSB_InitDevice;
SSBufferUSB_UnInitDevicePtr SSBufferUSB_UnInitDevice;
SSBufferUSB_GetModuleNoSerialNoPtr SSBufferUSB_GetModuleNoSerialNo;
SSBufferUSB_AddDeviceToWorkingSetPtr SSBufferUSB_AddDeviceToWorkingSet;
SSBufferUSB_RemoveDeviceFromWorkingSetPtr SSBufferUSB_RemoveDeviceFromWorkingSet;
SSBufferUSB_ActiveDeviceInWorkingSetPtr SSBufferUSB_ActiveDeviceInWorkingSet;
SSBufferUSB_StartCameraEngineExPtr SSBufferUSB_StartCameraEngineEx;
SSBufferUSB_StartCameraEnginePtr SSBufferUSB_StartCameraEngine;
SSBufferUSB_StopCameraEnginePtr SSBufferUSB_StopCameraEngine;
SSBufferUSB_SetUSBConnectMonitorPtr SSBufferUSB_SetUSBConnectMonitor;
SSBufferUSB_SetUSB30TransferSizePtr SSBufferUSB_SetUSB30TransferSize;
SSBufferUSB_GetCameraFirmwareVersionPtr SSBufferUSB_GetCameraFirmwareVersion;
SSBufferUSB_StartFrameGrabExPtr SSBufferUSB_StartFrameGrabEx;
SSBufferUSB_StopFrameGrabExPtr SSBufferUSB_StopFrameGrabEx;
SSBufferUSB_StartFrameGrabPtr SSBufferUSB_StartFrameGrab;
SSBufferUSB_StopFrameGrabPtr SSBufferUSB_StopFrameGrab;
SSBufferUSB_ShowFactoryControlPanelPtr SSBufferUSB_ShowFactoryControlPanel;
SSBufferUSB_HideFactoryControlPanelPtr SSBufferUSB_HideFactoryControlPanel;
SSBufferUSB_SetBayerFilterTypePtr SSBufferUSB_SetBayerFilterType;
SSBufferUSB_SetCameraWorkModePtr SSBufferUSB_SetCameraWorkMode;
SSBufferUSB_SetCustomizedResolutionExPtr SSBufferUSB_SetCustomizedResolutionEx;
SSBufferUSB_SetCustomizedResolutionPtr SSBufferUSB_SetCustomizedResolution;
SSBufferUSB_SetExposureTimePtr SSBufferUSB_SetExposureTime;
SSBufferUSB_SetFrameTimePtr SSBufferUSB_SetFrameTime;
SSBufferUSB_SetXYStartPtr SSBufferUSB_SetXYStart;
SSBufferUSB_SetGainsPtr SSBufferUSB_SetGains;
SSBufferUSB_SetGainRatiosPtr SSBufferUSB_SetGainRatios;
SSBufferUSB_SetGammaPtr SSBufferUSB_SetGamma;
SSBufferUSB_SetBWModePtr SSBufferUSB_SetBWMode;
SSBufferUSB_SetMinimumFrameDelayPtr SSBufferUSB_SetMinimumFrameDelay;
SSBufferUSB_SoftTriggerPtr SSBufferUSB_SoftTrigger;
SSBufferUSB_SetSensorBlankingsPtr SSBufferUSB_SetSensorBlankings;
SSBufferUSB_InstallFrameCallbackPtr SSBufferUSB_InstallFrameCallback;
SSBufferUSB_InstallUSBDeviceCallbackPtr SSBufferUSB_InstallUSBDeviceCallback;
SSBufferUSB_InstallFrameHookerPtr SSBufferUSB_InstallFrameHooker;
SSBufferUSB_InstallUSBDeviceHookerPtr SSBufferUSB_InstallUSBDeviceHooker;
SSBufferUSB_GetCurrentFramePtr SSBufferUSB_GetCurrentFrame;
SSBufferUSB_GetCurrentFrame16bitPtr SSBufferUSB_GetCurrentFrame16bit;
SSBufferUSB_GetCurrentFrameParaPtr SSBufferUSB_GetCurrentFramePara;
SSBufferUSB_GetDevicesErrorStatePtr SSBufferUSB_GetDevicesErrorState;
SSBufferUSB_IsUSBSuperSpeedPtr SSBufferUSB_IsUSBSuperSpeed;
SSBufferUSB_SetGPIOConfigPtr SSBufferUSB_SetGPIOConfig;
SSBufferUSB_SetGPIOOutPtr SSBufferUSB_SetGPIOOut;
SSBufferUSB_SetGPIOInOutPtr SSBufferUSB_SetGPIOInOut;
//////////////////////////////////////////////////////////////////////////

MMThreadLock g_imgPixelsLock_;
unsigned char *g_pImage;
int g_frameSize = SB_DEFAULT_SIZE;
int g_frameSize_width = SB_DEFAULT_WIDTH;
int g_deviceColorType = 0;
int g_InstanceCount = 0;
long g_xStart = 0;
HANDLE SingleFrameEvent = NULL;

/////////////////////////////////////////////////////////////////////////////
//CallBack Function

void CameraFaultCallBack(int DeviceID, int DeviceType )
{
	// Note: It's recommended to stop the engine and close the application
	SSBufferUSB_StopCameraEngine();
	SSBufferUSB_UnInitDevice();
}

void FrameCallBack(TProcessedDataProperty *Attributes, unsigned char *BytePtr)
{

	MMThreadGuard g_g(g_imgPixelsLock_);

	if (Attributes == NULL) return;

	if(g_frameSize_width == Attributes->Column)
	{
		if(BytePtr != NULL)
			memcpy(g_pImage, BytePtr, g_frameSize);
		if ( Attributes->WorkMode == 1 ) // Trigger mode generated (snap image)
                    SetEvent( SingleFrameEvent ); 
		return;
	}

}


int CMightex_SB_Camera::InitCamera()
{
	if(g_InstanceCount > 1)
		return DEVICE_ERR;

  HDll = LoadLibraryA("SSBuffer_USBCamera_SDK_Stdcall.dll");
  if (HDll)
  {
	SSBufferUSB_InitDevice = (SSBufferUSB_InitDevicePtr)GetProcAddress(HDll,"SSBufferUSB_InitDevice");
	SSBufferUSB_UnInitDevice = (SSBufferUSB_UnInitDevicePtr)GetProcAddress(HDll,"SSBufferUSB_UnInitDevice");
	SSBufferUSB_GetModuleNoSerialNo = (SSBufferUSB_GetModuleNoSerialNoPtr)GetProcAddress(HDll,"SSBufferUSB_GetModuleNoSerialNo");
	SSBufferUSB_AddDeviceToWorkingSet = (SSBufferUSB_AddDeviceToWorkingSetPtr) GetProcAddress(HDll,"SSBufferUSB_AddDeviceToWorkingSet");
	SSBufferUSB_RemoveDeviceFromWorkingSet = (SSBufferUSB_RemoveDeviceFromWorkingSetPtr)GetProcAddress(HDll,"SSBufferUSB_RemoveDeviceFromWorkingSet");
	SSBufferUSB_ActiveDeviceInWorkingSet = (SSBufferUSB_ActiveDeviceInWorkingSetPtr)GetProcAddress(HDll,"SSBufferUSB_ActiveDeviceInWorkingSet");
	SSBufferUSB_StartCameraEngineEx = (SSBufferUSB_StartCameraEngineExPtr)GetProcAddress(HDll, "SSBufferUSB_StartCameraEngineEx");
	SSBufferUSB_StartCameraEngine = (SSBufferUSB_StartCameraEnginePtr)GetProcAddress(HDll,"SSBufferUSB_StartCameraEngine");
	SSBufferUSB_StopCameraEngine = (SSBufferUSB_StopCameraEnginePtr)GetProcAddress(HDll,"SSBufferUSB_StopCameraEngine");
	SSBufferUSB_SetUSBConnectMonitor = (SSBufferUSB_SetUSBConnectMonitorPtr)GetProcAddress(HDll, "SSBufferUSB_SetUSBConnectMonitor");
	SSBufferUSB_SetUSB30TransferSize = (SSBufferUSB_SetUSB30TransferSizePtr)GetProcAddress(HDll, "SSBufferUSB_SetUSB30TransferSize");
	SSBufferUSB_GetCameraFirmwareVersion = (SSBufferUSB_GetCameraFirmwareVersionPtr)GetProcAddress(HDll, "SSBufferUSB_GetCameraFirmwareVersion");
	SSBufferUSB_StartFrameGrabEx = (SSBufferUSB_StartFrameGrabExPtr)GetProcAddress(HDll,"SSBufferUSB_StartFrameGrabEx");
	SSBufferUSB_StopFrameGrabEx = (SSBufferUSB_StopFrameGrabExPtr)GetProcAddress(HDll,"SSBufferUSB_StopFrameGrabEx");
	SSBufferUSB_StartFrameGrab = (SSBufferUSB_StartFrameGrabPtr)GetProcAddress(HDll, "SSBufferUSB_StartFrameGrab");
	SSBufferUSB_StopFrameGrab = (SSBufferUSB_StopFrameGrabPtr)GetProcAddress(HDll, "SSBufferUSB_StopFrameGrab");
	SSBufferUSB_SetBayerFilterType = (SSBufferUSB_SetBayerFilterTypePtr)GetProcAddress(HDll, "SSBufferUSB_SetBayerFilterType");
	SSBufferUSB_SetCameraWorkMode = (SSBufferUSB_SetCameraWorkModePtr)GetProcAddress(HDll, "SSBufferUSB_SetCameraWorkMode");
	SSBufferUSB_SetCustomizedResolutionEx = (SSBufferUSB_SetCustomizedResolutionExPtr)GetProcAddress(HDll, "SSBufferUSB_SetCustomizedResolutionEx");
	SSBufferUSB_SetCustomizedResolution = (SSBufferUSB_SetCustomizedResolutionPtr)GetProcAddress(HDll,"SSBufferUSB_SetCustomizedResolution");
	SSBufferUSB_SetExposureTime = (SSBufferUSB_SetExposureTimePtr)GetProcAddress(HDll,"SSBufferUSB_SetExposureTime");
	SSBufferUSB_SetFrameTime = (SSBufferUSB_SetFrameTimePtr)GetProcAddress(HDll,"SSBufferUSB_SetFrameTime");
	SSBufferUSB_SetXYStart = (SSBufferUSB_SetXYStartPtr)GetProcAddress(HDll,"SSBufferUSB_SetXYStart");
	SSBufferUSB_SetGains = (SSBufferUSB_SetGainsPtr)GetProcAddress(HDll,"SSBufferUSB_SetGains");
	SSBufferUSB_SetGainRatios = (SSBufferUSB_SetGainRatiosPtr)GetProcAddress(HDll, "SSBufferUSB_SetGainRatios");
	SSBufferUSB_SetGamma = (SSBufferUSB_SetGammaPtr)GetProcAddress(HDll,"SSBufferUSB_SetGamma");
	SSBufferUSB_SetBWMode = (SSBufferUSB_SetBWModePtr)GetProcAddress(HDll,"SSBufferUSB_SetBWMode");
	SSBufferUSB_SetMinimumFrameDelay = (SSBufferUSB_SetMinimumFrameDelayPtr)GetProcAddress(HDll, "SSBufferUSB_SetMinimumFrameDelay");
	SSBufferUSB_SoftTrigger = (SSBufferUSB_SoftTriggerPtr)GetProcAddress(HDll,"SSBufferUSB_SoftTrigger");
	SSBufferUSB_SetSensorBlankings = (SSBufferUSB_SetSensorBlankingsPtr)GetProcAddress(HDll, "SSBufferUSB_SetSensorBlankings");
	SSBufferUSB_InstallFrameCallback = (SSBufferUSB_InstallFrameCallbackPtr)GetProcAddress(HDll,"SSBufferUSB_InstallFrameCallback");
	SSBufferUSB_InstallUSBDeviceCallback = (SSBufferUSB_InstallUSBDeviceCallbackPtr)GetProcAddress(HDll,"SSBufferUSB_InstallUSBDeviceCallback");
	SSBufferUSB_InstallFrameHooker = (SSBufferUSB_InstallFrameHookerPtr)GetProcAddress(HDll, "SSBufferUSB_InstallFrameHooker");
	SSBufferUSB_InstallUSBDeviceHooker = (SSBufferUSB_InstallUSBDeviceHookerPtr)GetProcAddress(HDll, "SSBufferUSB_InstallUSBDeviceHooker");
	SSBufferUSB_GetCurrentFrame = (SSBufferUSB_GetCurrentFramePtr)GetProcAddress(HDll, "SSBufferUSB_GetCurrentFrame");
	SSBufferUSB_GetCurrentFrame16bit = (SSBufferUSB_GetCurrentFrame16bitPtr)GetProcAddress(HDll, "SSBufferUSB_GetCurrentFrame16bit");
	SSBufferUSB_GetCurrentFramePara = (SSBufferUSB_GetCurrentFrameParaPtr)GetProcAddress(HDll, "SSBufferUSB_GetCurrentFramePara");
	SSBufferUSB_GetDevicesErrorState = (SSBufferUSB_GetDevicesErrorStatePtr)GetProcAddress(HDll, "SSBufferUSB_GetDevicesErrorState");
	SSBufferUSB_IsUSBSuperSpeed = (SSBufferUSB_IsUSBSuperSpeedPtr)GetProcAddress(HDll, "SSBufferUSB_IsUSBSuperSpeed");
	SSBufferUSB_SetGPIOConfig = (SSBufferUSB_SetGPIOConfigPtr)GetProcAddress(HDll, "SSBufferUSB_SetGPIOConfig");
	SSBufferUSB_SetGPIOOut = (SSBufferUSB_SetGPIOOutPtr)GetProcAddress(HDll, "SSBufferUSB_SetGPIOOut");
	SSBufferUSB_SetGPIOInOut = (SSBufferUSB_SetGPIOInOutPtr)GetProcAddress(HDll, "SSBufferUSB_SetGPIOInOut");
  }
  else
		return DEVICE_ERR;

	int g_cameraCount = SSBufferUSB_InitDevice();
	if (g_cameraCount == 0)
	{
		SSBufferUSB_UnInitDevice();
		return DEVICE_NOT_CONNECTED;
	}

	char ModuleNo[32];
	char SerialNo[32];
	if(SSBufferUSB_GetModuleNoSerialNo(1, ModuleNo, SerialNo) == -1)
	{
		;
	}
	else
	{
		//remove string spaces
		char *s_ModuleNo = strchr(ModuleNo, ' ');
		if(s_ModuleNo)
			*s_ModuleNo = '\0';
		sprintf(camNames, "%s:%s\0", ModuleNo, SerialNo);
	}

		SSBufferUSB_AddDeviceToWorkingSet(1);
		SSBufferUSB_ActiveDeviceInWorkingSet(1, 1);

	if(SSBufferUSB_IsUSBSuperSpeed(1))
		SSBufferUSB_SetSensorBlankings(1, 1, 32);

	SSBufferUSB_StartCameraEngineEx( NULL, 8, 4, 1);
	SSBufferUSB_InstallFrameHooker( 1, FrameCallBack );
	SSBufferUSB_InstallUSBDeviceHooker( CameraFaultCallBack );
	SSBufferUSB_SetCameraWorkMode( 1, 1 ); // TRIGGER MODE
	SSBufferUSB_StartFrameGrab( GRAB_FRAME_FOREVER );

	//////////////////////////////////////////////////////////////////////////
	// GetDeviceType

	if (strstr(camNames, "B015") != NULL)
		deviceType = IMX273M;
	else if (strstr(camNames, "C015") != NULL)
		deviceType = IMX273C;
	else
		return DEVICE_ERR;

     switch (deviceType) {
		case IMX273C:
			deviceColorType = 1;
			break;
       default: 
			deviceColorType = 0;
	}

	//////////////////////////////////////////////////////////////////////////

	 if ((deviceType == IMX273C) || (deviceType == IMX273M))
		 MAX_RESOLUTION = SB_MAX_RESOLUTIONS_V015;

	s_MAX_RESOLUTION = MAX_RESOLUTION;
	
	int frameSize;
	if(deviceColorType)
	{
		g_frameSize = SB_DEFAULT_SIZE*3;
		frameSize = s_vidFrameSize[MAX_RESOLUTION].frameSize * 3;
	}
	else
		frameSize = s_vidFrameSize[MAX_RESOLUTION].frameSize * 2; //16 bit
	g_pImage = new BYTE[frameSize];
	ZeroMemory(g_pImage, frameSize);

	g_deviceColorType = deviceColorType;

	yStart = 0;
	h_Mirror = 0;
	v_Flip = 0;

     switch (deviceType) {
		case IMX273M:
		case IMX273C:
			p_frmSize = (struct FrmSize*)frameSize_IMX273;
			break;
		default:
			p_frmSize = NULL;
	}

	SSBufferUSB_SetCustomizedResolution(1, 
			s_vidFrameSize[MAX_RESOLUTION].width, SB_DEFAULT_HEIGHT, 0, GetCameraBufferCount(SB_DEFAULT_WIDTH, SB_DEFAULT_HEIGHT));

	is_initCamera = true;

	SingleFrameEvent = CreateEvent( NULL, FALSE, FALSE, "SingleFrameGrabbingEvent");

    return DEVICE_OK;
}

int CMightex_SB_Camera::GetCameraBufferCount(int width, int height)
{
	
	int frameSize = width * height;
	int tmpIndex = SB_MAX_RESOLUTIONS;
	while (frameSize <= s_vidFrameSize[tmpIndex].frameSize )
	{
		tmpIndex--;
		if(tmpIndex < 0) break;
	}
	
	if (deviceColorType == 0)
		return CAMERA_SB_BUFFER_BW[tmpIndex+1];	
	else
		return CAMERA_SB_BUFFER_COLOR[tmpIndex+1];	
}


///////////////////////////////////////////////////////////////////////////////
// CMightex_SB_Camera implementation
// ~~~~~~~~~~~~~~~~~~~~~~~~~~

/**
* CMightex_SB_Camera constructor.
* Setup default all variables and create device properties required to exist
* before intialization. In this case, no such properties were required. All
* properties will be created in the Initialize() method.
*
* As a general guideline Micro-Manager devices do not access hardware in the
* the constructor. We should do as little as possible in the constructor and
* perform most of the initialization in the Initialize() method.
*/
CMightex_SB_Camera::CMightex_SB_Camera() :
   CCameraBase<CMightex_SB_Camera> (),
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
	binMode_(0),
	binSize_(1),
	cameraXSize_(SB_DEFAULT_WIDTH),
	cameraYSize_(SB_DEFAULT_HEIGHT),
   cam_T_ (0.0),
   triggerDevice_(""),
   stopOnOverflow_(false),
	dropPixels_(false),
   fastImage_(false),
   saturatePixels_(false),
	fractionOfPixelsToDropOrSaturate_(0.002),
   pDemoResourceLock_(0),
   nComponents_(4)
{
   memset(testProperty_,0,sizeof(testProperty_));

   // call the base class method to set-up default error codes/messages
   InitializeDefaultErrorMessages();
   readoutStartTime_ = GetCurrentMMTime();
   pDemoResourceLock_ = new MMThreadLock();
   thd_ = new MySequenceThread(this);

	g_InstanceCount++;
	is_initCamera = false;
	HDll = NULL;
}

/**
* CMightex_SB_Camera destructor.
* If this device used as intended within the Micro-Manager system,
* Shutdown() will be always called before the destructor. But in any case
* we need to make sure that all resources are properly released even if
* Shutdown() was not called.
*/
CMightex_SB_Camera::~CMightex_SB_Camera()
{

   StopSequenceAcquisition();
   delete thd_;
   delete pDemoResourceLock_;

   g_InstanceCount--;
}

/**
* Obtains device name.
* Required by the MM::Device API.
*/
void CMightex_SB_Camera::GetName(char* name) const
{
   // Return the name used to referr to this device adapte
   CDeviceUtils::CopyLimitedString(name, g_Camera_SB_DeviceName);
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
int CMightex_SB_Camera::Initialize()
{
   if (initialized_)
      return DEVICE_OK;

   int nRet = InitCamera();
   if( nRet != DEVICE_OK)
      return nRet;

   // set property list
   // -----------------

   // Name
   nRet = CreateStringProperty(MM::g_Keyword_Name, g_Camera_SB_DeviceName, true);
   if (DEVICE_OK != nRet)
      return nRet;

   // Description
   nRet = CreateStringProperty(MM::g_Keyword_Description, "Mightex Super Speed USB camera(SB-Series) Device Adapter", true);
   if (DEVICE_OK != nRet)
      return nRet;

   // CameraName
   nRet = CreateStringProperty(MM::g_Keyword_CameraName, camNames, true);
   assert(nRet == DEVICE_OK);

   // CameraID
   nRet = CreateStringProperty(MM::g_Keyword_CameraID, "V1.0", true);
   assert(nRet == DEVICE_OK);

   // binning
   CPropertyAction *pAct = new CPropertyAction (this, &CMightex_SB_Camera::OnBinning);
   nRet = CreateIntegerProperty(MM::g_Keyword_Binning, 1, false, pAct);
   assert(nRet == DEVICE_OK);

   nRet = SetAllowedBinning(TRUE);
   if (nRet != DEVICE_OK)
      return nRet;

   pAct = new CPropertyAction(this, &CMightex_SB_Camera::OnBinMode);
   nRet = CreateIntegerProperty(g_Keyword_BinMode, 0, false, pAct);
   assert(nRet == DEVICE_OK);

   vector<string> BinModeValues;
   BinModeValues.push_back("0");
   if (!deviceColorType)
	   BinModeValues.push_back("1");

   nRet = SetAllowedValues(g_Keyword_BinMode, BinModeValues);
   if (nRet != DEVICE_OK)
	   return nRet;

   // pixel type
   pAct = new CPropertyAction (this, &CMightex_SB_Camera::OnPixelType);
   	if(deviceColorType)
	   nRet = CreateStringProperty(MM::g_Keyword_PixelType, g_PixelType_32bitRGB, false, pAct);
	else
	   nRet = CreateStringProperty(MM::g_Keyword_PixelType, g_PixelType_8bit, false, pAct);
   assert(nRet == DEVICE_OK);

   vector<string> pixelTypeValues;
   pixelTypeValues.push_back(g_PixelType_8bit);
   	if(deviceColorType)
		pixelTypeValues.push_back(g_PixelType_32bitRGB);
	else
		nComponents_ = 1;

   nRet = SetAllowedValues(MM::g_Keyword_PixelType, pixelTypeValues);
   if (nRet != DEVICE_OK)
      return nRet;

   pAct = new CPropertyAction (this, &CMightex_SB_Camera::OnBitDepth);
   nRet = CreateIntegerProperty("BitDepth", 8, false, pAct);
   assert(nRet == DEVICE_OK);

   vector<string> bitDepths;
   bitDepths.push_back("8");
   bitDepths.push_back("16");
   nRet = SetAllowedValues("BitDepth", bitDepths);
   if (nRet != DEVICE_OK)
      return nRet;

   // Exposure Time
   pAct = new CPropertyAction (this, &CMightex_SB_Camera::OnExposure);
   nRet = CreateIntegerProperty(MM::g_Keyword_Exposure, 5, false, pAct);
   assert(nRet == DEVICE_OK);
   
   // camera gain
   pAct = new CPropertyAction (this, &CMightex_SB_Camera::OnGain);
   nRet = CreateIntegerProperty("Gain Value", 1, false, pAct);
   assert(nRet == DEVICE_OK);
   SetPropertyLimits("Gain Value", 0, 48);

   // Resolution
   pAct = new CPropertyAction (this, &CMightex_SB_Camera::OnResolution);
   nRet = CreateStringProperty(g_Keyword_Resolution, g_Res[s_MAX_RESOLUTION], false, pAct);
   assert(nRet == DEVICE_OK);

   vector<string> ResValues;
	ResValues.push_back(g_Res[s_MAX_RESOLUTION]);
   nRet = SetAllowedValues(g_Keyword_Resolution, ResValues);
   if (nRet != DEVICE_OK)
      return nRet;

   // Resolution_Ry
   pAct = new CPropertyAction(this, &CMightex_SB_Camera::OnResolution_Ry);
   nRet = CreateIntegerProperty(g_Keyword_Resolution_Ry, s_vidFrameSize[MAX_RESOLUTION].height, false, pAct);
   assert(nRet == DEVICE_OK);

   // YStart
   pAct = new CPropertyAction (this, &CMightex_SB_Camera::OnYStart);
   nRet = CreateIntegerProperty(g_Keyword_YStart, 0, false, pAct);
   assert(nRet == DEVICE_OK);
   SetPropertyLimits(g_Keyword_YStart, 0, s_vidFrameSize[MAX_RESOLUTION].height - SB_DEFAULT_HEIGHT);

   // H_Mirror
   pAct = new CPropertyAction (this, &CMightex_SB_Camera::OnH_Mirror);
   nRet = CreateIntegerProperty("H_Mirror", 0, false, pAct);
   assert(nRet == DEVICE_OK);

   vector<string> h_Mirrors;
   h_Mirrors.push_back("0");
   h_Mirrors.push_back("1");
   nRet = SetAllowedValues("H_Mirror", h_Mirrors);
   if (nRet != DEVICE_OK)
      return nRet;

   // V_Flip
   pAct = new CPropertyAction (this, &CMightex_SB_Camera::OnV_Flip);
   nRet = CreateIntegerProperty("V_Flip", 0, false, pAct);
   assert(nRet == DEVICE_OK);

   vector<string> v_Flips;
   v_Flips.push_back("0");
   v_Flips.push_back("1");
   nRet = SetAllowedValues("V_Flip", v_Flips);
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
int CMightex_SB_Camera::Shutdown()
{
   initialized_ = false;

	if(HDll){
        SSBufferUSB_StopFrameGrab();
		SSBufferUSB_InstallFrameHooker( 1, NULL );
		SSBufferUSB_InstallUSBDeviceHooker( NULL );
		SSBufferUSB_StopCameraEngine();
		SSBufferUSB_RemoveDeviceFromWorkingSet(1);
		SSBufferUSB_UnInitDevice();

		 FreeLibrary( HDll );
		if ( SingleFrameEvent != NULL )
		        CloseHandle( SingleFrameEvent );
	}

   if(!is_initCamera)
		 return DEVICE_OK;

	if(g_pImage)
	{
		delete []g_pImage;
		g_pImage = NULL;
	}

	is_initCamera = false;

   return DEVICE_OK;
}

/**
* Performs exposure and grabs a single image.
* This function should block during the actual exposure and return immediately afterwards 
* (i.e., before readout).  This behavior is needed for proper synchronization with the shutter.
* Required by the MM::Camera API.
*/
int CMightex_SB_Camera::SnapImage()
{

	static int callCounter = 0;
	++callCounter;

   MM::MMTime startTime = GetCurrentMMTime();
   double exp = GetExposure();
   if (sequenceRunning_ && IsCapturing()) 
   {
      exp = GetSequenceExposure();
   }

   // We assume it must be in TRIGGER mode
   SSBufferUSB_SetCameraWorkMode(1, 1); // Set "TriggerEventCount" to 0.
   Sleep(20);
   SSBufferUSB_SoftTrigger(1); // Assert Once for single frame.
   ResetEvent(SingleFrameEvent);
   

   MM::MMTime s0(0,0);
   if( s0 < startTime )
   {
      while (exp > (GetCurrentMMTime() - startTime).getMsec())
      {
         CDeviceUtils::SleepMs(1);
      }		
   }
   else
   {
      std::cerr << "You are operating this device adapter without setting the core callback, timing functions aren't yet available" << std::endl;
      // called without the core callback probably in off line test program
      // need way to build the core in the test program

   }
   readoutStartTime_ = GetCurrentMMTime();

   WaitForSingleObject( SingleFrameEvent, 1000 );
   GetImageBuffer();

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
const unsigned char* CMightex_SB_Camera::GetImageBuffer()
{

   MMThreadGuard g(imgPixelsLock_);
   MM::MMTime readoutTime(readoutUs_);
   while (readoutTime > (GetCurrentMMTime() - readoutStartTime_)) {}		
   unsigned char *pB = (unsigned char*)(img_.GetPixels());

   MMThreadGuard g_g(g_imgPixelsLock_);
   if(bitDepth_ > 8)
   {
	   RAWtoImageJ();
	   return pB;
   }

   if (deviceColorType == 0)
		memcpy(img_.GetPixelsRW(), g_pImage, g_frameSize);
   else
   {
	   if (GetImageBytesPerPixel() == 4)
		   RGB3toRGB4((char*)g_pImage, (char*)img_.GetPixelsRW(), img_.Width(), img_.Height());
	   else
		   RGB3toRGB1((char*)g_pImage, (char*)img_.GetPixelsRW(), img_.Width(), img_.Height());
   }

   return pB;
}

/**
* Converts RAW image to ImageJ
*/
void CMightex_SB_Camera::RAWtoImageJ()
{
	unsigned char tmp;
	unsigned char *p = (unsigned char*)img_.GetPixelsRW();
	unsigned char *r = (unsigned char*)img_.GetPixelsRW();
	unsigned char *q = g_pImage + g_frameSize;
	for (int j = 0; j < g_frameSize; j++)
	{
		*r = *q;
		r++;
		q--;
	}

   for(int i = 0; i < g_frameSize/2; i++)
   {
	   tmp = *p;
	   *p = *(p+1);
	   *(p+1) = tmp;
	   *p = *p << 4;
	   p++;
	   p++;
   }
}

/**
* Converts three-byte image to four bytes
*/
void CMightex_SB_Camera::RGB3toRGB4(const char* srcPixels, char* destPixels, int width, int height)
{
   // nasty padding loop
   unsigned int srcOffset = 0;
   unsigned int dstOffset = 0;
   int totalSize = width * height;
   for(int i=0; i < totalSize; i++){
      memcpy(destPixels+dstOffset, srcPixels+srcOffset,3);
      srcOffset += 3;
      dstOffset += 4;
   }
}

/**
* Converts three-byte image to one bytes
*/
void CMightex_SB_Camera::RGB3toRGB1(const char* srcPixels, char* destPixels, int width, int height)
{
   // nasty padding loop
   unsigned int srcOffset = 0;
   unsigned int dstOffset = 0;
   int totalSize = width * height;
   for(int i=0; i < totalSize; i++){
      memcpy(destPixels+dstOffset, srcPixels+srcOffset,1);
      srcOffset += 3;
      dstOffset += 1;
   }
}


/**
* Returns image buffer X-size in pixels.
* Required by the MM::Camera API.
*/
unsigned CMightex_SB_Camera::GetImageWidth() const
{

   return img_.Width();
}

/**
* Returns image buffer Y-size in pixels.
* Required by the MM::Camera API.
*/
unsigned CMightex_SB_Camera::GetImageHeight() const
{

   return img_.Height();
}

/**
* Returns image buffer pixel depth in bytes.
* Required by the MM::Camera API.
*/
unsigned CMightex_SB_Camera::GetImageBytesPerPixel() const
{

   return img_.Depth();
} 

/**
* Returns the bit depth (dynamic range) of the pixel.
* This does not affect the buffer size, it just gives the client application
* a guideline on how to interpret pixel values.
* Required by the MM::Camera API.
*/
unsigned CMightex_SB_Camera::GetBitDepth() const
{

   return bitDepth_;
}

/**
* Returns the size in bytes of the image buffer.
* Required by the MM::Camera API.
*/
long CMightex_SB_Camera::GetImageBufferSize() const
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
* @param x - top-left corner coordinate
* @param y - top-left corner coordinate
* @param xSize - width
* @param ySize - height
*/
int CMightex_SB_Camera::SetROI(unsigned x, unsigned y, unsigned xSize, unsigned ySize)
{

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
int CMightex_SB_Camera::GetROI(unsigned& x, unsigned& y, unsigned& xSize, unsigned& ySize)
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
int CMightex_SB_Camera::ClearROI()
{

   ResizeImageBuffer();
   roiX_ = 0;
   roiY_ = 0;
      
   return DEVICE_OK;
}

/**
* Returns the current exposure setting in milliseconds.
* Required by the MM::Camera API.
*/
double CMightex_SB_Camera::GetExposure() const
{

   char buf[MM::MaxStrLength];
   int ret = GetProperty(MM::g_Keyword_Exposure, buf);
   if (ret != DEVICE_OK)
      return 0.0;
   return atof(buf);
}

/**
 * Returns the current exposure from a sequence and increases the sequence counter
 * Used for exposure sequences
 */
double CMightex_SB_Camera::GetSequenceExposure() 
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
void CMightex_SB_Camera::SetExposure(double exp)
{
   SetProperty(MM::g_Keyword_Exposure, CDeviceUtils::ConvertToString(exp));
   GetCoreCallback()->OnExposureChanged(this, exp);;
}

/**
* Returns the current binning factor.
* Required by the MM::Camera API.
*/
int CMightex_SB_Camera::GetBinning() const
{

   char buf[MM::MaxStrLength];
   int ret = GetProperty(MM::g_Keyword_Binning, buf);
   if (ret != DEVICE_OK)
      return 1;
   return atoi(buf);
}

/**
* Sets binning factor.
* Required by the MM::Camera API.
*/
int CMightex_SB_Camera::SetBinning(int binF)
{

   return SetProperty(MM::g_Keyword_Binning, CDeviceUtils::ConvertToString(binF));
}

int CMightex_SB_Camera::IsExposureSequenceable(bool& isSequenceable) const
{
   isSequenceable = isSequenceable_;
   return DEVICE_OK;
}

int CMightex_SB_Camera::GetExposureSequenceMaxLength(long& nrEvents)
{
   if (!isSequenceable_) {
      return DEVICE_UNSUPPORTED_COMMAND;
   }

   nrEvents = sequenceMaxLength_;
   return DEVICE_OK;
}

int CMightex_SB_Camera::StartExposureSequence()
{
   if (!isSequenceable_) {
      return DEVICE_UNSUPPORTED_COMMAND;
   }

   // may need thread lock
   sequenceRunning_ = true;
   return DEVICE_OK;
}

int CMightex_SB_Camera::StopExposureSequence()
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
int CMightex_SB_Camera::ClearExposureSequence()
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
int CMightex_SB_Camera::AddToExposureSequence(double exposureTime_ms) 
{
   if (!isSequenceable_) {
      return DEVICE_UNSUPPORTED_COMMAND;
   }

   exposureSequence_.push_back(exposureTime_ms);
   return DEVICE_OK;
}

int CMightex_SB_Camera::SendExposureSequence() const {
   if (!isSequenceable_) {
      return DEVICE_UNSUPPORTED_COMMAND;
   }

   return DEVICE_OK;
}

int CMightex_SB_Camera::SetAllowedBinning(int isBinning) 
{

   vector<string> binValues;
   binValues.push_back("1");

   if(isBinning == 1)
   {
	   binValues.push_back("2");
   }

   return SetAllowedValues(MM::g_Keyword_Binning, binValues);
}


/**
 * Required by the MM::Camera API
 * Please implement this yourself and do not rely on the base class implementation
 * The Base class implementation is deprecated and will be removed shortly
 */
int CMightex_SB_Camera::StartSequenceAcquisition(double interval) {

	SSBufferUSB_SetCameraWorkMode(1, 0); // Set to NORMAL mode

   return StartSequenceAcquisition(LONG_MAX, interval, false);            
}

/**                                                                       
* Stop and wait for the Sequence thread finished                                   
*/                                                                        
int CMightex_SB_Camera::StopSequenceAcquisition()                                     
{
   if (IsCallbackRegistered())
   {
	   ;
   }

  if(is_initCamera){
  	  SSBufferUSB_SetCameraWorkMode(1, 1); // Set to TRIGGER mode		
	}

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
int CMightex_SB_Camera::StartSequenceAcquisition(long numImages, double interval_ms, bool stopOnOverflow)
{

   if (IsCapturing())
      return DEVICE_CAMERA_BUSY_ACQUIRING;

   int ret = GetCoreCallback()->PrepareForAcq(this);
   if (ret != DEVICE_OK)
      return ret;
   sequenceStartTime_ = GetCurrentMMTime();
   imageCounter_ = 0;
   thd_->Start(numImages,interval_ms);
   stopOnOverflow_ = stopOnOverflow;
   return DEVICE_OK;
}

/*
 * Inserts Image and MetaData into MMCore circular Buffer
 */
int CMightex_SB_Camera::InsertImage()
{

   MM::MMTime timeStamp = this->GetCurrentMMTime();
   char label[MM::MaxStrLength];
   this->GetLabel(label);
 
   // Important:  metadata about the image are generated here:
   Metadata md;
   md.put(MM::g_Keyword_Metadata_CameraLabel, label);
   md.put(MM::g_Keyword_Elapsed_Time_ms, CDeviceUtils::ConvertToString((timeStamp - sequenceStartTime_).getMsec()));
   md.put(MM::g_Keyword_Metadata_ROI_X, CDeviceUtils::ConvertToString( (long) roiX_)); 
   md.put(MM::g_Keyword_Metadata_ROI_Y, CDeviceUtils::ConvertToString( (long) roiY_)); 

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
   } else
      return ret;
}

/*
 * Do actual capturing
 * Called from inside the thread  
 */
int CMightex_SB_Camera::ThreadRun (MM::MMTime startTime)
{

   int ret=DEVICE_ERR;
   
   // Trigger
   if (triggerDevice_.length() > 0) {
      MM::Device* triggerDev = GetDevice(triggerDevice_.c_str());
      if (triggerDev != 0) {
      	LogMessage("trigger requested");
      	triggerDev->SetProperty("Trigger","+");
      }
   }

   ret = InsertImage();
     

   while (((double) (this->GetCurrentMMTime() - startTime).getMsec() / imageCounter_) < this->GetSequenceExposure())
   {
      CDeviceUtils::SleepMs(1);
   }

   if (ret != DEVICE_OK)
   {
      return ret;
   }
   return ret;
};

bool CMightex_SB_Camera::IsCapturing() {
   return !thd_->IsStopped();
}

/*
 * called from the thread function before exit 
 */
void CMightex_SB_Camera::OnThreadExiting() throw()
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


MySequenceThread::MySequenceThread(CMightex_SB_Camera* pCam)
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
   actualDuration_ = MM::MMTime{};
   startTime_= camera_->GetCurrentMMTime();
   lastFrameTime_ = MM::MMTime{};
}

bool MySequenceThread::IsStopped(){
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
   int ret=DEVICE_ERR;
   try 
   {
      do
      {  
         ret=camera_->ThreadRun(startTime_);
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
// CMightex_SB_Camera Action handlers
///////////////////////////////////////////////////////////////////////////////

/**
* Handles "Binning" property.
*/
int CMightex_SB_Camera::OnBinning(MM::PropertyBase* pProp, MM::ActionType eAct)
{

   int ret = DEVICE_ERR;
   switch(eAct)
   {
   case MM::AfterSet:
      {
         if(IsCapturing())
            return DEVICE_CAMERA_BUSY_ACQUIRING;

         // the user just set the new value for the property, so we have to
         // apply this value to the 'hardware'.
         long binFactor;
		 vector<string> pixelTypeValues;
		 pProp->Get(binFactor);
			{
			 if (binFactor == binSize_) return DEVICE_OK;

			 if(binFactor == 1)
			{
				int h = s_vidFrameSize[MAX_RESOLUTION].height;
				if (binMode_ == 1)
					h = p_frmSize[1].height; //544

				if (SSBufferUSB_SetCustomizedResolutionEx(1,
					s_vidFrameSize[MAX_RESOLUTION].width, h, binFactor - 1,
					binMode_, GetCameraBufferCount(s_vidFrameSize[MAX_RESOLUTION].width, h) > 0, 0))
				{
					unsigned int bytesPerPixel = 1;
					if (bitDepth_ > 8)
						bytesPerPixel = 2;
					g_frameSize = s_vidFrameSize[MAX_RESOLUTION].width * h * bytesPerPixel;
					if (deviceColorType)
					{
						if (bitDepth_ == 8)
						{
							g_frameSize = g_frameSize * 3;
							if (nComponents_ == 4)
								bytesPerPixel = 4;
						}
					}
					img_.Resize(s_vidFrameSize[MAX_RESOLUTION].width, h, bytesPerPixel);
					binSize_ = binFactor;
				}
				else
					return DEVICE_ERR;
			}
			else
			{
				if(SSBufferUSB_SetCustomizedResolutionEx(1, 
					s_vidFrameSize[MAX_RESOLUTION].width, s_vidFrameSize[MAX_RESOLUTION].height,
					binFactor-1, binMode_, GetCameraBufferCount(s_vidFrameSize[MAX_RESOLUTION].width, s_vidFrameSize[MAX_RESOLUTION].height) > 0, 0))
				{
 					unsigned int bytesPerPixel = 1;
					if (bitDepth_ > 8)
						bytesPerPixel = 2;
					g_frameSize = p_frmSize[binFactor-1].width * p_frmSize[binFactor-1].height * bytesPerPixel;
					if (deviceColorType)
					{
						if (bitDepth_ == 8)
						{
							g_frameSize = g_frameSize * 3;
							if (nComponents_ == 4)
								bytesPerPixel = 4;
						}
					}
					img_.Resize(p_frmSize[binFactor-1].width, p_frmSize[binFactor-1].height, bytesPerPixel);
					binSize_ = binFactor;
				}
				else
					return DEVICE_ERR;

				SetPropertyLimits(g_Keyword_YStart, 0, 0);
				SetProperty(g_Keyword_YStart, "0");
				yStart = 0;

				vector<string> ResValues;
				ResValues.push_back(g_Res[s_MAX_RESOLUTION]);
				SetAllowedValues(g_Keyword_Resolution, ResValues);
				SetProperty(g_Keyword_Resolution, g_Res[s_MAX_RESOLUTION]);
			}
			ret=DEVICE_OK;
			}
      }break;
   case MM::BeforeGet:
      {
         ret=DEVICE_OK;
		 pProp->Set(binSize_);
      }break;
   default:
      break;
   }
   return ret; 
}

/**
* Handles "BinMode" property.
*/
int CMightex_SB_Camera::OnBinMode(MM::PropertyBase* pProp, MM::ActionType eAct)
{

	int ret = DEVICE_ERR;
	switch (eAct)
	{
	case MM::AfterSet:
	{
		if (IsCapturing())
			return DEVICE_CAMERA_BUSY_ACQUIRING;

		// the user just set the new value for the property, so we have to
		// apply this value to the 'hardware'.
		int h = s_vidFrameSize[MAX_RESOLUTION].height;
		long binFactor;
		pProp->Get(binFactor);
		{
			if (binFactor == binMode_) return DEVICE_OK;

			if (binFactor == 1 && binSize_ == 1)
				h = p_frmSize[1].height; //544

			if (SSBufferUSB_SetCustomizedResolutionEx(1,
				s_vidFrameSize[MAX_RESOLUTION].width, h, binSize_ - 1,
				binFactor, GetCameraBufferCount(s_vidFrameSize[MAX_RESOLUTION].width, h) > 0, 0))
			{
				unsigned int bytesPerPixel = 1;
				if (bitDepth_ > 8 )
					bytesPerPixel = 2;

				if (binSize_ > 1)
					h = p_frmSize[1].height; //544
				img_.Resize(p_frmSize[binSize_ - 1].width, h, bytesPerPixel);
				g_frameSize = p_frmSize[binSize_ - 1].width * h * bytesPerPixel;

				binMode_ = binFactor;
				if (binFactor > 0)
				{				
					SetPropertyLimits(g_Keyword_YStart, 0, 0);
					SetProperty(g_Keyword_YStart, "0");
					yStart = 0;

					vector<string> ResValues;
					ResValues.push_back(g_Res[s_MAX_RESOLUTION]);
					SetAllowedValues(g_Keyword_Resolution, ResValues);
					SetProperty(g_Keyword_Resolution, g_Res[s_MAX_RESOLUTION]);
				}
			}
			else
				return DEVICE_ERR;

			ret = DEVICE_OK;
		}
	}break;
	case MM::BeforeGet:
	{
		ret = DEVICE_OK;
		pProp->Set(binMode_);
	}break;
	default:
		break;
	}
	return ret;
}

/**
* Handles "PixelType" property.
*/
int CMightex_SB_Camera::OnPixelType(MM::PropertyBase* pProp, MM::ActionType eAct)
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
            ret=DEVICE_OK;
         }
		else if ( pixelType.compare(g_PixelType_32bitRGB) == 0)
		{
			nComponents_ = 4;
			img_.Resize(img_.Width(), img_.Height(), 4);
			ret=DEVICE_OK;
		}
		else if ( pixelType.compare(g_PixelType_16bit) == 0)
		{
			nComponents_ = 1;
			img_.Resize(img_.Width(), img_.Height(), 2);
			ret=DEVICE_OK;
		}
         else
         {
            // on error switch to default pixel type
            nComponents_ = 1;
            img_.Resize(img_.Width(), img_.Height(), 1);
            pProp->Set(g_PixelType_8bit);
            ret = ERR_UNKNOWN_MODE;
         }
      } break;
   case MM::BeforeGet:
      {
         ret=DEVICE_OK;
      } break;
   default:
      break;
   }
   return ret; 
}

/**
* Handles "BitDepth" property.
*/
int CMightex_SB_Camera::OnBitDepth(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   int ret = DEVICE_ERR;
   switch(eAct)
   {
   case MM::AfterSet:
      {
         if(IsCapturing())
            return DEVICE_CAMERA_BUSY_ACQUIRING;

         long bitDepth;
         pProp->Get(bitDepth);

		 if (bitDepth == bitDepth_) return DEVICE_OK;

			unsigned int bytesPerComponent;
 			unsigned int bytesPerPixel = 1;
			vector<string> pixelTypeValues;
			int h = img_.Height();
			if (binSize_ == 2)
				h = s_vidFrameSize[MAX_RESOLUTION].height; //1088
			switch (bitDepth) {
            case 8:
				bytesPerComponent = 1;
               bitDepth_ = 8;

			   pixelTypeValues.push_back(g_PixelType_8bit);
   				if(deviceColorType)
					pixelTypeValues.push_back(g_PixelType_32bitRGB);
				else
					nComponents_ = 1;
			   ret = SetAllowedValues(MM::g_Keyword_PixelType, pixelTypeValues);
			   if (ret != DEVICE_OK)
				  return ret;
   				if(deviceColorType)
				{
					SetProperty(MM::g_Keyword_PixelType, g_PixelType_32bitRGB);
					g_frameSize = GetImageBufferSize()/4*3;
				}
				else
				{
					SetProperty(MM::g_Keyword_PixelType, g_PixelType_8bit);
					g_frameSize = GetImageBufferSize();
				}

				SSBufferUSB_StopFrameGrab();
				SSBufferUSB_StopCameraEngine();
				SSBufferUSB_StartCameraEngineEx(NULL, 8, 4, 1);
				SSBufferUSB_InstallFrameHooker( 1, FrameCallBack ); //RGB Type
				SSBufferUSB_InstallUSBDeviceHooker( CameraFaultCallBack );
				SSBufferUSB_SetCameraWorkMode( 1, 1 ); // TRIGGER MODE
				SSBufferUSB_SetCustomizedResolutionEx(1,
					s_vidFrameSize[MAX_RESOLUTION].width, h, binSize_-1, binMode_, GetCameraBufferCount(s_vidFrameSize[MAX_RESOLUTION].width, h) > 0, 0);
				SSBufferUSB_StartFrameGrab( GRAB_FRAME_FOREVER );

			   ret=DEVICE_OK;
            break;
            case 10:
				bytesPerComponent = 2;
				bytesPerPixel = 2;
				bitDepth_ = 10;
               ret=DEVICE_OK;
            break;
            case 12:
				bytesPerComponent = 2;
				bytesPerPixel = 2;
				bitDepth_ = 12;
               ret=DEVICE_OK;
            break;
			case 16:
				bytesPerComponent = 2;
				bytesPerPixel = 2;
				bitDepth_ = 16;

				pixelTypeValues.push_back(g_PixelType_16bit);
				ret = SetAllowedValues(MM::g_Keyword_PixelType, pixelTypeValues);
				if (ret != DEVICE_OK)
					return ret;
				SetProperty(MM::g_Keyword_PixelType, g_PixelType_16bit);

				g_frameSize = GetImageBufferSize();

				SSBufferUSB_StopFrameGrab();
				SSBufferUSB_StopCameraEngine();
				SSBufferUSB_StartCameraEngineEx(NULL, 16, 4, 1);
				SSBufferUSB_InstallFrameHooker(0, FrameCallBack); //RAW Type
				SSBufferUSB_InstallUSBDeviceHooker(CameraFaultCallBack);
				SSBufferUSB_SetCameraWorkMode(1, 1); // TRIGGER MODE
				SSBufferUSB_SetCustomizedResolutionEx(1,
					s_vidFrameSize[MAX_RESOLUTION].width, h, binSize_ - 1, binMode_, GetCameraBufferCount(s_vidFrameSize[MAX_RESOLUTION].width, h) > 0, 0);
				SSBufferUSB_StartFrameGrab(GRAB_FRAME_FOREVER);

				ret = DEVICE_OK;
				break;
			default:
               // on error switch to default pixel type
					bytesPerComponent = 1;

               pProp->Set((long)8);
               bitDepth_ = 8;
               ret = ERR_UNKNOWN_MODE;
            break;
         }
      } break;
   case MM::BeforeGet:
      {
         pProp->Set((long)bitDepth_);
         ret=DEVICE_OK;
      } break;
   default:
      break;
   }
 
   return ret; 
}

///////////////////////////////////////////////////////////////////////////////
// Private CMightex_SB_Camera methods
///////////////////////////////////////////////////////////////////////////////

/**
* Sync internal image buffer size to the chosen property values.
*/
int CMightex_SB_Camera::ResizeImageBuffer()
{

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
	else if ( pixelType.compare(g_PixelType_16bit) == 0)
	{
      byteDepth = 2;
	}
	else if ( pixelType.compare(g_PixelType_32bitRGB) == 0)
	{
      byteDepth = 4;
	}

   img_.Resize(cameraXSize_/binSize_, cameraYSize_/binSize_, byteDepth);
   return DEVICE_OK;
}

void CMightex_SB_Camera::GenerateEmptyImage(ImgBuffer& img)
{
   MMThreadGuard g(imgPixelsLock_);
   if (img.Height() == 0 || img.Width() == 0 || img.Depth() == 0)
      return;
   unsigned char* pBuf = const_cast<unsigned char*>(img.GetPixels());
   memset(pBuf, 0, img.Height()*img.Width()*img.Depth());
}

void CMightex_SB_Camera::TestResourceLocking(const bool recurse)
{
   MMThreadGuard g(*pDemoResourceLock_);
   if(recurse)
      TestResourceLocking(false);
}

///////////////////////////////////////////////////////////////////////////////
// new: OnExposure 
///////////////////////////////////////////////////////////////////////////////
int CMightex_SB_Camera::OnExposure(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   int ret = DEVICE_ERR;
   switch(eAct)
   {
   case MM::AfterSet:
      {
	     long exposure;
 	     long exposure_pos;
         pProp->Get(exposure_pos);
		 
		 exposure = 1000 * exposure_pos / 50;
		 if(SSBufferUSB_SetExposureTime(1, exposure) == -1)
		    return DEVICE_ERR;

         ret=DEVICE_OK;
      }break;
   case MM::BeforeGet:
      {
         ret=DEVICE_OK;
      } break;
   case MM::NoAction:
      break;
   case MM::IsSequenceable:
   case MM::AfterLoadSequence:
   case MM::StartSequence:
   case MM::StopSequence:
      return DEVICE_PROPERTY_NOT_SEQUENCEABLE;
      break;
   }
   return ret; 
}

// handles gain property

int CMightex_SB_Camera::OnGain(MM::PropertyBase* pProp, MM::ActionType eAct)
{

   int ret = DEVICE_ERR;
   switch(eAct)
   {
   case MM::AfterSet:
      {
         long gain;
         pProp->Get(gain);
		if(SSBufferUSB_SetGains(1, gain, gain, gain) == -1)
			return DEVICE_ERR;
		 ret=DEVICE_OK;
      }break;
   case MM::BeforeGet:
      {
		 ret=DEVICE_OK;
      }break;
   case MM::NoAction:
      break;
   case MM::IsSequenceable:
   case MM::AfterLoadSequence:
   case MM::StartSequence:
   case MM::StopSequence:
      return DEVICE_PROPERTY_NOT_SEQUENCEABLE;
      break;
   }
   return ret; 
}

/**
* Handles "Resolution" property.
*/
int CMightex_SB_Camera::OnResolution(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   int ret = DEVICE_ERR;
   if(binSize_ > 1 || binMode_ > 0) return DEVICE_OK;
   switch(eAct)
   {
   case MM::AfterSet:
      {
       if(IsCapturing())
          return DEVICE_CAMERA_BUSY_ACQUIRING;

		 std::string resolution;
       pProp->Get(resolution);

		 std::istringstream iss(resolution);
		 string width, height;
		 getline(iss,width,'*');
		 getline(iss,height);
		 
		 long w = atoi(width.c_str());
		 long h = atoi(height.c_str());

		 if(h < s_vidFrameSize[MAX_RESOLUTION].height) return DEVICE_OK;

		if(SSBufferUSB_SetCustomizedResolutionEx(1, 
			s_vidFrameSize[MAX_RESOLUTION].width, h, 0, 0, GetCameraBufferCount(w, h) > 0, 0))
		{
			unsigned int bytesPerPixel = 1;
			if (bitDepth_ > 8)
				bytesPerPixel = 2;
			g_frameSize = s_vidFrameSize[MAX_RESOLUTION].width * h * bytesPerPixel;
			if (deviceColorType)
			{
				if (bitDepth_ == 8)
				{
					g_frameSize = g_frameSize * 3;
					if(nComponents_ == 4)
						bytesPerPixel = 4;
				}
			}
			img_.Resize(s_vidFrameSize[MAX_RESOLUTION].width, h, bytesPerPixel);

			SetPropertyLimits(g_Keyword_YStart, 0, 0);
			SetProperty(g_Keyword_YStart, "0");
			yStart = 0;

			vector<string> ResValues;
			ResValues.push_back(g_Res[s_MAX_RESOLUTION]);
			SetAllowedValues(g_Keyword_Resolution, ResValues);

			ret = DEVICE_OK;
		}
		else
			 return DEVICE_ERR;
      } break;
   case MM::BeforeGet:
      {
         ret=DEVICE_OK;
      } break;
   case MM::NoAction:
      break;
   case MM::IsSequenceable:
   case MM::AfterLoadSequence:
   case MM::StartSequence:
   case MM::StopSequence:
      return DEVICE_PROPERTY_NOT_SEQUENCEABLE;
      break;
   }
   return ret; 
}

/**
* Handles "Resolution_Ry" property.
*/
int CMightex_SB_Camera::OnResolution_Ry(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	int ret = DEVICE_ERR;
	if (binSize_ > 1 || binMode_ > 0) return DEVICE_OK;
	switch (eAct)
	{
	case MM::AfterSet:
	{
		if (IsCapturing())
			return DEVICE_CAMERA_BUSY_ACQUIRING;

		long h = s_vidFrameSize[MAX_RESOLUTION].height;
		pProp->Get(h);

		if (h < 64 || h > s_vidFrameSize[MAX_RESOLUTION].height) return DEVICE_OK;

		if (SSBufferUSB_SetCustomizedResolutionEx(1,
			s_vidFrameSize[MAX_RESOLUTION].width, h, 0, 0, GetCameraBufferCount(s_vidFrameSize[MAX_RESOLUTION].width, h) > 0, 0))
		{
			unsigned int bytesPerPixel = 1;
			if (bitDepth_ > 8)
				bytesPerPixel = 2;
			g_frameSize = s_vidFrameSize[MAX_RESOLUTION].width * h * bytesPerPixel;
			if (deviceColorType)
			{			
				if (bitDepth_ == 8)
				{
					g_frameSize = g_frameSize * 3;
					if(nComponents_ == 4)
						bytesPerPixel = 4;
				}
			}
			img_.Resize(s_vidFrameSize[MAX_RESOLUTION].width, h, bytesPerPixel);

			SetPropertyLimits(g_Keyword_YStart, 0, s_vidFrameSize[MAX_RESOLUTION].height - h);
			SetProperty(g_Keyword_YStart, "0");
			yStart = 0;

			char s_resolution[256];
			sprintf(s_resolution, "%d*%d", s_vidFrameSize[MAX_RESOLUTION].width, h);
			vector<string> ResValues;
			ResValues.push_back(g_Res[s_MAX_RESOLUTION]);
			if(h < s_vidFrameSize[MAX_RESOLUTION].height)
				ResValues.push_back(s_resolution);
			SetAllowedValues(g_Keyword_Resolution, ResValues);
			SetProperty(g_Keyword_Resolution, s_resolution);

			ret = DEVICE_OK;
		}
		else
			return DEVICE_ERR;

	} break;
	case MM::BeforeGet:
	{
		ret = DEVICE_OK;
	} break;
	case MM::NoAction:
		break;
	case MM::IsSequenceable:
	case MM::AfterLoadSequence:
	case MM::StartSequence:
	case MM::StopSequence:
		return DEVICE_PROPERTY_NOT_SEQUENCEABLE;
		break;
	}
	return ret;
}

/**
* Handles "YStart" property.
*/
int CMightex_SB_Camera::OnYStart(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   int ret = DEVICE_ERR;
   switch(eAct)
   {
   case MM::AfterSet:
      {
         long y;
         pProp->Get(y);
		 int m = y % 8;
		 if(m < 4)
			 y -= m;
		 else
			 y += 8-m;
		if(SSBufferUSB_SetXYStart(1, 0, y) == -1)
			return DEVICE_ERR;
		 yStart = y;
		 ret=DEVICE_OK;
      }break;
   case MM::BeforeGet:
      {
		 ret=DEVICE_OK;
      }break;
   case MM::NoAction:
      break;
   case MM::IsSequenceable:
   case MM::AfterLoadSequence:
   case MM::StartSequence:
   case MM::StopSequence:
      return DEVICE_PROPERTY_NOT_SEQUENCEABLE;
      break;
   }
   return ret; 
}

/**
* Handles "H_Mirror" property.
*/
int CMightex_SB_Camera::OnH_Mirror(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   int ret = DEVICE_ERR;
   switch(eAct)
   {
   case MM::AfterSet:
      {
         long h;
         pProp->Get(h);
		if(SSBufferUSB_SetBWMode(1, 0, h, v_Flip) == -1)
			return DEVICE_ERR;
		 h_Mirror = h;

		 ret=DEVICE_OK;
      }break;
   case MM::BeforeGet:
      {
		 ret=DEVICE_OK;
      }break;
   case MM::NoAction:
      break;
   case MM::IsSequenceable:
   case MM::AfterLoadSequence:
   case MM::StartSequence:
   case MM::StopSequence:
      return DEVICE_PROPERTY_NOT_SEQUENCEABLE;
      break;
   }
   return ret; 
}

/**
* Handles "V_Flip" property.
*/
int CMightex_SB_Camera::OnV_Flip(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   int ret = DEVICE_ERR;
   switch(eAct)
   {
   case MM::AfterSet:
      {
         long v;
         pProp->Get(v);
		if(SSBufferUSB_SetBWMode(1, 0, h_Mirror, v) == -1)
			return DEVICE_ERR;
		 v_Flip = v;
		 ret=DEVICE_OK;
      }break;
   case MM::BeforeGet:
      {
		 ret=DEVICE_OK;
      }break;
   case MM::NoAction:
      break;
   case MM::IsSequenceable:
   case MM::AfterLoadSequence:
   case MM::StartSequence:
   case MM::StopSequence:
      return DEVICE_PROPERTY_NOT_SEQUENCEABLE;
      break;
   }
   return ret; 
}
