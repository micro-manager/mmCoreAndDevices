#include "MyASICam2.h"



///////////////////////////////////////////////////////////////////////////////
// FILE:          CMyASICam.cpp
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



using namespace std;
#ifdef _VEROPTICS
const char* g_CameraName = "MM_VERCam";
#else
const char* g_CameraName = "MM_ASICam";
#endif
const char* g_StateDeviceName = "MM_EFW";

const char* g_PixelType_RAW8 = "RAW8";
const char* g_PixelType_RAW12 = "RAW12";
const char* g_PixelType_RAW16 = "RAW16";
const char* g_PixelType_Y8 = "Y8";
const char* g_PixelType_RGB24 = "RGB24";
const char* g_PixelType_RGB48 = "RGB48";

const char* g_DeviceIndex = "Selected Device";
const char* g_Keyword_USBTraffic = "USBTraffic";
const char* g_Keyword_USBTraffic_Auto = "USBTraffic Auto";

const char* g_Keyword_IsHeaterOn = "Anti-dew Switch";
const char* g_Keyword_IsCoolerOn = "Cooler Switch";
const char* g_Keyword_TargetTemp = "Target Temperature";
const char* g_Keyword_CoolPowerPerc = "Cooler Power Percentage";
const char* g_Keyword_WB_R = "White Balance Red";
const char* g_Keyword_WB_B = "White Balance Blue";
const char* g_Keyword_AutoWB = "White Balance Auto";

const char* g_Keyword_on = "on";
const char* g_Keyword_off = "off";

const char* g_Keyword_Gamma = "Gamma";
const char* g_Keyword_AutoExp = "Exp Auto";
const char* g_Keyword_AutoGain = "Gain Auto";
const char* g_Keyword_Flip = "Flip";
const char* g_Keyword_HighSpeedMode = "High Speed Mode";
const char* g_Keyword_HardwareBin = "Hardware Bin";
const char* g_Keyword_USBHost = "USB Host";

///////////////////////////////////////////////////////////////////////////////
// Exported MMDevice API
///////////////////////////////////////////////////////////////////////////////

/* List all supported hardware devices here
*/
inline static void OutputDbgPrint(const char* strOutPutString, ...)
{
#ifdef _DEBUG
	char strBuf[128] = {0};
	sprintf(strBuf, "<%s> ", "MM_ASI");
	va_list vlArgs;
	va_start(vlArgs, strOutPutString);
	vsnprintf((char*)(strBuf+strlen(strBuf)), sizeof(strBuf)-strlen(strBuf), strOutPutString, vlArgs);
	va_end(vlArgs);

#ifdef _WINDOWS
	OutputDebugStringA(strBuf);
#elif defined _LIN
	printf("%s",strBuf);
#endif

#endif
}
MODULE_API void InitializeModuleData()
{
#ifdef _VEROPTICS
	RegisterDevice(g_CameraName, MM::CameraDevice, "Micro-manager Veroptics camera");//出现在device list里
#else
	RegisterDevice(g_CameraName, MM::CameraDevice, "ZWO ASI camera");//出现在device list里
#endif
	RegisterDevice(g_StateDeviceName, MM::StateDevice, "ZWO EFW filter wheel");
}

MODULE_API MM::Device* CreateDevice(const char* deviceName)
{
	if (deviceName == 0)
		return 0;

	// decide which device class to create based on the deviceName parameter
	if (strcmp(deviceName, g_CameraName) == 0)
	{
		// create camera
		return new CMyASICam();
	}

	if (strcmp(deviceName, g_StateDeviceName) == 0)
	{
		// create camera
		return new CMyEFW();
	}

	// ...supplied name not recognized
	return 0;
}

MODULE_API void DeleteDevice(MM::Device* pDevice)
{
	delete pDevice;
}

static void ToLower(char *sz)
{
	int i = 0;
	while (sz[i] != 0)
	{
		if(sz[i] >= 'A' && sz[i] <= 'Z')
			sz[i] += 32;
		i++;
	}
}

static bool StrReplace(char *dst,char *src, char* bePlaced, char* replaceTo, bool bMatchCase = true)
{	
	char szTemp[256] = {0}, szSrcTemp[256] = {0};
	char* pszMid = NULL;
	char*pszEnd = NULL;
	strcpy(szSrcTemp, src);
	if(!bMatchCase)
	{
		ToLower(szSrcTemp);
		ToLower(bePlaced);
	}

	pszMid = strstr(szSrcTemp, bePlaced);//ASI174MC-C
	strcpy(szSrcTemp, src);
	if(!pszMid)
	{
		if(dst != src)
			strcpy(dst, src);
		return false;
	}
	strncpy(szTemp, szSrcTemp, pszMid - szSrcTemp);
	szTemp[pszMid - szSrcTemp] = 0;

	pszEnd = pszMid + strlen(bePlaced);
	strcat(szTemp, replaceTo);
	strcat(szTemp, pszEnd);
	strcpy(dst, szTemp);
	return true;
}
///////////////////////////////////////////////////////////////////////////////
// CMyASICam implementation
// ~~~~~~~~~~~~~~~~~~~~~~~

/**
* CMyASICam constructor.
* Setup default all variables and create device properties required to exist
* before intialization. In this case, no such properties were required. All
* properties will be created in the Initialize() method.
*
* As a general guideline Micro-Manager devices do not access hardware in the
* the constructor. We should do as little as possible in the constructor and
* perform most of the initialization in the Initialize() method.
*/
CMyASICam::CMyASICam() :
	iBin (1),
	initialized_(false),
	iROIWidth(0),
	iROIHeight(0),
	Status(closed),
	ImgType(ASI_IMG_RAW8),
	uc_pImg(0),
	imageCounter_(0),
	thd_(0),
	pControlCaps(0),
	pRGB32(0),
	pRGB64(0),
	b12RAW(false),
	bRGB48(false),
	ImgFlip(ASI_FLIP_NONE)
	{
	// call the base class method to set-up default error codes/messages
	InitializeDefaultErrorMessages();
	ASICameraInfo.CameraID = -1;
	
	// Description property
#ifdef _VEROPTICS
	int ret = CreateProperty(MM::g_Keyword_Description, "Veroptics Camera Device Adapter", MM::String, true);
#else
	int ret = CreateProperty(MM::g_Keyword_Description, "ASICamera Camera Device Adapter", MM::String, true);
#endif
	assert(ret == DEVICE_OK);

	iConnectedCamNum = ASIGetNumOfConnectedCameras();


	vector<string> CamIndexValues;
	for(int i = 0; i < iConnectedCamNum; i++)
	{		
		ASIGetCameraProperty(&ASICameraInfo, i);
#ifdef _VEROPTICS
		StrReplace(ASICameraInfo.Name, ASICameraInfo.Name, "ZWO", "Veroptics");//区分大小写
		StrReplace(ASICameraInfo.Name, ASICameraInfo.Name, "ASI", "VER");//区分大小写
		StrReplace(ASICameraInfo.Name, ASICameraInfo.Name, "mc", "C", false);//不区分大小写
		StrReplace(ASICameraInfo.Name, ASICameraInfo.Name, "mm", "M", false);//不区分大小写
#endif
		strcpy(ConnectedCamName[i], ASICameraInfo.Name);//保存连接的摄像头名字		
		CamIndexValues.push_back(ConnectedCamName[i]);
	}

	CPropertyAction *pAct = new CPropertyAction (this, &CMyASICam::OnSelectCamIndex);//通过名字选择打开的序号
	if(iConnectedCamNum > 0)
	{
		strcpy(sz_ModelIndex, ConnectedCamName[0]);//默认打开第一个camera
		//iCamIndex = 0;
		ASIGetCameraProperty(&ASICameraInfo, 0);
	}
	else
	{
#ifdef _VEROPTICS
	strcpy(sz_ModelIndex,"no Veroptics camera connected");
#else
	strcpy(sz_ModelIndex,"no ASI camera connected");
#endif
		
	}
//	strcpy(sz_ModelIndex, "DropDown");
	ret = CreateProperty(g_DeviceIndex, sz_ModelIndex, MM::String, false, pAct, true); //选择摄像头序号
	SetAllowedValues(g_DeviceIndex, CamIndexValues);
	assert(ret == DEVICE_OK);

	strcpy(FlipArr[ASI_FLIP_BOTH], "both");
	strcpy(FlipArr[ASI_FLIP_HORIZ], "horz");
	strcpy(FlipArr[ASI_FLIP_VERT], "vert");
	strcpy(FlipArr[ASI_FLIP_NONE], "none");
	// camera type pre-initialization property

	//  create live video thread
	thd_ = new SequenceThread(this);
}

/**
* CMyASICam destructor.
* If this device used as intended within the Micro-Manager system,
* Shutdown() will be always called before the destructor. But in any case
* we need to make sure that all resources are properly released even if
* Shutdown() was not called.
*/
CMyASICam::~CMyASICam()
{
	//  if (initialized_)
	if(Status != closed)
		Shutdown();

	DeleteImgBuf();
	if(thd_)
		delete thd_;
}

/**
* Obtains device name.
* Required by the MM::Device API.
*/
void CMyASICam::GetName(char* name) const
{
	// We just return the name we use for referring to this
	// device adapter.
	CDeviceUtils::CopyLimitedString(name, g_CameraName);
}

/**
* Intializes the hardware.
* Typically we access and initialize hardware at this point.
* Device properties are typically created here as well.
* Required by the MM::Device API.
*/
int CMyASICam::Initialize()
{
	OutputDbgPrint("open camera ID: %d\n", ASICameraInfo.CameraID);
	if(ASICameraInfo.CameraID < 0)
		return DEVICE_NOT_CONNECTED;
	if(ASICameraInfo.CameraID >= 0)
	{
		if(ASIOpenCamera(ASICameraInfo.CameraID)!= ASI_SUCCESS)
			return DEVICE_NOT_CONNECTED;

		if(ASIInitCamera(ASICameraInfo.CameraID)!= ASI_SUCCESS)
			return DEVICE_NOT_CONNECTED;

		// CameraName
		
	//	ASIGetCameraProperty(&ASICameraInfo, iCamIndex);
#ifdef _VEROPTICS
		StrReplace(ASICameraInfo.Name, ASICameraInfo.Name, "ZWO", "Veroptics");//区分大小写
		StrReplace(ASICameraInfo.Name, ASICameraInfo.Name, "ASI", "VER");//区分大小写
		StrReplace(ASICameraInfo.Name, ASICameraInfo.Name, "mc", "C", false);//不区分大小写
		StrReplace(ASICameraInfo.Name, ASICameraInfo.Name, "mm", "M", false);//不区分大小写
#endif
		char *sz_Name = ASICameraInfo.Name;
		int nRet = CreateStringProperty(MM::g_Keyword_CameraName, sz_Name, true);
		assert(nRet == DEVICE_OK);


	   iROIWidth = ASICameraInfo.MaxWidth/iBin/8*8;// 2->1, *2
		iROIHeight = ASICameraInfo.MaxHeight/iBin/2*2;//1->2. *0.5
#ifdef _VEROPTICS
		if(ASICameraInfo.IsColorCam)
		{
			ImgType = ASI_IMG_RGB24;
			bRGB48 = true;			
		}
		else
		{
			if(isImgTypeSupported(ASI_IMG_RAW16))		
				ImgType = ASI_IMG_RAW16;			
			else			
				ImgType = ASI_IMG_RAW8;			
			
		}
#endif
		ASISetROIFormat(ASICameraInfo.CameraID, iROIWidth, iROIHeight, iBin, ImgType);
		iSetWid = iROIWidth;
		iSetHei = iROIHeight;
		iSetBin = iBin;
		iSetX = 0;
		iSetY = 0; 

		long lVal;
		ASI_BOOL bAuto;

#ifdef _VEROPTICS
		ASISetControlValue(ASICameraInfo.CameraID, ASI_TARGET_TEMP, -40, ASI_FALSE);
		ASISetControlValue(ASICameraInfo.CameraID, ASI_COOLER_ON, 1, ASI_FALSE);
#endif

		lExpMs = GetExposure();

		Status = opened;


		ASIGetNumOfControls(ASICameraInfo.CameraID, &iCtrlNum);
		DeletepControlCaps(ASICameraInfo.CameraID);
		MallocControlCaps(ASICameraInfo.CameraID);
	}

	if (initialized_)
		return DEVICE_OK;

	OutputDbgPrint("Init property\n");
	// set property list
	// -----------------
	vector<string> boolValues;
	boolValues.push_back(g_Keyword_off);
	boolValues.push_back(g_Keyword_on);
	// binning
	CPropertyAction *pAct = new CPropertyAction (this, &CMyASICam::OnBinning);
	int ret = CreateProperty(MM::g_Keyword_Binning, "1", MM::Integer, false, pAct);
	assert(ret == DEVICE_OK);

	vector<string> binningValues;

	int i = 0;
	char cBin[2];
	while(ASICameraInfo.SupportedBins[i] > 0)
	{
		sprintf(cBin, "%d", ASICameraInfo.SupportedBins[i]);
		binningValues.push_back(cBin);
		i++;
	}

	ret = SetAllowedValues(MM::g_Keyword_Binning, binningValues);
	assert(ret == DEVICE_OK);

	// pixel type
	pAct = new CPropertyAction (this, &CMyASICam::OnPixelType);
	ret = CreateProperty(MM::g_Keyword_PixelType, g_PixelType_Y8, MM::String, false, pAct);
	assert(ret == DEVICE_OK);

	vector<string> pixelTypeValues;
	if(isImgTypeSupported(ASI_IMG_RAW8))
		pixelTypeValues.push_back(g_PixelType_RAW8);
	if(isImgTypeSupported(ASI_IMG_RAW16))
	{
		pixelTypeValues.push_back(g_PixelType_RAW16);
		pixelTypeValues.push_back(g_PixelType_RAW12);
	}
	if(isImgTypeSupported(ASI_IMG_Y8))
		pixelTypeValues.push_back(g_PixelType_Y8);
	if(isImgTypeSupported(ASI_IMG_RGB24))
	{
		pixelTypeValues.push_back(g_PixelType_RGB24);
		pixelTypeValues.push_back(g_PixelType_RGB48);
	}

	ret = SetAllowedValues(MM::g_Keyword_PixelType, pixelTypeValues);
	assert(ret == DEVICE_OK);

	//gain
	int iMin, iMax;

	ASI_CONTROL_CAPS* pOneCtrlCap = GetOneCtrlCap(ASI_GAIN);
	if(pOneCtrlCap)
	{
		pAct = new CPropertyAction (this, &CMyASICam::OnGain);
		ret = CreateProperty(MM::g_Keyword_Gain, "1", MM::Integer, false, pAct);
		assert(ret == DEVICE_OK);
		
		iMin = pOneCtrlCap->MinValue;
		iMax = pOneCtrlCap->MaxValue;
		SetPropertyLimits(MM::g_Keyword_Gain, iMin, iMax);
	}

	//brightness

	pOneCtrlCap = GetOneCtrlCap(ASI_BRIGHTNESS);
	if(pOneCtrlCap)
	{
		pAct = new CPropertyAction (this, &CMyASICam::OnBrightness);
		ret = CreateProperty(MM::g_Keyword_Offset, "1", MM::Integer, false, pAct);
		assert(ret == DEVICE_OK);
		iMin = pOneCtrlCap->MinValue;
		iMax = pOneCtrlCap->MaxValue;
		SetPropertyLimits(MM::g_Keyword_Offset, iMin, iMax);
	}

	//USBTraffic
	pOneCtrlCap = GetOneCtrlCap(ASI_BANDWIDTHOVERLOAD);
	if(pOneCtrlCap)
	{
		pAct = new CPropertyAction (this, &CMyASICam::OnUSBTraffic);
		ret = CreateProperty(g_Keyword_USBTraffic, "1", MM::Integer, false, pAct);
		assert(ret == DEVICE_OK);
		iMin = pOneCtrlCap->MinValue;
		iMax = pOneCtrlCap->MaxValue;
		SetPropertyLimits(g_Keyword_USBTraffic, iMin, iMax);

		pAct = new CPropertyAction (this, &CMyASICam::OnUSB_Auto);
		ret = CreateProperty(g_Keyword_USBTraffic_Auto, g_Keyword_off, MM::String, false, pAct);
		assert(ret == DEVICE_OK);
		ret = SetAllowedValues(g_Keyword_USBTraffic_Auto, boolValues);
		assert(ret == DEVICE_OK);
	}
	//Temperature
	
	if(GetOneCtrlCap(ASI_TEMPERATURE))
	{

		pAct = new CPropertyAction (this, &CMyASICam::OnTemperature);
		ret = CreateProperty(MM::g_Keyword_CCDTemperature, "0", MM::Float, true, pAct);
		assert(ret == DEVICE_OK);

	}

	// white balance red
	if(ASICameraInfo.IsColorCam)
	{
		pOneCtrlCap = GetOneCtrlCap(ASI_WB_R);
		pAct = new CPropertyAction (this, &CMyASICam::OnWB_R);
		ret = CreateProperty(g_Keyword_WB_R, "1", MM::Integer, false, pAct);
		assert(ret == DEVICE_OK);
		iMin = pOneCtrlCap->MinValue;
		iMax = pOneCtrlCap->MaxValue;
		SetPropertyLimits(g_Keyword_WB_R, iMin, iMax);

		// white balance blue
		pOneCtrlCap = GetOneCtrlCap(ASI_WB_B);
		pAct = new CPropertyAction (this, &CMyASICam::OnWB_B);
		ret = CreateProperty(g_Keyword_WB_B, "1", MM::Integer, false, pAct);
		assert(ret == DEVICE_OK);
		iMin = pOneCtrlCap->MinValue;
		iMax = pOneCtrlCap->MaxValue;
		SetPropertyLimits(g_Keyword_WB_B, iMin, iMax);

		//auto white balance blue
		pAct = new CPropertyAction (this, &CMyASICam::OnAutoWB);
		ret = CreateProperty(g_Keyword_AutoWB, g_Keyword_off, MM::String, false, pAct);
		assert(ret == DEVICE_OK);
		ret = SetAllowedValues(g_Keyword_AutoWB, boolValues);
	}

	//cool
	if(ASICameraInfo.IsCoolerCam)
	{
		//Cooler Switch
		pAct = new CPropertyAction (this, &CMyASICam::OnCoolerOn);
		ret = CreateProperty(g_Keyword_IsCoolerOn, g_Keyword_off, MM::String, false, pAct);
		assert(ret == DEVICE_OK);
		vector<string> coolerValues;
		coolerValues.push_back(g_Keyword_off);
		coolerValues.push_back(g_Keyword_on); 
		ret = SetAllowedValues(g_Keyword_IsCoolerOn, coolerValues);
		assert(ret == DEVICE_OK);

		//Target Temperature
		pOneCtrlCap = GetOneCtrlCap(ASI_TARGET_TEMP);
		pAct = new CPropertyAction (this, &CMyASICam::OnTargetTemp);
		ret = CreateProperty(g_Keyword_TargetTemp, "0", MM::Integer, false, pAct);
		assert(ret == DEVICE_OK);
		iMin = pOneCtrlCap->MinValue;
		iMax = pOneCtrlCap->MaxValue;
		SetPropertyLimits(g_Keyword_TargetTemp, iMin, iMax);
		assert(ret == DEVICE_OK);

		//power percentage
		pOneCtrlCap = GetOneCtrlCap(ASI_COOLER_POWER_PERC);
		pAct = new CPropertyAction (this, &CMyASICam::OnCoolerPowerPerc);
		ret = CreateProperty(g_Keyword_CoolPowerPerc, "0", MM::Integer, true, pAct);
		assert(ret == DEVICE_OK);
		iMin = pOneCtrlCap->MinValue;
		iMax = pOneCtrlCap->MaxValue;
		SetPropertyLimits(g_Keyword_CoolPowerPerc, iMin, iMax);
		assert(ret == DEVICE_OK);

		//Anti dew
		pAct = new CPropertyAction (this, &CMyASICam::OnHeater);
		ret = CreateProperty(g_Keyword_IsHeaterOn, g_Keyword_off, MM::String, false, pAct);
		assert(ret == DEVICE_OK);		
		ret = SetAllowedValues(g_Keyword_IsHeaterOn, coolerValues);
		assert(ret == DEVICE_OK);
	}

	//gamma
	pOneCtrlCap = GetOneCtrlCap(ASI_GAMMA);
	if(pOneCtrlCap)
	{
		pAct = new CPropertyAction (this, &CMyASICam::OnGamma);
		ret = CreateProperty(g_Keyword_Gamma, "1", MM::Integer, false, pAct);
		assert(ret == DEVICE_OK);
		iMin = pOneCtrlCap->MinValue;
		iMax = pOneCtrlCap->MaxValue;
		SetPropertyLimits(g_Keyword_Gamma, iMin, iMax);
	}

	//auto exposure
	pOneCtrlCap = GetOneCtrlCap(ASI_EXPOSURE);
	if(pOneCtrlCap)
	{
		pAct = new CPropertyAction (this, &CMyASICam::OnAutoExp);
		ret = CreateProperty(g_Keyword_AutoExp, "1", MM::String, false, pAct);
		assert(ret == DEVICE_OK);
		SetAllowedValues(g_Keyword_AutoExp, boolValues);
	}
	//auto gain
	pOneCtrlCap = GetOneCtrlCap(ASI_GAIN);
	if(pOneCtrlCap)
	{
		pAct = new CPropertyAction (this, &CMyASICam::OnAutoGain);
		ret = CreateProperty(g_Keyword_AutoGain, "1", MM::String, false, pAct);
		assert(ret == DEVICE_OK);
		SetAllowedValues(g_Keyword_AutoGain, boolValues);

	}

	//flip
	pOneCtrlCap = GetOneCtrlCap(ASI_FLIP);
	if(pOneCtrlCap)
	{
		pAct = new CPropertyAction (this, &CMyASICam::OnFlip);
		ret = CreateProperty(g_Keyword_Flip, "1", MM::String, false, pAct);
		assert(ret == DEVICE_OK);
		boolValues.clear();
		for(int i = 0; i < 4; i++)
			boolValues.push_back(FlipArr[i]);
	
		SetAllowedValues(g_Keyword_Flip, boolValues);

	}

	//high speed mode
	pOneCtrlCap = GetOneCtrlCap(ASI_HIGH_SPEED_MODE);
	if(pOneCtrlCap)
	{
		pAct = new CPropertyAction (this, &CMyASICam::OnHighSpeedMod);
		ret = CreateProperty(g_Keyword_HighSpeedMode, "1", MM::String, false, pAct);
		assert(ret == DEVICE_OK);
		boolValues.clear();
		boolValues.push_back(g_Keyword_off);
		boolValues.push_back(g_Keyword_on);
		SetAllowedValues(g_Keyword_HighSpeedMode, boolValues);

	}

	//hardware bin
	pOneCtrlCap = GetOneCtrlCap(ASI_HARDWARE_BIN);
	if(pOneCtrlCap)
	{
		pAct = new CPropertyAction (this, &CMyASICam::OnHardwareBin);
		ret = CreateProperty(g_Keyword_HardwareBin, "1", MM::String, false, pAct);
		assert(ret == DEVICE_OK);
		boolValues.clear();
		boolValues.push_back(g_Keyword_off);
		boolValues.push_back(g_Keyword_on);
		SetAllowedValues(g_Keyword_HardwareBin, boolValues);

	}

	//USB3 host
	char USBHost[16] = {0};
	if(ASICameraInfo.IsUSB3Host)
		strcpy(USBHost, "USB3");
	else
		strcpy(USBHost, "USB2");
	ret = CreateProperty(g_Keyword_USBHost, USBHost, MM::String, true);
	assert(ret == DEVICE_OK);


	// synchronize all properties
	// --------------------------
	ret = UpdateStatus();
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
int CMyASICam::Shutdown()
{
	initialized_ = false;
	StopSequenceAcquisition();

	if(Status == capturing)
		ASIStopVideoCapture(ASICameraInfo.CameraID);
	if(Status == snaping)
		ASIStopExposure(ASICameraInfo.CameraID);
	ASICloseCamera(ASICameraInfo.CameraID);
	
	if(pControlCaps)
	{
		delete pControlCaps;
		pControlCaps = 0;
	}
	DeleteImgBuf();
	Status = closed;
	return DEVICE_OK;
}

/**
* Performs exposure and grabs a single image.
* This function should block during the actual exposure and return immediately afterwards 
* (i.e., before readout).  This behavior is needed for proper synchronization with the shutter.
* Required by the MM::Camera API.
*/
int CMyASICam::SnapImage()//曝光期间要阻塞
{
	//  GenerateImage();
//	ASIGetStartPos(iCamIndex, &iStartXImg, &iStartYImg);
	ASIStartExposure(ASICameraInfo.CameraID, ASI_FALSE);
	Status = snaping;
	unsigned long time = GetTickCount(), deltaTime = 0;
	ASI_EXPOSURE_STATUS exp_status;
	do 
	{
		ASIGetExpStatus(ASICameraInfo.CameraID, &exp_status);
		deltaTime = GetTickCount() - time;
		if(deltaTime > 10000 && GetTickCount() - time > 3*lExpMs)
		{
			OutputDbgPrint("delta %d ms, stop snap\n", deltaTime);
			ASIStopExposure(ASICameraInfo.CameraID);
			break;
		}
			
		Sleep(1);
	} while (exp_status == ASI_EXP_WORKING);

	Status = opened;

	if(uc_pImg == 0)
	{
		iBufSize = GetImageBufferSize();
		uc_pImg = new unsigned char[iBufSize];
	}
	if(exp_status == ASI_EXP_SUCCESS)
	{
		ASIGetDataAfterExp(ASICameraInfo.CameraID, uc_pImg, iBufSize);
		ASI_BOOL bAuto;
		long lVal;
		ASIGetControlValue(ASICameraInfo.CameraID, ASI_FLIP, &lVal, &bAuto);
		ImgFlip = (ASI_FLIP_STATUS)lVal;
		ASI_IMG_TYPE imgType;
		ASIGetROIFormat(ASICameraInfo.CameraID, &ImgWid, &ImgHei, &ImgBin, &imgType);
		ASIGetStartPos(ASICameraInfo.CameraID, &ImgStartX, &ImgStartY);
	}

	OutputDbgPrint("exp_status %d\n", (int)exp_status);
	if(exp_status == ASI_EXP_SUCCESS)
		return DEVICE_OK;
	else
		return DEVICE_SNAP_IMAGE_FAILED;
}
void CMyASICam::Conv16RAWTo12RAW()
{
	unsigned long line0;
//	unsigned int *pBuf16 = (unsigned int*)uc_pImg;
#ifdef _WINDOWS
	UINT16 *pBuf16 = (UINT16 *)uc_pImg;//unsigned short
#else
	uint16_t *pBuf16 = (uint16_t *)uc_pImg;//unsigned short
#endif
	
	for(int y = 0; y < iROIHeight; y++)
	{
		line0 = iROIWidth*y;
		for(int x = 0; x < iROIWidth; x++)
		{
			pBuf16[line0 + x] /= 16;
		}
	}
}
void CMyASICam::ConvRGB2RGBA32()
{
	if(!pRGB32)
	{
		pRGB32 = new unsigned char[iROIWidth*iROIHeight*4];
	}
	unsigned long index32, index24, line0;
	for(int y = 0; y < iROIHeight; y++)
	{
		line0 = iROIWidth*y;
		for(int x = 0; x < iROIWidth; x++)
		{
			index32 = (line0 + x)*4;
			index24 = (line0 + x)*3;
			pRGB32[index32 + 0] = uc_pImg[index24+0];
			pRGB32[index32 + 1] = uc_pImg[index24+1];
			pRGB32[index32 + 2] = uc_pImg[index24+2];
			pRGB32[index32 + 3] = 0;
		}
	}
}

void CMyASICam::ConvRGB2RGBA64()
{
	if(!pRGB64)
	{
		pRGB64 = new unsigned char[iROIWidth*iROIHeight*4*2];
		memset(pRGB64, 0, iROIWidth*iROIHeight*4*2);
	}
	unsigned long index64, index24, line0;
	for(int y = 0; y < iROIHeight; y++)
	{
		line0 = iROIWidth*y;
		for(int x = 0; x < iROIWidth; x++)
		{
			index64 = (line0 + x)*8;
			index24 = (line0 + x)*3;
			pRGB64[index64 + 1] = uc_pImg[index24+0];
			pRGB64[index64 + 3] = uc_pImg[index24+1];
			pRGB64[index64 + 5] = uc_pImg[index24+2];
//			pRGB64[index64 + 6] = 0;
		}
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
const unsigned char* CMyASICam::GetImageBuffer()
{
	//  return const_cast<unsigned char*>(img_.GetPixels());
	if(ImgType == ASI_IMG_RGB24)
	{
		if(bRGB48)
		{
			ConvRGB2RGBA64();
			return pRGB64;
		}
		else
		{
			ConvRGB2RGBA32();
			return pRGB32;
		}
	
	}
	else if(ImgType == ASI_IMG_RAW16 && b12RAW)
		Conv16RAWTo12RAW();
		return uc_pImg;
}

/**
* Returns image buffer X-size in pixels.
* Required by the MM::Camera API.
*/
unsigned CMyASICam::GetImageWidth() const
{
	return iROIWidth;
}

/**
* Returns image buffer Y-size in pixels.
* Required by the MM::Camera API.
*/
unsigned CMyASICam::GetImageHeight() const
{
	return iROIHeight;
}

/**
* Returns image buffer pixel depth in bytes.
* Required by the MM::Camera API.
*/
unsigned CMyASICam::GetImageBytesPerPixel() const //每个像素的字节数
{
	return iPixBytes;
} 

/**
* Returns the bit depth (dynamic range) of the pixel.
* This does not affect the buffer size, it just gives the client application
* a guideline on how to interpret pixel values.
* Required by the MM::Camera API.
*/
unsigned CMyASICam::GetBitDepth() const//颜色的范围 8bit 或 16bit
{
	if(ImgType == ASI_IMG_RAW16)
	{
		if(b12RAW)
			return 12;
		else
			return 16;
	}
	else
	{
		if(ImgType == ASI_IMG_RGB24 && bRGB48)
			return 16;
		else
			return 8;
	}
}

/**
* Returns the size in bytes of the image buffer.
* Required by the MM::Camera API.
*/
long CMyASICam::GetImageBufferSize() const
{
	return iROIWidth*iROIHeight*iPixBytes;
}
 /**
       * Returns the name for each component 
       */
int CMyASICam::GetComponentName(unsigned component, char* name)
{
	if(iComponents != 1)
	{
		switch (component)
		{
		case 1:
			strcpy(name, "red");
			break;
		case 2:
			strcpy(name, "green");
			break;
		case 3:
			strcpy(name, "blue");
			break;
		case 4:
			strcpy(name, "0");
			break;
		default:
			strcpy(name, "error");
			break;
		}
	}
	else
		strcpy(name, "grey");
	return DEVICE_OK;
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
void CMyASICam::DeleteImgBuf()
{
	if(uc_pImg)
	{
		delete[] uc_pImg;
		uc_pImg = 0;
		iBufSize = 0;
		OutputDbgPrint("clr\n");
	}
	if(pRGB32)
	{
		delete[] pRGB32;
		pRGB32 = 0;
	}
	if(pRGB64)
	{
		delete[] pRGB64;
		pRGB64 = 0;
	}
}
int CMyASICam::SetROI(unsigned x, unsigned y, unsigned xSize, unsigned ySize)//bin2时的值是相对于bin2后图像上的, 而ASISetStartPos都是相对于bin1的
{
	if (xSize == 0 && ySize == 0)
		;
	else
	{
		/*20160107
		设置ROI是以显示图片为参照的,程序传进来的起始点(x, y)是用已显示图片的起始点(调用GetROI()得到)加上鼠标选择的偏移，
		尺寸和起始点都以ImgBin/iSetBin缩放, iSetBin是要设置的bin值
		如果有翻转, 则要换算回正常的起点坐标
		*/
		switch(ImgFlip)
		{
		case ASI_FLIP_NONE:		
			break;
		case ASI_FLIP_HORIZ:
			x = ASICameraInfo.MaxWidth/ImgBin - x - xSize;
			break;
		case ASI_FLIP_VERT:
			y = ASICameraInfo.MaxHeight/ImgBin - y - ySize;
			break;
		case ASI_FLIP_BOTH:
			x = ASICameraInfo.MaxWidth/ImgBin - x - xSize;
			y = ASICameraInfo.MaxHeight/ImgBin - y - ySize;
			break;
		}	
	
		iSetWid = xSize*ImgBin/iSetBin;// 2->1, *2
		iSetHei = ySize*ImgBin/iSetBin;//1->2. *0.5
		iSetWid = iSetWid/8*8;
		iSetHei = iSetHei/2*2;


		iSetX = x*ImgBin/iSetBin;//bin改变后, startpos是相对于bin后的画面的，也要按照比例改变
		iSetY = y*ImgBin/iSetBin;		
		iSetX = iSetX/4*4;
		iSetY = iSetY/2*2;	

		if(ASISetROIFormat(ASICameraInfo.CameraID, iSetWid, iSetHei, iSetBin, ImgType)  == ASI_SUCCESS)//如果设置成功
		{
			OutputDbgPrint("wid:%d hei:%d bin:%d\n", xSize, ySize, iBin);
			DeleteImgBuf();//buff大小改变
			ASISetStartPos(ASICameraInfo.CameraID, iSetX, iSetY);			
		}
		ASIGetROIFormat(ASICameraInfo.CameraID, &iROIWidth, &iROIHeight, &iBin, &ImgType);
	}
	return DEVICE_OK;
}

/**
* Returns the actual dimensions of the current ROI.
* Required by the MM::Camera API.
*/
int CMyASICam::GetROI(unsigned& x, unsigned& y, unsigned& xSize, unsigned& ySize)//程序调用这里得到当前ROI起点，加上ROI里的矩形起点，得到ROI再ROI的起点
{
	/* 20160107
	得到显示图像的ROI信息
		如果有翻转, 要换算成反方向边的坐标, 方便程序相加得到新ROI,再换算回正常方向的坐标*/

	x = ImgStartX;
	y = ImgStartY;
	switch(ImgFlip)
	{
	case ASI_FLIP_NONE:		
		break;
	case ASI_FLIP_HORIZ:
		x = ASICameraInfo.MaxWidth/ImgBin - ImgStartX - ImgWid;
		break;
	case ASI_FLIP_VERT:
		y = ASICameraInfo.MaxHeight/ImgBin - ImgStartY - ImgHei;
		break;
	case ASI_FLIP_BOTH:
		x = ASICameraInfo.MaxWidth/ImgBin - ImgStartX - ImgWid;
		y = ASICameraInfo.MaxHeight/ImgBin - ImgStartY - ImgHei;
		break;
	}	
	xSize = ImgWid;
	ySize = ImgHei;
	return DEVICE_OK;
}

/**
* Resets the Region of Interest to full frame.
* Required by the MM::Camera API.
*/
int CMyASICam::ClearROI()
{
	//  ResizeImageBuffer();
	iSetWid = iROIWidth = ASICameraInfo.MaxWidth/iBin/8*8;
	iSetHei = iROIHeight = ASICameraInfo.MaxHeight/iBin/2*2;
	
	if(ASISetROIFormat(ASICameraInfo.CameraID, iROIWidth, iROIHeight, iBin, ImgType) == ASI_SUCCESS)
	{
		ASISetStartPos(ASICameraInfo.CameraID, 0, 0);
		iSetX = iSetY = 0;
		DeleteImgBuf();
	}
	return DEVICE_OK;
}

/**
* Returns the current exposure setting in milliseconds.
* Required by the MM::Camera API.
*/
double CMyASICam::GetExposure() const
{
//	return lExpMs;
	long lVal;
	ASI_BOOL bAuto;
	ASIGetControlValue(ASICameraInfo.CameraID, ASI_EXPOSURE, &lVal, &bAuto);
	return lVal/1000;

}

/**
* Sets exposure in milliseconds.
* Required by the MM::Camera API.
*/
void CMyASICam::SetExposure(double exp)
{
	lExpMs = exp;
	ASISetControlValue(ASICameraInfo.CameraID, ASI_EXPOSURE, exp*1000, ASI_FALSE);
}

/**
* Returns the current binning factor.
* Required by the MM::Camera API.
*/
int CMyASICam::GetBinning() const
{
	return iBin;
}

/**
* Sets binning factor.
* Required by the MM::Camera API.
*/
int CMyASICam::SetBinning(int binF)
{
	return SetProperty(MM::g_Keyword_Binning, CDeviceUtils::ConvertToString(binF));//就是onBinning(, afterSet)
}

int CMyASICam::PrepareSequenceAcqusition()
{
	if (IsCapturing())
		return DEVICE_CAMERA_BUSY_ACQUIRING;
	/*   int ret = GetCoreCallback()->PrepareForAcq(this);
	if (ret != DEVICE_OK)
	return ret;*/
	return DEVICE_OK;
}


/**
* Required by the MM::Camera API
* Please implement this yourself and do not rely on the base class implementation
* The Base class implementation is deprecated and will be removed shortly
*/
int CMyASICam::StartSequenceAcquisition(double interval) {
	return StartSequenceAcquisition(LONG_MAX, interval, false);            
}
/**
* Simple implementation of Sequence Acquisition
* A sequence acquisition should run on its own thread and transport new images
* coming of the camera into the MMCore circular buffer.
*/
int CMyASICam::StartSequenceAcquisition(long numImages, double interval_ms, bool stopOnOverflow)
{
	if (IsCapturing())
		return DEVICE_CAMERA_BUSY_ACQUIRING;

	OutputDbgPrint("StartCap\n");

	ASIStartVideoCapture(ASICameraInfo.CameraID);
	Status = capturing;

	OutputDbgPrint("StartSeqAcq\n");
	thd_->Start(numImages,interval_ms);//开始线程

	return DEVICE_OK;
}

/*
* Do actual capturing
* Called from inside the thread  
*/
int CMyASICam::RunSequenceOnThread(MM::MMTime startTime)
{
	int ret=DEVICE_ERR;
	
	if(ASIGetVideoData(ASICameraInfo.CameraID, uc_pImg, iBufSize, 2*lExpMs) == ASI_SUCCESS)
	{
		ret = InsertImage();
		ASI_BOOL bAuto;
		long lVal;
		ASIGetControlValue(ASICameraInfo.CameraID, ASI_FLIP, &lVal, &bAuto);
		ImgFlip = (ASI_FLIP_STATUS)lVal;
	
		ASI_IMG_TYPE imgType;
		ASIGetROIFormat(ASICameraInfo.CameraID, &ImgWid, &ImgHei, &ImgBin, &imgType);
		ASIGetStartPos(ASICameraInfo.CameraID, &ImgStartX, &ImgStartY);
	}
	return ret;
}

/*
* Inserts Image and MetaData into MMCore circular Buffer
*/
int CMyASICam::InsertImage()
{
	MM::MMTime timeStamp = this->GetCurrentMMTime();
	char label[MM::MaxStrLength];
	this->GetLabel(label);

	// Important:  metadata about the image are generated here:
	Metadata md;
	md.put("Camera", label);

	char buf[MM::MaxStrLength];
	GetProperty(MM::g_Keyword_Binning, buf);
	md.put(MM::g_Keyword_Binning, buf);

	//   MMThreadGuard g(imgPixelsLock_);

	const unsigned char* pI;
	pI = GetImageBuffer();
	int ret = 0;
	ret  = GetCoreCallback()->InsertImage(this, pI, iROIWidth, iROIHeight, iPixBytes, md.Serialize().c_str());
	if (ret == DEVICE_BUFFER_OVERFLOW)//缓冲区满了要清空, 否则不能继续插入图像而卡住
	{
		// do not stop on overflow - just reset the buffer
		GetCoreCallback()->ClearImageBuffer(this);
		// don't process this same image again...
		return GetCoreCallback()->InsertImage(this, pI, iROIWidth, iROIHeight, iPixBytes, md.Serialize().c_str(), false);
	} else
		return ret;
}


/**                                                                       
* Stop and wait for the Sequence thread finished                                   
*/                                                                        
int CMyASICam::StopSequenceAcquisition()                                     
{                                                                         
	if (!thd_->IsStopped())
	{
		thd_->Stop();//停止线程
		OutputDbgPrint("StopSeqAcq bf wait\n");
//		if(!thd_->IsStopped())
		thd_->wait();//等待线程退出
		OutputDbgPrint("StopSeqAcq af wait\n");
	}                                                                    
//	if(Status == capturing)
//	{
	ASIStopVideoCapture(ASICameraInfo.CameraID);
	Status = opened;
//	}

	return DEVICE_OK;                                                      
} 



bool CMyASICam::IsCapturing() 
{
	//  return !thd_->IsStopped();
	if(Status == capturing || Status == snaping)
		return true;
	else
		return false;
}


///////////////////////////////////////////////////////////////////////////////
// CMyASICam Action handlers
///////////////////////////////////////////////////////////////////////////////

/**
* Handles "Binning" property.
*/
int CMyASICam::OnBinning(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::AfterSet)
	{
		long binSize;
		pProp->Get(binSize);
		char binF;
		binF = binSize;
		
		if( !thd_->IsStopped() )//micro manager主面板里bin时会先由程序停止再设置，而在property里设置bin不会停止，导致错误，所以不停止时不能设置
		   return DEVICE_CAMERA_BUSY_ACQUIRING;
		/* bin后的 起始点和尺寸是 把设置值按照 old Bin/new Bin 缩放的*/
		 iSetWid = iSetWid*iSetBin/binF;// 2->1, *2
		iSetHei = iSetHei*iSetBin/binF;//1->2. *0.5
		iSetWid = iSetWid/8*8;
		iSetHei = iSetHei/2*2;	

		iSetX = iSetX*iSetBin/binF;//bin改变后, startpos是相对于bin后的画面的，也要按照比例改变
		iSetY = iSetY*iSetBin/binF;

		if(ASISetROIFormat(ASICameraInfo.CameraID, iSetWid, iSetHei, binF, ImgType) == ASI_SUCCESS)
		{
			DeleteImgBuf();
			ASISetStartPos(ASICameraInfo.CameraID, iSetX, iSetY);//会重新计算startx 和starty，和所选区域不同，因此要重新设置
		}
		ASIGetROIFormat(ASICameraInfo.CameraID, &iROIWidth, &iROIHeight, &iBin, &ImgType);
		iSetBin = binF;
	}
	else if (eAct == MM::BeforeGet)
	{
		ASIGetROIFormat(ASICameraInfo.CameraID, &iROIWidth, &iROIHeight, &iBin, &ImgType);
		pProp->Set((long)iBin);
	}

	return DEVICE_OK;
}

/**
* Handles "PixelType" property.
*/
int CMyASICam::OnPixelType(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::AfterSet)//从控件得到选定的值
	{
		string val;
		pProp->Get(val);
		if(Status == capturing)
			return DEVICE_CAMERA_BUSY_ACQUIRING;
		if (val.compare(g_PixelType_RAW8) == 0)
			ImgType = ASI_IMG_RAW8;
		else if (val.compare(g_PixelType_RAW16) == 0)
		{
			ImgType = ASI_IMG_RAW16;
			b12RAW = false;
		}
		else if (val.compare(g_PixelType_RAW12) == 0 )
		{
			ImgType = ASI_IMG_RAW16;
			b12RAW = true;
		}
		else if (val.compare(g_PixelType_Y8) == 0)
			ImgType = ASI_IMG_Y8;
		else if (val.compare(g_PixelType_RGB24) == 0)
		{
			ImgType = ASI_IMG_RGB24;
			bRGB48 = false;
		}
		else if (val.compare(g_PixelType_RGB48) == 0)
		{
			ImgType = ASI_IMG_RGB24;
			bRGB48 = true;
		}
		RefreshImgType();
		OutputDbgPrint("w%d h%d b%d t%d\n", iROIWidth, iROIHeight, iBin, ImgType);
		int iStartX, iStartY;
		ASIGetStartPos(ASICameraInfo.CameraID, &iStartX, &iStartY);
		if(ASISetROIFormat(ASICameraInfo.CameraID, iROIWidth, iROIHeight, iBin, ImgType) == ASI_SUCCESS)
		{
			ASISetStartPos(ASICameraInfo.CameraID, iStartX, iStartY);
			DeleteImgBuf();
		}


	}
	else if (eAct == MM::BeforeGet)//值给控件显示
	{
		ASIGetROIFormat(ASICameraInfo.CameraID, &iROIWidth, &iROIHeight, &iBin, &ImgType);

		if(ImgType == ASI_IMG_RAW8)
			pProp->Set(g_PixelType_RAW8);
		else if(ImgType == ASI_IMG_RAW16)
		{
			if(b12RAW)
				pProp->Set(g_PixelType_RAW12);
			else
				pProp->Set(g_PixelType_RAW16);
		}
		else if(ImgType == ASI_IMG_Y8)
			pProp->Set(g_PixelType_Y8);
		else if(ImgType == ASI_IMG_RGB24)
		{
			if(bRGB48)
				pProp->Set(g_PixelType_RGB48);
			else
				pProp->Set(g_PixelType_RGB24);
		}
		else
			assert(false); // this should never happen

		RefreshImgType();

	}

	return DEVICE_OK;
}

/**
* Handles "Gain" property.
*/
int CMyASICam::OnGain(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	long lVal;
	ASI_BOOL bAuto;
	if (eAct == MM::AfterSet)
	{
		pProp->Get(lVal);
		ASISetControlValue(ASICameraInfo.CameraID, ASI_GAIN, lVal, ASI_FALSE);
	}
	else if (eAct == MM::BeforeGet)
	{
		ASIGetControlValue(ASICameraInfo.CameraID, ASI_GAIN, &lVal, &bAuto);
		pProp->Set(lVal);
	}

	return DEVICE_OK;
}

/**
* Handles "CamIndex" property.
*/
int CMyASICam::OnSelectCamIndex(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	string str;
	if (eAct == MM::AfterSet)//从控件得到选定的值
	{
		pProp->Get(str);
		for(int i = 0; i < iConnectedCamNum; i++)
		{
			if(!str.compare(ConnectedCamName[i]))
			{
			//	iCamIndex = i;
				ASIGetCameraProperty(&ASICameraInfo, i);
				strcpy(sz_ModelIndex, ConnectedCamName[i]);
				break;
			}
		}
	}
	else if (eAct == MM::BeforeGet)//值给控件显示
	{
		pProp->Set(sz_ModelIndex);
	}

	return DEVICE_OK;
}
/**
* Handles "Temperature" property.
*/
int CMyASICam::OnTemperature(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	long lVal;
	ASI_BOOL bAuto;
	if (eAct == MM::AfterSet)
	{
		pProp->Get(lVal);
	}
	else if (eAct == MM::BeforeGet)
	{
		ASIGetControlValue(ASICameraInfo.CameraID, ASI_TEMPERATURE, &lVal, &bAuto);
		pProp->Set((double)lVal/10);
	}


	return DEVICE_OK;
}
/**
* Handles "Brightness" property.
*/
int CMyASICam::OnBrightness(MM::PropertyBase* pProp,MM::ActionType eAct)
{
	long lVal;
	ASI_BOOL bAuto;
	if (eAct == MM::AfterSet)//从控件得到选定的值
	{
		pProp->Get(lVal);
		ASISetControlValue(ASICameraInfo.CameraID,ASI_BRIGHTNESS, lVal, ASI_FALSE);
	}
	else if (eAct == MM::BeforeGet)//值给控件显示
	{
		ASIGetControlValue(ASICameraInfo.CameraID, ASI_BRIGHTNESS, &lVal, &bAuto);
		pProp->Set(lVal);
	}

	return DEVICE_OK;
}

/**
* Handles "USBTraffic" property.
*/
int CMyASICam::OnUSBTraffic(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	long lVal;
	ASI_BOOL bAuto;
	if (eAct == MM::AfterSet)//从控件得到选定的值
	{
		pProp->Get(lVal);
		ASISetControlValue(ASICameraInfo.CameraID,ASI_BANDWIDTHOVERLOAD, lVal, ASI_FALSE);
	}
	else if (eAct == MM::BeforeGet)//值给控件显示
	{
		ASIGetControlValue(ASICameraInfo.CameraID,ASI_BANDWIDTHOVERLOAD, &lVal, &bAuto);
		pProp->Set(lVal);
	}

	return DEVICE_OK;
}

/**
* Handles "USBTraffic Auto" property.
*/
int CMyASICam::OnUSB_Auto(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	long lVal;
	ASI_BOOL bAuto;
	if (eAct == MM::AfterSet)//从控件得到选定的值
	{
		ASIGetControlValue(ASICameraInfo.CameraID,ASI_BANDWIDTHOVERLOAD, &lVal, &bAuto);
		string strVal;
		pProp->Get(strVal);
		bAuto = strVal.compare(g_Keyword_on)?ASI_FALSE:ASI_TRUE;
		ASISetControlValue(ASICameraInfo.CameraID,ASI_BANDWIDTHOVERLOAD, lVal, bAuto);
//		SetPropertyReadOnly(g_Keyword_USBTraffic, bAuto);
	}
	else if (eAct == MM::BeforeGet)//值给控件显示
	{
		ASIGetControlValue(ASICameraInfo.CameraID,ASI_BANDWIDTHOVERLOAD, &lVal, &bAuto);
		pProp->Set(bAuto==ASI_TRUE?g_Keyword_on:g_Keyword_off);
//		SetPropertyReadOnly(g_Keyword_USBTraffic,bAuto);
	}

	return DEVICE_OK;
}
/**
* Handles "Cooler switch" property.
*/
int CMyASICam::OnCoolerOn(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	long lVal;
	ASI_BOOL bAuto;
	if (eAct == MM::AfterSet)
	{
	//	ASIGetControlValue(iCamIndex, ASI_TARGET_TEMP, &lVal, &bAuto);
		string strVal;
		pProp->Get(strVal);//从控件得到选定的值
		lVal = !strVal.compare(g_Keyword_on);
		ASISetControlValue(ASICameraInfo.CameraID, ASI_COOLER_ON, lVal, ASI_FALSE);
	}
	else if (eAct == MM::BeforeGet)//值给控件显示
	{
		ASIGetControlValue(ASICameraInfo.CameraID, ASI_COOLER_ON, &lVal, &bAuto);
		pProp->Set(lVal > 0?g_Keyword_on:g_Keyword_off);
	}
	return DEVICE_OK;
}
/**
* Handles "Heater switch" property.
*/
int CMyASICam::OnHeater(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	long lVal;
	ASI_BOOL bAuto;
	if (eAct == MM::AfterSet)
	{
		//	ASIGetControlValue(iCamIndex, ASI_TARGET_TEMP, &lVal, &bAuto);
		string strVal;
		pProp->Get(strVal);//从控件得到选定的值
		lVal = !strVal.compare(g_Keyword_on);
		ASISetControlValue(ASICameraInfo.CameraID, ASI_ANTI_DEW_HEATER, lVal, ASI_FALSE);
	}
	else if (eAct == MM::BeforeGet)//值给控件显示
	{
		ASIGetControlValue(ASICameraInfo.CameraID, ASI_ANTI_DEW_HEATER, &lVal, &bAuto);
		pProp->Set(lVal > 0?g_Keyword_on:g_Keyword_off);
	}
	return DEVICE_OK;
}
/**
* Handles "Target Temperature" property.
*/
int CMyASICam::OnTargetTemp(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	long lVal;
	ASI_BOOL bAuto;
	if (eAct == MM::AfterSet)
	{
		ASIGetControlValue(ASICameraInfo.CameraID,ASI_TARGET_TEMP, &lVal, &bAuto);
		pProp->Get(lVal);//从控件得到选定的值->变量
		ASISetControlValue(ASICameraInfo.CameraID,ASI_TARGET_TEMP, lVal, bAuto);
	}
	else if (eAct == MM::BeforeGet)//变量值->控件显示
	{
		ASIGetControlValue(ASICameraInfo.CameraID,ASI_TARGET_TEMP, &lVal, &bAuto);
		pProp->Set(lVal);
	}
	return DEVICE_OK;
}
/**
* Handles "power percentage" property.
*/
int CMyASICam::OnCoolerPowerPerc(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	long lVal;
	ASI_BOOL bAuto;
	if (eAct == MM::AfterSet)
	{
		pProp->Get(lVal);//从控件得到选定的值->变量

	}
	else if (eAct == MM::BeforeGet)//变量值->控件显示
	{
		ASIGetControlValue(ASICameraInfo.CameraID,ASI_COOLER_POWER_PERC, &lVal, &bAuto);
		pProp->Set(lVal);
	}
	return DEVICE_OK;
}
/**
* Handles "white balance red" property.
*/
int CMyASICam::OnWB_R(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	long lVal;
	ASI_BOOL bAuto;
	if (eAct == MM::AfterSet)
	{
		pProp->Get(lVal);//从控件得到选定的值->变量
		ASISetControlValue(ASICameraInfo.CameraID,ASI_WB_R, lVal, ASI_FALSE);
	}
	else if (eAct == MM::BeforeGet)//变量值->控件显示
	{
		ASIGetControlValue(ASICameraInfo.CameraID,ASI_WB_R, &lVal, &bAuto);
		pProp->Set(lVal);
	}
	return DEVICE_OK;
}
/**
* Handles "white balance blue" property.
*/
int CMyASICam::OnWB_B(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	long lVal;
	ASI_BOOL bAuto;
	if (eAct == MM::AfterSet)
	{
		pProp->Get(lVal);//从控件得到选定的值->变量
		ASISetControlValue(ASICameraInfo.CameraID,ASI_WB_B, lVal, ASI_FALSE);
	}
	else if (eAct == MM::BeforeGet)//变量值->控件显示
	{
		ASIGetControlValue(ASICameraInfo.CameraID,ASI_WB_B, &lVal, &bAuto);
		pProp->Set(lVal);
	}
	return DEVICE_OK;
}
	/**
* Handles "auto white balance" property.
*/
int CMyASICam::OnAutoWB(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	string strVal;
	long lVal;
	ASI_BOOL bAuto;
	if (eAct == MM::AfterSet)
	{
		ASIGetControlValue(ASICameraInfo.CameraID,ASI_WB_B, &lVal, &bAuto);
		pProp->Get(strVal);//从控件得到选定的值->变量
		bAuto = strVal.compare(g_Keyword_on)?ASI_FALSE:ASI_TRUE;
		ASISetControlValue(ASICameraInfo.CameraID,ASI_WB_B, lVal, bAuto);
	//	SetPropertyReadOnly(g_Keyword_WB_R,bAuto );
	//	SetPropertyReadOnly(g_Keyword_WB_B,bAuto );
		
	}
	else if (eAct == MM::BeforeGet)//变量值->控件显示
	{
		ASIGetControlValue(ASICameraInfo.CameraID,ASI_WB_B, &lVal, &bAuto);
		pProp->Set(bAuto?g_Keyword_on:g_Keyword_off);
//		SetPropertyReadOnly(g_Keyword_WB_R,bAuto );
//		SetPropertyReadOnly(g_Keyword_WB_B,bAuto );
	}
	return DEVICE_OK;
}
	/**
* Handles "auto white balance" property.
*/
int CMyASICam::OnGamma(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	long lVal;
	ASI_BOOL bAuto;
	if (eAct == MM::AfterSet)//从控件得到选定的值->变量
	{
		
		pProp->Get(lVal);
		ASISetControlValue(ASICameraInfo.CameraID,ASI_GAMMA, lVal, ASI_FALSE);
	}
	else if(eAct == MM::BeforeGet)//变量值->控件显示
	{
		ASIGetControlValue(ASICameraInfo.CameraID,ASI_GAMMA, &lVal, &bAuto);
		pProp->Set(lVal);
	}
	return DEVICE_OK;
}
	/**
* Handles "auto exposure" property.
*/
int CMyASICam::OnAutoExp(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	long lVal;
	ASI_BOOL bAuto;
	if (eAct == MM::AfterSet)//从控件得到选定的值->变量
	{
		ASIGetControlValue(ASICameraInfo.CameraID,ASI_EXPOSURE, &lVal, &bAuto);
		string strVal;
		pProp->Get(strVal);
		bAuto = strVal.compare(g_Keyword_on)?ASI_FALSE:ASI_TRUE;
		ASISetControlValue(ASICameraInfo.CameraID,ASI_EXPOSURE, lVal, bAuto);
	}
	else if(eAct == MM::BeforeGet)//变量值->控件显示
	{
		ASIGetControlValue(ASICameraInfo.CameraID,ASI_EXPOSURE, &lVal, &bAuto);
		pProp->Set(bAuto?g_Keyword_on:g_Keyword_off);
//		SetPropertyReadOnly(MM::g_Keyword_Exposure,bAuto );
	}
	return DEVICE_OK;
}
	/**
* Handles "auto gain" property.
*/
int CMyASICam::OnAutoGain(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	long lVal;
	ASI_BOOL bAuto;
	if (eAct == MM::AfterSet)//从控件得到选定的值->变量
	{
		ASIGetControlValue(ASICameraInfo.CameraID,ASI_GAIN, &lVal, &bAuto);
		string strVal;
		pProp->Get(strVal);
		bAuto = strVal.compare(g_Keyword_on)?ASI_FALSE:ASI_TRUE;
		ASISetControlValue(ASICameraInfo.CameraID,ASI_GAIN, lVal, bAuto);
	}
	else if(eAct == MM::BeforeGet)//变量值->控件显示
	{
		ASIGetControlValue(ASICameraInfo.CameraID,ASI_GAIN, &lVal, &bAuto);
		pProp->Set(bAuto?g_Keyword_on:g_Keyword_off);
	//	SetPropertyReadOnly(MM::g_Keyword_Gain,bAuto );
	}
	return DEVICE_OK;
}
/**
* Handles "flip" property.
*/
int CMyASICam::OnFlip(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	long lVal;
	ASI_BOOL bAuto;
	if (eAct == MM::AfterSet)//从控件得到选定的值->变量
	{
		ASIGetControlValue(ASICameraInfo.CameraID, ASI_FLIP, &lVal, &bAuto);
		string strVal;
		pProp->Get(strVal);
		for(int i = 0; i < 4; i++)
		{
			if( !strVal.compare(FlipArr[i]) )
			{
				ASISetControlValue(ASICameraInfo.CameraID,ASI_FLIP, i, ASI_FALSE);
				break;
			}
		}
		
	}
	else if(eAct == MM::BeforeGet)//变量值->控件显示
	{
		ASIGetControlValue(ASICameraInfo.CameraID,ASI_FLIP, &lVal, &bAuto);
		pProp->Set(FlipArr[lVal]);
	}
	return DEVICE_OK;
}
/**
* Handles "hight speed mode" property.
*/
int CMyASICam::OnHighSpeedMod(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	long lVal;
	ASI_BOOL bAuto;
	if (eAct == MM::AfterSet)//从控件得到选定的值->变量
	{
		string strVal;
		pProp->Get(strVal);
		lVal = strVal.compare(g_Keyword_on)?0:1;
		ASISetControlValue(ASICameraInfo.CameraID,ASI_HIGH_SPEED_MODE, lVal, ASI_FALSE);
	}
	else if(eAct == MM::BeforeGet)//变量值->控件显示
	{
		ASIGetControlValue(ASICameraInfo.CameraID,ASI_HIGH_SPEED_MODE, &lVal, &bAuto);
		pProp->Set(lVal?g_Keyword_on:g_Keyword_off);
	}
	return DEVICE_OK;
}
/**
* Handles "hardware bin" property.
*/
int CMyASICam::OnHardwareBin(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	long lVal;
	ASI_BOOL bAuto;
	if (eAct == MM::AfterSet)//从控件得到选定的值->变量
	{
		string strVal;
		pProp->Get(strVal);
		lVal = strVal.compare(g_Keyword_on)?0:1;
		ASISetControlValue(ASICameraInfo.CameraID,ASI_HARDWARE_BIN, lVal, ASI_FALSE);
	}
	else if(eAct == MM::BeforeGet)//变量值->控件显示
	{
		ASIGetControlValue(ASICameraInfo.CameraID,ASI_HARDWARE_BIN, &lVal, &bAuto);
		pProp->Set(lVal?g_Keyword_on:g_Keyword_off);
	}
	return DEVICE_OK;
}
///////////////////////////////////////////////////////////////////////////////
// Private CMyASICam methods
///////////////////////////////////////////////////////////////////////////////

void CMyASICam::MallocControlCaps(int iCamindex)
{
	if(pControlCaps == NULL)
	{
		pControlCaps = new ASI_CONTROL_CAPS[iCtrlNum];
		for(int i = 0; i<iCtrlNum; i++) 
		{
			ASIGetControlCaps(iCamindex, i, &pControlCaps[i]);
		}
	}
}
void CMyASICam::DeletepControlCaps(int iCamindex)
{
	if(iCamindex < 0)
		return ;
	if(pControlCaps)
	{		
		delete[] pControlCaps;
		pControlCaps = NULL;
	}
}

bool CMyASICam::isImgTypeSupported(ASI_IMG_TYPE ImgType)
{
	int i = 0;
	while (ASICameraInfo.SupportedVideoFormat[i] != ASI_IMG_END)
	{
		if(ASICameraInfo.SupportedVideoFormat[i] == ImgType)
			return true;
		i++;
	}
	return false;
}


ASI_CONTROL_CAPS* CMyASICam::GetOneCtrlCap(int CtrlID)
{
	if(pControlCaps == 0)
		return 0;
	for(int i = 0; i < iCtrlNum; i++)
	{
		if(pControlCaps[i].ControlType == CtrlID)
			return &pControlCaps[i];
	}
	return 0;
}
void CMyASICam::RefreshImgType()
{
	if(ImgType == ASI_IMG_RAW16)
	{
		iPixBytes = 2;
		iComponents = 1;
	}
	else if(ImgType == ASI_IMG_RGB24)
	{
		if(bRGB48)
		{
			iPixBytes = 8;
			iComponents = 4;
		}
		else
		{
			iPixBytes = 4;
			iComponents = 4;
		}
	
	}
	else
	{
		iPixBytes = 1;
		iComponents = 1;
	}
}


///////////////////////////////////////////////////////////////////////////////
// EFW implementation
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CMyEFW::CMyEFW() : 
	initialized_(false),
	bPosWait(false)
{
	InitializeDefaultErrorMessages();

	EFWInfo.ID = -1;

	// Description property
	int ret = CreateProperty(MM::g_Keyword_Description, "ZWO EFW filter wheel Device Adapter", MM::String, true);

	assert(ret == DEVICE_OK);

	iConnectedEFWNum = EFWGetNum();


	vector<string> EFWIndexValues;
	for(int i = 0; i < iConnectedEFWNum; i++)
	{		
		EFWGetID(i, &EFWInfo.ID);
		sprintf(ConnectedEFWName[i], "EFW (ID %d)", EFWInfo.ID);//保存名字		
		EFWIndexValues.push_back(ConnectedEFWName[i]);
	}

	CPropertyAction *pAct = new CPropertyAction (this, &CMyEFW::OnSelectEFWIndex);//通过名字选择打开的序号
	if(iConnectedEFWNum > 0)
	{
		strcpy(sz_ModelIndex, ConnectedEFWName[0]);//默认打开第一个
		//iCamIndex = 0;
		EFWGetID(0, &EFWInfo.ID);
	}
	else
	{
		strcpy(sz_ModelIndex,"no EFW connected");
	}
	//	strcpy(sz_ModelIndex, "DropDown");
	ret = CreateProperty(g_DeviceIndex, sz_ModelIndex, MM::String, false, pAct, true); //选择摄像头序号
	SetAllowedValues(g_DeviceIndex, EFWIndexValues);
	assert(ret == DEVICE_OK);
}

CMyEFW::~CMyEFW()
{
	Shutdown();
}

void CMyEFW::GetName(char* Name) const
{
	CDeviceUtils::CopyLimitedString(Name, g_StateDeviceName);
}


int CMyEFW::Initialize()
{

	if (initialized_)
		return DEVICE_OK;
	if(EFWInfo.ID < 0)
		return DEVICE_NOT_CONNECTED;
	
	if(EFWOpen(EFWInfo.ID)!= ASI_SUCCESS)
		return DEVICE_NOT_CONNECTED;
		
	EFWGetProperty(EFWInfo.ID, &EFWInfo);
	// set property list
	// -----------------
	
	// Name
	int ret = CreateStringProperty(MM::g_Keyword_Name, g_StateDeviceName, true);
	if (DEVICE_OK != ret)
		return ret;

	// Description
/*	ret = CreateStringProperty(MM::g_Keyword_Description, "EFW driver", true);
	if (DEVICE_OK != ret)
		return ret;*/

	// create default positions and labels
	const int bufSize = 64;
	char buf[bufSize];
	for (int i=0; i<EFWInfo.slotNum; i++)
	{
		snprintf(buf, bufSize, "position-%d", i + 1);
		SetPositionLabel(i, buf);
//		AddAllowedValue(MM::g_Keyword_Closed_Position, buf);
	}

	// State
	// -----
	CPropertyAction* pAct = new CPropertyAction (this, &CMyEFW::OnState);
	ret = CreateProperty(MM::g_Keyword_State, "0", MM::Integer, false, pAct);
	assert(ret == DEVICE_OK);	
	SetPropertyLimits(MM::g_Keyword_State, 0, EFWInfo.slotNum - 1);
	if (ret != DEVICE_OK)
		return ret;

	// Label
	// -----
	pAct = new CPropertyAction (this, &CStateBase::OnLabel);
	ret = CreateStringProperty(MM::g_Keyword_Label, "", false, pAct);
	if (ret != DEVICE_OK)
		return ret;

	ret = UpdateStatus();
	if (ret != DEVICE_OK)
		return ret;

	initialized_ = true;

	return DEVICE_OK;
}

bool CMyEFW::Busy()//返回true时不刷新label和state
{
	if(bPosWait)//
	{
		MM::MMTime interval = GetCurrentMMTime() - changedTime_;
		if (interval < MM::MMTime::fromMs(500))
			return true;
	}
	int pos;
	EFW_ERROR_CODE err = EFWGetPosition(EFWInfo.ID, &pos);
	if(err != EFW_SUCCESS)
		return false;
	if (pos == -1)
	{
		//Sleep(500);
		changedTime_ = GetCurrentMMTime();
		bPosWait = true;
		return true;
	}
	else
	{
		bPosWait = false;
		return false;	
	}
}


int CMyEFW::Shutdown()
{
	if (initialized_)
	{
		initialized_ = false;
	}
	EFWClose(EFWInfo.ID);
	return DEVICE_OK;
}

///////////////////////////////////////////////////////////////////////////////
// Action handlers
///////////////////////////////////////////////////////////////////////////////

int CMyEFW::OnState(MM::PropertyBase* pProp, MM::ActionType eAct)//CStateDeviceBase::OnLabel 会调用这里
{
	if (eAct == MM::BeforeGet)//值给控件显示
	{
		int pos;
		EFWGetPosition(EFWInfo.ID, &pos);	
		if(pos == -1)
			pProp->Set(lLastPos);
		else
		{
			lLastPos = pos;
			pProp->Set(lLastPos);
		}
		// nothing to do, let the caller to use cached property
	}
	else if (eAct == MM::AfterSet)//从控件得到选定的值->变量
	{
		// Set timer for the Busy signal
//		changedTime_ = GetCurrentMMTime();

		long pos;
		pProp->Get(pos);
		if (pos >= EFWInfo.slotNum || pos < 0)
		{
		//	pProp->Set(position_); // revert
			return DEVICE_INVALID_PROPERTY_VALUE;
		}
		else		
			EFWSetPosition(EFWInfo.ID, pos);	
	}

	return DEVICE_OK;
}


/**
* Handles "EFWIndex" property.
*/
int CMyEFW::OnSelectEFWIndex(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	string str;
	if (eAct == MM::AfterSet)//从控件得到选定的值
	{
		pProp->Get(str);
		for(int i = 0; i < iConnectedEFWNum; i++)
		{
			if(!str.compare(ConnectedEFWName[i]))
			{			
				EFWGetID(i, &EFWInfo.ID);

				strcpy(sz_ModelIndex, ConnectedEFWName[i]);
				break;
			}
		}
	}
	else if (eAct == MM::BeforeGet)//值给控件显示
	{
		pProp->Set(sz_ModelIndex);
	}

	return DEVICE_OK;
}
unsigned long CMyEFW::GetNumberOfPositions() const
{ 
	return EFWInfo.slotNum;
}

