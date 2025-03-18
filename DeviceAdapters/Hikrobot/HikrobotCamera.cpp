
#include <sstream>
#include <math.h>
#include "ModuleInterface.h"
#include "DeviceUtils.h"
#include <vector>
#include "HikrobotCamera.h"
#include "DeviceBase.h"
#include <process.h>
#include <algorithm>   //std::sort
using namespace std;

const char* g_HikrobotCameraDeviceName = "HikrobotCamera";

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



#define MONO_COMPONENTS  1			//mono8 占用1个组件，每个通道是1个组件
#define MONO_CONVERTED_DEPTH 8		//  mono8占用8字节
#define MONO_IMAGE_BYTES_PERPIXEL  1	// 使用mono8的字节数
#define MONO_CONVERTED_FORMAT	 PixelType_Gvsp_Mono8


#define COLOR_COMPONENTS  4				//RGBA 占用4个组件，每个通道是1个组件
#define COLOR_CONVERTED_DEPTH 32		//RGBA 的字节数量
#define COLOR_IMAGE_BYTES_PERPIXEL  4	// 使用RGBA8的字节数
#define COLOR_CONVERTED_FORMAT  PixelType_Gvsp_RGBA8_Packed			//32位， 可能底层 CircularBuffer::InsertMultiChannel 限制 或者其他原因，导致 转换成RGB32异常，待分析


#if 0
/* 下面配置 pok  */
#define COLOR_COMPONENTS  3				//RGB 每个通道是1个组件
#define COLOR_CONVERTED_DEPTH 24
#define COLOR_IMAGE_BYTES_PERPIXEL  3 
#define COLOR_CONVERTED_FORMAT  PixelType_Gvsp_RGB8_Packed
#endif 




///////////////////////////////////////////////////////////////////////////////
// Exported MMDevice API
///////////////////////////////////////////////////////////////////////////////

MODULE_API void InitializeModuleData()
{
	RegisterDevice(g_HikrobotCameraDeviceName, MM::CameraDevice, "Hikrobot  Camera");
}

MODULE_API MM::Device* CreateDevice(const char* deviceName)
{
	if (deviceName == 0)
		return 0;

	// decide which device class to create based on the deviceName parameter
	if (strcmp(deviceName, g_HikrobotCameraDeviceName) == 0) {
		// create camera
		return new HikrobotCamera();
	}
	// ...supplied name not recognized
	return 0;
}

MODULE_API void DeleteDevice(MM::Device* pDevice)
{
	delete pDevice;
}

///////////////////////////////////////////////////////////////////////////////
// HikrobotCamera implementation
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/**
* Constructor.
*/
HikrobotCamera::HikrobotCamera() :
	CLegacyCameraBase<HikrobotCamera>(),
	maxWidth_(0),
	maxHeight_(0),
	exposure_us_(0),
	exposureMax_(0),
	exposureMin_(0),
	gainMax_(0),
	gainMin_(0),
	m_nbitDepth(MONO_CONVERTED_DEPTH),
	temperatureState_("Undefined"),
	reverseX_("0"),
	reverseY_("0"),
	imgBuffer_(NULL),
	pixelType_("Undefined"),
	sensorReadoutMode_("Undefined"),
	shutterMode_("None"),
	m_bInitialized(false),
	m_pCamera(new CMvCamera()),
	m_bGrabbing(false),
	m_hImageRecvThreadHandle(NULL),
	m_bRecvRuning(false),
	m_pConvertData(NULL),
	m_nConvertDataLen(0),
	m_nComponents(MONO_COMPONENTS)
{
	// call the base class method to set-up default error codes/messages
	InitializeDefaultErrorMessages();
	SetErrorText(ERR_SERIAL_NUMBER_REQUIRED, "Serial number is required");
	SetErrorText(ERR_SERIAL_NUMBER_NOT_FOUND, "No camera with the given serial number was found");
	SetErrorText(ERR_CANNOT_CONNECT, "Cannot connect to camera; it may be in use");

	CreateStringProperty("SerialNumber", "Undefined", false, 0, true);

	EnumDevice();
}

HikrobotCamera::~HikrobotCamera()
{

	m_bRecvRuning = false;
	if (m_hImageRecvThreadHandle)
	{
		WaitForSingleObject(m_hImageRecvThreadHandle, INFINITE);
		CloseHandle(m_hImageRecvThreadHandle);
		m_hImageRecvThreadHandle = NULL;
	}

	if (NULL !=  m_pCamera)
	{
		delete m_pCamera;
		m_pCamera = NULL;
	}

	if (NULL !=  imgBuffer_)
	{
		free(imgBuffer_);
		imgBuffer_ = NULL;
	}

	if (NULL != m_pConvertData)
	{
		free(m_pConvertData);
		m_pConvertData = NULL;
	}

}

int HikrobotCamera::EnumDevice()
{
	memset(&m_stDevList, 0, sizeof(MV_CC_DEVICE_INFO_LIST));
	CMvCamera::EnumDevices(MV_GIGE_DEVICE|MV_USB_DEVICE, &m_stDevList);
	bool bFirst = true;
	if (0 == m_stDevList.nDeviceNum)
	{
		MvWriteLog(__FILE__, __LINE__, m_chDevID, "No camera present.");
	}

	for (unsigned int i = 0; i < m_stDevList.nDeviceNum; i++)
	{
		MV_CC_DEVICE_INFO* pDeviceInfo = m_stDevList.pDeviceInfo[i];
		if (NULL == pDeviceInfo)
		{
			continue;
		}

		string chSerialNumber;

		if (MV_GIGE_DEVICE == pDeviceInfo->nTLayerType)
		{
			chSerialNumber = (char*)m_stDevList.pDeviceInfo[i]->SpecialInfo.stGigEInfo.chSerialNumber;
		}
		else if (MV_USB_DEVICE == pDeviceInfo->nTLayerType)
		{
			chSerialNumber = (char*)m_stDevList.pDeviceInfo[i]->SpecialInfo.stUsb3VInfo.chSerialNumber;
		}
		else
		{
			continue;  // not support.
		}

		AddAllowedValue("SerialNumber", chSerialNumber.c_str());
		if (bFirst)
		{
			SetProperty("SerialNumber", chSerialNumber.c_str());
			bFirst = false;
		}
	}

	return DEVICE_OK;
}


/**
* Obtains device name.
*/
void HikrobotCamera::GetName(char* name) const
{
	CDeviceUtils::CopyLimitedString(name, g_HikrobotCameraDeviceName);		//增加说明： 此处是插件的名字，具体相机序列号是需要 （GetProperty("SerialNumber", serialNumber);） 中选择的。 （basler也是同样的逻辑）
}

/**
* Initializes the hardware.
*/
int HikrobotCamera::Initialize()
{
	if (m_bInitialized)
	{
		MvWriteLog(__FILE__, __LINE__, m_chDevID, "Initialize has already m_bInitialized");
		return DEVICE_OK;
	}

	char serialNumber[MM::MaxStrLength] = {0};
	GetProperty("SerialNumber", serialNumber);
	if (strlen(serialNumber) == 0 || strcmp(serialNumber, "Undefined") == 0)
		return ERR_SERIAL_NUMBER_REQUIRED;

	int nIdx = 0;

	string chCurrentSerialNumber;

	for (unsigned int i = 0; i < m_stDevList.nDeviceNum; i++)
	{
		chCurrentSerialNumber = "";


		if (MV_GIGE_DEVICE == m_stDevList.pDeviceInfo[i]->nTLayerType)
		{
			chCurrentSerialNumber = (char*)m_stDevList.pDeviceInfo[i]->SpecialInfo.stGigEInfo.chSerialNumber;
		}
		else
		{
			chCurrentSerialNumber = (char*)m_stDevList.pDeviceInfo[i]->SpecialInfo.stUsb3VInfo.chSerialNumber;
		}

		if (strcmp(chCurrentSerialNumber.c_str(), serialNumber) == 0)
		{
			nIdx = i;
			break;
		}
	}
	
	if (!m_pCamera->IsDeviceAccessible(m_stDevList.pDeviceInfo[nIdx], MV_ACCESS_Exclusive))
	{
		MvWriteLog(__FILE__, __LINE__, m_chDevID, "Device is not Accessible");
		return DEVICE_ERR;
	}

	stringstream msg;
	string strUsrDefName;
	string  strManufactureName;
	string  strModeName;
	string  strSerialNumber;

	if (MV_GIGE_DEVICE == m_stDevList.pDeviceInfo[nIdx]->nTLayerType)
	{
		strUsrDefName = (char*)m_stDevList.pDeviceInfo[nIdx]->SpecialInfo.stGigEInfo.chUserDefinedName;
		strManufactureName = (char*)m_stDevList.pDeviceInfo[nIdx]->SpecialInfo.stGigEInfo.chManufacturerName;
		strModeName = (char*)m_stDevList.pDeviceInfo[nIdx]->SpecialInfo.stGigEInfo.chModelName;
		strSerialNumber = (char*)m_stDevList.pDeviceInfo[nIdx]->SpecialInfo.stGigEInfo.chSerialNumber;
	}
	else
	{
		strUsrDefName = (char*)m_stDevList.pDeviceInfo[nIdx]->SpecialInfo.stUsb3VInfo.chUserDefinedName;
		strManufactureName = (char*)m_stDevList.pDeviceInfo[nIdx]->SpecialInfo.stUsb3VInfo.chManufacturerName;
		strModeName = (char*)m_stDevList.pDeviceInfo[nIdx]->SpecialInfo.stUsb3VInfo.chModelName;
		strSerialNumber = (char*)m_stDevList.pDeviceInfo[nIdx]->SpecialInfo.stUsb3VInfo.chSerialNumber;
	}


	MvWriteLog(__FILE__, __LINE__, m_chDevID,"Begin Connect Camera UsrName[%s] ManufactureName[%s] strModeName[%s] strSerialNumber[%s]", strUsrDefName, strManufactureName, strModeName, strSerialNumber);

	// Name
	int ret = CreateProperty(MM::g_Keyword_Name, g_HikrobotCameraDeviceName, MM::String, true);
	if (DEVICE_OK != ret)
	{
		return ret;
	}


	// Description
	ret = CreateProperty(MM::g_Keyword_Description, "Hikrobot Camera device adapter", MM::String, true);
	if (DEVICE_OK != ret)
	{
		return ret;
	}

	// Serial Number
	ret = CreateProperty(MM::g_Keyword_CameraID, strSerialNumber.c_str(), MM::String, true);
	if (DEVICE_OK != ret)
	{
		return ret;
	}
	
	assert(nIdx < m_stDevList.nDeviceNum);
	int nRet = m_pCamera->Open(m_stDevList.pDeviceInfo[nIdx]);
	if (MV_OK != nRet)
	{
		return DEVICE_ERR;
	}

	string strBasiceLog = " [ "+ strModeName + " " + strSerialNumber + " ] ";
	SetLogBasicInfo(strBasiceLog);

	sprintf_s(m_chDevID, sizeof(m_chDevID), " %s(%s) ", strModeName.c_str(), strSerialNumber.c_str());	//保存相机的序列号，型号信息


	//Sensor size
	MVCC_INTVALUE_EX WidthMax = { 0 };
	m_pCamera->GetIntValue("WidthMax", &WidthMax);
	maxWidth_ = WidthMax.nCurValue;
	
	MVCC_INTVALUE_EX stParam = { 0 };
	m_pCamera->GetIntValue("Width", &stParam);

	if (IsAvailable("Width"))
	{
		CPropertyAction* pAct = new CPropertyAction(this, &HikrobotCamera::OnWidth);
		ret = CreateProperty("SensorWidth", CDeviceUtils::ConvertToString((int)stParam.nCurValue), MM::Integer, false, pAct);
		SetPropertyLimits("SensorWidth", (double)stParam.nMin, (double)stParam.nMax);
		assert(ret == DEVICE_OK);
	}


	MVCC_INTVALUE_EX HeightMax = {0};
	m_pCamera->GetIntValue("HeightMax", &HeightMax);
	maxHeight_ = HeightMax.nCurValue;

	memset(&stParam, 0, sizeof(MVCC_INTVALUE_EX));
	m_pCamera->GetIntValue("Height", &stParam);
	if (IsAvailable("Height"))
	{
		CPropertyAction* pAct = new CPropertyAction(this, &HikrobotCamera::OnHeight);
		ret = CreateProperty("SensorHeight", CDeviceUtils::ConvertToString((int)stParam.nCurValue), MM::Integer, false, pAct);
		SetPropertyLimits("SensorHeight", (double)stParam.nMin, (double)stParam.nMax);
		assert(ret == DEVICE_OK);
	}
	//end of Sensor size

	MVCC_FLOATVALUE stFloatValue = { 0.0 };
	m_pCamera->GetFloatValue("ExposureTime", &stFloatValue);
	exposureMin_ = stFloatValue.fMin;
	exposureMax_ = stFloatValue.fMax;
	exposure_us_ = stFloatValue.fCurValue;

	//Pixel type
	CPropertyAction* pAct = new CPropertyAction(this, &HikrobotCamera::OnPixelType);
	ret = CreateProperty(MM::g_Keyword_PixelType, "NA", MM::String, false, pAct);
	assert(ret == DEVICE_OK);

	vector<string> pixelTypeValues;

	MVCC_ENUMVALUE stEnumValue = { 0 };
	MVCC_ENUMENTRY stPixelFormatInfo = { 0 };


	m_pCamera->GetEnumValue("PixelFormat", &stEnumValue);
	for (int i = stEnumValue.nSupportedNum - 1; i >= 0; i--)
		//for (int i = 0; i < stEnumValue.nSupportedNum; i++)
	{
		stPixelFormatInfo.nValue = stEnumValue.nSupportValue[i];
		m_pCamera->GetEnumEntrySymbolic("PixelFormat", &stPixelFormatInfo);
		string strPixelFormatInfo = (char*)stPixelFormatInfo.chSymbolic;
		pixelTypeValues.push_back(strPixelFormatInfo);

		if (stEnumValue.nCurValue == stEnumValue.nSupportValue[i])
		{
			pixelType_ = strPixelFormatInfo;
			MvWriteLog(__FILE__, __LINE__, m_chDevID, "Camera Default pixelType_ %s", strPixelFormatInfo.c_str());
		}

		MvWriteLog(__FILE__, __LINE__, m_chDevID, "GetEnumEntrySymbolic and pixelType_ %s  nCurValue[%d] nSupportValue [%d]", strPixelFormatInfo.c_str(), stEnumValue.nCurValue, stEnumValue.nSupportValue[i]);
	}


	SetAllowedValues(MM::g_Keyword_PixelType, pixelTypeValues);

	/////TestPattern//////
	MVCC_ENUMVALUE stTestPattern = { 0 };
	m_pCamera->GetEnumValue("TestPattern", &stTestPattern);
	if (IsWritable("TestPattern"))
	{
		if (IsAvailable("TestPattern"))
		{
			pAct = new CPropertyAction(this, &HikrobotCamera::OnTestPattern);
			ret = CreateProperty("TestPattern", "NA", MM::String, false, pAct);
			vector<string> TestPatternVals;
			TestPatternVals.push_back("Off");		//参考basler，先把off放入vector； 在循环中放入off，应该也OK； 
			MVCC_ENUMENTRY Entry = { 0 };

			for (unsigned int i = 0; i < stTestPattern.nSupportedNum; i++)
			{
				Entry.nValue = stTestPattern.nSupportValue[i];
				m_pCamera->GetEnumEntrySymbolic("TestPattern", &Entry);
				string strValue = Entry.chSymbolic;
				if (IsAvailable(strValue.c_str()) && strValue != "Off")
				{
					TestPatternVals.push_back(strValue);
				}
			}
			SetAllowedValues("TestPattern", TestPatternVals);
		}
	}


	/////AutoGain//////
	MVCC_ENUMVALUE gainAuto = {0};
	m_pCamera->GetEnumValue("GainAuto", &gainAuto);
	if (IsWritable("GainAuto"))
	{

		if (/*gainAuto != NULL && */IsAvailable("GainAuto"))
		{
			pAct = new CPropertyAction(this, &HikrobotCamera::OnAutoGain);
			ret = CreateProperty("GainAuto", "NA", MM::String, false, pAct);
			vector<string> LSPVals;
			LSPVals.push_back("Off");
			//gainAuto->GetEntries(entries);
			MVCC_ENUMENTRY gainEntry = { 0 };

			for (unsigned int i = 0; i < gainAuto.nSupportedNum; i++)
			{
				gainEntry.nValue = gainAuto.nSupportValue[i];
				m_pCamera->GetEnumEntrySymbolic("GainAuto", &gainEntry);
				string strValue = gainEntry.chSymbolic;
				if (IsAvailable(strValue.c_str()) && strValue != "Off")
				{
					LSPVals.push_back(strValue);
				}
			}
			SetAllowedValues("GainAuto", LSPVals);
		}
	}


	/////AutoExposure//////
	MVCC_ENUMVALUE ExposureAuto = { 0 };
	m_pCamera->GetEnumValue("ExposureAuto", &ExposureAuto);
	if (IsWritable("ExposureAuto"))
	{

		if (/*ExposureAuto != NULL && */IsAvailable("ExposureAuto"))
		{
			pAct = new CPropertyAction(this, &HikrobotCamera::OnAutoExpore);
			ret = CreateProperty("ExposureAuto", "NA", MM::String, false, pAct);
			vector<string> LSPVals;
			LSPVals.push_back("Off");

			MVCC_ENUMENTRY entry;
			
			for (unsigned int i = 0;i < ExposureAuto.nSupportedNum; i++)
			{
				entry.nValue = ExposureAuto.nSupportValue[i];
				m_pCamera->GetEnumEntrySymbolic("ExposureAuto", &entry);
				string strValue = entry.chSymbolic;
				if (IsAvailable(strValue.c_str()) && strValue != "Off")
				{
					LSPVals.push_back(strValue);
				}
			}
			SetAllowedValues("ExposureAuto", LSPVals);
		}
	}

	//get gain limits and value
	if (IsAvailable("Gain"))
	{
		MVCC_FLOATVALUE gain;
		m_pCamera->GetFloatValue("Gain", &gain);

		gainMax_ = gain.fMax;
		gainMin_ = gain.fMin;
		gain_ = gain.fCurValue;
	}
	else if (IsAvailable("GainRaw"))
	{
		MVCC_FLOATVALUE GainRaw;
		m_pCamera->GetFloatValue("GainRaw", &GainRaw);
		gainMax_ = (double)GainRaw.fMax;
		gainMin_ = (double)GainRaw.fMin;
		gain_ = (double)GainRaw.fCurValue;
	}

	//make property
	pAct = new CPropertyAction(this, &HikrobotCamera::OnGain);
	ret = CreateProperty(MM::g_Keyword_Gain, "1.0", MM::Float, false, pAct);
	SetPropertyLimits(MM::g_Keyword_Gain, gainMin_, gainMax_);

	/////Offset//////
	MVCC_FLOATVALUE BlackLevel = {0};
	m_pCamera->GetFloatValue("BlackLevel", &BlackLevel);
	MVCC_FLOATVALUE BlackLevelRaw = { 0 };
	m_pCamera->GetFloatValue("BlackLevelRaw", &BlackLevelRaw);
	

	if (IsAvailable("BlackLevel"))
	{
		offsetMax_ = BlackLevel.fMax;
		offsetMin_ = BlackLevel.fMin;
		offset_ = BlackLevel.fCurValue;

	}
	else if (IsAvailable("BlackLevelRaw"))
	{
		offsetMax_ = (double)BlackLevelRaw.fMax;
		offsetMin_ = (double)BlackLevelRaw.fMin;
		offset_ = (double)BlackLevelRaw.fCurValue;
	}

	//make property
	pAct = new CPropertyAction(this, &HikrobotCamera::OnOffset);
	ret = CreateProperty(MM::g_Keyword_Offset, "1.0", MM::Float, false, pAct);
	SetPropertyLimits(MM::g_Keyword_Offset, offsetMin_, offsetMax_);


	////Sensor readout//////
	if (IsAvailable("SensorReadoutMode"))
	{
		pAct = new CPropertyAction(this, &HikrobotCamera::OnSensorReadoutMode);
		ret = CreateProperty("SensorReadoutMode", "NA", MM::String, false, pAct);  
		vector<string> vals;
		// vals.push_back("Off");	// 海康相机无这个节点，不会走到这个分支中; 参考basler分支，basler无添加off节点
		MVCC_ENUMVALUE SensorReadoutMode = { 0 };
		m_pCamera->GetEnumValue("SensorReadoutMode", &SensorReadoutMode);
		
		for (unsigned int i = 0; i < SensorReadoutMode.nSupportedNum; i++)
		{
			MVCC_ENUMENTRY entry;
			entry.nValue = SensorReadoutMode.nSupportValue[i];
			m_pCamera->GetEnumEntrySymbolic("SensorReadoutMode", &entry);
			string strValue = entry.chSymbolic;
			if (IsAvailable(strValue.c_str()) && strValue != "Off")
			{
				vals.push_back(strValue);
			}
		}
		SetAllowedValues("SensorReadoutMode", vals);
	}

	MVCC_ENUMVALUE LightSourcePreset = { 0 };
	m_pCamera->GetEnumValue("LightSourcePreset", &LightSourcePreset);
	if (/*LightSourcePreset != NULL && */IsAvailable("LightSourcePreset"))
	{
		pAct = new CPropertyAction(this, &HikrobotCamera::OnLightSourcePreset);
		ret = CreateProperty("LightSourcePreset", "NA", MM::String, false, pAct);
		vector<string> LSPVals;
		LSPVals.push_back("Off");
		MVCC_ENUMENTRY entry;
		for (unsigned int i = 0;i <  LightSourcePreset.nSupportedNum; i++)
		{
			entry.nValue = LightSourcePreset.nSupportValue[i];
			m_pCamera->GetEnumEntrySymbolic("LightSourcePreset", &entry);
			string strValue = entry.chSymbolic;
			if (IsAvailable(strValue.c_str()) && strValue != "Off")
			{
				LSPVals.push_back(strValue);
			}
		}
		SetAllowedValues("LightSourcePreset", LSPVals);
	}


	/////Trigger Mode//////
	MVCC_ENUMVALUE TriggerMode = {0};
	m_pCamera->GetEnumValue("TriggerMode", &TriggerMode);
	if (IsAvailable("TriggerMode"))
	{
		pAct = new CPropertyAction(this, &HikrobotCamera::OnTriggerMode);
		ret = CreateProperty("TriggerMode", "Off", MM::String, false, pAct);
		vector<string> LSPVals;
		LSPVals.push_back("Off");
		LSPVals.push_back("On");
		SetAllowedValues("TriggerMode", LSPVals);
	}

	/////Trigger Source//////
	MVCC_ENUMVALUE triggersource = {0};
	m_pCamera->GetEnumValue("TriggerSource", &triggersource);
	MVCC_ENUMENTRY triggersourceEntry = { 0 };

	if (IsWritable("TriggerSource"))
	{
		if (IsAvailable("TriggerSource"))
		{
			pAct = new CPropertyAction(this, &HikrobotCamera::OnTriggerSource);
			ret = CreateProperty("TriggerSource", "NA", MM::String, false, pAct);
			vector<string> LSPVals;
			for (unsigned int i = 0; i < triggersource.nSupportedNum; i++)
			{
				triggersourceEntry.nValue = triggersource.nSupportValue[i];
				m_pCamera->GetEnumEntrySymbolic("TriggerSource", &triggersourceEntry);
				string strEntry = triggersourceEntry.chSymbolic;

				if (IsAvailable(strEntry.c_str()) /*&& strEntry.find("Software") == std::string::npos*/ )
				{
					MvWriteLog(__FILE__, __LINE__, m_chDevID, "Init find OnTriggerSource: %s ", strEntry.c_str());

					LSPVals.push_back(strEntry);
				}
			}
			SetAllowedValues("TriggerSource", LSPVals);
		}
	}

	////Shutter mode//////	
	MVCC_ENUMVALUE shutterMode = {0};
	m_pCamera->GetEnumValue("ShutterMode",&shutterMode);
	if (IsAvailable("ShutterMode"))
	{
		pAct = new CPropertyAction(this, &HikrobotCamera::OnShutterMode);
		ret = CreateProperty("ShutterMode", "NA", MM::String, false, pAct);
		vector<string> shutterVals;

		MVCC_ENUMENTRY entry = { 0 };
		for (unsigned int i = 0; i < shutterMode.nSupportedNum; i++)
		{
			entry.nValue = shutterMode.nSupportValue[i];
			m_pCamera->GetEnumEntrySymbolic("ShutterMode", &entry);
		}

		std::string strValue = entry.chSymbolic;
		if ( strValue.compare("Global") && IsAvailable("Global"))
		{
			shutterVals.push_back("Global");
		}
		if (strValue.compare("Rolling") && IsAvailable("Rolling"))
		{
			shutterVals.push_back("Rolling");
		}
		if (strValue.compare("GlobalResetRelease") && IsAvailable("GlobalResetRelease"))
		{
			shutterVals.push_back("GlobalResetRelease");
		}
		SetAllowedValues("ShutterMode", shutterVals);
	}

	/////Reverse X//////
	if (IsAvailable("ReverseX"))
	{
		pAct = new CPropertyAction(this, &HikrobotCamera::OnReverseX);
		ret = CreateProperty("ReverseX", "0", MM::String, false, pAct);
		vector<string> reverseXVals;
		reverseXVals.push_back("0");
		reverseXVals.push_back("1");
		SetAllowedValues("ReverseX", reverseXVals);
	}

	/////Reverse Y//////
	if (IsAvailable("ReverseY"))
	{
		pAct = new CPropertyAction(this, &HikrobotCamera::OnReverseY);
		ret = CreateProperty("ReverseY", "0", MM::String, false, pAct);
		vector<string> reverseYVals;
		reverseYVals.push_back("0");
		reverseYVals.push_back("1");
		SetAllowedValues("ReverseY", reverseYVals);
	}

	//////ResultingFramerate
	if (IsAvailable("ResultingFrameRateAbs"))
	{
	
		MVCC_INTVALUE_EX ResultingFrameRatePrevious = {0};
		m_pCamera->GetIntValue("ResultingFrameRateAbs", &ResultingFrameRatePrevious);

		std::ostringstream oss;
		oss << ResultingFrameRatePrevious.nCurValue;
		pAct = new CPropertyAction(this, &HikrobotCamera::OnResultingFramerate);
		ret = CreateProperty("ResultingFrameRateAbs", oss.str().c_str(), MM::String, true, pAct);
		if (DEVICE_OK != ret)
		{
			return ret;
		}

	}

	//////ResultingFramerate
	if (IsAvailable("ResultingFrameRate"))
	{

		MVCC_INTVALUE_EX ResultingFrameRatePrevious = { 0 };
		m_pCamera->GetIntValue("ResultingFrameRate", &ResultingFrameRatePrevious);

		std::ostringstream oss;
		oss << ResultingFrameRatePrevious.nCurValue;
		pAct = new CPropertyAction(this, &HikrobotCamera::OnResultingFramerate);
		ret = CreateProperty("ResultingFrameRate", oss.str().c_str(), MM::String, true, pAct);
		if (DEVICE_OK != ret)
		{
			return ret;
		}

	}

	/////Set Acquisition AcquisitionFrameRateEnable//////
	if (IsAvailable("AcquisitionFrameRateEnable"))
	{
		pAct = new CPropertyAction(this, &HikrobotCamera::OnAcqFramerateEnable);
		ret = CreateProperty("AcquisitionFramerateEnable", "0", MM::String, false, pAct);
		vector<string> setAcqFrmVals;
		setAcqFrmVals.push_back("0");
		setAcqFrmVals.push_back("1");
		SetAllowedValues("AcquisitionFramerateEnable", setAcqFrmVals);
	}

	/////Acquisition Frame rate//////
	{
		if (IsAvailable("AcquisitionFrameRate"))
		{
			MVCC_INTVALUE_EX AcquisitionFrameRate = { 0 };
			m_pCamera->GetIntValue("AcquisitionFrameRate", &AcquisitionFrameRate); 
			// it is not necessary to use full range to 
			acqFramerateMax_ = AcquisitionFrameRate.nMax;
			acqFramerateMin_ = AcquisitionFrameRate.nMin;
			acqFramerate_ = AcquisitionFrameRate.nCurValue;

		}
		else if (IsAvailable("AcquisitionFrameRateAbs"))
		{
			MVCC_INTVALUE_EX AcquisitionFrameRateAbs = { 0 };
			m_pCamera->GetIntValue("AcquisitionFrameRateAbs", &AcquisitionFrameRateAbs);
			acqFramerateMax_ = AcquisitionFrameRateAbs.nMax;
			acqFramerateMin_ = AcquisitionFrameRateAbs.nMin;
			acqFramerate_ = AcquisitionFrameRateAbs.nCurValue;

		}
		pAct = new CPropertyAction(this, &HikrobotCamera::OnAcqFramerate);
		ret = CreateProperty("AcquisitionFramerate", "100", MM::String, false, pAct);
		//SetPropertyLimits("AcquisitionFramerate", acqFramerateMin_, acqFramerateMax_);
		assert(ret == DEVICE_OK);
	}

	//// binning
	pAct = new CPropertyAction(this, &HikrobotCamera::OnBinning);
	ret = CreateProperty(MM::g_Keyword_Binning, "1", MM::Integer, false, pAct);
	SetPropertyLimits(MM::g_Keyword_Binning, 1, 1);
	assert(ret == DEVICE_OK);
	vector<string> binValues;
	MVCC_ENUMVALUE BinningHorizontal = { 0 };
	m_pCamera->GetEnumValue("BinningHorizontal", &BinningHorizontal);

	MVCC_ENUMVALUE BinningVertical = { 0 };
	m_pCamera->GetEnumValue("BinningVertical", &BinningVertical);

	if (IsAvailable("BinningHorizontal") && IsAvailable("BinningVertical"))
	{

		//assumed that BinningHorizontal and BinningVertical allow same steps
		int64_t min = 0;// = BinningHorizontal->GetMin();
		int64_t max = 0;// = BinningHorizontal->GetMax();
		const int num = BinningHorizontal.nSupportedNum;
		std::vector<unsigned int> vec;
		for (unsigned int i = 0; i < BinningHorizontal.nSupportedNum; i++)
		{
			vec.push_back(BinningHorizontal.nSupportValue[i]);
		}
		std::sort(vec.begin(), vec.end());
		min = vec[0];
		max = vec[BinningHorizontal.nSupportedNum - 1];

		MvWriteLog(__FILE__, __LINE__, m_chDevID, "binning range: %lld - %lld", min, max);

		SetPropertyLimits(MM::g_Keyword_Binning, (double)min, (double)max);

		for (int x = 1; x <= max; x++)
		{
			std::ostringstream oss;
			oss << x;
			binValues.push_back(oss.str());
			AddAllowedValue(MM::g_Keyword_Binning, oss.str().c_str());
		}
		binningFactor_.assign(CDeviceUtils::ConvertToString((long)BinningHorizontal.nCurValue));
		CheckForBinningMode(pAct);
	}
	else
	{
		binValues.push_back("1");
		binningFactor_.assign("1");
	}

	if (m_pCamera->EnumerateTls() & MV_GIGE_DEVICE)
	{
		MVCC_INTVALUE_EX GevSCPD = { 0 };
		m_pCamera->GetIntValue("GevSCPD", &GevSCPD);
		if (IsAvailable("GevSCPD"))
		{
			pAct = new CPropertyAction(this, &HikrobotCamera::OnInterPacketDelay);
			ret = CreateProperty("InterPacketDelay", CDeviceUtils::ConvertToString((long)GevSCPD.nCurValue), MM::Integer, false, pAct);
			SetPropertyLimits("InterPacketDelay", (double)GevSCPD.nMin, (double)GevSCPD.nMax);
			assert(ret == DEVICE_OK);
		}
	}

	// synchronize all properties
	// --------------------------
	ret = UpdateStatus();
	if (DEVICE_OK != ret)
	{
		return ret;
	}

	//preparation for snaps
	ResizeSnapBuffer();
	m_bInitialized = true;

	MvWriteLog(__FILE__, __LINE__, m_chDevID, "HikrobotCamera::Initialize");

	return DEVICE_OK;
}


int HikrobotCamera::CheckForBinningMode(CPropertyAction* pAct)
{
	// Binning Mode
	MVCC_ENUMVALUE BinningModeHorizontal = { 0 };
	m_pCamera->GetEnumValue("BinningModeHorizontal", &BinningModeHorizontal);

	MVCC_ENUMVALUE BinningModeVertical = { 0 };
	m_pCamera->GetEnumValue("BinningModeVertical", &BinningModeVertical);

	if (IsAvailable("BinningModeVertical") && IsAvailable("BinningModeHorizontal"))
	{
		pAct = new CPropertyAction(this, &HikrobotCamera::OnBinningMode);

		vector<string> LSPVals;
		// assumed BinningHorizontalMode & BinningVerticalMode same entries
		for (unsigned int i = 0;i < BinningModeVertical.nSupportedNum;i ++)
		{
			MVCC_ENUMENTRY EnumEntry;
			EnumEntry.nValue = BinningModeVertical.nSupportValue[i];
			if (i == 0)
			{
				CreateProperty("BinningMode", EnumEntry.chSymbolic, MM::String, false, pAct);
			}

			LSPVals.push_back(EnumEntry.chSymbolic);
		}
		SetAllowedValues("BinningMode", LSPVals);
		return DEVICE_OK;
	}
	return DEVICE_CAN_NOT_SET_PROPERTY;
}


int HikrobotCamera::SetProperty(const char* name, const char* value)
{
	int nRet = __super::SetProperty( name, value );
	return nRet;
} 

/**
* Shuts down (unloads) the device.
*/
int HikrobotCamera::Shutdown()
{
	if (!m_pCamera)
	{
		m_pCamera->Close();
		delete m_pCamera;
	}
	m_bInitialized = false;
	MvWriteLog(__FILE__, __LINE__, m_chDevID, "Shutdown set m_bInitialized false");


	return DEVICE_OK;
}

int HikrobotCamera::SnapImage()
{

	/*
	basler在snapimage中调用的是 virtual void StartGrabbing( size_t maxImages, EGrabStrategy strategy = GrabStrategy_OneByOne, EGrabLoop grabLoopType = GrabLoop_ProvidedByUser );
	这个接口的描述是：”Extends the StartGrabbing(EStrategy, EGrabLoop) by a number of images to grab. If the passed count of images has been reached, StopGrabbing is called
    automatically. The images are counted according to the grab strategy. Skipped images are not taken into account.“；  就是说获取图像个数满足后，后台会自动停止取流；
	海康SDK无此类接口，所以需要 start ，获取图像， stop 
	*/

	MvWriteLog(__FILE__, __LINE__, m_chDevID, "SnapImage Begin");

	m_pCamera->SetGrabStrategy(MV_GrabStrategy_OneByOne);
	m_pCamera->StartGrabbing();

	m_bGrabbing = true;
	int nRet = MV_E_UNKNOW;
	MV_FRAME_OUT stOutFrame = { 0 };

	do 
	{
		//此处暂时设定1s, 若相机帧率过低，则可能异常，需要调整; 
		// 超时时间不能太长，容易导致接口卡死异常;
		nRet = m_pCamera->GetImageBuffer(&stOutFrame, 1000);		
		if (nRet == MV_OK)
		{
			MvWriteLog( __FILE__, __LINE__, m_chDevID, "Get One Frame: Width[%d], Height[%d], FrameNum[%d]",
				stOutFrame.stFrameInfo.nWidth, stOutFrame.stFrameInfo.nHeight, stOutFrame.stFrameInfo.nFrameNum);
		}
		else
		{
			MvWriteLog(__FILE__, __LINE__, m_chDevID, "Get Image fail!");
			break;
		}

		ResizeSnapBuffer();	//分配内存空间
		CopyToImageBuffer(&stOutFrame);

		nRet = m_pCamera->FreeImageBuffer(&stOutFrame);
		if (nRet != MV_OK)
		{
			MvWriteLog(__FILE__, __LINE__, m_chDevID, "Free Image Buffer fail [%#x]!", nRet);
		}

		
		break;	// 获取一张图像结束.
	} while (0);

	m_pCamera->StopGrabbing();
	m_bGrabbing = false;

	MvWriteLog(__FILE__, __LINE__, m_chDevID, "SnapImage End");
	return DEVICE_OK;
}


void HikrobotCamera::CopyToImageBuffer(MV_FRAME_OUT* pstFrameOut)
{
	if (NULL == pstFrameOut || NULL == pstFrameOut->pBufAddr || NULL  == imgBuffer_)
	{
		MvWriteLog(__FILE__, __LINE__, m_chDevID, "CopyToImageBuffer param invalid.");
		return;
	}

	if (pstFrameOut->stFrameInfo.enPixelType == PixelType_Gvsp_Mono8)
	{
		// Workaround : OnPixelType call back will not be fired always.
		m_nComponents = MONO_COMPONENTS;
		m_nbitDepth = MONO_CONVERTED_DEPTH;
		SetProperty(MM::g_Keyword_PixelType, g_PixelType_8bit);

		memcpy(imgBuffer_, pstFrameOut->pBufAddr, pstFrameOut->stFrameInfo.nFrameLen);
	}
	else
	{
		int nRet = MV_OK;
		unsigned char* pConvertData = NULL;
		unsigned int nConvertDataSize = 0;
		MvGvspPixelType enDstPixelType = PixelType_Gvsp_Undefined;
		unsigned int nChannelNum = 0;


		nRet = PixTypeProc(pstFrameOut->stFrameInfo.enPixelType, nChannelNum, enDstPixelType);
		if (MV_OK != nRet)
		{
			MvWriteLog(__FILE__, __LINE__, m_chDevID, "PixTypeProc Failed,errcode [%#x]!", nRet);
			return;
		}

		int nNeedSize = pstFrameOut->stFrameInfo.nWidth * pstFrameOut->stFrameInfo.nHeight * nChannelNum;

		if (m_nConvertDataLen < nNeedSize || (NULL == m_pConvertData))
		{
			if (m_pConvertData)
			{
				free(m_pConvertData);
				m_pConvertData = NULL;
			}

			m_pConvertData = (unsigned char*)malloc(nNeedSize);
			if (NULL == m_pConvertData)
			{
				MvWriteLog(__FILE__, __LINE__, m_chDevID, "malloc pConvertData len [%d] fail!",nNeedSize);
				nRet = MV_E_RESOURCE;
				return;
			}
			m_nConvertDataLen = nNeedSize;
		}

		// ch:像素格式转换 | en:Convert pixel format 
		MV_CC_PIXEL_CONVERT_PARAM stConvertParam = { 0 };

		stConvertParam.nWidth = pstFrameOut->stFrameInfo.nWidth;                 //ch:图像宽 | en:image width
		stConvertParam.nHeight = pstFrameOut->stFrameInfo.nHeight;               //ch:图像高 | en:image height
		stConvertParam.pSrcData = pstFrameOut->pBufAddr;                         //ch:输入数据缓存 | en:input data buffer
		stConvertParam.nSrcDataLen = pstFrameOut->stFrameInfo.nFrameLen;         //ch:输入数据大小 | en:input data size
		stConvertParam.enSrcPixelType = pstFrameOut->stFrameInfo.enPixelType;    //ch:输入像素格式 | en:input pixel format
		stConvertParam.enDstPixelType = enDstPixelType;                         //ch:输出像素格式 | en:output pixel format
		stConvertParam.pDstBuffer = m_pConvertData;                               //ch:输出数据缓存 | en:output data buffer
		stConvertParam.nDstBufferSize = nNeedSize;                       //ch:输出缓存大小 | en:output buffer size
		nRet = GetCamera()->ConvertPixelType(&stConvertParam);
		if (MV_OK != nRet)
		{
			MvWriteLog(__FILE__, __LINE__, m_chDevID, "Convert Pixel Type fail, errcode [%#x]!", nRet);
		}

		memcpy(imgBuffer_, m_pConvertData, stConvertParam.nDstLen);	
		
	}
}


unsigned HikrobotCamera::PixTypeProc(MvGvspPixelType enPixelType, unsigned int & nChannelNum, MvGvspPixelType & enDstPixelType)
{
	int nRet = MV_OK;

	//如果是彩色则转成RGB8
	if (IsColor(enPixelType))
	{
		nChannelNum = COLOR_CONVERTED_DEPTH / 8;
		enDstPixelType = COLOR_CONVERTED_FORMAT;;
		SetProperty(MM::g_Keyword_PixelType, g_PixelType_8bitRGBA);

		m_nComponents = COLOR_COMPONENTS;
		m_nbitDepth = COLOR_CONVERTED_DEPTH;

	}
	//如果是黑白则转换成Mono8
	else if (IsMono(enPixelType))
	{
		nChannelNum = 1;
		enDstPixelType = PixelType_Gvsp_Mono8;
		SetProperty(MM::g_Keyword_PixelType, g_PixelType_8bit);

		m_nComponents = MONO_COMPONENTS;
		m_nbitDepth = MONO_CONVERTED_DEPTH;
	}
	else
	{
		MvWriteLog(__FILE__, __LINE__, m_chDevID, "[%d] Don't support to convert.", enPixelType);

		return MV_E_PARAMETER;
	}
	
	return nRet;

}


/**
* Returns pixel data.
*/
const unsigned char* HikrobotCamera::GetImageBuffer()
{
	return (unsigned char*)imgBuffer_;
}

unsigned HikrobotCamera::GetImageWidth() const
{

	MVCC_INTVALUE_EX stParam = { 0 };
	m_pCamera->GetIntValue("Width", &stParam);
	return stParam.nCurValue;
}

unsigned HikrobotCamera::GetImageHeight() const
{
	MVCC_INTVALUE_EX stParam = { 0 };
	m_pCamera->GetIntValue("Height", &stParam);
	return stParam.nCurValue;
}


/**
* Returns image buffer pixel depth in bytes.
*/
unsigned HikrobotCamera::GetImageBytesPerPixel() const
{
	const char* subject("Bayer");
	std::size_t found = pixelType_.find(subject);
	unsigned int ret = 0;

	//mono统一转换成mon8,其他类型转换为RGBA32
	if (pixelType_ == "Mono8" || pixelType_ == "Mono10" || pixelType_ == "Mono12" || pixelType_ == "Mono10Packed" || pixelType_ == "Mono12Packed" || pixelType_ == "Mono16")
	{
		ret = MONO_IMAGE_BYTES_PERPIXEL;

	}
	else if (pixelType_ == "BayerGB8" || pixelType_ == "BayerGB12Packed" || pixelType_ == "BayerGB12" || pixelType_ == "BayerGB8" || pixelType_ == "RGB8Packed" ||
		pixelType_ == "YUV422_8_UYVY" || pixelType_ == "YUV422_8")
	{
		ret = COLOR_IMAGE_BYTES_PERPIXEL;
	}
	else
	{
		ret = COLOR_IMAGE_BYTES_PERPIXEL;
	}
	

	MvWriteLog(__FILE__, __LINE__, (char *)m_chDevID, "pixelType_ [%s] GetImageBytesPerPixel [%d].", pixelType_.c_str(), ret);

	return ret;
}

/**
* Returns the bit depth (dynamic range) of the pixel.
*/
unsigned int HikrobotCamera::GetBitDepth() const
{
	const char* subject("Bayer");
	std::size_t found = pixelType_.find(subject);
	unsigned int ret = 0;
	//mono统一转换成mon8,其他类型转换为RGBA32
	if (pixelType_ == "Mono8" || pixelType_ == "Mono10" || pixelType_ == "Mono12" || pixelType_ == "Mono10Packed" || pixelType_ == "Mono12Packed" || pixelType_ == "Mono16")
	{
		ret = MONO_CONVERTED_DEPTH;

	}
	else if (pixelType_ == "BayerGB8" || pixelType_ == "BayerGB12Packed" || pixelType_ == "BayerGB12" || pixelType_ == "BayerGB8" || pixelType_ == "RGB8Packed" ||
		pixelType_ == "YUV422_8_UYVY" || pixelType_ == "YUV422_8")
	{
		ret = COLOR_CONVERTED_DEPTH;
	}
	else
	{
		ret = COLOR_CONVERTED_DEPTH;
	}

	
	MvWriteLog(__FILE__, __LINE__, (char*)m_chDevID, "pixelType_ [%s] GetBitDepth [%d].", pixelType_.c_str(), ret);
	return ret;
}


int HikrobotCamera::OnPixelType(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	MVCC_ENUMVALUE pixelFormat = { 0 };
	MVCC_ENUMENTRY entry = { 0 };

	if (eAct == MM::AfterSet) {

		if (m_bGrabbing)
		{
			MvWriteLog(__FILE__, __LINE__, m_chDevID, "OnPixelType  Already start, StopGrab first. ");
			m_pCamera->StopGrabbing();
		}

		pProp->Get(pixelType_);
		m_pCamera->SetEnumValueByString("PixelFormat", pixelType_.c_str());


		MvWriteLog(__FILE__, __LINE__, m_chDevID, "OnPixelType Set PixelFormat value [%s] ", pixelType_.c_str());

		MVCC_FLOATVALUE offset = { 0 };
		m_pCamera->GetFloatValue("BlackLevel", &offset);
		offsetMax_ = offset.fMax;
		offsetMin_ = offset.fMin;
		SetPropertyLimits(MM::g_Keyword_Offset, offsetMin_, offsetMax_);


		if (m_bGrabbing)
		{
			MvWriteLog(__FILE__, __LINE__, m_chDevID, "OnPixelType  Already start, Recovery StartGrabing. ");
			m_pCamera->StartGrabbing();
		}

	}
	else if (eAct == MM::BeforeGet) {

		m_pCamera->GetEnumValue("PixelFormat", &pixelFormat);
		entry.nValue = pixelFormat.nCurValue;
		m_pCamera->GetEnumEntrySymbolic("PixelFormat", &entry);

		MvWriteLog(__FILE__, __LINE__, m_chDevID, "OnPixelType Get PixelFormat value [%s]", entry.chSymbolic);

		pixelType_.assign(entry.chSymbolic);
		pProp->Set(pixelType_.c_str());
	}


	m_pCamera->GetEnumValue("PixelFormat", &pixelFormat);
	entry.nValue = pixelFormat.nCurValue;
	m_pCamera->GetEnumEntrySymbolic("PixelFormat", &entry);
	std::string strPixelFormatInfo(entry.chSymbolic);
	const char* subject("Bayer");
	std::size_t found = strPixelFormatInfo.find(subject);

	if (strPixelFormatInfo.compare("Mono8") == 0 
		|| strPixelFormatInfo.compare("Mono10") == 0 || strPixelFormatInfo.compare("Mono10Packed") == 0
		|| strPixelFormatInfo.compare("Mono12") == 0 || strPixelFormatInfo.compare("Mono12Packed") == 0
		|| strPixelFormatInfo.compare("Mono16") == 0)
	{
		m_nComponents = MONO_COMPONENTS;
		m_nbitDepth = MONO_CONVERTED_DEPTH;
		SetProperty(MM::g_Keyword_PixelType, g_PixelType_8bit);


	}

	else if (strPixelFormatInfo.compare("BayerGB12Packed") == 0 || strPixelFormatInfo.compare("BayerGB12") == 0 || 
		strPixelFormatInfo.compare("BGR8") == 0 || strPixelFormatInfo.compare("RGB8") == 0)
	{
		m_nComponents = COLOR_COMPONENTS;
		m_nbitDepth = COLOR_CONVERTED_DEPTH;
		SetProperty(MM::g_Keyword_PixelType, g_PixelType_8bitRGBA);
	}
	else
	{
		m_nComponents = COLOR_COMPONENTS;
		m_nbitDepth = COLOR_CONVERTED_DEPTH;
		SetProperty(MM::g_Keyword_PixelType, g_PixelType_8bitRGBA);
	}


	MvWriteLog(__FILE__, __LINE__, m_chDevID, "OnPixelType   [%s]  nComponents_ [%d]", entry.chSymbolic, m_nComponents);

	return DEVICE_OK;
}
/**
* Returns the size in bytes of the image buffer.
*/
long HikrobotCamera::GetImageBufferSize() const
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
int HikrobotCamera::SetROI(unsigned x, unsigned y, unsigned xSize, unsigned ySize)
{

	MVCC_INTVALUE_EX width = { 0 };
	m_pCamera->GetIntValue("Width", &width);
	MVCC_INTVALUE_EX height = { 0 };
	m_pCamera->GetIntValue("Height", &height);
	MVCC_INTVALUE_EX offsetX = { 0 };
	m_pCamera->GetIntValue("OffsetX", &offsetX);
	MVCC_INTVALUE_EX offsetY = { 0 };
	m_pCamera->GetIntValue("OffsetY", &offsetY);


	x -= (x % offsetX.nInc);
	y -= (y % offsetY.nInc);
	xSize -= (xSize % width.nInc);
	ySize -= (ySize % height.nInc);
	if (xSize < width.nMin) {
		xSize = (unsigned int)width.nMin;
	}
	if (ySize < height.nMin) {
		ySize = (unsigned int)height.nMin;
	}
	if (x < offsetX.nMin) {
		x = (unsigned int)offsetX.nMin;
	}
	if (y < offsetY.nMin) {
		y = (unsigned int)offsetY.nMin;
	}

	m_pCamera->SetIntValue("Width", xSize);
	m_pCamera->SetIntValue("Height", ySize);
	m_pCamera->SetIntValue("OffsetX", x);
	m_pCamera->SetIntValue("OffsetY", y);


	MvWriteLog(__FILE__, __LINE__, m_chDevID, "Set roi Width %d, Height %d, OffsetX %d, OffsetY %d", xSize, ySize, x, y);

	return DEVICE_OK;
}

/**
* Returns the actual dimensions of the current ROI.
*/
int HikrobotCamera::GetROI(unsigned& x, unsigned& y, unsigned& xSize, unsigned& ySize)
{
	MVCC_INTVALUE_EX width = { 0 };
	m_pCamera->GetIntValue("Width", &width);
	MVCC_INTVALUE_EX height = { 0 };
	m_pCamera->GetIntValue("Height", &height);
	MVCC_INTVALUE_EX offsetX = { 0 };
	m_pCamera->GetIntValue("OffsetX", &offsetX);
	MVCC_INTVALUE_EX offsetY = { 0 };
	m_pCamera->GetIntValue("OffsetY", &offsetY);

	x = (unsigned int)offsetX.nCurValue;
	y = (unsigned int)offsetY.nCurValue;
	xSize = (unsigned int)width.nCurValue;
	ySize = (unsigned int)height.nCurValue;

#if 0
	MvWriteLog(__FILE__, __LINE__, m_chDevID, "Get roi Width %d, Height %d, OffsetX %d, OffsetY %d\n", xSize, ySize, x, y);
#endif

	return DEVICE_OK;
}

/**
* Resets the Region of Interest to full frame.
*/
int HikrobotCamera::ClearROI()
{
	m_pCamera->SetIntValue("OffsetX", 0);
	m_pCamera->SetIntValue("OffsetY", 0);
	m_pCamera->SetIntValue("Width", maxWidth_);
	m_pCamera->SetIntValue("Height", maxHeight_);

	MvWriteLog(__FILE__, __LINE__, m_chDevID, "Clear roi Width %d, Height %d, OffsetX %d, OffsetY %d\n", maxWidth_, maxHeight_, 0, 0);

	return DEVICE_OK;
}

/**
* Returns the current exposure setting in milliseconds.
* Required by the MM::Camera API.
*/
double HikrobotCamera::GetExposure() const
{
	MVCC_FLOATVALUE stFloatValue = { 0.0 };
	m_pCamera->GetFloatValue("ExposureTime", &stFloatValue);
	return stFloatValue.fCurValue / 1000.0;
}

/**
* Sets exposure in milliseconds.
* Required by the MM::Camera API.
*/
void HikrobotCamera::SetExposure(double exp)
{
	exp *= 1000; //convert to us
	if (exp > exposureMax_) {
		exp = exposureMax_;
	}
	else if (exp < exposureMin_) {
		exp = exposureMin_;
	}

	m_pCamera->SetEnumValue("ExposureAuto", MV_EXPOSURE_AUTO_MODE_OFF);
	m_pCamera->SetFloatValue("ExposureTime", exp);
	exposure_us_ = exp;

}

/**
* Returns the current binning factor.
*/
int HikrobotCamera::GetBinning() const
{
	return  std::atoi(binningFactor_.c_str());
}

int HikrobotCamera::SetBinning(int binFactor)
{
	cout << "SetBinning called\n";
	if (binFactor > 1 && binFactor < 4) {
		return DEVICE_OK;
	}
	return DEVICE_OK;
}

bool HikrobotCamera::IsColor(MvGvspPixelType enType)
{
	switch (enType)
	{
	case PixelType_Gvsp_RGB8_Packed:
	case PixelType_Gvsp_BGR8_Packed:
	case PixelType_Gvsp_YUV422_Packed:
	case PixelType_Gvsp_YUV422_YUYV_Packed:
	case PixelType_Gvsp_BayerGR8:
	case PixelType_Gvsp_BayerRG8:
	case PixelType_Gvsp_BayerGB8:
	case PixelType_Gvsp_BayerBG8:
	case PixelType_Gvsp_BayerGB10:
	case PixelType_Gvsp_BayerGB10_Packed:
	case PixelType_Gvsp_BayerBG10:
	case PixelType_Gvsp_BayerBG10_Packed:
	case PixelType_Gvsp_BayerRG10:
	case PixelType_Gvsp_BayerRG10_Packed:
	case PixelType_Gvsp_BayerGR10:
	case PixelType_Gvsp_BayerGR10_Packed:
	case PixelType_Gvsp_BayerGB12:
	case PixelType_Gvsp_BayerGB12_Packed:
	case PixelType_Gvsp_BayerBG12:
	case PixelType_Gvsp_BayerBG12_Packed:
	case PixelType_Gvsp_BayerRG12:
	case PixelType_Gvsp_BayerRG12_Packed:
	case PixelType_Gvsp_BayerGR12:
	case PixelType_Gvsp_BayerGR12_Packed:
		return true;
	default:
		return false;
	}
}

bool HikrobotCamera::IsMono(MvGvspPixelType enType)
{
	switch (enType)
	{
	case PixelType_Gvsp_Mono8:
	case PixelType_Gvsp_Mono8_Signed:
	case PixelType_Gvsp_Mono10:
	case PixelType_Gvsp_Mono10_Packed:
	case PixelType_Gvsp_Mono12:
	case PixelType_Gvsp_Mono12_Packed:
	case PixelType_Gvsp_Mono14:
	case PixelType_Gvsp_Mono16:
		return true;
	default:
		return false;
	}
}


int HikrobotCamera::StartSequenceAcquisition(long numImages, double /* interval_ms */, bool /* stopOnOverflow */) {


	UNREFERENCED_PARAMETER(numImages);
	MvWriteLog(__FILE__, __LINE__, m_chDevID, "GStartSequenceAcquisition , not support ,just return.");

	return DEVICE_OK;

}

int HikrobotCamera::StartSequenceAcquisition(double /* interval_ms */) {

	if (m_bGrabbing)
	{
		MvWriteLog(__FILE__, __LINE__, m_chDevID, "StartSequenceAcquisition Begin, but Already Start.");
		return DEVICE_NOT_SUPPORTED;   //设备已经start，不能再次start; ImageJ 中截图和取流不能同时使用  [截图后，快速start可能会报错]
	}

	MvWriteLog(__FILE__, __LINE__, m_chDevID, "StartSequenceAcquisition Begin");

	StopSequenceAcquisition();
	m_pCamera->SetGrabStrategy(MV_GrabStrategy_OneByOne);
	m_pCamera->StartGrabbing();


	m_bRecvRuning = true;	//取流线程工作
	unsigned int nThreadID = 0;
	if (NULL == m_hImageRecvThreadHandle)
	{
		m_hImageRecvThreadHandle = (void*)_beginthreadex(NULL, 0, ImageRecvThread, this, 0, &nThreadID);
		if (NULL == m_hImageRecvThreadHandle)
		{
			MvWriteLog(__FILE__, __LINE__, m_chDevID, "Create ImageRecvThread failed.");
			return DEVICE_ERR;
		}
	}

	m_bGrabbing = true; //取流状态
	MvWriteLog(__FILE__, __LINE__, m_chDevID, "StartSequenceAcquisition End");

	return DEVICE_OK;
}

// 取流处理线程
unsigned int  __stdcall HikrobotCamera::ImageRecvThread(void* pUser)
{
	if (NULL == pUser)
	{
		return 0;
	}
	HikrobotCamera* pThis = (HikrobotCamera*)pUser;

	pThis->MvWriteLog(__FILE__, __LINE__, pThis->m_chDevID, "ImageRecvThreadProc Start.");
	pThis->ImageRecvThreadProc();
	pThis->MvWriteLog(__FILE__, __LINE__, pThis->m_chDevID, "ImageRecvThreadProc End.");

}


void HikrobotCamera::ImageRecvThreadProc()
{
	int nRet = MV_OK;
	MV_FRAME_OUT stOutFrame = { 0 };

	MvWriteLog(__FILE__, __LINE__, m_chDevID,"ImageRecvThreadProc Begin");



	while (m_bRecvRuning)
	{
		nRet = GetCamera()->GetImageBuffer(&stOutFrame, 1000);
		if (MV_OK == nRet)
		{
			MvWriteLog(__FILE__, __LINE__, m_chDevID, "Get One Frame: Width[%d], Height[%d], FrameNum[%d] FrameLen[%d] enPixelType[%lld]",
				stOutFrame.stFrameInfo.nWidth, stOutFrame.stFrameInfo.nHeight, stOutFrame.stFrameInfo.nFrameNum, stOutFrame.stFrameInfo.nFrameLen, stOutFrame.stFrameInfo.enPixelType);


			MvGvspPixelType enDstPixelType = PixelType_Gvsp_Undefined;
			unsigned int nChannelNum = 0;

			nRet = PixTypeProc(stOutFrame.stFrameInfo.enPixelType, nChannelNum, enDstPixelType);
			if (MV_OK != nRet)
			{
				MvWriteLog(__FILE__, __LINE__, m_chDevID, "PixTypeProc Failed,errcode [%#x]!", nRet);
				return;
			}
			
			int nNeedSize = stOutFrame.stFrameInfo.nWidth * stOutFrame.stFrameInfo.nHeight * nChannelNum;
			if (m_nConvertDataLen < nNeedSize || (NULL == m_pConvertData))
			{
				if (m_pConvertData)
				{
					free(m_pConvertData);
					m_pConvertData = NULL;
				}

				m_pConvertData = (unsigned char*)malloc(nNeedSize);
				if (NULL == m_pConvertData)
				{
					nRet = GetCamera()->FreeImageBuffer(&stOutFrame);
					if (MV_OK != nRet)
					{
						MvWriteLog(__FILE__, __LINE__, m_chDevID, "FreeImageBuffer failed [%#x]", nRet);
					}

					MvWriteLog(__FILE__, __LINE__, m_chDevID, "Malloc pConvertData Len %d  fail!", nNeedSize);
					nRet = MV_E_RESOURCE;

					break;
				}
				m_nConvertDataLen = nNeedSize;

				MvWriteLog(__FILE__, __LINE__, m_chDevID, "Malloc pConvertData len [%d].", nNeedSize);
			}


			// ch:像素格式转换 | en:Convert pixel format 
			MV_CC_PIXEL_CONVERT_PARAM stConvertParam = { 0 };
			stConvertParam.nWidth = stOutFrame.stFrameInfo.nWidth;                 //ch:图像宽 | en:image width
			stConvertParam.nHeight = stOutFrame.stFrameInfo.nHeight;               //ch:图像高 | en:image height
			stConvertParam.pSrcData = stOutFrame.pBufAddr;                         //ch:输入数据缓存 | en:input data buffer
			stConvertParam.nSrcDataLen = stOutFrame.stFrameInfo.nFrameLen;         //ch:输入数据大小 | en:input data size
			stConvertParam.enSrcPixelType = stOutFrame.stFrameInfo.enPixelType;    //ch:输入像素格式 | en:input pixel format
			stConvertParam.enDstPixelType = enDstPixelType;                         //ch:输出像素格式 | en:output pixel format
			stConvertParam.pDstBuffer = m_pConvertData;                               //ch:输出数据缓存 | en:output data buffer
			stConvertParam.nDstBufferSize = m_nConvertDataLen;                       //ch:输出缓存大小 | en:output buffer size
			nRet = GetCamera()->ConvertPixelType(&stConvertParam);
			if (MV_OK != nRet)
			{
				MvWriteLog(__FILE__, __LINE__, m_chDevID, "Convert Pixel Type fail, Errcode [%#x]!", nRet);
				break;
			}

			//!fix , md must assign something
			Metadata md;
			md.put(MM::g_Keyword_Metadata_CameraLabel, "");

			nRet = GetCoreCallback()->InsertImage(this, (const unsigned char*)stConvertParam.pDstBuffer,
				stOutFrame.stFrameInfo.nWidth,
				stOutFrame.stFrameInfo.nHeight, GetImageBytesPerPixel(), 1, md.Serialize().c_str(), FALSE);
			if (nRet == DEVICE_BUFFER_OVERFLOW)
			{
				//if circular buffer overflows, just clear it and keep putting stuff in so live mode can continue
				GetCoreCallback()->ClearImageBuffer(this);
				MvWriteLog(__FILE__, __LINE__, m_chDevID, "InsertImage clear!");


			}
			if (nRet == DEVICE_OK)
			{
				MvWriteLog(__FILE__, __LINE__, m_chDevID, "Success InsertImage Width[%d], Height[%d], FrameNum[%d] FrameLen[%d] enPixelType[%lld]",
					stOutFrame.stFrameInfo.nWidth, stOutFrame.stFrameInfo.nHeight, stOutFrame.stFrameInfo.nFrameNum, stOutFrame.stFrameInfo.nFrameLen, stOutFrame.stFrameInfo.enPixelType);
			}
	

			nRet = GetCamera()->FreeImageBuffer(&stOutFrame);
			if (nRet != MV_OK)
			{
				MvWriteLog(__FILE__, __LINE__, m_chDevID, "FreeImageBuffer failed!");
			}

		}
		else
		{
			//DebugInfo("HikrobotCamera::Get Image fail! nRet [0x%x]\n", nRet);
		}

	}


	GetCoreCallback()->ClearImageBuffer(this);



	MvWriteLog(__FILE__, __LINE__, m_chDevID, "ImageRecvThreadProc End!");
	return;
}



bool HikrobotCamera::IsCapturing()
{
	return 	m_bGrabbing;
}

int HikrobotCamera::StopSequenceAcquisition()
{
	MvWriteLog(__FILE__, __LINE__, m_chDevID, "StopSequenceAcquisition Begin");


	m_pCamera->StopGrabbing();

	m_bRecvRuning = false;
	if (m_hImageRecvThreadHandle)
	{
		WaitForSingleObject(m_hImageRecvThreadHandle, INFINITE);
		CloseHandle(m_hImageRecvThreadHandle);
		m_hImageRecvThreadHandle = NULL;
	}
	m_bGrabbing = false;
	MvWriteLog(__FILE__, __LINE__, m_chDevID, "StopSequenceAcquisition End");

	return DEVICE_OK;
}

int HikrobotCamera::PrepareSequenceAcqusition()
{
	// nothing to prepare
	return DEVICE_OK;
}

void HikrobotCamera::ResizeSnapBuffer() {

	long bytes = GetImageBufferSize();

	if (bytes > imgBufferSize_)
	{
		if (imgBuffer_)
		{
			free(imgBuffer_);
			imgBuffer_ = NULL;
		}

		imgBuffer_ = malloc(bytes);
		imgBufferSize_ = bytes;

		MvWriteLog(__FILE__, __LINE__, m_chDevID, "imgBufferSize_ %d ", imgBufferSize_);
	}
}


//////
// Action handlers
///////////////////////////////////////////////////////////////////////////////

int HikrobotCamera::OnTriggerSource(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	string TriggerSource_;
	if (eAct == MM::AfterSet) {
		pProp->Get(TriggerSource_);
		m_pCamera->SetEnumValueByString("TriggerSource", TriggerSource_.c_str());

		MvWriteLog(__FILE__, __LINE__, m_chDevID, "OnTriggerSource  MM::AfterSet %s ", TriggerSource_.c_str());
	}
	else if (eAct == MM::BeforeGet) {
	
		MVCC_ENUMVALUE stEnumValue = { 0 };
		m_pCamera->GetEnumValue("TriggerSource", &stEnumValue);
		if(MV_TRIGGER_SOURCE_SOFTWARE == stEnumValue.nCurValue)
		{
			MvWriteLog(__FILE__, __LINE__, m_chDevID, "OnTriggerSource  MM::BeforeGet  Software");

			const char* s = "Software";
			pProp->Set(s);
		}
	}
	return DEVICE_OK;
}

int HikrobotCamera::OnBinningMode(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	MVCC_ENUMVALUE BinningHorizontal = {0};
	m_pCamera->GetEnumValue("BinningHorizontal", &BinningHorizontal);
	MVCC_ENUMVALUE BinningVertical = {0};
	m_pCamera->GetEnumValue("BinningHorizontal", &BinningVertical);

	if (eAct == MM::AfterSet)
	{
		if (IsAvailable("BinningModeVertical") && IsAvailable("BinningModeVertical"))
		{
		
				string binningMode;
				pProp->Get(binningMode);
				m_pCamera->SetStringValue("BinningModeHorizontal", binningMode.c_str());
				m_pCamera->SetStringValue("BinningModeVertical", binningMode.c_str());
			
		
		}
	}
	else if (eAct == MM::BeforeGet)
	{
		if (IsAvailable("BinningModeVertical") && IsAvailable("BinningModeVertical"))
		{
			std::ostringstream strValue;
			strValue << BinningHorizontal.nCurValue;
			pProp->Set(strValue.str().c_str());
		}

	}
	return DEVICE_OK;
}

int HikrobotCamera::OnHeight(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	MVCC_INTVALUE_EX Height = { 0 };
	m_pCamera->GetIntValue("Height", &Height);

	std::string strval;
	if (eAct == MM::AfterSet)
	{

		if (IsAvailable("Height"))
		{
			try
			{
				if (m_bGrabbing)
				{
					MvWriteLog(__FILE__, __LINE__, m_chDevID, "OnHeight  Already start, StopGrab first. ");
					m_pCamera->StopGrabbing();
				}
				pProp->Get(strval);
				int64_t val = std::atoi(strval.c_str());
				int64_t inc = Height.nInc;
				m_pCamera->SetIntValue("Height", val - (val % inc));

				if (m_bGrabbing)
				{
					MvWriteLog(__FILE__, __LINE__, m_chDevID, "OnHeight  Already start, Recovery StartGrabing. ");
					m_pCamera->StartGrabbing();
				}
				//pProp->Set(Width->GetValue());

				MvWriteLog(__FILE__, __LINE__, m_chDevID, "OnHeight  AfterSet: %ld ", long(val - (val % inc)));
			}
			catch (...)
			{
				// Error handling.
				MvWriteLog(__FILE__, __LINE__, m_chDevID, "error handle");

			}
		}
	}
	else if (eAct == MM::BeforeGet) {
		try {
			if (IsAvailable("Height"))
			{
				pProp->Set((long)Height.nCurValue);
				MvWriteLog(__FILE__, __LINE__, m_chDevID, "OnHeight BeforeGet: %d", Height.nCurValue);
			}
		}
		catch (...)
		{
			// Error handling.
			MvWriteLog(__FILE__, __LINE__, m_chDevID, "Set Height, Error.");
		}
	}
	return DEVICE_OK;
}

int HikrobotCamera::OnWidth(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	MVCC_INTVALUE_EX Width = { 0 };
	m_pCamera->GetIntValue("Width", &Width);

	std::string strval;
	if (eAct == MM::AfterSet)
	{
		bool Isgrabbing = m_bGrabbing;

		if (IsAvailable("Width"))
		{
			try
			{
				if (Isgrabbing)
				{
					m_pCamera->StopGrabbing();
				}
				pProp->Get(strval);
				int64_t val = std::atoi(strval.c_str());
				int64_t inc = Width.nInc;
				m_pCamera->SetIntValue("Width", val - (val % inc));
				if (Isgrabbing)
				{
					m_pCamera->StartGrabbing();
				}
				//pProp->Set(Width->GetValue());

				MvWriteLog(__FILE__, __LINE__, m_chDevID, "OnWidth AfterSet get: %ld ", long(val - (val % inc)));

			}
			catch (...)
			{
				// Error handling.
				MvWriteLog(__FILE__, __LINE__, m_chDevID, "An exception occurred when set width");

			}
		}
	}
	else if (eAct == MM::BeforeGet) {
		try {
			if (IsAvailable("Width"))
			{
				pProp->Set((long)Width.nCurValue);

				MvWriteLog(__FILE__, __LINE__, m_chDevID, "OnWidth before get: %ld ", long(Width.nCurValue));
			}
		}
		catch (...)
		{
			// Error handling.
			MvWriteLog(__FILE__, __LINE__, m_chDevID, "An exception occurred when get width");
		}
	}
	return DEVICE_OK;
}

int HikrobotCamera::OnExposure(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::AfterSet)
	{
		if (IsWritable("ExposureTime") || IsWritable("ExposureTimeAbs"))
		{
			try
			{
				pProp->Get(exposure_us_);

				exposure_us_ = m_pCamera->SetFloatValue("ExposureTime", exposure_us_);
				exposure_us_ = m_pCamera->SetFloatValue("ExposureTimeAbs", exposure_us_);
			}
			catch (...)
			{
				// Error handling.
				MvWriteLog(__FILE__, __LINE__, m_chDevID, "An exception occurred when set ExposureTime or ExposureTimeAbs");
			}
		}
	}
	else if (eAct == MM::BeforeGet) {

		try {
			if (IsAvailable("ExposureTime") && IsAvailable("ExposureTimeAbs"))
			{
				MVCC_FLOATVALUE stValue;
				m_pCamera->GetFloatValue("ExposureTime", &stValue);
				exposure_us_ = stValue.fCurValue;

				m_pCamera->GetFloatValue("ExposureTimeAbs", &stValue);
				exposure_us_ = stValue.fCurValue;

				pProp->Set(exposure_us_);
			}
		}
		catch (...)
		{
			// Error handling.
			MvWriteLog(__FILE__, __LINE__, m_chDevID, "An exception occurred when get ExposureTime or ExposureTimeAbs");
		}
	}

	return DEVICE_OK;
}

int HikrobotCamera::OnBinning(MM::PropertyBase* pProp, MM::ActionType eAct)
{

	MVCC_ENUMVALUE BinningHorizontal = {0};
	m_pCamera->GetEnumValue("BinningHorizontal", &BinningHorizontal);
	MVCC_ENUMVALUE BinningVertical = { 0 };
	m_pCamera->GetEnumValue("BinningHorizontal", &BinningVertical);

	if (eAct == MM::AfterSet)
	{
		bool Isgrabbing = m_bGrabbing;

		if (IsAvailable("BinningHorizontal") && IsAvailable("BinningHorizontal"))
		{
			try
			{
				if (Isgrabbing)
				{
					m_pCamera->StopGrabbing();
				}
				pProp->Get(binningFactor_);
				int64_t val = std::atoi(binningFactor_.c_str());
				m_pCamera->SetIntValue("BinningHorizontal", val);
				m_pCamera->SetIntValue("BinningVertical", val);
				if (Isgrabbing)
				{
					m_pCamera->StartGrabbing();
				}
				pProp->Set(binningFactor_.c_str());
			}
			catch (...)
			{
				MvWriteLog(__FILE__, __LINE__, m_chDevID, "An exception occurred when set BinningHorizontal or BinningVertical");

			}
		}
	}
	else if (eAct == MM::BeforeGet) {

		try {
			if (IsAvailable("BinningHorizontal") && IsAvailable("BinningHorizontal") )
			{
				binningFactor_ = CDeviceUtils::ConvertToString((long)BinningHorizontal.nCurValue);
				pProp->Set((long)BinningHorizontal.nCurValue);
			}
			else
			{
				pProp->Set("1");
			}
		}
		catch (...)
		{
			MvWriteLog(__FILE__, __LINE__, m_chDevID, "An exception occurred when get BinningHorizontal or BinningVertical");
		}
	}

	return DEVICE_OK;
}

unsigned  HikrobotCamera::GetNumberOfComponents() const
{
	std::string s = CDeviceUtils::ConvertToString(long(m_nComponents));
	MvWriteLog(__FILE__, __LINE__, (char *)m_chDevID, "GetNumberOfComponents: %ld" , m_nComponents);

	return m_nComponents;
};


int HikrobotCamera::OnSensorReadoutMode(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (IsAvailable("SensorReadoutMode"))
	{
		string Sensormode = "";
		if (eAct == MM::AfterSet) {
			pProp->Get(Sensormode);


			m_pCamera->SetStringValue("SensorReadoutMode", Sensormode.c_str());

			MVCC_STRINGVALUE strMode;
			m_pCamera->GetStringValue("SensorReadoutMode", &strMode);
			pProp->Set(strMode.chCurValue);
		}
		else if (eAct == MM::BeforeGet) {
			MVCC_STRINGVALUE strMode;
			m_pCamera->GetStringValue("SensorReadoutMode", &strMode);
			pProp->Set(strMode.chCurValue);
		}
	}
	return DEVICE_OK;
}

int HikrobotCamera::OnTriggerMode(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	MVCC_ENUMVALUE TriggerMode = { 0 };
	string TriggerMode_;

	m_pCamera->GetEnumValue("TriggerMode", &TriggerMode);
	MVCC_ENUMENTRY entry;
	entry.nValue = TriggerMode.nCurValue;
	m_pCamera->GetEnumEntrySymbolic("TriggerMode", &entry);

	if (IsAvailable("TriggerMode"))
	{
		if (eAct == MM::AfterSet)
		{
			pProp->Get(TriggerMode_);
			m_pCamera->SetEnumValueByString("TriggerMode", TriggerMode_.c_str());

			pProp->Set(TriggerMode_.c_str());

			MvWriteLog(__FILE__, __LINE__, m_chDevID, "OnTriggerMode  MM::AfterSet %s", TriggerMode_.c_str());
		}
		else if (eAct == MM::BeforeGet)
		{

			MvWriteLog(__FILE__, __LINE__, m_chDevID, "OnTriggerMode  MM::BeforeGet %s", entry.chSymbolic);

			pProp->Set(entry.chSymbolic);
		}
	}

	return DEVICE_OK;
}

int HikrobotCamera::OnTemperature(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	UNREFERENCED_PARAMETER(pProp);
	UNREFERENCED_PARAMETER(eAct);
	return DEVICE_NOT_SUPPORTED;
}

int HikrobotCamera::OnTemperatureState(MM::PropertyBase* pProp, MM::ActionType eAct)
{ 
	//FIX me
	if (eAct == MM::BeforeGet) {
		MVCC_ENUMENTRY ptrtemperatureState_;
		m_pCamera->GetEnumEntrySymbolic("TemperatureState", &ptrtemperatureState_);

		temperatureState_.assign(ptrtemperatureState_.chSymbolic);
		pProp->Set(temperatureState_.c_str());
	}
	return DEVICE_OK;
}

int HikrobotCamera::OnReverseX(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::AfterSet) {
		pProp->Get(reverseX_);

		bool reverseX = false;
		m_pCamera->GetBoolValue("ReverseX", &reverseX);
		//reverseX->FromString(reverseX_.c_str());
		istringstream(reverseX_) >> boolalpha >> reverseX;//boolalpha>>必须要加 
		m_pCamera->SetBoolValue("ReverseX", &reverseX);
	}
	else if (eAct == MM::BeforeGet) {
		//CBooleanPtr reverseX(nodeMap_->GetNode("ReverseX"));

		bool reverseX = false;
		m_pCamera->GetBoolValue("ReverseX", &reverseX);
		//reverseX_.assign(reverseX);
		pProp->Set(reverseX_.c_str());
	}
	
	return DEVICE_OK;
}

int HikrobotCamera::OnReverseY(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::AfterSet) {
		pProp->Get(reverseY_);
		//CBooleanPtr reverseY(nodeMap_->GetNode("ReverseY"));
		//reverseY->FromString(reverseY_.c_str());
		bool ReverseY = false;
		m_pCamera->GetBoolValue("ReverseY", &ReverseY);
		istringstream(reverseX_) >> boolalpha >> ReverseY;//boolalpha>>必须要加 
		m_pCamera->SetBoolValue("ReverseX", &ReverseY);
	}
	else if (eAct == MM::BeforeGet) {
		//CBooleanPtr reverseY(nodeMap_->GetNode("ReverseY"));
		//reverseY_.assign(reverseY->ToString().c_str());
		bool reverseX = false;
		m_pCamera->GetBoolValue("ReverseX", &reverseX);
		pProp->Set(reverseY_.c_str());
	}
	return DEVICE_OK;
}

int HikrobotCamera::OnAcqFramerateEnable(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::AfterSet) {
		pProp->Get(setAcqFrm_);


		//CBooleanPtr setAcqFrm(nodeMap_->GetNode("AcquisitionFrameRateEnable"));
		//setAcqFrm->FromString(setAcqFrm_.c_str());

		bool setAcqFrm = false;
		m_pCamera->GetBoolValue("AcquisitionFrameRateEnable", &setAcqFrm);
		istringstream(setAcqFrm_) >> boolalpha >> setAcqFrm;//boolalpha>>必须要加 
		m_pCamera->SetBoolValue("AcquisitionFrameRateEnable", &setAcqFrm);

	}
	else if (eAct == MM::BeforeGet) {
		//CBooleanPtr setAcqFrm(nodeMap_->GetNode("AcquisitionFrameRateEnable"));
		//setAcqFrm_.assign(setAcqFrm->ToString().c_str());
		bool setAcqFrm = false;
		m_pCamera->SetBoolValue("AcquisitionFrameRateEnable", &setAcqFrm);
		setAcqFrm_ =  std::to_string(setAcqFrm);
		
		pProp->Set(setAcqFrm_.c_str());
	}
	return DEVICE_OK;
}

int HikrobotCamera::OnAcqFramerate(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::AfterSet) {
		pProp->Get(acqFramerate_);
		//m_pCamera->AcquisitionFrameRateAbs.TrySetValue(acqFramerate_);
		//m_pCamera->AcquisitionFrameRate.TrySetValue(acqFramerate_);

		m_pCamera->SetFloatValue("AcquisitionFrameRateAbs", acqFramerate_);
		m_pCamera->SetFloatValue("AcquisitionFrameRate", acqFramerate_);
	}
	else if (eAct == MM::BeforeGet) {
		if (IsAvailable("AcquisitionFrameRate"))
		{
			MVCC_FLOATVALUE value;
			m_pCamera->GetFloatValue("AcquisitionFrameRate",&value);
			acqFramerate_ = value.fCurValue;
		}
		else if (IsAvailable("AcquisitionFrameRateAbs"))
		{
			MVCC_FLOATVALUE value;
			m_pCamera->GetFloatValue("AcquisitionFrameRateAbs", &value);
			acqFramerate_ = value.fCurValue;
			//acqFramerate_ = m_pCamera->AcquisitionFrameRateAbs.GetValue();
		}
		std::ostringstream oss;
		//oss << std::fixed << std::setfill('0') << std::setprecision(2) << acqFramerate_;
		oss << acqFramerate_;
		pProp->Set(oss.str().c_str());
	}
	return DEVICE_OK;
}

int HikrobotCamera::OnTestPattern(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	string TestPattern_;
	MVCC_ENUMVALUE stEnumValue = { 0 };
	MVCC_ENUMENTRY entry = { 0 };

	if (eAct == MM::AfterSet) 
	{
		pProp->Get(TestPattern_);

		m_pCamera->SetEnumValueByString("TestPattern", TestPattern_.c_str());

		MvWriteLog(__FILE__, __LINE__, m_chDevID, "OnTestPattern MM::AfterSet %s", TestPattern_.c_str());
	}
	else if (eAct == MM::BeforeGet) 
	{
		m_pCamera->GetEnumValue("TestPattern", &stEnumValue);
		entry.nValue = stEnumValue.nCurValue;
		m_pCamera->GetEnumEntrySymbolic("TestPattern", &entry);

		MvWriteLog(__FILE__, __LINE__, m_chDevID, "OnTestPattern MM::BeforeGet %s", entry.chSymbolic);

		TestPattern_.assign(entry.chSymbolic);
		pProp->Set(TestPattern_.c_str());
	}

	return DEVICE_OK;
}

int HikrobotCamera::OnAutoGain(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	string GainAuto_;
	if (eAct == MM::AfterSet) {
		pProp->Get(GainAuto_);

		MVCC_ENUMVALUE GainAuto = { 0 };
		m_pCamera->GetEnumValue("GainAuto", &GainAuto);

		pProp->Set(CDeviceUtils::ConvertToString((long)GainAuto.nCurValue));
	}
	else if (eAct == MM::BeforeGet) {

		MVCC_ENUMVALUE GainAuto = { 0 };
		m_pCamera->GetEnumValue("GainAuto", &GainAuto);
		pProp->Set(CDeviceUtils::ConvertToString((long)GainAuto.nCurValue));

	}
	return DEVICE_OK;
}




int HikrobotCamera::OnResultingFramerate(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet) {
		if (IsAvailable("ResultingFrameRateAbs"))
		{
			MVCC_STRINGVALUE value;
			m_pCamera->GetStringValue("ResultingFrameRateAbs", &value);
			pProp->Set(value.chCurValue);
		}
		else if (IsAvailable("ResultingFrameRate"))
		{
			MVCC_STRINGVALUE value;
			m_pCamera->GetStringValue("ResultingFrameRate", &value);
			pProp->Set(value.chCurValue);
		}
	}
	return DEVICE_OK;
}
int HikrobotCamera::OnAutoExpore(MM::PropertyBase* pProp, MM::ActionType eAct)
{

	string ExposureAuto_;
	if (eAct == MM::AfterSet) {
		pProp->Get(ExposureAuto_);
		
		
		MVCC_ENUMVALUE ExposureAuto = { 0 };
		m_pCamera->GetEnumValue("ExposureAuto", &ExposureAuto);
		pProp->Set(CDeviceUtils::ConvertToString((long)ExposureAuto.nCurValue));

	}
	else if (eAct == MM::BeforeGet) {
		
		MVCC_ENUMVALUE ExposureAuto = { 0 };
		m_pCamera->GetEnumValue("ExposureAuto", &ExposureAuto);
		pProp->Set(CDeviceUtils::ConvertToString((long)ExposureAuto.nCurValue));
	}

	return DEVICE_OK;
}



int HikrobotCamera::OnLightSourcePreset(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	
	string LightSourcePreset_;
	if (eAct == MM::AfterSet) {
		pProp->Get(LightSourcePreset_);
		
		MVCC_ENUMVALUE LightSourcePreset = { 0 };
		m_pCamera->GetEnumValue("LightSourcePreset", &LightSourcePreset);
		pProp->Set(CDeviceUtils::ConvertToString((long)LightSourcePreset.nCurValue));
	}
	else if (eAct == MM::BeforeGet) {
		MVCC_ENUMVALUE LightSourcePreset = { 0 };
		m_pCamera->GetEnumValue("LightSourcePreset", &LightSourcePreset);
		pProp->Set(CDeviceUtils::ConvertToString((long)LightSourcePreset.nCurValue));
	}
	return DEVICE_OK;
}

int HikrobotCamera::OnShutterMode(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	
	if (eAct == MM::AfterSet) {
		pProp->Get(shutterMode_);

		MVCC_ENUMVALUE shutterMode = { 0 };
		m_pCamera->GetEnumValue("ShutterMode", &shutterMode);
		pProp->Set(CDeviceUtils::ConvertToString((long)shutterMode.nCurValue));
	}
	else if (eAct == MM::BeforeGet) {

		MVCC_ENUMVALUE shutterMode = { 0 };
		m_pCamera->GetEnumValue("ShutterMode", &shutterMode);
		pProp->Set(CDeviceUtils::ConvertToString((long)shutterMode.nCurValue));
	}
	return DEVICE_OK;
}

int HikrobotCamera::OnDeviceLinkThroughputLimit(MM::PropertyBase* pProp, MM::ActionType eAct)
{

	MVCC_INTVALUE_EX DeviceLinkThroughputLimit = {0};
	m_pCamera->GetIntValue("DeviceLinkThroughputLimit", &DeviceLinkThroughputLimit);
	if (IsAvailable("DeviceLinkThroughputLimit"))
	{
		if (eAct == MM::AfterSet && IsWritable("DeviceLinkThroughputLimit"))
		{
			long val;
			pProp->Get(val);
			m_pCamera->SetIntValue("DeviceLinkThroughputLimit", val);
			DeviceLinkThroughputLimit_ = DeviceLinkThroughputLimit.nCurValue;
		}
		else if (eAct == MM::BeforeGet)
		{
			DeviceLinkThroughputLimit_ = DeviceLinkThroughputLimit.nCurValue;
			pProp->Set(CDeviceUtils::ConvertToString((long)DeviceLinkThroughputLimit_));
		}
	}
	return DEVICE_OK;
}

int HikrobotCamera::OnInterPacketDelay(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	
	MVCC_INTVALUE_EX GevSCPD = {0};
	m_pCamera->GetIntValue("GevSCPD", &GevSCPD);
	if (IsAvailable("GevSCPD"))
	{
		if (eAct == MM::AfterSet && IsWritable("GevSCPD"))
		{
			long val;
			pProp->Get(val);
			m_pCamera->SetIntValue("GevSCPD", val);

			InterPacketDelay_ = GevSCPD.nCurValue;
		}
		else if (eAct == MM::BeforeGet)
		{
			InterPacketDelay_ = GevSCPD.nCurValue;
			pProp->Set(CDeviceUtils::ConvertToString((long)InterPacketDelay_));
		}
	}
	return DEVICE_OK;
}

int HikrobotCamera::OnGain(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	try
	{

		MVCC_FLOATVALUE gain;
		m_pCamera->GetFloatValue("Gain", &gain);

		MVCC_INTVALUE_EX GainRaw = {0};
		m_pCamera->GetIntValue("GainRaw", &GainRaw);

		if (eAct == MM::AfterSet) {
			pProp->Get(gain_);
			if (gain_ > gainMax_) {
				gain_ = gainMax_;
			}
			if (gain_ < gainMin_) {
				gain_ = gainMin_;
			}
			if (IsAvailable("Gain"))
			{
				// the range gain depends on Pixel format sometimes.
				if (gain.fMin <= gain_ && gain.fMax >= gain_)
				{
					m_pCamera->SetFloatValue("Gain", gain_);

				}
				else
				{
					MvWriteLog(__FILE__, __LINE__, m_chDevID, "gain value out of range");
					gainMax_ = gain.fMax;
					gainMin_ = gain.fMin;
					gain_ = gain.fCurValue;
					SetPropertyLimits(MM::g_Keyword_Gain, gainMin_, gainMax_);
					pProp->Set(gain_);
				}
			}
			else if (IsAvailable("GainRaw"))
			{
				// the range gain depends on Pixel format sometimes.
				if (GainRaw.nMin <= gain_ && GainRaw.nMax >= gain_)
				{
					m_pCamera->SetFloatValue("GainRaw", (int64_t)(gain_));
				}
				else
				{
					MvWriteLog(__FILE__, __LINE__, m_chDevID, "gain value out of range");
					gainMax_ = gain.fMax;
					gainMin_ = gain.fMin;
					gain_ = gain.fCurValue;
					SetPropertyLimits(MM::g_Keyword_Gain, gainMin_, gainMax_);
					pProp->Set(gain_);
				}
			}
		}
		else if (eAct == MM::BeforeGet) {

			if (IsAvailable("Gain"))
			{
				gain_ = gain.fCurValue;
				pProp->Set(gain_);
			}
			else if (IsAvailable("GainRaw"))
			{
				gain_ = (double)GainRaw.nCurValue;
				pProp->Set(gain_);
				cout << "Gain Raw set successfully" << gain_ << endl;
			}
		}
	}
	catch (...)
	{
		// Error handling.
		MvWriteLog(__FILE__, __LINE__, m_chDevID, "OnGain unkonwn error");

		return DEVICE_ERR;
	}
	return DEVICE_OK;
}

int HikrobotCamera::OnOffset(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	MVCC_FLOATVALUE offset;
	m_pCamera->GetFloatValue("BlackLevel", &offset);
	MVCC_FLOATVALUE offsetRaw;
	m_pCamera->GetFloatValue("BlackLevelRaw", &offsetRaw);

	if (eAct == MM::AfterSet) {
		pProp->Get(offset_);
		if (offset_ > offsetMax_) {
			offset_ = offsetMax_;
		}
		if (offset_ < offsetMin_) {
			offset_ = offsetMin_;
		}
		if (IsAvailable("BlackLevel"))
		{
			m_pCamera->SetFloatValue("BlackLevel", offset_);
		}
		else if (IsAvailable("BlackLevelRaw"))
		{
			m_pCamera->SetFloatValue("BlackLevelRaw", offset_);

		}
	}
	else if (eAct == MM::BeforeGet) {
		if (IsAvailable("BlackLevel"))
		{
			offset_ = offset.fCurValue;
			pProp->Set(offset_);
		}
		else if (IsAvailable("BlackLevelRaw"))
		{
			offset_ = offsetRaw.fCurValue;
			pProp->Set(offset_);
		}
	}
	return DEVICE_OK;
}

void HikrobotCamera::ReduceImageSize(int64_t Width, int64_t Height)
{
	// This function is just for debug purpose
	/*if (!m_pCamera->IsOpen())
	{
		m_pCamera->Open();
	}*/


	MvWriteLog(__FILE__, __LINE__, m_chDevID, "HikrobotCamera::ReduceImageSize width %lld, height %lld \r\n", Width, Height);

	int64_t inc = 1;
	MVCC_INTVALUE_EX width = { 0 };
	m_pCamera->GetIntValue("Width", &width);
	if (width.nMax >= Width)
	{
		inc = width.nInc;
		m_pCamera->SetIntValue("Width",(Width - (Width % inc)));
	}

	MVCC_INTVALUE_EX height = { 0 };
	m_pCamera->GetIntValue("Width", &height);
	if (height.nMax >= Height)
	{
		inc = height.nInc;
		m_pCamera->SetIntValue("Width", (Height - (Height % inc)));
	}
}


void HikrobotCamera::SetLogBasicInfo(std::string msg) 
{
	// 记录型号，序列号;
	m_strBasiceLog = msg;
}



void HikrobotCamera::AddToLog(std::string msg) const
{
	// 增加下型号，序列号;
	LogMessage(m_strBasiceLog + msg, false);
}

char* GetFileName(char* strPath)
{
	char* endPos = NULL;
	if (strPath)
	{
#ifdef WIN32
		if ((endPos = const_cast<char*>(strrchr(strPath, '\\'))) != NULL)
#else
		if ((endPos = const_cast<char*>(strrchr(strPath, '/'))) != NULL)
#endif
		{
			endPos += 1;
		}
	}

	return endPos;
}


void HikrobotCamera::MvWriteLog(char* file, int line, char* pDevID, const char* fmt, ...) const
{
	va_list args;
	char szInfo[1024];

	sprintf_s(szInfo, 1024, "DevID:%s  File:%s Line:-L%04d Description:", pDevID, GetFileName(file), line);


	va_start(args, fmt);
	unsigned int nLen = strlen(szInfo);

	if ((nLen > 0) && (nLen < 1024))
	{
		vsnprintf_s(szInfo + nLen, 1024 - nLen, 1024 - nLen, fmt, args);
		va_end(args);
	}


	LogMessage(szInfo, false);
}




void HikrobotCamera::SetPixConfig(bool bMono)
{
	if (bMono)
	{
		SetProperty(MM::g_Keyword_PixelType, g_PixelType_8bit);
		m_nComponents = MONO_COMPONENTS;
		m_nbitDepth = MONO_CONVERTED_DEPTH;
	}
	else
	{
		SetProperty(MM::g_Keyword_PixelType, g_PixelType_8bitRGBA);

		m_nComponents = COLOR_COMPONENTS;
		m_nbitDepth = COLOR_CONVERTED_DEPTH;
	}
}


void HikrobotCamera::UpdateTemperature()
{
	return;
}

bool HikrobotCamera::IsAvailable(const char* strName) {
	//! Tests if available, FIX!
	MV_XML_AccessMode AccessMode = AM_Undefined;
	m_pCamera->GetNodeAccessMode(strName, &AccessMode);
	bool bRet = !(AccessMode == AM_NA || AccessMode == AM_NI);


#ifdef _DEBUG
	MvWriteLog(__FILE__, __LINE__, m_chDevID, "Node name [%s] IsWritable [%d], accessmode [%d]\n", strName, bRet, AccessMode);
#endif

	return bRet;
}

bool HikrobotCamera::IsWritable(const char* strName)
{
	MV_XML_AccessMode AccessMode = AM_Undefined;
	m_pCamera->GetNodeAccessMode(strName, &AccessMode);
	bool bRet = (AccessMode == AM_WO || AccessMode == AM_RW);

#ifdef _DEBUG

	MvWriteLog(__FILE__, __LINE__, m_chDevID, "Node name [%s] IsWritable [%d], accessmode [%d]", strName, bRet, AccessMode);

#endif

	return bRet;
}


