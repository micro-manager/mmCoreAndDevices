///////////////////////////////////////////////////////////////////////////////
// FILE:          MeadowlarkLC.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   MeadowlarkLC Device Adapter for Meadowlark Optics D5020 liquid crystal controller
// 
// AUTHOR:		  Amitabh Verma
//				  Ivan Ivanov
// 
// COPYRIGHT:	  Marine Biological Laboratory (2011 - 2017)
//				  Chan Zuckerberg Biohub San Francisco (2017 - 2023)
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



#include <cstdio>
#include <string>
#include <math.h>
#include <sstream>
#include <tchar.h>
#include <iostream>
#include <windows.h>
#include <array>
#include <fstream>


//////// MeadowlarkLC //////////
#include "MeadowlarkLC.h"
#include "usbdrvd.h"
#include "resource.h"
#include "InternalCalibrationCurve.h"

//////// Micro-manager interface //////////
#include "ModuleInterface.h"

//////////////////////////////////////////////////////////////////////////////////////////

const char* g_ControllerName = "MeadowlarkLC";
const char* g_ControllerAbout = "Meadowlark Optics D5020 Liquid Crystal Controller";
const char* g_ControllerDevices = "Select Device #";
const char* g_ControllerDeviceType = "Select Device Interface";
const char* g_ControllerDeviceType10V_D5020 = "D5020 Controller (untested)";
const char* g_ControllerDeviceType20V_D5020 = "D5020-20V Controller";
const double g_ControllerDeviceType10VFac_D5020 = 1000;
const double g_ControllerDeviceType20VFac_D5020 = 1000;
const char* g_ControllerTotalLCs = "Total Number of LCs";

const double g_ControllerDeviceRetardanceLowLimit = 0.001;
const double g_ControllerDeviceRetardanceHighLimit = 1.6;

const double g_ControllerDeviceRetardanceAbsRetLow = 0.0;
const double g_ControllerDeviceRetardanceAbsRetHigh = 1200.0;

const char* g_ControllerLCType_Internal = "Internal (Single generic 546nm curve)";
const char* g_ControllerLCType_F001 = "File (mmgr_dal_MeadowlarkLC.csv)"; //"Loaded From File (mmgr_dal_MeadowlarkLC.csv)";

const char* g_ControllerLCType_F001_Curves = "mmgr_dal_MeadowlarkLC.csv";
const char* g_ControllerLCType_F002_Curves = ".csv";

const char* g_ControllerDescription = "Description";
const char* g_ControllerDescriptionInfo = "Description Info";
const std::string g_Nonedetected = "None detected";

const char* g_Dev_Adapter_Ver = "2.0.0";
using namespace std;

const HMODULE GetCurrentModule()
{
	DWORD flags = GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS;
	HMODULE hm = 0;
	::GetModuleHandleEx(flags, reinterpret_cast<LPCTSTR>(GetCurrentModule), &hm);
	return hm;
}


// Default config values will be used at startup
const double ConfigDataDefaults[5][3] = {
	{546,0.25,0.5},
	{546,0.25,0.47},
	{546,0.25,0.53},
	{546,0.22,0.5},
	{546,0.28,0.5}
};

///////////////////////////////////////////////////////////////////////////////
// Exported MMDevice API
///////////////////////////////////////////////////////////////////////////////
MODULE_API void InitializeModuleData()
{
	RegisterDevice(g_ControllerName, MM::GenericDevice, g_ControllerAbout);
	//AddAvailableDeviceName(g_ControllerName, g_ControllerAbout);
}

MODULE_API MM::Device* CreateDevice(const char* deviceName)
{
	if (deviceName == 0) return 0;
	if (strcmp(deviceName, g_ControllerName) == 0)
	{
		return new MeadowlarkLC();
	}
	return 0;
}

MODULE_API void DeleteDevice(MM::Device* pDevice)
{
	delete pDevice;
}

///////////////////////////////////////////////////////////////////////////////
// MeadowlarkLC Hub
///////////////////////////////////////////////////////////////////////////////

MeadowlarkLC::MeadowlarkLC() :

	answerTimeoutMs_(1000),
	initialized_(false),
	interruptExercise_(false),
	tneDuration_(0),
	tneAmplitude_(0),
	numActiveLCs_(2),
	//  numTotalLCs_(4),
	numPalEls_(5),
	cur_dev_name("0"),
	cur_dev(0),
	g_ControllerDeviceTypeVFac(0),
	VoltageToRetFactor(0),
	RetToVoltageFactor(0),
	wavelength_(546.0), // the cached value	
	numberofCurves(1), // default is 1, will change based on import
	ArrayLength2(198)

{
	cur_dev = 0;
	controllerType_ = g_ControllerDeviceType20V_D5020; // default to D5020-20V
	controllerLCType_ = g_ControllerLCType_F001; // use mmgr_dal_MeadowlarkLC.csv file by default
	description_ = "Not found";
	serialnum_ = "Undefined";
	cur_dev_name = "1";

	loadDefault();
	InitializeDefaultErrorMessages();
	// add custom messages

	SetErrorText(ERR_PORT_CHANGE_FORBIDDEN, "You can't change the port after device has been initialized.");
	SetErrorText(ERR_INVALID_DEVICE, "The selected plugin does not fit for the device.");
	SetErrorText(ERR_INVALID_SERIAL_NUMBER, "Invalid Serial Number. Please refer to your Meadowlark device unit.");
	SetErrorText(ERR_INVALID_LCSERIAL_NUMBER, "Invalid LC-Serial Number. Calibration curve for this LC does not exist.");
	SetErrorText(ERR_INVALID_LC_UNPAIRED, "Invalid LC-Controller Pair. This Controller is not paired with an LC. Enter LC-S/N");
	SetErrorText(ERR_INVALID_LC_SELECTION, "Invalid Selection. The calibration curve file (mmgr_dal_MeadowlarkLC.csv) does not exist or is invalid.");

	std::string str = IntToString(numTotalLCs_).c_str();
	totalLCsMsg = "Invalid Active LCs. Number of Active LCs cannot be more than Total of " + str + " LCs";
	SetErrorText(ERR_INVALID_ACTIVE_LCS, totalLCsMsg.c_str());

	// pre-initialization properties

	// Description from device
	CPropertyAction* pAct = new CPropertyAction(this, &MeadowlarkLC::GetDesc);
	CreateProperty(g_ControllerDescriptionInfo, description_.c_str(), MM::String, false, pAct, true);

	// Select device #
	pAct = new CPropertyAction(this, &MeadowlarkLC::OnNumDevices);
	CreateProperty(g_ControllerDevices, g_Nonedetected.c_str(), MM::String, false, pAct, true);

	EnableDelay();

	theGUID.Data1 = 0xa22b5b8b;
	theGUID.Data2 = 0xc670;
	theGUID.Data3 = 0x4198;
	theGUID.Data4[0] = 0x93;
	theGUID.Data4[1] = 0x85;
	theGUID.Data4[2] = 0xaa;
	theGUID.Data4[3] = 0xba;
	theGUID.Data4[4] = 0x9d;
	theGUID.Data4[5] = 0xfc;
	theGUID.Data4[6] = 0x7d;
	theGUID.Data4[7] = 0x2b;

	USB_PID = 0x139C;
	devcnt = USBDRVD_GetDevCount(USB_PID);

	BYTE ver_cmd[] = { 'v', 'e', 'r', ':', '?', '\n' };
	BYTE status[64];
	//  description_[64];

	int devCount = devcnt;

	UINT pipeCount = 0;
	int pipeCountInt = pipeCount;

	ClearAllowedValues(g_ControllerDevices); // clear old vals

	if (devCount > 0) {

		for (int devIdx = 1; devIdx < (devCount + 1); devIdx++)
		{
			dev_Handle = USBDRVD_OpenDevice(devIdx, flagsandattrs, USB_PID); // get the device handle
			pipe0 = USBDRVD_PipeOpen(1, 0, flagsandattrs, &theGUID);
			pipe1 = USBDRVD_PipeOpen(1, 1, flagsandattrs, &theGUID);

			USBDRVD_BulkWrite(dev_Handle, 1, ver_cmd, sizeof(ver_cmd)); /* send ver:? command */
			USBDRVD_BulkRead(dev_Handle, 0, status, sizeof(status)); /* read status response */

			/* output status until a <CR> is found */
			std::string device = IntToString(devIdx);
			std::string temp = ((char*)status);

			if (dev_Handle >= 0) {
				AddAllowedValue(g_ControllerDevices, device.c_str());
				devNameList.push_back(device.c_str());
				descriptionUnparsed_ = temp + '\n';
				description_ = temp;
				description_ = RemoveChars(description_, ".");
				description_ = RemoveChars(description_, ",");
				cur_dev_name = device;

				pipeCount = USBDRVD_GetPipeCount(dev_Handle);
				pipeCountInt = pipeCount;
			}

			USBDRVD_CloseDevice(dev_Handle);
		}

		SetProperty(g_ControllerDevices, cur_dev_name.c_str());

		// LC Controller Type
		pAct = new CPropertyAction(this, &MeadowlarkLC::OnControllerType);
		CreateProperty(g_ControllerDeviceType, g_ControllerDeviceType20V_D5020, MM::String, false, pAct, true);
		AddAllowedValue(g_ControllerDeviceType, g_ControllerDeviceType20V_D5020);
		//AddAllowedValue(g_ControllerDeviceType, g_ControllerDeviceType10V_D5020);

		// LC (Non-Bonded, Bonded, etc...)
		pAct = new CPropertyAction(this, &MeadowlarkLC::OnControllerLCType);
		CreateProperty("Select LC Type", g_ControllerLCType_F001, MM::String, false, pAct, true);
		AddAllowedValue("Select LC Type", g_ControllerLCType_F001);
		AddAllowedValue("Select LC Type", g_ControllerLCType_Internal);
		

		pAct = new CPropertyAction(this, &MeadowlarkLC::OnSerialNumber);
		CreateProperty("Controller S/N", "Undefined", MM::String, false, pAct, true);

		pAct = new CPropertyAction(this, &MeadowlarkLC::OnNumTotalLCs);
		CreateProperty(g_ControllerTotalLCs, "0", MM::Integer, true, pAct, true);

		pAct = new CPropertyAction(this, &MeadowlarkLC::OnNumActiveLCs);
		CreateProperty("Number of Active LCs", "0", MM::Integer, false, pAct, true);

		pAct = new CPropertyAction(this, &MeadowlarkLC::OnNumPalEls);
		CreateProperty("Total Number of Palette Elements", "5", MM::Integer, false, pAct, true);
	}
	else {
		std::string device = "None detected";
		AddAllowedValue(g_ControllerDevices, device.c_str());
		devNameList.push_back(device.c_str());
		SetProperty(g_ControllerDevices, device.c_str());
	}
}


MeadowlarkLC::~MeadowlarkLC()
{
	Shutdown();
}

void MeadowlarkLC::GetName(char* name) const
{
	CDeviceUtils::CopyLimitedString(name, g_ControllerName);
}

int MeadowlarkLC::Initialize()
{

	if (cur_dev < 1) {
		return DEVICE_NOT_CONNECTED;
	}

	if (serialnum_ == "Undefined" || serialnum_ == "") {
		return ERR_INVALID_SERIAL_NUMBER;
	}

	if (controllerLCType_ == "Undefined" || controllerLCType_ == "") {
		return ERR_INVALID_LCSERIAL_NUMBER;
	}

	if (numActiveLCs_ > numTotalLCs_) {
		std::string str = IntToString(numTotalLCs_).c_str();
		totalLCsMsg = "Invalid Active LCs. Number of Active LCs cannot be more than Total of " + str + " LCs";
		SetErrorText(ERR_INVALID_ACTIVE_LCS, totalLCsMsg.c_str());

		return ERR_INVALID_ACTIVE_LCS;
	}

	loadDefault();
	palEl_[numPalEls_];
	retardance_[numTotalLCs_];
	voltage_[numTotalLCs_];

	for (long i = 0; i < numTotalLCs_; ++i) {
		retardance_[i] = 0.5;
		voltage_[i] = 555;
	}

	for (long i = 0; i < numPalEls_; i++) {

		for (int i2 = 0; i2 < numActiveLCs_; i2++) {
			palette_[i][i2] = 0.25;
		}
		string str;
		for (int i2 = 0; i2 < numActiveLCs_; i2++) {
			str = str + " " + DoubleToString(retardance_[i2]);
		}
		palEl_[i] = DoubleToString(wavelength_) + str;
	}

	if (controllerType_ == g_ControllerDeviceType20V_D5020) {
		g_ControllerDeviceTypeVFac = g_ControllerDeviceType20VFac_D5020;
		tneAmplitude_ = 20;
	}
	else if (controllerType_ == g_ControllerDeviceType10V_D5020) {
		g_ControllerDeviceTypeVFac = g_ControllerDeviceType10VFac_D5020;
		tneAmplitude_ = 10;
	}

	// load curve based on LC Type
	controllerLcTypeChange();

	initialized_ = true;

	dev_Handle = USBDRVD_OpenDevice(cur_dev, flagsandattrs, USB_PID); // get the device handle

	// Name
	int ret = CreateProperty(MM::g_Keyword_Name, g_ControllerName, MM::String, true);
	if (DEVICE_OK != ret)
		return ret;

	// Select device #
	ret = CreateProperty("Device ID", IntToString(cur_dev).c_str(), MM::String, true);
	if (DEVICE_OK != ret)
		return ret;

	// Description
	ret = CreateProperty(MM::g_Keyword_Description, g_ControllerAbout, MM::String, true);
	if (DEVICE_OK != ret)
		return ret;

	// Description from device	
	ret = CreateProperty("Firmare Info", description_.c_str(), MM::String, true);
	if (DEVICE_OK != ret)
		return ret;

	// Controller Serial number
	CPropertyAction* pAct = new CPropertyAction(this, &MeadowlarkLC::OnSerialNumber);
	ret = CreateProperty("Controller Serial Number", "Serial Number Not Found", MM::String, true, pAct);
	if (ret != DEVICE_OK)
		return ret;

	// Device Adapter Version number
	pAct = new CPropertyAction(this, &MeadowlarkLC::OnDevAdapterVersionNumber);
	ret = CreateProperty("Device Adapter Version Number", "Device Adapter Version Number Not Found", MM::String, true, pAct);
	if (ret != DEVICE_OK)
		return ret;

	// Controller Type
	pAct = new CPropertyAction(this, &MeadowlarkLC::OnControllerType);
	ret = CreateProperty("Controller Type", controllerType_.c_str(), MM::String, true, pAct);
	if (ret != DEVICE_OK)
		return ret;

	// Controller LC Type
	pAct = new CPropertyAction(this, &MeadowlarkLC::OnControllerLCType);
	ret = CreateProperty("Controller LC Type", controllerLCType_.c_str(), MM::String, false, pAct);
	AddAllowedValue("Controller LC Type", g_ControllerLCType_F001);
	AddAllowedValue("Controller LC Type", g_ControllerLCType_Internal);
	if (ret != DEVICE_OK)
		return ret;

	// Wavelength
	pAct = new CPropertyAction(this, &MeadowlarkLC::OnWavelength);
	ret = CreateProperty("Wavelength", "546.0", MM::Float, false, pAct);
	if (ret != DEVICE_OK)
		return ret;
	SetPropertyLimits("Wavelength", 425., 700.);

	// Delay
	pAct = new CPropertyAction(this, &MeadowlarkLC::OnDelay);
	ret = CreateProperty("Device Delay (ms.)", "200.0", MM::Float, false, pAct);
	if (ret != DEVICE_OK)
		return ret;
	SetPropertyLimits("Device Delay (ms.)", 0.0, 200.0);

	// Temparature
	pAct = new CPropertyAction(this, &MeadowlarkLC::OnTemperature);
	ret = CreateProperty("Temperature (C)", "230.0", MM::Float, true, pAct);
	if (ret != DEVICE_OK)
		return ret;
	//SetPropertyLimits("Temperature", 0., 273.);

	// Active LC number
	pAct = new CPropertyAction(this, &MeadowlarkLC::OnNumActiveLCs);
	ret = CreateProperty("Active LCs", "0", MM::Integer, true, pAct);
	if (ret != DEVICE_OK)
		return ret;

	// Total LC number
	pAct = new CPropertyAction(this, &MeadowlarkLC::OnNumTotalLCs);
	ret = CreateProperty("Total LCs", "0", MM::Integer, true, pAct);
	if (ret != DEVICE_OK)
		return ret;

	// TNE Duration
	pAct = new CPropertyAction(this, &MeadowlarkLC::OnTneDuaration);
	ret = CreateProperty("TNE Duration (ms)", IntToString(tneDuration_).c_str(), MM::Integer, false, pAct);
	SetPropertyLimits("TNE Duration (ms)", 0, 255);
	SetProperty("TNE Duration (ms)", "0");
	if (ret != DEVICE_OK)
		return ret;

	// TNE Amplitude
	pAct = new CPropertyAction(this, &MeadowlarkLC::OnTneAmplitude);
	ret = CreateProperty("TNE Amplitude (V)", IntToString(tneAmplitude_).c_str(), MM::Float, false, pAct);

	SetPropertyLimits("TNE Amplitude (V)", 0.0, tneAmplitude_);
	//SetProperty("TNE Amplitude (V)", DoubleToString(tneAmplitude_).c_str());

	if (ret != DEVICE_OK)
		return ret;

	CPropertyActionEx* pActX = 0;
	//	 create an extended (i.e. array) properties 0 through 1	   

	// Voltage controls
	for (long i = 0; i < numActiveLCs_; ++i) {
		ostringstream s;
		s << "Voltage (V) LC-" << char(65 + i);
		pActX = new CPropertyActionEx(this, &MeadowlarkLC::OnVoltage, i);
		CreateProperty(s.str().c_str(), "7.500", MM::Float, false, pActX);

		if (controllerType_ == g_ControllerDeviceType10V_D5020) {
			SetPropertyLimits(s.str().c_str(), 0, 10.0);
		}
		else {
			SetPropertyLimits(s.str().c_str(), 0, 20.0);
		}
	}

	// Retardance controls -- after Voltage controls
	for (long i = 0; i < numActiveLCs_; ++i) {
		ostringstream s;
		s << "Retardance LC-" << char(65 + i) << " [in waves]";
		pActX = new CPropertyActionEx(this, &MeadowlarkLC::OnRetardance, i);
		CreateProperty(s.str().c_str(), "0.5", MM::Float, false, pActX);
		SetPropertyLimits(s.str().c_str(), g_ControllerDeviceRetardanceLowLimit, g_ControllerDeviceRetardanceHighLimit);
	}

	// Absolute Retardance controls -- after Voltage controls
	for (long i = 0; i < numActiveLCs_; ++i) {
		ostringstream s;
		s << "Retardance LC-" << char(65 + i) << " [in nm]";
		pActX = new CPropertyActionEx(this, &MeadowlarkLC::OnAbsRetardance, i);
		CreateProperty(s.str().c_str(), "100", MM::Float, true, pActX);
	}

	// Palettes
	for (long i = 0; i < numPalEls_; ++i) {
		ostringstream s;
		std::string number;

		std::stringstream strstream;
		strstream << i;
		strstream >> number;
		if (i < 10) {
			number = "0" + number;
		}

		s << "Pal. elem. " << number << "; enter 0 to define; 1 to activate";
		pActX = new CPropertyActionEx(this, &MeadowlarkLC::OnPalEl, i);
		CreateProperty(s.str().c_str(), "", MM::String, false, pActX);
	}

	pAct = new CPropertyAction(this, &MeadowlarkLC::OnSendToMeadowlarkLC);
	ret = CreateProperty("String send to -", "", MM::String, false, pAct);
	if (ret != DEVICE_OK)
		return ret;

	pAct = new CPropertyAction(this, &MeadowlarkLC::OnGetFromMeadowlarkLC);
	ret = CreateProperty("String from -", "", MM::String, true, pAct);
	if (ret != DEVICE_OK)
		return ret;

	importConfig();

	// Needed for Busy flag
	// changedTime_ = GetCurrentMMTime();
	SetErrorText(99, "Device set busy for ");

	return DEVICE_OK;
}


int MeadowlarkLC::Shutdown()
{
	USBDRVD_CloseDevice(dev_Handle);
	initialized_ = false;
	return DEVICE_OK;
}



//////////////// Action Handlers (MeadowlarkLC) /////////////////

int MeadowlarkLC::OnNumTotalLCs(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set(numTotalLCs_);
	}
	else if (eAct == MM::AfterSet)
	{
		pProp->Get(numTotalLCs_);
	}
	return DEVICE_OK;
}

int MeadowlarkLC::OnNumActiveLCs(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set(numActiveLCs_);
	}
	else if (eAct == MM::AfterSet)
	{
		pProp->Get(numActiveLCs_);
	}
	return DEVICE_OK;
}

int MeadowlarkLC::OnNumPalEls(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set(numPalEls_);
	}
	else if (eAct == MM::AfterSet)
	{
		pProp->Get(numPalEls_);
	}

	return DEVICE_OK;
}

int MeadowlarkLC::GetDesc(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set(description_.c_str());
	}
	else if (eAct == MM::AfterSet)
	{
		pProp->Get(description_);
	}
	return DEVICE_OK;
}

int MeadowlarkLC::OnNumDevices(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	std::string temp;
	temp = IntToString(cur_dev);

	if (eAct == MM::BeforeGet)
	{
		pProp->Set(temp.c_str());
	}
	else if (eAct == MM::AfterSet)
	{
		pProp->Get(temp);
	}


	if (temp == g_Nonedetected) {
		cur_dev = 0;
	}
	else {
		cur_dev = StringToInt(temp);
	}

	if (description_ == "Not found") {
		temp = g_Nonedetected;
		pProp->Set(temp.c_str());
	}

	return DEVICE_OK;
}

int MeadowlarkLC::OnTneDuaration(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{

		BYTE query_volt[] = { 't', 'n', 'e' , ':', char(49), ',', '?', '\r' };
		BYTE statusRead[64];

		USBDRVD_BulkWrite(dev_Handle, 1, query_volt, sizeof(query_volt)); /* send ld:n,? command */

		USBDRVD_BulkRead(dev_Handle, 0, statusRead, sizeof(statusRead)); /* read status response */

		std::string ans = ((char*)statusRead);

		size_t length;
		char buffer[64];
		int chN = 9;
		for (int j = 7; j < 10; j++) {
			if (ans[j] == ',') {
				chN = j;
			}
		}

		length = ans.copy(buffer, chN - 6, 6);
		buffer[length] = '\n';

		int tneP = StringToInt(buffer);

		if (tneP >= 0 && tneP < 256) {
			tneDuration_ = tneP;
		}

		int sizeC = ans.length();
		size_t length2;
		char buffer2[64];
		int chN2 = 9;
		for (int j2 = 7; j2 < 11; j2++) {
			if (ans[j2] == ',') {
				chN2 = j2 + 1;
			}
		}

		length2 = ans.copy(buffer2, sizeC - chN2, chN2);
		buffer2[length2] = '\n';

		int tneA = StringToInt(buffer2);

		if (tneA >= 0 && tneA <= 65535) {
			tneAmplitude_ = roundN(tneA / g_ControllerDeviceTypeVFac, 2);
		}

		pProp->Set(tneDuration_);
	}
	else if (eAct == MM::AfterSet)
	{
		pProp->Get(tneDuration_);

		int amp = roundN(tneAmplitude_ * g_ControllerDeviceTypeVFac, 1);

		SetTneToDevice(amp, tneDuration_);
	}
	return DEVICE_OK;
}

int MeadowlarkLC::OnTneAmplitude(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (initialized_) {
		if (eAct == MM::BeforeGet)
		{

			BYTE query_volt[] = { 't', 'n', 'e' , ':', char(49), ',', '?', '\r' };
			BYTE statusRead[64];

			USBDRVD_BulkWrite(dev_Handle, 1, query_volt, sizeof(query_volt)); /* send ld:n,? command */

			USBDRVD_BulkRead(dev_Handle, 0, statusRead, sizeof(statusRead)); /* read status response */

			std::string ans = ((char*)statusRead);

			size_t length;
			char buffer[64];
			int chN = 9;
			for (int j = 7; j < 10; j++) {
				if (ans[j] == ',') {
					chN = j;
				}
			}

			length = ans.copy(buffer, chN - 6, 6);
			buffer[length] = '\n';

			int tneP = StringToInt(buffer);

			if (tneP >= 0 && tneP < 256) {
				tneDuration_ = tneP;
			}

			int sizeC = ans.length();
			size_t length2;
			char buffer2[64];
			int chN2 = 9;
			for (int j2 = 7; j2 < 11; j2++) {
				if (ans[j2] == ',') {
					chN2 = j2 + 1;
				}
			}

			length2 = ans.copy(buffer2, sizeC - chN2, chN2);
			buffer2[length2] = '\n';

			int tneA = StringToInt(buffer2);

			if (tneA >= 0 && tneA <= 65535) {
				tneAmplitude_ = roundN(tneA / g_ControllerDeviceTypeVFac, 2);
			}

			pProp->Set(tneAmplitude_);
		}
		else if (eAct == MM::AfterSet)
		{
			pProp->Get(valTNE);

			tneAmplitude_ = valTNE * g_ControllerDeviceTypeVFac;

			SetTneToDevice(tneAmplitude_, tneDuration_);
		}
	}
	return DEVICE_OK;
}

int MeadowlarkLC::OnRetardance(MM::PropertyBase* pProp, MM::ActionType eAct, long index)
{
	if (eAct == MM::BeforeGet)
	{

		if (index + 1 > 0) 
		{
			double voltage = GetVoltage(index); // in mV
			//double voltage = voltage_[index];
			double maxV = 10000;
			if (controllerType_ == g_ControllerDeviceType20V_D5020) {
				maxV = 20000;
			}

			if (voltage >= 0 && voltage <= maxV) {
				double retardance = VoltageToRetardance(voltage, index);

				//if the new retardance is the same as the old retardance, then do nothing
				if (retardance_[index] == retardance) {
					return DEVICE_OK;
				}
				retardance_[index] = retardance;
				pProp->Set(retardance);

				//update abs retardance property
				ostringstream s;
				s << "Retardance LC-" << char(65 + index) << " [in nm]";
				std::string s2 = DoubleToString(retardance * wavelength_);
				SetProperty(s.str().c_str(), s2.c_str());
			}
		}
		else {
			return DEVICE_INVALID_PROPERTY_VALUE;
		}
	}
	else if (eAct == MM::AfterSet)
	{
		double retardance = retardance_[index];
		double retardanceT;

		// read value from property
		pProp->Get(retardanceT);

		boolean retLimitCheck = false;
		boolean voltLimitCheck = false;

		if (retardanceT * wavelength_ >= g_ControllerDeviceRetardanceAbsRetLow && retardanceT * wavelength_ <= g_ControllerDeviceRetardanceAbsRetHigh) {
			retLimitCheck = true;
		}

		// write retardance out to device....	  
		retardanceT = ceilf(retardanceT * 10000) / 10000;
		double voltage = round(RetardanceToVoltage(retardanceT, index));

		if (controllerType_ == g_ControllerDeviceType10V_D5020) {
			if (voltage >= 0 && voltage <= 10000) {
				voltLimitCheck = true;
			}
		}
		else {
			if (voltage >= 0 && voltage <= 20000) {
				voltLimitCheck = true;
			}
		}

		if (retLimitCheck && voltLimitCheck) {
			retardance = retardanceT;
		}

		retardance_[index] = retardance;

		if (voltage < 0 || voltage > 20000) {
			retardance_[index] = VoltageToRetardance(voltage, index);
		}
		if (voltage < 0) {
			voltage = 0;
		}
		else if (controllerType_ == g_ControllerDeviceType10V_D5020) {
			if (voltage > 10000) {
				voltage = 10000;
			}
		}
		else {
			if (voltage > 20000) {
				voltage = 20000;
			}
		}

		if (voltage_[index] == voltage) {
			return DEVICE_OK;
		}
		voltage_[index] = voltage;

		//update Voltage property
		ostringstream s;
		s << "Voltage (V) LC-" << char(65+index);
		std::string s2 = DoubleToString(voltage/1000); // in V
		SetProperty(s.str().c_str(), s2.c_str());

		// write voltage out to device....
		int volt16bit = static_cast<int>(voltage);
		SendVoltageToDevice(volt16bit, index);

		// changedTime_ = GetCurrentMMTime();
	}
	return DEVICE_OK;
}

int MeadowlarkLC::OnAbsRetardance(MM::PropertyBase* pProp, MM::ActionType eAct, long index)
{
	if (eAct == MM::BeforeGet)
	{
		if (index + 1 > 0) {
			pProp->Set(retardance_[index] * wavelength_);
		}
		else {
			return DEVICE_INVALID_PROPERTY_VALUE;
		}
	}
	else if (eAct == MM::AfterSet)
	{

	}
	return DEVICE_OK;
}

int MeadowlarkLC::OnVoltage(MM::PropertyBase* pProp, MM::ActionType eAct, long index) {

	if (initialized_) {
		if (eAct == MM::BeforeGet)
		{
			double voltage = GetVoltage(index); // in mV

			double maxV = 10000;
			if (controllerType_ == g_ControllerDeviceType20V_D5020) {
				maxV = 20000;
			}

			if (index + 1 >= 0 && voltage >= 0 && voltage <= maxV) {

				//if the new voltage is the same as the old voltage, then do nothing
				if (voltage_[index] == voltage) {
					return DEVICE_OK;
				}
				voltage_[index] = voltage;
				pProp->Set(voltage / 1000); // pProp is in V
			}
		}
		else if (eAct == MM::AfterSet)
		{
			double voltage;

			// read value from property, in Volts
			pProp->Get(voltage);
			voltage *= 1000; // now in mV

			double maxV = 10000;
			if (controllerType_ == g_ControllerDeviceType20V_D5020) {
				maxV = 20000;
			}

			if (index+1 >= 0 && voltage >= 0 && voltage <= maxV) {

				//if the new voltage is the same as the old voltage, then do nothing
				if (voltage_[index] == voltage) {
					return DEVICE_OK;
				}
				voltage_[index] = voltage;

				//send voltage to device
				double v = voltage / 1000 * g_ControllerDeviceTypeVFac;
				int volt16bit = static_cast<int>(v);
				SendVoltageToDevice(volt16bit, index);
			}
		}
	}
	return DEVICE_OK;
}


int MeadowlarkLC::OnWavelength(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		//pProp->Set(wavelength_);
		//ArrayLcVoltagesRetCurve[4][0] = wavelength_;
	}
	else if (eAct == MM::AfterSet)
	{
		double wavelength;
		// read value from property
		pProp->Get(wavelength);
		// write wavelength out to device....	  

		wavelength_ = wavelength;
		//ArrayLcVoltagesRetCurve[4][1] = wavelength_;
	}
	generateCurve();
	return DEVICE_OK;
}

int MeadowlarkLC::OnDelay(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		double delayT = GetDelayMs();
		double delay = delayT * 1000;
		pProp->Set(delayT);
	}
	else if (eAct == MM::AfterSet)
	{
		double delayT;
		pProp->Get(delayT);
		double delay = delayT * 1000;
		SetDelayMs(delayT);
	}

	return DEVICE_OK;
}

int MeadowlarkLC::OnSerialNumber(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set(serialnum_.c_str());
	}
	else if (eAct == MM::AfterSet)
	{
		pProp->Get(serialnum_);
	}
	return DEVICE_OK;
}

int MeadowlarkLC::OnDevAdapterVersionNumber(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set(g_Dev_Adapter_Ver);
	}
	return DEVICE_OK;
}

int MeadowlarkLC::OnControllerType(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set(controllerType_.c_str());
	}
	else if (eAct == MM::AfterSet)
	{
		pProp->Get(controllerType_);

		if (controllerType_ == g_ControllerDeviceType20V_D5020) {
			numTotalLCs_ = 2;
		}
		else {
			numTotalLCs_ = 4;
		}
		SetProperty(g_ControllerTotalLCs, IntToString(numTotalLCs_).c_str());
	}
	return DEVICE_OK;
}

int MeadowlarkLC::OnControllerLCType(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set(controllerLCType_.c_str());
		controllerLcTypeChange();
	}
	else if (eAct == MM::AfterSet)
	{
		pProp->Get(controllerLCType_);
	}
	return DEVICE_OK;
}

int MeadowlarkLC::OnTemperature(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (initialized_ == true) {
		if (eAct == MM::BeforeGet)
		{

			BYTE query_tmp[] = { 't', 'm', 'p', ':', '?', '\n' };
			BYTE status[64];

			USBDRVD_BulkWrite(dev_Handle, 1, query_tmp, sizeof(query_tmp)); /* send tmp:?, command */
			USBDRVD_BulkRead(dev_Handle, 0, status, sizeof(status)); /* read status response */

			std::string ans = ((char*)status);

			size_t length;
			char buffer[64];
			length = ans.copy(buffer, 5, 4);
			buffer[length] = '\0';

			int tempInt = StringToInt(buffer);
			double ret = (tempInt * 500 / 65535) - 273.15;

			pProp->Set(ret);
		}
		else if (eAct == MM::AfterSet)
		{
			double temp;
			// read value from property

			pProp->Get(temp);

			int tempInt = (temp * 273.15) * (16384 / 500);

			std::string tempStr = (string("tsp:") + IntToString(tempInt));

			std::vector<char> query_voltc(tempStr.begin(), tempStr.end());
			// write temp out to device....	

			BYTE status[64];
			BYTE query_tmp[sizeof(query_voltc)];

			int ii = 0;
			for (int i = 0; i < sizeof(query_voltc); i++) {
				query_tmp[i] = (char)query_voltc[i];
				ii = i;
			}

			query_tmp[ii + 1] = (BYTE)'\n';
			query_tmp[ii + 2] = (BYTE)'\0';

			USBDRVD_BulkWrite(dev_Handle, 1, query_tmp, sizeof(query_tmp)); /* send tsp:t command */

			USBDRVD_BulkRead(dev_Handle, 0, status, sizeof(status)); /* read status response */
			std::string ans = ((char*)status);

			temperature_ = temp;
		}
	}
	return DEVICE_OK;
}

int MeadowlarkLC::OnPalEl(MM::PropertyBase* pProp, MM::ActionType eAct, long index)
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set(palEl_[index].c_str());
	}
	else if (eAct == MM::AfterSet)
	{
		long setPalEl = 0;
		pProp->Get(setPalEl);

		if (setPalEl == 0) {
			string str = "";
			for (int i2 = 0; i2 < numActiveLCs_; i2++) {
				str = str + " " + DoubleToString(retardance_[i2]);
			}
			palEl_[index] = DoubleToString(wavelength_) + str;

			for (int i2 = 0; i2 < numActiveLCs_; i2++) {
				palette_[index][i2] = retardance_[i2];
			}

		}
		else {
			for (int i2 = 0; i2 < numActiveLCs_; i2++) {
				ostringstream s0;
				s0 << "Retardance LC-" << char(65 + i2) << " [in waves]";
				std::string s02 = DoubleToString(palette_[index][i2]);
				SetProperty(s0.str().c_str(), s02.c_str());
			}
		}
	}
	return DEVICE_OK;
}

int MeadowlarkLC::OnSendToMeadowlarkLC(MM::PropertyBase* pProp, MM::ActionType eAct)
{

	if (eAct == MM::BeforeGet)
	{
		//      pProp->Set(sendToMeadowlarkLC_.c_str());
	}
	else if (eAct == MM::AfterSet)
	{
		// read value from property
		pProp->Get(sendToMeadowlarkLC_);
		// write retardance out to device....  
		int len = strlen(sendToMeadowlarkLC_.c_str());
		char state[6];
		char p[2];

		std::string propstring = "String from -";
		bool searchQuestionMark = false;
		for (int q = 0; q < len; q++) {
			if (sendToMeadowlarkLC_[q] == 63) {
				searchQuestionMark = true;
			}
		}

		if (len > 5) {
			strncpy(state, sendToMeadowlarkLC_.c_str(), 5);
			state[5] = '\0';
		}

		strncpy(p, sendToMeadowlarkLC_.c_str(), 1);
		p[1] = '\0';


		if (!searchQuestionMark && ((std::string)state == "State" || (std::string)state == "state") && len < 8) {

			int index = 0;
			int number = 0;
			if (len == 6) {
				number = (sendToMeadowlarkLC_[5] - 48);
				index = number;
			}
			else {
				number = (sendToMeadowlarkLC_[5] - 48) * 10;
				index = number + (sendToMeadowlarkLC_[6] - 48) * 1;
			}

			string pre = "";
			if (index < 10) {
				pre = "0";
			}

			//changedTime_ = GetCurrentMMTime();			 

			ostringstream s0;
			s0 << "Pal. elem. " << pre << index << "; enter 0 to define; 1 to activate";

			SetProperty(s0.str().c_str(), "1");

			getFromMeadowlarkLC_ = sendToMeadowlarkLC_;
			SetProperty(propstring.c_str(), sendToMeadowlarkLC_.c_str());

		}
		else if (!searchQuestionMark && ((std::string)p == "P" || (std::string)p == "p") && len < 4) {

			int index = 0;
			int number = 0;
			if (len == 2) {
				number = (sendToMeadowlarkLC_[1] - 48);
				index = number;
			}
			else {
				number = (sendToMeadowlarkLC_[1] - 48) * 10;
				index = number + (sendToMeadowlarkLC_[2] - 48) * 1;
			}

			string pre = "";
			if (index < 10) {
				pre = "0";
			}

			//changedTime_ = GetCurrentMMTime();

			ostringstream s0;
			s0 << "Pal. elem. " << pre << index << "; enter 0 to define; 1 to activate";

			SetProperty(s0.str().c_str(), "1");

			getFromMeadowlarkLC_ = sendToMeadowlarkLC_;
			SetProperty(propstring.c_str(), sendToMeadowlarkLC_.c_str());

		}
		else if (!searchQuestionMark && ((std::string)p == "E" || (std::string)p == "e") && len < 4) {

			int numCycles = 0;
			int number = 0;
			if (len == 2) {
				number = (sendToMeadowlarkLC_[1] - 48);
				numCycles = number * 1;
			}
			else if (len == 3 && sendToMeadowlarkLC_[1] == ' ') {
				number = (sendToMeadowlarkLC_[2] - 48);
				numCycles = number * 1;
			}
			else {
				number = (sendToMeadowlarkLC_[1] - 48) * 10;
				numCycles = number + (sendToMeadowlarkLC_[2] - 48);
			}

			doExercise(numCycles * 10);

			//changedTime_ = GetCurrentMMTime();			 

			ostringstream sss; sss << "Excercising for " << number << " cycles please wait.";
			SetProperty(propstring.c_str(), sss.str().c_str());

			double oldVals[10];
			for (int i2 = 0; i2 < numActiveLCs_; i2++) {
				oldVals[i2] = retardance_[i2];
			}

			if ((sendToMeadowlarkLC_ != "R1" || sendToMeadowlarkLC_ != "R 1" || sendToMeadowlarkLC_ != "r1" || sendToMeadowlarkLC_ != "r 1")) {

				interruptExercise_ = true;

				for (int i2 = 0; i2 < numActiveLCs_; i2++) {
					ostringstream s0;
					s0 << "Retardance LC-" << char(65 + i2) << " [in waves]";
					SetProperty(s0.str().c_str(), "0.5");
				}
				Sleep(100);
				for (int i2 = 0; i2 < numActiveLCs_; i2++) {
					ostringstream s0;
					s0 << "Retardance LC-" << char(65 + i2) << " [in waves]";
					SetProperty(s0.str().c_str(), "1.5");
				}
				CDeviceUtils::SleepMs(100);

				interruptExercise_ = false;
			}

			for (int i2 = 0; i2 < numActiveLCs_; i2++) {
				ostringstream s0;
				s0 << "Retardance LC-" << char(65 + i2) << " [in waves]";
				SetProperty(s0.str().c_str(), DoubleToString(oldVals[i2]).c_str());
			}

			getFromMeadowlarkLC_ = sendToMeadowlarkLC_;
			SetProperty(propstring.c_str(), sendToMeadowlarkLC_.c_str());

		}
		else if ((sendToMeadowlarkLC_ == "R?" || sendToMeadowlarkLC_ == "R ?" || sendToMeadowlarkLC_ == "r?" || sendToMeadowlarkLC_ == "r ?")) {

			sendToMeadowlarkLC_ = "R0";
			getFromMeadowlarkLC_ = sendToMeadowlarkLC_;
			SetProperty(propstring.c_str(), sendToMeadowlarkLC_.c_str());

		}
		else if (!searchQuestionMark && ((std::string)p == "L" || (std::string)p == "l")) {

			std::string myStr = "L " + sendToMeadowlarkLC_.substr(1, len);
			vector<double> numbers = getNumbersFromMessage(myStr, briefModeQ_);
			//changedTime_ = GetCurrentMMTime();			 

			for (int i2 = 0; i2 < numActiveLCs_; i2++) {
				ostringstream s0;
				s0 << "Retardance LC-" << char(65 + i2) << " [in waves]";
				SetProperty(s0.str().c_str(), DoubleToString(numbers[i2]).c_str());
			}

			getFromMeadowlarkLC_ = sendToMeadowlarkLC_;
			SetProperty(propstring.c_str(), sendToMeadowlarkLC_.c_str());

		}
		else if (!searchQuestionMark && ((std::string)p == "D" || (std::string)p == "d") && len < 4) {

			int index = 0;
			int number = 0;
			if (len == 2) {
				number = (sendToMeadowlarkLC_[1] - 48);
				index = number;
			}
			else {
				number = (sendToMeadowlarkLC_[1] - 48) * 10;
				index = number + (sendToMeadowlarkLC_[2] - 48) * 1;
			}

			string pre = "";
			if (index < 10) {
				pre = "0";
			}

			//changedTime_ = GetCurrentMMTime();

			ostringstream s0;
			s0 << "Pal. elem. " << pre << index << "; enter 0 to define; 1 to activate";

			SetProperty(s0.str().c_str(), "0");

			getFromMeadowlarkLC_ = sendToMeadowlarkLC_;
			SetProperty(propstring.c_str(), sendToMeadowlarkLC_.c_str());

		}
		else if (!searchQuestionMark && len == 5 && ((sendToMeadowlarkLC_.substr(0, 2) == "SP") || (sendToMeadowlarkLC_.substr(0, 2) == "sp"))) {

			int w = StringToInt(sendToMeadowlarkLC_.substr(2));

			for (int i = 0; i < 250; i++)
			{
				if (ConfigData[i][0] == w) {
					// Palettes
					SetProperty("Wavelength", DoubleToString(ConfigData[i][0]).c_str());

					for (long i = 0; i < numPalEls_; ++i) {
						ostringstream s;
						std::string number;

						std::stringstream strstream;
						strstream << i;
						strstream >> number;
						if (i < 10) {
							number = "0" + number;
						}

						for (int i2 = 0; i2 < numActiveLCs_; i2++) {
							ostringstream s0;
							s0 << "Retardance LC-" << char(65 + i2) << " [in waves]";
							SetProperty(s0.str().c_str(), DoubleToString(ConfigData[i][i2 + 1]).c_str());
						}

						s << "Pal. elem. " << number << "; enter 0 to define; 1 to activate";
						SetProperty(s.str().c_str(), "0");
					}
				}
			}

		}
		else if (!searchQuestionMark && ((std::string)p == "W" || (std::string)p == "w")) {

			std::string myStr = sendToMeadowlarkLC_.substr(1, len).c_str();
			vector<double> numbers = getNumbersFromMessage(myStr, briefModeQ_);
			//changedTime_ = GetCurrentMMTime();			 

			SetProperty("Wavelength", DoubleToString(numbers[0]).c_str());

			getFromMeadowlarkLC_ = "W " + sendToMeadowlarkLC_;
			SetProperty(propstring.c_str(), sendToMeadowlarkLC_.c_str());

		}
		else if (searchQuestionMark && (sendToMeadowlarkLC_ == "W ?" || sendToMeadowlarkLC_ == "W?" || sendToMeadowlarkLC_ == "w ?" || sendToMeadowlarkLC_ == "w?")) {

			getFromMeadowlarkLC_ = "W " + DoubleToString(wavelength_);
			SetProperty(propstring.c_str(), getFromMeadowlarkLC_.c_str());

		}
		else if (searchQuestionMark && (sendToMeadowlarkLC_ == "L ?" || sendToMeadowlarkLC_ == "L?" || sendToMeadowlarkLC_ == "l ?" || sendToMeadowlarkLC_ == "l?")) {

			string str = "";
			for (int i2 = 0; i2 < numActiveLCs_; i2++) {
				str = str + " " + DoubleToString(retardance_[i2]);
			}

			getFromMeadowlarkLC_ = "L " + str;
			SetProperty(propstring.c_str(), getFromMeadowlarkLC_.c_str());

		}
		else if (searchQuestionMark && (sendToMeadowlarkLC_ == "V ?" || sendToMeadowlarkLC_ == "V?" || sendToMeadowlarkLC_ == "v?" || sendToMeadowlarkLC_ == "v?")) {

			getFromMeadowlarkLC_ = description_;
			SetProperty(propstring.c_str(), description_.c_str());

		}

		else if (!searchQuestionMark && ((sendToMeadowlarkLC_ == "import") || sendToMeadowlarkLC_ == "IMPORT")) {

			loadDefault();
			import(controllerLCType_Curve);

		}
		else if (!searchQuestionMark && ((sendToMeadowlarkLC_ == "export") || sendToMeadowlarkLC_ == "EXPORT")) {

			exportCurve();

		}
		else if (!searchQuestionMark && ((sendToMeadowlarkLC_ == "generate") || sendToMeadowlarkLC_ == "GENERATE")) {

			generateCurve();

		}
		else if (!searchQuestionMark && ((sendToMeadowlarkLC_ == "exportloaded") || sendToMeadowlarkLC_ == "EXPORTLOADED")) {

			exportloadedCurve();

		}
		else if (!searchQuestionMark && ((sendToMeadowlarkLC_ == "importdefault") || sendToMeadowlarkLC_ == "IMPORTDEFAULT")) {

			loadDefault();

		}
		else if (!searchQuestionMark && (sendToMeadowlarkLC_ == "")) {

			SetProperty(propstring.c_str(), "");

		}
		else {
			getFromMeadowlarkLC_ = sendToMeadowlarkLC_;
		}
	}
	return DEVICE_OK;
}

int MeadowlarkLC::OnGetFromMeadowlarkLC(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		//   GetSerialAnswer (port_.c_str(), "\r", getFromMeadowlarkLC_);
		pProp->Set(getFromMeadowlarkLC_.c_str());
	}
	else if (eAct == MM::AfterSet)
	{

	}
	return DEVICE_OK;
}

bool MeadowlarkLC::Busy()
{
	if (delay.getMsec() > 0.0) {
		MM::MMTime interval = GetCurrentMMTime() - changedTime_;
		if (interval.getMsec() < delay.getMsec()) {
			return true;
		}
	}

	return false;
}


std::vector<double> MeadowlarkLC::getNumbersFromMessage(std::string variLCmessage, bool briefMode) {
	std::istringstream variStream(variLCmessage);
	std::string prefix;
	double val;
	std::vector<double> values;

	if (!briefMode) {
		variStream >> prefix;
	}
	for (;;) {
		variStream >> val;
		if (!variStream.fail()) {
			values.push_back(val);
		}
		else {
			break;
		}
	}

	return values;
}

std::string  MeadowlarkLC::IntToString(int N)
{
	ostringstream ss("");
	ss << N;
	return ss.str();
}

std::string  MeadowlarkLC::DoubleToString(double N)
{
	ostringstream ss("");
	ss << N;
	return ss.str();
}

int MeadowlarkLC::StringToInt(std::string str)
{
	std::istringstream variStream(str);
	int val;

	variStream >> val;

	return val;
}

double MeadowlarkLC::StringToDouble(std::string str)
{
	std::istringstream variStream(str);
	double val;

	variStream >> val;

	return val;
}

std::string MeadowlarkLC::RemoveChars(const std::string& source, const std::string& chars) {
	std::string result = "";
	for (unsigned int i = 0; i < source.length(); i++) {
		bool foundany = false;
		for (unsigned int j = 0; j < chars.length() && !foundany; j++) {
			foundany = (source[i] == chars[j]);
		}
		if (!foundany) {
			result += source[i];
		}
	}
	return result;
}

void MeadowlarkLC::hexconvert(char* text, unsigned char bytes[])
{
	int i;
	int temp;

	for (i = 0; i < 4; ++i) {
		sscanf(text + 2 * i, "%2x", &temp);
		bytes[i] = temp;
	}
}

vector<unsigned char> intToBytes(int paramInt)
{
	vector<unsigned char> arrayOfByte(4);
	for (int i = 0; i < 4; i++)
		arrayOfByte[3 - i] = (paramInt >> (i * 8));
	return arrayOfByte;
}

double MeadowlarkLC::round(double number)
{
	return number < 0.0 ? ceil(number - 0.5) : floor(number + 0.5);
}

double MeadowlarkLC::roundN(double val, int precision)
{
	std::stringstream s;
	s << std::setprecision(precision) << std::setiosflags(std::ios_base::fixed) << val;
	s >> val;
	return val;
}

double MeadowlarkLC::GetVoltage(long index) {
	// returns voltage in mV

	double voltage = -1;

	if (index + 1 > 0) {
		BYTE query_volt[] = { 'l', 'd', ':', char(index + 49), ',', '?', '\r' };
		BYTE statusRead[64];

		USBDRVD_BulkWrite(dev_Handle, 1, query_volt, sizeof(query_volt)); /* send ld:n,? command */
		USBDRVD_BulkRead(dev_Handle, 0, statusRead, sizeof(statusRead)); /* read status response */

		std::string ans = ((char*)statusRead);

		size_t length;
		char buffer[64];
		length = ans.copy(buffer, 5, 5);
		buffer[length] = '\n';

		voltage = round(StringToInt(buffer) * 1000 / g_ControllerDeviceTypeVFac); // in mV
	}

	return voltage;
}

double MeadowlarkLC::VoltageToRetardance(double volt, long index) {
	//TODO: voltage to retardance conversion does not work for low voltages (<2 V)

	double retardance = 0;
	double FacRetardance = 0;

	double myRetardance = 0;
	double diff = 0;

	double upper = 0;
	double lower = 0;

	for (int i = 0; i < ArrayLength2 - 1; i++) {

		if ((volt > round(ArrayLcVoltagesRetCurve[i][0]) && volt < round(ArrayLcVoltagesRetCurve[i + 1][0]))) {
			if (i == 0) {
				FacRetardance = 1 - (round(ArrayLcVoltagesRetCurve[0][0]) / volt);
				myRetardance = ArrayLcVoltagesRetCurve[0][1 + index] + ((ArrayLcVoltagesRetCurve[0][1 + index] - ArrayLcVoltagesRetCurve[0][1 + index]) * FacRetardance);
				break;
			}
			else if (i == ArrayLength2 - 1) {
				FacRetardance = 1 - (round(ArrayLcVoltagesRetCurve[ArrayLength2 - 1][0]) / volt);
				diff = ((round(ArrayLcVoltagesRetCurve[ArrayLength2 - 1][0]) - round(ArrayLcVoltagesRetCurve[ArrayLength2 - 2][0])) * FacRetardance);

				myRetardance = ArrayLcVoltagesRetCurve[ArrayLength2 - 2][1 + index] + diff;
				break;
			}
			else {
				upper = ArrayLcVoltagesRetCurve[i][1 + index];
				lower = ArrayLcVoltagesRetCurve[i + 1][1 + index];

				FacRetardance = ArrayLcVoltagesRetCurve[i][0] / volt;
				diff = upper - lower;

				double v1 = ArrayLcVoltagesRetCurve[i + 1][0] - volt;
				double v2 = volt - ArrayLcVoltagesRetCurve[i][0];
				double totDiff = ArrayLcVoltagesRetCurve[i + 1][0] - ArrayLcVoltagesRetCurve[i][0];

				if (v1 < v2) {
					double fac = v1 / totDiff;
					myRetardance = lower + (diff * fac);
					retardance = myRetardance / wavelength_;
				}
				else {
					double fac = v2 / totDiff;
					myRetardance = upper - (diff * fac);
					retardance = myRetardance / wavelength_;
				}
				return retardance;
			}
		}
		else if (ArrayLcVoltagesRetCurve[i][0] == 20000) {
			myRetardance = ArrayLcVoltagesRetCurve[i][1 + index];
		}
	}

	retardance = myRetardance / wavelength_;

	return retardance;
}

double MeadowlarkLC::RetardanceToVoltage(double retard, long index) {
	// retardance to voltage conversion does not work for high retardances (>= 1.6).
	// voltage cannot be set to Vmax

	double voltage = 0;
	double FacRetardance = 0;
	double diff = 0;
	double myAbsRetardance = (wavelength_ * retard);

	double upper = 0;
	double lower = 0;

	for (int i = 0; i < ArrayLength2; i++) {
		if (myAbsRetardance >= ArrayLcVoltagesRetCurve[i][1 + index]) {
			if (i == 0) {
				FacRetardance = 1 - (ArrayLcVoltagesRetCurve[0][1 + index] / myAbsRetardance);
				voltage = round(ArrayLcVoltagesRetCurve[0][0]) - ((round(ArrayLcVoltagesRetCurve[1][0]) - round(ArrayLcVoltagesRetCurve[0][0])) * FacRetardance);
				break;
			}
			else if (i == ArrayLength2 - 1) {
				FacRetardance = 1 - (ArrayLcVoltagesRetCurve[ArrayLength2 - 1][1 + index] / myAbsRetardance);
				voltage = round(ArrayLcVoltagesRetCurve[ArrayLength2 - 1][0]) - ((round(ArrayLcVoltagesRetCurve[ArrayLength2 - 1][0]) - round(ArrayLcVoltagesRetCurve[ArrayLength2 - 2][0])) * FacRetardance);
				break;
			}
			else {
				lower = ArrayLcVoltagesRetCurve[i][0];
				upper = ArrayLcVoltagesRetCurve[i - 1][0];

				if (upper == 20000) {
					return 20001; // instruct out of range
				}

				FacRetardance = myAbsRetardance / ArrayLcVoltagesRetCurve[i][1 + index];
				diff = lower - upper;

				double v1 = ArrayLcVoltagesRetCurve[i - 1][1 + index] - myAbsRetardance;
				double v2 = myAbsRetardance - ArrayLcVoltagesRetCurve[i][1 + index];
				double totDiff = ArrayLcVoltagesRetCurve[i - 1][1 + index] - ArrayLcVoltagesRetCurve[i][1 + index];

				if (v1 > v2) {
					double fac = v1 / totDiff;
					voltage = upper + (diff * fac);
				}
				else {
					double fac = v2 / totDiff;
					voltage = lower - (diff * fac);
				}

				break;
			}
		}
		else if (ArrayLcVoltagesRetCurve[i][0] == 20000 && myAbsRetardance < ArrayLcVoltagesRetCurve[i][1 + index]) {
			return 20001; // instruct out of range
		}
	}

	return round(voltage);
}


void MeadowlarkLC::SendVoltageToDevice(int volt16bit, long index) {

	if (index + 1 > 0 && volt16bit >= 0 && volt16bit <= 65535) {

		std::string voltStr = IntToString(volt16bit);
		std::vector<char> query_voltc(voltStr.begin(), voltStr.end());

		query_voltc.push_back('\0');

		// write temp out to device....				  	  			  
		BYTE status[64];

		BYTE a1 = '\r';
		BYTE a2 = '\r';
		BYTE a3 = '\r';
		BYTE a4 = '\r';
		BYTE a5 = '\r';
		BYTE a6 = '\r';

		int sizeC = voltStr.length();
		int s = 0;

		switch (sizeC) {
		case 5:
			a1 = voltStr[0];
			a2 = voltStr[1];
			a3 = voltStr[2];
			a4 = voltStr[3];
			a5 = voltStr[4];
			break;
		case 4:
			a1 = '0';
			a2 = voltStr[0];
			a3 = voltStr[1];
			a4 = voltStr[2];
			a5 = voltStr[3];
			s = 4;
			break;
		case 3:
			a1 = '0';
			a2 = '0';
			a3 = voltStr[0];
			a4 = voltStr[1];
			a5 = voltStr[2];
			s = 8;
			break;
		case 2:
			a1 = '0';
			a2 = '0';
			a3 = '0';
			a4 = voltStr[0];
			a5 = voltStr[1];
			s = 12;
			break;
		case 1:
			a1 = '0';
			a2 = '0';
			a3 = '0';
			a4 = '0';
			a5 = voltStr[0];
			s = 16;
			break;
		}

		BYTE test_query_volt[] = { 'l', 'd', ':', char(index + 49), ',' ,char(a1),char(a2),char(a3),char(a4),char(a5),a6 };


		USBDRVD_BulkWrite(dev_Handle, 1, test_query_volt, sizeof(test_query_volt)); /* send ld:n,i command */

		USBDRVD_BulkRead(dev_Handle, 0, status, sizeof(status)); /* read status response */

		/* output status until a <CR> is found */

		// Needed for Busy flag
		changedTime_ = GetCurrentMMTime();
	}
}

void MeadowlarkLC::SetTneToDevice(int amplitude, double duration) {

	if (tneDuration_ >= 0 && tneDuration_ <= 255) {

		std::string tneStrD = IntToString(duration);

		// write temp out to device....				  	  			  
		BYTE status[64];

		BYTE a1 = '\r';
		BYTE a2 = '\r';
		BYTE a3 = '\r';
		BYTE a4 = '\r';

		int sizeC = tneStrD.length();

		switch (sizeC) {
		case 3:
			a1 = tneStrD[0];
			a2 = tneStrD[1];
			a3 = tneStrD[2];
			break;
		case 2:
			a1 = '0';
			a2 = tneStrD[0];
			a3 = tneStrD[1];
			break;
		case 1:
			a1 = '0';
			a2 = '0';
			a3 = tneStrD[0];
			break;
		}

		std::string tneStrA = IntToString(amplitude);

		BYTE b1 = '\r';
		BYTE b2 = '\r';
		BYTE b3 = '\r';
		BYTE b4 = '\r';
		BYTE b5 = '\r';

		sizeC = tneStrA.length();

		switch (sizeC) {
		case 5:
			b1 = tneStrA[0];
			b2 = tneStrA[1];
			b3 = tneStrA[2];
			b4 = tneStrA[3];
			b5 = tneStrA[4];
			break;
		case 4:
			b1 = '0';
			b2 = tneStrA[0];
			b3 = tneStrA[1];
			b4 = tneStrA[2];
			b5 = tneStrA[3];
			break;
		case 3:
			b1 = '0';
			b2 = '0';
			b3 = tneStrA[0];
			b4 = tneStrA[1];
			b5 = tneStrA[2];
			break;
		case 2:
			b1 = '0';
			b2 = '0';
			b3 = '0';
			b4 = tneStrA[0];
			b5 = tneStrA[1];
			break;
		case 1:
			b1 = '0';
			b2 = '0';
			b3 = '0';
			b4 = '0';
			b5 = tneStrA[0];
			break;
		}

		BYTE test_query_tne[] = { 't', 'n', 'e', ':', char(0 + 49), ',' ,char(a1),char(a2),char(a3), ',',char(b1),char(b2),char(b3),char(b4),char(b5), a4 };
		USBDRVD_BulkWrite(dev_Handle, 1, test_query_tne, sizeof(test_query_tne)); /* send ld:n,i command */
		USBDRVD_BulkRead(dev_Handle, 0, status, sizeof(status)); /* read status response */

		BYTE test_query_tne2[] = { 't', 'n', 'e', ':', char(1 + 49), ',' ,char(a1),char(a2),char(a3), ',',char(b1),char(b2),char(b3),char(b4),char(b5), a4 };
		USBDRVD_BulkWrite(dev_Handle, 1, test_query_tne2, sizeof(test_query_tne2)); /* send ld:n,i command */
		USBDRVD_BulkRead(dev_Handle, 0, status, sizeof(status)); /* read status response */

		/* output status until a <CR> is found */

		//changedTime_ = GetCurrentMMTime();
	}
}

void MeadowlarkLC::doExercise(int seconds) {


	for (int i = 0; i < seconds; i++) {
		SendVoltageToDevice(0, 0);
		SendVoltageToDevice(0, 1);

		CDeviceUtils::SleepMs(500);

		if (interruptExercise_) {
			break;
		}

		SendVoltageToDevice(65535, 0);
		SendVoltageToDevice(65535, 1);
	}

	interruptExercise_ = false;
	//changedTime_ = GetCurrentMMTime();
}

void  MeadowlarkLC::exportCurve() {
	ostringstream s;
	s << serialnum_ << "_MeadowlarkLcCalib_Export_" << wavelength_ << ".csv";
	ofstream arrayData(s.str()); // File Creation(on C drive) // overwrite

	//Outputs array to txtFile
	arrayData << "Volt(mv),LC-A,LC-B" << endl;
	arrayData << "-,-,-" << endl;
	arrayData << "0," << wavelength_ << "," << wavelength_ << endl;

	for (int i = 0; i < ArrayLength2 - 1; i++)
	{
		arrayData << ArrayLcVoltagesRetCurve[i][0] << "," << ArrayLcVoltagesRetCurve[i][1] << "," << ArrayLcVoltagesRetCurve[i][2] << endl; //Outputs array to txtFile
	}
	arrayData << "-,-,-" << endl;

	arrayData.close();
}

void  MeadowlarkLC::exportConfig() {
	ostringstream s;
	s << "MeadowlarkLc_" << serialnum_ << "_Config.csv";
	ofstream arrayData(s.str()); // File Creation(on C drive) // overwrite

	//Outputs array to txtFile
	arrayData << "Edit or Define new Palettes. 1st is used as default when Micro-Manager loads. Please keep starting/ending delimiters \"-\" intact." << endl;
	arrayData << "Wavelength,LC-A,LC-B" << endl;
	arrayData << "-,-,-" << endl;

	if (ConfigData[0][0] == 0) {
		for (int i = 0; i < 5; i++)
		{
			arrayData << roundN(ConfigDataDefaults[i][0], 2) << "," << roundN(ConfigDataDefaults[i][1], 2) << "," << roundN(ConfigDataDefaults[i][2], 2) << endl; //Outputs array to txtFile
		}

	}
	else {
		for (int i = 0; i < 250; i++)
		{
			if (ConfigData[i][0] == 0) {
				break;
			}
			else {
				arrayData << ConfigData[i][0] << "," << ConfigData[i][1] << "," << ConfigData[i][2] << endl; //Outputs array to txtFile
			}
		}
	}

	arrayData << "-,-,-" << endl;

	arrayData.close();
}

void  MeadowlarkLC::exportloadedCurve() {
	ostringstream s;
	s << serialnum_ << "_MeadowlarkLcCalib_Export_Loaded.csv";
	ofstream arrayData(s.str()); // File Creation(on C drive) // overwrite

	//Outputs array to txtFile
	arrayData << "Volt(mv),LC-A,LC-B,Volt(mv),LC-A,LC-B,Volt(mv),LC-A,LC-B" << endl;
	arrayData << "-,-,-,-,-,-,-,-,-" << endl;

	for (int r = 0; r < ArrayLength2 - 1; r++)
		for (int c = 0; c < numberofCurves; c++) {
			{
				if (c == 0 && r == 0) {
					for (int w = 0; w < numberofCurves; w++) {
						if (w == numberofCurves - 1) {
							arrayData << Wavelengths[w] << endl;
						}
						else {
							arrayData << Wavelengths[w] << ",";
						}
					}
				}
				if (c == numberofCurves - 1) {
					arrayData << ArrayLcVoltagesRetLoaded[r][c] << "," << endl;
				}
				else {
					arrayData << ArrayLcVoltagesRetLoaded[r][c] << ","; //Outputs array to txtFile
				}
			}
		}
	arrayData << "-,-,-" << endl;

	arrayData.close();
}

void  MeadowlarkLC::clearConfig() {

	for (int i = 0; i < 250; i++)
	{
		for (int i2 = 0; i2 < 3; i2++) {
			ConfigData[i][i2] == 0;
		}
	}
}

void MeadowlarkLC::convertStringtoStringArray(std::string str) {

	stringstream infile(str);

	vector <vector <string> > data;
	int c = 0;
	bool start = false;
	bool end = false;
	bool hasDeterminedNumberOfWavelength = false;

	while (infile.good())
	{

		string s;
		if (!getline(infile, s)) break;

		if (s[0] == '-' && !start) {
			start = true;
		}
		else if (s[0] == '-' && !end) {
			end = true;
		}

		if (!hasDeterminedNumberOfWavelength && start && s[0] != '-') {
			istringstream ss(s);
			vector <string> record;

			int c2 = 0;
			while (ss)
			{
				string s;
				if (!getline(ss, s, ',')) break;
				record.push_back(s);
				Wavelengths[c2] = StringToDouble(s);
				c2++;
			}

			numberofCurves = (c2) / 3; // 3 columns used by each curve
			hasDeterminedNumberOfWavelength = true;

		}
		else if (start && hasDeterminedNumberOfWavelength && !end) {
			istringstream ss(s);
			vector <string> record;

			int c2 = 0;
			while (ss)
			{
				string s;
				if (!getline(ss, s, ',')) break;
				record.push_back(s);
				ArrayLcVoltagesRetLoaded[c][c2] = StringToDouble(s);
				c2++;
			}

			data.push_back(record);
			c++;

			ArrayLength2 = c + 1;
		}

		generateCurve();
	}

	if (!infile.eof())
	{
		cerr << "End of Line!\n";
	}
}

void  MeadowlarkLC::import(std::string calibCurveFilename) {

	if (!checkCalibFile(calibCurveFilename)) { // check file exists and is valid // ToDo: Better check conditions
		return;
	}

	vector <vector <string> > data;
	ifstream infile(calibCurveFilename);
	int c = 0;
	bool start = false;
	bool end = false;
	bool hasDeterminedNumberOfWavelength = false;

	while (infile)
	{

		string s;
		if (!getline(infile, s)) break;

		if (s[0] == '-' && !start) {
			start = true;
		}
		else if (s[0] == '-' && !end) {
			end = true;
		}

		if (!hasDeterminedNumberOfWavelength && start && s[0] != '-') {
			istringstream ss(s);
			vector <string> record;

			int c2 = 0;
			while (ss)
			{
				string s;
				if (!getline(ss, s, ',')) break;
				record.push_back(s);
				Wavelengths[c2] = StringToDouble(s);
				c2++;
			}

			numberofCurves = (c2) / 3; // 3 columns used by each curve
			hasDeterminedNumberOfWavelength = true;

		}
		else if (start && hasDeterminedNumberOfWavelength && !end) {
			istringstream ss(s);
			vector <string> record;

			int c2 = 0;
			while (ss)
			{
				string s;
				if (!getline(ss, s, ',')) break;
				record.push_back(s);
				ArrayLcVoltagesRetLoaded[c][c2] = StringToDouble(s);
				c2++;
			}

			data.push_back(record);
			c++;

			ArrayLength2 = c + 1;
		}

		generateCurve();
	}

	if (!infile.eof())
	{
		cerr << "End of Line!\n";
	}
}

void  MeadowlarkLC::importConfig() {

	ostringstream s;
	s << "MeadowlarkLc_" << serialnum_ << "_Config.csv";
	string configFilename = s.str();

	if (!checkConfigFile(configFilename)) { // check file exists and is valid
		exportConfig();
		return;
	}

	clearConfig();

	vector <vector <string> > data;
	ifstream infile(configFilename);
	int c = 0;
	bool start = false;
	bool end = false;
	bool skip = true;

	while (infile)
	{

		string s;
		if (!getline(infile, s)) break;

		if (s[0] == '-' && !start) {
			start = true; skip = true;
		}
		else if (s[0] == '-' && !end) {
			end = true; skip = true;
		}
		else {
			skip = false;
		}

		if (start && !end && !skip) {
			istringstream ss(s);
			vector <string> record;

			int c2 = 0;
			while (ss)
			{
				string s;
				if (!getline(ss, s, ',')) break;
				record.push_back(s);
				ConfigData[c][c2] = StringToDouble(s);
				c2++;
			}

			data.push_back(record);
			c++;
		}
	}

	if (!infile.eof())
	{
		cerr << "End of Line!\n";
	}

	// Palettes
	for (long i = 0; i < numPalEls_; ++i) {
		ostringstream s;
		std::string number;

		std::stringstream strstream;
		strstream << i;
		strstream >> number;
		if (i < 10) {
			number = "0" + number;
		}

		SetProperty("Wavelength", DoubleToString(ConfigData[i][0]).c_str());

		for (int i2 = 0; i2 < numActiveLCs_; i2++) {
			ostringstream s0;
			s0 << "Retardance LC-" << char(65 + i2) << " [in waves]";
			SetProperty(s0.str().c_str(), DoubleToString(ConfigData[i][i2 + 1]).c_str());
		}

		s << "Pal. elem. " << number << "; enter 0 to define; 1 to activate";
		SetProperty(s.str().c_str(), "0");
	}
}

bool MeadowlarkLC::checkCalibFile(std::string calibCurveFilename) {
	vector <vector <string> > data;
	ifstream infile(calibCurveFilename);

	if (!infile) {
		controllerLCType_ = g_ControllerLCType_Internal;
		return false;
	}

	string s0;
	if (!getline(infile, s0));
	if ((s0[0] != 'V' || s0[s0.length() - 1] != 'B') && (s0[0] != 'L')) { // starting line of calib file has words Voltage and ends with 'B'
		controllerLCType_ = g_ControllerLCType_Internal;
		return false;
	}

	string s; int c = 0;
	while (infile) {
		if (!getline(infile, s)) {
			break;
		}
		else {
			c++;
			s0 = s;
		};
	}
	if (s0[0] != '-' || s0[s0.length() - 1] != '-') { // EOF has '-'
		controllerLCType_ = g_ControllerLCType_Internal;
		return false;
	}
	if (c < 100) { // calibration file is expected to have more than 100 voltage points (Meadowlark provides files with min. 197 points)
		controllerLCType_ = g_ControllerLCType_Internal;
		return false;
	}

	return true;
}

bool MeadowlarkLC::checkConfigFile(std::string configFilename) {
	vector <vector <string> > data;
	ifstream infile(configFilename);

	string s; int c = 0;
	while (infile) {
		if (!getline(infile, s)) {
			break;
		}
		else {
			c++;
		};
	}

	if (c < 4) {
		return false;
	}

	return true;
}

void MeadowlarkLC::controllerLcTypeChange() {
	// Add multiple curves switch and load from resource ID
	//
	if (controllerLCType_ == g_ControllerLCType_F001) {
		controllerLCType_Curve = g_ControllerLCType_F001_Curves;
		loadDefault();
		import(controllerLCType_Curve);
	}
	else if (controllerLCType_ == g_ControllerLCType_Internal) {
		loadDefault();
	}
	generateCurve();
}

void  MeadowlarkLC::generateCurve() {

	int curve = 1; boolean lowerThanCurve = false; boolean higherThanCurve = false; boolean isCurve = false; int maxCurves = (numberofCurves * 3) - 1;

	if (numberofCurves == 1) {
		for (int i = 0; i < ArrayLength2; i++) {
			ArrayLcVoltagesRetCurve[i][0] = ArrayLcVoltagesRetLoaded[i][0];
			ArrayLcVoltagesRetCurve[i][1] = ArrayLcVoltagesRetLoaded[i][1];
			ArrayLcVoltagesRetCurve[i][2] = ArrayLcVoltagesRetLoaded[i][2];
		}
	}
	else {
		for (int c = 0; c < (numberofCurves * 3); c++) {
			if ((int)wavelength_ == (int)Wavelengths[c + 1]) {
				isCurve = true; curve = c; break;
			}
			else if (c > 0 && (int)wavelength_ > (int)Wavelengths[c - 1] && (int)wavelength_ < (int)Wavelengths[c + 1] && c < (numberofCurves * 3)) { // within
				curve = c; break;
			}
			else if ((int)wavelength_ < (int)Wavelengths[1]) { // below lowest
				curve = c; lowerThanCurve = true; break;
			}
			else if ((int)wavelength_ > (int)Wavelengths[maxCurves]) { // above highest
				curve = maxCurves - 2; higherThanCurve = true; break;
			}
		}

		if (lowerThanCurve) {
			double w_diff = Wavelengths[curve + 1] - wavelength_;
			double new_wavelength = Wavelengths[curve + 1] + w_diff;
			double fact1 = (Wavelengths[4] - new_wavelength) / (Wavelengths[4] - Wavelengths[2]);
			double fact2 = (new_wavelength - Wavelengths[2]) / (Wavelengths[4] - Wavelengths[2]);

			for (int i = 0; i < ArrayLength2; i++) {
				double val = ArrayLcVoltagesRetLoaded[i][curve];
				ArrayLcVoltagesRetCurve[i][0] = val;
				double curve1aVal = getValueFromArray(val, 1, 0);
				double curve1bVal = getValueFromArray(val, 2, 0);
				double curve2aVal = getValueFromArray(val, 4, 3);
				double curve2bVal = getValueFromArray(val, 5, 3);
				ArrayLcVoltagesRetCurve[i][1] = 2 * curve1aVal - ((curve1aVal * fact1) + (curve2aVal * fact2));
				ArrayLcVoltagesRetCurve[i][2] = 2 * curve1bVal - ((curve1bVal * fact1) + (curve2bVal * fact2));
			}

		}
		else if (higherThanCurve) {
			double w_diff = wavelength_ - Wavelengths[curve + 1];
			double new_wavelength = Wavelengths[curve + 1] - w_diff;
			double fact1 = (Wavelengths[curve + 1] - new_wavelength) / (Wavelengths[curve + 1] - Wavelengths[curve - 1]);
			double fact2 = (new_wavelength - Wavelengths[curve - 1]) / (Wavelengths[curve + 1] - Wavelengths[curve - 1]);

			for (int i = 0; i < ArrayLength2; i++) {
				double val = ArrayLcVoltagesRetLoaded[i][curve];
				ArrayLcVoltagesRetCurve[i][0] = val;
				double curve1aVal = getValueFromArray(val, curve - 2, curve - 3);
				double curve1bVal = getValueFromArray(val, curve - 1, curve - 3);
				double curve2aVal = getValueFromArray(val, curve + 1, curve);
				double curve2bVal = getValueFromArray(val, curve + 2, curve);
				ArrayLcVoltagesRetCurve[i][1] = 2 * curve2aVal - ((curve1aVal * fact1) + (curve2aVal * fact2));
				ArrayLcVoltagesRetCurve[i][2] = 2 * curve2bVal - ((curve1bVal * fact1) + (curve2bVal * fact2));
			}

		}
		else if (isCurve) {
			for (int i = 0; i < ArrayLength2; i++) {
				double val = ArrayLcVoltagesRetLoaded[i][curve];
				ArrayLcVoltagesRetCurve[i][0] = val;
				double curveVal1 = getValueFromArray(val, curve + 1, curve);
				double curveVal2 = getValueFromArray(val, curve + 2, curve);
				ArrayLcVoltagesRetCurve[i][1] = curveVal1;
				ArrayLcVoltagesRetCurve[i][2] = curveVal2;
			}
		}
		else {

			double fact1 = (Wavelengths[curve + 1] - wavelength_) / (Wavelengths[curve + 1] - Wavelengths[curve - 1]);
			double fact2 = (wavelength_ - Wavelengths[curve - 1]) / (Wavelengths[curve + 1] - Wavelengths[curve - 1]);

			for (int i = 0; i < ArrayLength2; i++) {
				double val = ArrayLcVoltagesRetLoaded[i][curve];
				ArrayLcVoltagesRetCurve[i][0] = val;
				double curve1aVal = getValueFromArray(val, curve - 2, curve - 3);
				double curve1bVal = getValueFromArray(val, curve - 1, curve - 3);
				double curve2aVal = getValueFromArray(val, curve + 1, curve);
				double curve2bVal = getValueFromArray(val, curve + 2, curve);
				if (fact1 == 1 || fact1 == 0) {
					ArrayLcVoltagesRetCurve[i][1] = (curve1aVal * fact1) + (curve2aVal * fact2);
					ArrayLcVoltagesRetCurve[i][2] = (curve1bVal * fact1) + (curve2bVal * fact2);
				}
				else if (fact1 > fact2) {
					ArrayLcVoltagesRetCurve[i][1] = (curve1aVal * fact1) + (curve2aVal * fact2);
					ArrayLcVoltagesRetCurve[i][2] = (curve1bVal * fact1) + (curve2bVal * fact2);
				}
				else if (fact1 < fact2) {
					ArrayLcVoltagesRetCurve[i][1] = (curve1aVal * fact1) + (curve2aVal * fact2);
					ArrayLcVoltagesRetCurve[i][2] = (curve1bVal * fact1) + (curve2bVal * fact2);
				}
				else {
					ArrayLcVoltagesRetCurve[i][1] = (curve1aVal * fact1) + (curve2aVal * fact2);
					ArrayLcVoltagesRetCurve[i][2] = (curve1bVal * fact1) + (curve2bVal * fact2);
				}
			}
		}
	}
	/*ArrayLcVoltagesRetCurve[2][0] = curve;
	ArrayLcVoltagesRetCurve[2][1] = isCurve;
	ArrayLcVoltagesRetCurve[2][2] = wavelength_;
	ArrayLcVoltagesRetCurve[3][0] = Wavelengths[2];
	ArrayLcVoltagesRetCurve[3][1] = Wavelengths[4];
	ArrayLcVoltagesRetCurve[3][2] = Wavelengths[7];*/
}

double MeadowlarkLC::getValueFromArray(double val, int x, int curve_idx) {
	for (int i = 0; i < ArrayLength2; i++) {
		if (val == ArrayLcVoltagesRetLoaded[i][curve_idx]) {
			return ArrayLcVoltagesRetLoaded[i][x];
		}
		else if (val > ArrayLcVoltagesRetLoaded[i][curve_idx] && val < ArrayLcVoltagesRetLoaded[i + 1][curve_idx]) {
			double diff = ArrayLcVoltagesRetLoaded[i + 1][curve_idx] - ArrayLcVoltagesRetLoaded[i][curve_idx];
			double retDiff = ArrayLcVoltagesRetLoaded[i][x] - ArrayLcVoltagesRetLoaded[i + 1][x];
			double excess = retDiff * ((val - ArrayLcVoltagesRetLoaded[i][curve_idx]) / diff);
			return  ArrayLcVoltagesRetLoaded[i][x] - excess;
		}
	}
	return 0.5;
}

void  MeadowlarkLC::loadDefault() {

	for (int i = 0; i < ArrayLengthDefault; i++) {
		ArrayLcVoltagesRetLoaded[i][0] = ArrayDefaultLcVoltagesRet[i][0];
		ArrayLcVoltagesRetLoaded[i][1] = ArrayDefaultLcVoltagesRet[i][1];
		ArrayLcVoltagesRetLoaded[i][2] = ArrayDefaultLcVoltagesRet[i][2];

		ArrayLcVoltagesRetCurve[i][0] = ArrayDefaultLcVoltagesRet[i][0];
		ArrayLcVoltagesRetCurve[i][1] = ArrayDefaultLcVoltagesRet[i][1];
		ArrayLcVoltagesRetCurve[i][2] = ArrayDefaultLcVoltagesRet[i][2];
	}

	ArrayLength2 = ArrayLengthDefault;
}

void  MeadowlarkLC::loadResource(int ID) {
	lcCurve_ = "";

	HMODULE handle = GetCurrentModule();
	HRSRC hRes = FindResource(handle, MAKEINTRESOURCE(ID), MAKEINTRESOURCE(TEXTFILE));

	if (NULL != hRes)
	{
		HGLOBAL hData = LoadResource(handle, hRes);
		if (NULL != hData)
		{
			DWORD dataSize = SizeofResource(handle, hRes);
			char* data = (char*)LockResource(hData);
			lcCurve_.assign(data, dataSize);

			//::OutputDebugString(lcCurve_.c_str()); // Print as ASCII text

			//char* buffer = new char[dataSize+1];
			//::memcpy(buffer, data, dataSize);
			//buffer[dataSize] = 0; // NULL terminator
			//::OutputDebugString(buffer); // Print as ASCII text
			//delete[] buffer;
		}
	}
}

//////////////// ---- ToDo ---- //////////////////
//
// 1. Added Exercise routine in a new thread so that it does not stop device interface



