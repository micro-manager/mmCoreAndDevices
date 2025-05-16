// Micro-Manager device adapter for VisiTech iSIM
//
// Copyright (C) 2016 Open Imaging, Inc.
//
// This library is free software; you can redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation; version 2.1.
//
// This library is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License
// for more details.
//
// IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this library; if not, write to the Free Software Foundation,
// Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
//
//
// Author: Mark Tsuchida <mark@open-imaging.com>


#include "VTiSIM.h"

#include "ModuleInterface.h"
#include <VisiSDK.h>
#include <algorithm>
#include <utility>
#include <vector>

const char* const g_DeviceName_Hub = "VTiSIMHub";
const char* const g_DeviceName_LaserShutter = "LaserShutter";
const char* const g_DeviceName_Lasers = "Lasers";
const char* const g_DeviceName_Scanner = "Scanner";
const char* const g_DeviceName_PinholeArray = "PinholeArray";
const char* const g_DeviceName_Dichroic = "Dichroic";
const char* const g_DeviceName_BarrierFilter = "BarrierFilter";
const char* const g_DeviceName_FRAP ="FRAP"; // Ver 2.1.0.0 - FRAP - start
//const char* const g_DeviceName_Pifoc ="Pifoc Controls"; // Ver 2.2.0.0 - Pifoc
const char* const g_DeviceName_Pifoc ="VT Piezo Control"; // Ver 2.4.0.0 - Pifoc

const char* const g_PropName_LaserModulation = "Modulation";
const char* const g_PropName_LaserName0 = "State-0 Name";
const char* const g_PropName_LaserName1 = "State-1 Name";
const char* const g_PropName_LaserName2 = "State-2 Name";
const char* const g_PropName_LaserName3 = "State-3 Name";
const char* const g_PropName_LaserName4 = "State-4 Name";
const char* const g_PropName_LaserName5 = "State-5 Name";
const char* const g_PropName_LaserName6 = "State-6 Name";
const char* const g_PropName_LaserName7 = "State-7 Name";
const char* const g_PropName_Scanning = "Scanning";
const char* const g_PropName_ScanRate = "Scan Rate (Hz)";
const char* const g_PropName_ScanWidth = "Scan Width";
const char* const g_PropName_ScanOffset = "Scan Offset";
const char* const g_PropName_ActualRate = "Actual Scan Rate (Hz)";
const char* const g_PropName_FinePosition = "Fine Step Position";
const char* const g_PropName_PinholeSize = "Pinhole Size (um)";
const char* const g_PropName_Backlash = "Backlash Compensation";
const char* const g_PropName_DichroicPosition = "Dichroic Position";
const char* const g_PropName_DichroicPos1="Dichroic Pos 1";
const char* const g_PropName_DichroicPos2="Dichroic Pos 2";
const char* const g_PropName_DichroicPos3="Dichroic Pos 3";
const char* const g_PropName_DichroicLabel = "Dichroic Label";
const char* const g_PropName_BarrierFilterPos = "Filter Position";
const char* const g_PropName_BarrierFilterLabel = "Filter Label";
const char* const g_PropName_BarrierFilterPos1="Filter Pos 1";
const char* const g_PropName_BarrierFilterPos2="Filter Pos 2";
const char* const g_PropName_BarrierFilterPos3="Filter Pos 3";
const char* const g_PropName_BarrierFilterPos4="Filter Pos 4";
const char* const g_PropName_BarrierFilterPos5="Filter Pos 5";
const char* const g_PropName_BarrierFilterPos6="Filter Pos 6";
const char* const g_PropVal_Off = "Off";
const char* const g_PropVal_On = "On";
const char* const g_PropVal_ConstIntensity = "Constant Intensity";
const char* const g_PropVal_RampIntensity = "Ramp Intensity";
const char* const g_PropVal_RampFastInten = "RampFast Intensity";

// Ver 2.1.0.0 - FRAP - start
const char* const g_PropVal_ActivateFRAP = "FRAP Enable";
const char* const g_PropVal_FRAPX = "X";
const char* const g_PropVal_FRAPY = "Y";
const char* const g_PropVal_FRAPXRange = "Range of X";	// Ver 2.3
const char* const g_PropVal_FRAPYRange = "Range of Y";	// Ver 2.3
const char* const g_PropVal_FRAPXMin = "Minimum of X";	// Ver 2.3
const char* const g_PropVal_FRAPYMin = "Minimum of Y";	// Ver 2.3
const char* const g_PropName_FRAPXOffset = "X Offset";	// Ver 2.3.2.0
const char* const g_PropName_FRAPYOffset = "Y Offset";	// Ver 2.3.2.0

#define GALVO_X_RANGE 930  // Ver 2.1.1.0 - range update
#define GALVO_Y_RANGE 620
#define GALVO_X_MIN 1653
#define GALVO_Y_MIN 1714
#define GALVO_X_MAX 2583
#define GALVO_Y_MAX 2334
// Ver 2.1.0.0 - FRAP - end

// Ver 2.4.0.0 - Start
/*// Ver 2.2.0.0 - Pifoc - Start
const char* g_PropVal_AxisName = "Axis";
const char* g_PropVal_AxisLimitUm = "Limit_um";
const char* g_PropVal_InvertTravelRange = "Invert travel range";
const char* g_PropVal_StageType = "Stage";
const char* g_PropVal_StepSize = "StepSizeUm";
const char*g_PropVal_ControllerName = "Analogue Output";
// Ver 2.2.0.0 - Pifoc - End*/
const char* g_PropVal_PiezoTravelRangeUm = "Piezo Travel Range (um)";
// Ver 2.4.0.0 - End

MODULE_API void InitializeModuleData()
{
	RegisterDevice(g_DeviceName_Hub, MM::HubDevice, "VT-iSIM system");
}


MODULE_API MM::Device* CreateDevice(const char* name)
{
	if (!name)
		return 0;

	if (strcmp(name, g_DeviceName_Hub) == 0)
		return new VTiSIMHub();
	if (strcmp(name, g_DeviceName_LaserShutter) == 0)
		return new VTiSIMLaserShutter();
	if (strcmp(name, g_DeviceName_Lasers) == 0)
		return new VTiSIMLasers();
	if (strcmp(name, g_DeviceName_Scanner) == 0)
		return new VTiSIMScanner();
	if (strcmp(name, g_DeviceName_PinholeArray) == 0)
		return new VTiSIMPinholeArray();
	if (strcmp(name, g_DeviceName_Dichroic) == 0)
		return new VTiSIMDichroic();
	if (strcmp(name, g_DeviceName_BarrierFilter) == 0)
		return new VTiSIMBarrierFilter();
	// Ver 2.1.0.0 - FRAP - start
	if (strcmp(name, g_DeviceName_FRAP) == 0)
		return new VTiSIMFRAP();
	// Ver 2.1.0.0 - FRAP - end
	// Ver 2.2.0.0 - Pifoc - start
	if (strcmp(name, g_DeviceName_Pifoc) == 0)
		return new VTiSIMPifoc();
	// Ver 2.2.0.0 - Pifoc - end

	return 0;
}


MODULE_API void DeleteDevice(MM::Device* pDevice)
{
	delete pDevice;
}


VTiSIMHub::VTiSIMHub() :
hAotfControl_(0),
	hScanAndMotorControl_(0)
{
	SetErrorText(VTI_ERR_TIMEOUT_OCCURRED, "Timeout occurred");
	SetErrorText(VTI_ERR_DEVICE_NOT_FOUND, "Device not found");
	SetErrorText(VTI_ERR_NOT_INITIALISED, "Device not initialized, or initialization failed");
	SetErrorText(VTI_ERR_ALREADY_INITIALISED, "Device already initialized");
}


VTiSIMHub::~VTiSIMHub()
{
}


int VTiSIMHub::Initialize()
{
	if (!hAotfControl_)
	{
		DWORD err = vti_Initialise(VTI_HARDWARE_AOTF_USB, &hAotfControl_);
		if (err != VTI_SUCCESS)
			return err;
	}

	if (!hScanAndMotorControl_)
	{
		DWORD err = vti_Initialise(VTI_HARDWARE_VTINFINITY_4, &hScanAndMotorControl_);
		if (err != VTI_SUCCESS)
			return err;
	}

	LONG major, minor, rev, build;
	DWORD err = vti_GetDllVersionInfo(&major, &minor, &rev, &build);
	if (err == VTI_SUCCESS)
	{
		char s[MM::MaxStrLength + 1];
		snprintf(s, MM::MaxStrLength, "%d.%d.%d.%d",
			(int)major, (int)minor, (int)rev, (int)build);
		err = CreateStringProperty("VisiSDK Version", s, true);
		if (err != DEVICE_OK)
			return err;
		// Ver 2.4.0.0 - 3. Start
		LONG MMmajor, MMminor, MMrev, MMbuild;
		MMmajor = 2;
		MMminor = 5;
		MMrev	= 0;
		MMbuild	= 0;
		snprintf(s, MM::MaxStrLength, "%d.%d.%d.%d",
			(int)MMmajor, (int)MMminor, (int)MMrev, (int)MMbuild);
		err = CreateStringProperty("Device Adaptor Version", s, true);
		if (err != DEVICE_OK)
			return err;
		// Ver 2.4.0.0 - 3. End
	}

	return DEVICE_OK;
}


int VTiSIMHub::Shutdown()
{
	int lastErr = DEVICE_OK;

	if (hScanAndMotorControl_)
	{
		DWORD err = vti_UnInitialise(&hScanAndMotorControl_);
		if (err != VTI_SUCCESS)
		{
			lastErr = err;
		}
	}

	if (hAotfControl_)
	{
		DWORD err = vti_UnInitialise(&hAotfControl_);
		if (err != VTI_SUCCESS)
		{
			lastErr = err;
		}
	}

	return lastErr;
}


void VTiSIMHub::GetName(char* name) const
{
	CDeviceUtils::CopyLimitedString(name, g_DeviceName_Hub);
}


bool VTiSIMHub::Busy()
{
	return false;
}


int VTiSIMHub::DetectInstalledDevices()
{
	ClearInstalledDevices();

	MM::Device* pDev = new VTiSIMLaserShutter();
	if (pDev)
		AddInstalledDevice(pDev);

	pDev = new VTiSIMLasers();
	if (pDev)
		AddInstalledDevice(pDev);

	pDev = new VTiSIMScanner();
	if (pDev)
		AddInstalledDevice(pDev);

	pDev = new VTiSIMPinholeArray();
	if (pDev)
		AddInstalledDevice(pDev);

	pDev = new VTiSIMDichroic();
	if (pDev)
		AddInstalledDevice(pDev);

	pDev = new VTiSIMBarrierFilter();
	if (pDev)
		AddInstalledDevice(pDev);

	// Ver 2.1.0.0 - FRAP - start
	pDev = new VTiSIMFRAP();
	if (pDev)
		AddInstalledDevice(pDev);
	// Ver 2.1.0.0 - FRAP - end

	// Ver 2.2.0.0 - Pifoc - start
	pDev = new VTiSIMPifoc();
	if (pDev)
		AddInstalledDevice(pDev);
	// Ver 2.2.0.0 - Pifoc - end
	return DEVICE_OK;
}


VTiSIMLaserShutter::VTiSIMLaserShutter() :
isOpen_(false)
{
	SetErrorText(VTI_ERR_TIMEOUT_OCCURRED, "Timeout occurred");
	SetErrorText(VTI_ERR_DEVICE_NOT_FOUND, "Device not found");
	SetErrorText(VTI_ERR_NOT_INITIALISED, "Device not initialized");
	SetErrorText(VTI_ERR_NOT_SUPPORTED, "Operation not supported");
	SetErrorText(VTI_ERR_INCORRECT_MODE, "AOTF is in manual mode");
}


VTiSIMLaserShutter::~VTiSIMLaserShutter()
{
}


int VTiSIMLaserShutter::Initialize()
{
	int err = CreateIntegerProperty(MM::g_Keyword_State, 0, false,
		new CPropertyAction(this, &VTiSIMLaserShutter::OnState));
	if (err != DEVICE_OK)
		return err;
	err = SetPropertyLimits(MM::g_Keyword_State, 0, 1);
	if (err != DEVICE_OK)
		return err;

	// Sync with our memory of state
	return DoSetOpen(isOpen_);
}


int VTiSIMLaserShutter::Shutdown()
{
	// Always turn off on shutdown
	return SetOpen(false);
}


void VTiSIMLaserShutter::GetName(char* name) const
{
	CDeviceUtils::CopyLimitedString(name, g_DeviceName_LaserShutter);
}


bool VTiSIMLaserShutter::Busy()
{
	return false;
}


int VTiSIMLaserShutter::GetOpen(bool& open)
{
	open = isOpen_;
	return DEVICE_OK;
}


int VTiSIMLaserShutter::SetOpen(bool open)
{
	if (open == isOpen_)
		return DEVICE_OK;
	return DoSetOpen(open);
}


int VTiSIMLaserShutter::OnState(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set(isOpen_ ? 1L : 0L);
	}
	else if (eAct == MM::AfterSet)
	{
		long v;
		pProp->Get(v);
		return SetOpen(v != 0);
	}
	return DEVICE_OK;
}


VTiSIMHub* VTiSIMLaserShutter::VTiHub()
{
	return static_cast<VTiSIMHub*>(GetParentHub());
}


int VTiSIMLaserShutter::DoSetOpen(bool open)
{
	DWORD err = vti_SetShutter(VTiHub()->GetAOTFHandle(), open);
	if (err != VTI_SUCCESS)
		return err;
	isOpen_ = open;

	int mmerr = OnPropertyChanged(MM::g_Keyword_State, open ? "1" : "0");
	if (mmerr != DEVICE_OK)
		return mmerr;

	return DEVICE_OK;
}


VTiSIMLasers::VTiSIMLasers() :
curChan_(0),
	Bitmask(0)
{
	char s[MM::MaxStrLength + 1];

	memset(intensities_, 0, sizeof(intensities_));

	SetErrorText(VTI_ERR_TIMEOUT_OCCURRED, "Timeout occurred");
	SetErrorText(VTI_ERR_DEVICE_NOT_FOUND, "Device not found");
	SetErrorText(VTI_ERR_NOT_INITIALISED, "Device not initialized");
	SetErrorText(VTI_ERR_NOT_SUPPORTED, "Operation not supported");
	SetErrorText(VTI_ERR_INCORRECT_MODE, "AOTF is in manual mode");

	strLaserName[0]= "405nm";
	strLaserName[1]= "442nm";
	strLaserName[2]= "488nm";
	strLaserName[3]= "514/532nm";
	strLaserName[4]= "561nm";
	strLaserName[5]= "642nm";
	strLaserName[6]= "0nm";
	strLaserName[7]= "0nm";


	for (long i = 0; i < nChannels; ++i)
	{
		snprintf(s, MM::MaxStrLength, "State Name-%ld", i+1);
		CreateStringProperty(s, strLaserName[i].c_str(), false,
			new CPropertyActionEx(this, &VTiSIMLasers::OnLaserNameState,i), true);
	}

}

VTiSIMLasers::~VTiSIMLasers()
{
}

int VTiSIMLasers::Initialize()
{
	//DWORD vterr;
	char s[MM::MaxStrLength + 1];
	char s2[MM::MaxStrLength + 1];

	Bitmask = 0; // Initilised the bit mask here.
	// Comment it out to remove the Laser state Parameters
	/* for (long i = 0; i < nChannels; ++i)
	{
	snprintf(s, MM::MaxStrLength, "Laser-%ld", i);
	SetPositionLabel(i, s);
	} 

	int err = CreateIntegerProperty(MM::g_Keyword_State, 0, false,
	new CPropertyAction(this, &VTiSIMLasers::OnState));
	if (err != DEVICE_OK)
	return err;
	err = SetPropertyLimits(MM::g_Keyword_State, 0, nChannels - 1);
	if (err != DEVICE_OK)
	return err;

	err = CreateStringProperty(MM::g_Keyword_Label, "", false,
	new CPropertyAction(this, &CStateBase::OnLabel));
	if (err != DEVICE_OK)
	return err;*/

	for (long i = 0; i < nChannels; ++i)
	{
		// To change the name of laser Intensity according to the Laser Name
		//snprintf(s, MM::MaxStrLength, "Intensity-%ld", LaserName[i]);
		snprintf(s, MM::MaxStrLength, "Intensity-%d-%s",i+1, (strLaserName[i]).c_str());

		int err = CreateIntegerProperty(s, intensities_[i], false,
			new CPropertyActionEx(this, &VTiSIMLasers::OnIntensity, i));
		if (err != DEVICE_OK)
			return err;
		err = SetPropertyLimits(s, 0, 100);
		if (err != DEVICE_OK)
			return err;

		err = DoSetIntensity(i, intensities_[i]);
		if (err != DEVICE_OK)
			return err;

		// Set individual Laser Enable 
		snprintf(s2, MM::MaxStrLength, "Enable-%d-%s",i+1, (strLaserName[i]).c_str());
		LogMessage("Initialize");
		LogMessage(s2);
		err = CreateStringProperty(s2, g_PropVal_Off, false,
			new CPropertyActionEx(this,&VTiSIMLasers::OnLaserState,i)); // allow to turn on  mutliple laser at the same time. ));
		err = AddAllowedValue(s2, g_PropVal_Off);
		if (err != DEVICE_OK)
			return err;
		err = AddAllowedValue(s2, g_PropVal_On);
		if (err != DEVICE_OK)
			return err;
		for (long j = 0; j < nChannels; ++j)
		{
			err = DoUploadTTLBitmask(j,0);
			if (err != DEVICE_OK)
				return err;
		}
	}

	// Set Laser Modulation Type (Constant, Ramp , Ramp Fast Intensity)
	int err = CreateStringProperty(g_PropName_LaserModulation, g_PropVal_ConstIntensity, false,
		new CPropertyAction(this, &VTiSIMLasers::OnModulation));
	if (err != DEVICE_OK)
		return err;

	err = AddAllowedValue(g_PropName_LaserModulation, g_PropVal_ConstIntensity);
	if (err != DEVICE_OK)
		return err;

	err = AddAllowedValue(g_PropName_LaserModulation, g_PropVal_RampIntensity);
	if (err != DEVICE_OK)
		return err;

	err = AddAllowedValue(g_PropName_LaserModulation, g_PropVal_RampFastInten);
	if (err != DEVICE_OK)
		return err;
	// Sync with our memory of state
	// return DoSetChannel(curChan_);

	return DoSetModulation(0); // Set the Constant intensity modulation
}

int VTiSIMLasers::Shutdown()
{
	return DEVICE_OK;
}


void VTiSIMLasers::GetName(char* name) const
{
	CDeviceUtils::CopyLimitedString(name, g_DeviceName_Lasers);
}


bool VTiSIMLasers::Busy()
{
	return false;
}


unsigned long VTiSIMLasers::GetNumberOfPositions() const
{
	return nChannels;
}


int VTiSIMLasers::OnState(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set(static_cast<long>(curChan_));
	}
	else if (eAct == MM::AfterSet)
	{
		long v;
		pProp->Get(v);
		if (v == curChan_)
			return DEVICE_OK;
		return DoSetChannel(v);
	}
	return DEVICE_OK;
}

int VTiSIMLasers::OnLaserState(MM::PropertyBase* pProp, MM::ActionType eAct,long channel)
{
	//int channel;
	if (eAct == MM::BeforeGet)
	{

	}
	else if (eAct == MM::AfterSet)
	{
		std::string s,Name;
		char ss[MM::MaxStrLength + 1];
		pProp->Get(s);
		int shouldLaserOn = (s == g_PropVal_On);
		Name = pProp->GetName();
		snprintf(ss, MM::MaxStrLength, "Channel-%ld", channel);
		LogMessage("OnLaserState");
		LogMessage(ss);
		return DoUploadTTLBitmask(channel,shouldLaserOn);
	}
	return DEVICE_OK;
}

int VTiSIMLasers::OnLaserNameState(MM::PropertyBase* pProp, MM::ActionType eAct,long chan)
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set(strLaserName[chan].c_str());
	}
	else if (eAct == MM::AfterSet)
	{
		std::string str;
		pProp->Get(str);
		strLaserName[chan]=str;
	}
	return DEVICE_OK;
}

int VTiSIMLasers::OnIntensity(MM::PropertyBase* pProp, MM::ActionType eAct, long chan)
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set(static_cast<long>(intensities_[chan]));
	}
	else if (eAct == MM::AfterSet)
	{
		long v;
		pProp->Get(v);
		if (v == intensities_[chan])
			return DEVICE_OK;
		return DoSetIntensity(chan, v);
	}
	return DEVICE_OK;
}

int VTiSIMLasers::OnModulation(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{

	}
	else if (eAct == MM::AfterSet)
	{
		std::string ss;
		int Mode = 0;
		pProp->Get(ss);
		if (ss == g_PropVal_ConstIntensity)
			Mode = 0;
		else if(ss == g_PropVal_RampIntensity)
			Mode = 1;
		else if(ss == g_PropVal_RampFastInten)
			Mode =2;

		char s[MM::MaxStrLength + 1];
		snprintf(s, MM::MaxStrLength, "OnModulation-%ld",Mode);
		LogMessage(s);
		return DoSetModulation(Mode);
	}
	return DEVICE_OK;
}

int VTiSIMLasers::DoSetModulation(int mode)
{
	if(mode<0 || mode>2)
		return DEVICE_ERR; // Shouldn't happen

	char s[MM::MaxStrLength + 1];
	snprintf(s, MM::MaxStrLength, "Mode-%ld",mode);
	LogMessage(s);

	VTI_EX_PARAM Param;
	switch (mode) 
	{
	case 0:
		Param.ParamOption = VTI_RECTIFIED_INTENSITY_CONSTANT;
		break;

	case 1:
		Param.ParamOption = VTI_RECTIFIED_INTENSITY_RAMPED_NORMAL;
		break;

	case 2:
		Param.ParamOption = VTI_RECTIFIED_INTENSITY_RAMPED_FAST;
		break;

	default:
		Param.ParamOption = VTI_RECTIFIED_INTENSITY_CONSTANT;
		break;
	}
	Param.pArray = NULL;
	Param.ArrayBytes = 0;
	DWORD err =vti_SetExtendedFeature(VTiHub()->GetAOTFHandle(), VTI_FEATURE_RECTIFIED_INTENSITY, &Param, sizeof(Param));
	if (err != VTI_SUCCESS)
	{
		snprintf(s, MM::MaxStrLength, "OnMoudulationERROR-%ld",err);
		LogMessage(s);
		return err;
	}
	return DEVICE_OK;
}

VTiSIMHub* VTiSIMLasers::VTiHub()
{
	return static_cast<VTiSIMHub*>(GetParentHub());
}


int VTiSIMLasers::DoSetChannel(int chan)
{
	if (chan < 0 || chan >= nChannels)
		return DEVICE_ERR; // Shouldn't happen

	DWORD err = vti_SetTTLBitmask(VTiHub()->GetAOTFHandle(), 1 << chan);
	if (err != VTI_SUCCESS)
		return err;
	curChan_ = chan;
	return DEVICE_OK;
}

int VTiSIMLasers::DoUploadTTLBitmask(int channel , int ShouldOn)
{
	/*for (int Channel = 0; Channel < 8; Channel++)
	{
	int Checked = m_CheckTTL[Channel].GetCheck();
	Bitmask |= Checked << Channel;
	}*/
	char s[MM::MaxStrLength + 1];
	int result;
	bool IsBypassToggled = FALSE; // Ver 2.0.0.0
	int temp; // Ver 2.0.0.0
	int PreBitmask;

	// Ver 2.0.0.0
	PreBitmask = Bitmask;
	//snprintf(s, MM::MaxStrLength, "Pre update_Bitmask-%ld",PreBitmask);
	//LogMessage(s);
	// Ver 2.0.0.0

	//snprintf(s, MM::MaxStrLength, "ShouldOn-%ld",ShouldOn);
	//LogMessage(s);

	result = 1 << channel;

	// Ver 2.0.0.0 Start
	temp = Bitmask>>channel;
	//snprintf(s, MM::MaxStrLength, "condition checks-%ld",temp); 
	//LogMessage(s);


	if (ShouldOn && (Bitmask>>(channel))!= 1) // Turn ON
	{
		Bitmask = PreBitmask|result;
		//snprintf(s, MM::MaxStrLength, "turn on_result-%ld",result);
		//LogMessage(s);
		
		if (Bitmask!=PreBitmask && result == 64)
			IsBypassToggled = TRUE;
		else
			IsBypassToggled = FALSE;
	}

	else if(!ShouldOn && (Bitmask>>(channel))!= 0) //turn off
	{
		result = ~result;
		Bitmask =PreBitmask&result;
		//snprintf(s, MM::MaxStrLength, "turn off_result-%ld",result);
		//LogMessage(s);

		if (Bitmask!=PreBitmask && result == -65)
			IsBypassToggled = TRUE;
		else
			IsBypassToggled = FALSE;
	}
	// Ver 2.0.0.0 End

	/*if (ShouldOn && (Bitmask>>(channel))!= 1) // Turn ON
	{
		Bitmask|= result;
		snprintf(s, MM::MaxStrLength, "turn on_result-%ld",result);
		LogMessage(s);
	}

	else if(!ShouldOn && (Bitmask>>(channel))!= 0) //turn off
	{
		//02_01_2019 - bug fixed on laser disable - start
		result = ~result;
		Bitmask &=result;
		snprintf(s, MM::MaxStrLength, "turn off_result-%ld",result);
		LogMessage(s);
		//02_01_2019 - bug fixed on laser disable - end
	}*/

	//snprintf(s, MM::MaxStrLength, "Post update_Bitmask-%ld",Bitmask);
	//LogMessage(s);
	DWORD err = vti_SetTTLBitmask(VTiHub()->GetAOTFHandle(), Bitmask);
	if (err != VTI_SUCCESS)
		return err;

		// Ver 2.0.0.0 - start
	if(IsBypassToggled)
	{
		snprintf(s, MM::MaxStrLength, "Delay start");
		LogMessage(s);
		//MM::MMTime delay(800*1000.0);
		CDeviceUtils::SleepMs(800);
		snprintf(s, MM::MaxStrLength, "Delay End");
		LogMessage(s);
	}
	IsBypassToggled = FALSE;
	// Ver 2.0.0.0 - End

	return DEVICE_OK;
}

int VTiSIMLasers::DoSetIntensity(int chan, int percentage)
{
	if (chan < 0 || chan >= nChannels)
		return DEVICE_ERR; // Shouldn't happen

	if (percentage < 0) // Shouldn't happen
		percentage = 0;
	if (percentage > 100) // Shouldn't happen
		percentage = 100;

	DWORD err = vti_SetIntensity(VTiHub()->GetAOTFHandle(), chan, percentage);
	if (err != VTI_SUCCESS)
		return err;
	intensities_[chan] = percentage;
	return DEVICE_OK;
}

VTiSIMScanner::VTiSIMScanner() :
minRate_(1),
	maxRate_(1000),
	minWidth_(0),
	maxWidth_(4095),
	scanRate_(150),//150
	scanWidth_(1600),
	scanOffset_(0), // 0 is always an allowed offset
	actualRate_(0.0f)
{
	SetErrorText(VTI_ERR_TIMEOUT_OCCURRED, "Timeout occurred");
	SetErrorText(VTI_ERR_DEVICE_NOT_FOUND, "Device not found");
	SetErrorText(VTI_ERR_NOT_INITIALISED, "Device not initialized");
	SetErrorText(VTI_ERR_NOT_SUPPORTED, "Operation not supported");
}


VTiSIMScanner::~VTiSIMScanner()
{
}


int VTiSIMScanner::Initialize()
{
	int err = CreateStringProperty(g_PropName_Scanning, g_PropVal_Off, false,
		new CPropertyAction(this, &VTiSIMScanner::OnStartStop));
	if (err != DEVICE_OK)
		return err;
	err = AddAllowedValue(g_PropName_Scanning, g_PropVal_Off);
	if (err != DEVICE_OK)
		return err;
	err = AddAllowedValue(g_PropName_Scanning, g_PropVal_On);
	if (err != DEVICE_OK)
		return err;

	DWORD vterr = vti_GetScanRateRange(VTiHub()->GetScanAndMotorHandle(),
		&minRate_, &maxRate_);
	if (vterr != VTI_SUCCESS)
		return vterr;
	if (scanRate_ < minRate_)
		scanRate_ = minRate_;
	if (scanRate_ > maxRate_)
		scanRate_ = maxRate_;
	err = CreateIntegerProperty(g_PropName_ScanRate, scanRate_, false,
		new CPropertyAction(this, &VTiSIMScanner::OnScanRate));
	if (err != DEVICE_OK)
		return err;
	err = SetPropertyLimits(g_PropName_ScanRate, minRate_, maxRate_);
	if (err != DEVICE_OK)
		return err;

	vterr = vti_GetScanWidthRange(VTiHub()->GetScanAndMotorHandle(),
		&minWidth_, &maxWidth_);
	if (vterr != VTI_SUCCESS)
		return vterr;
	if (scanWidth_ < minWidth_)
		scanWidth_ = minWidth_;
	if (scanWidth_ > maxWidth_)
		scanWidth_ = maxWidth_;
	err = CreateIntegerProperty(g_PropName_ScanWidth, scanWidth_, false,
		new CPropertyAction(this, &VTiSIMScanner::OnScanWidth));
	if (err != DEVICE_OK)
		return err;
	err = SetPropertyLimits(g_PropName_ScanWidth, minWidth_, maxWidth_);
	if (err != DEVICE_OK)
		return err;

	err = CreateFloatProperty(g_PropName_ActualRate, actualRate_, true,
		new CPropertyAction(this, &VTiSIMScanner::OnActualScanRate));
	if (err != DEVICE_OK)
		return err;

	err = CreateIntegerProperty(g_PropName_ScanOffset, scanOffset_, false,
		new CPropertyAction(this, &VTiSIMScanner::OnScanOffset));
	if (err != DEVICE_OK)
		return err;
	// For negative Polarity  err = SetPropertyLimits(g_PropName_ScanOffset, 0, GetMaxOffset());
	int minOffset;
	minOffset= -1 * GetMaxOffset();
	err = SetPropertyLimits(g_PropName_ScanOffset, minOffset, GetMaxOffset());

	if (err != DEVICE_OK)
		return err;

	return DoStartStopScan(false);
}


int VTiSIMScanner::Shutdown()
{
	return DoStartStopScan(false);
}


void VTiSIMScanner::GetName(char* name) const
{
	CDeviceUtils::CopyLimitedString(name, g_DeviceName_Scanner);
}


bool VTiSIMScanner::Busy()
{
	return false;
}


int VTiSIMScanner::OnScanRate(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set(static_cast<long>(scanRate_));
	}
	else if (eAct == MM::AfterSet)
	{
		long v;
		pProp->Get(v);
		if (v == scanRate_)
			return DEVICE_OK;
		return DoSetScanRate(v);
	}
	return DEVICE_OK;
}


int VTiSIMScanner::OnScanWidth(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set(static_cast<long>(scanWidth_));
	}
	else if (eAct == MM::AfterSet)
	{
		long v;
		pProp->Get(v);
		if (v == scanWidth_)
			return DEVICE_OK;
		return DoSetScanWidth(v);
	}
	return DEVICE_OK;
}


int VTiSIMScanner::OnScanOffset(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set(static_cast<long>(scanOffset_));
	}
	else if (eAct == MM::AfterSet)
	{
		long v;
		pProp->Get(v);
		if (v == scanOffset_)
			return DEVICE_OK;
		return DoSetScanOffset(v);
	}
	return DEVICE_OK;
}


int VTiSIMScanner::OnStartStop(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		bool scanning;
		int err = DoGetScanning(scanning);
		if (err != DEVICE_OK)
			return err;
		pProp->Set(scanning ? g_PropVal_On : g_PropVal_Off);
	}
	else if (eAct == MM::AfterSet)
	{
		std::string s;
		pProp->Get(s);
		bool shouldScan = (s == g_PropVal_On);
		bool scanning;
		int err = DoGetScanning(scanning);
		if (err != DEVICE_OK)
			return err;
		if (shouldScan == scanning)
			return DEVICE_OK;
		return DoStartStopScan(shouldScan);
	}
	return DEVICE_OK;
}


int VTiSIMScanner::OnActualScanRate(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set(static_cast<double>(actualRate_));
	}
	return DEVICE_OK;
}


VTiSIMHub* VTiSIMScanner::VTiHub()
{
	return static_cast<VTiSIMHub*>(GetParentHub());
}


int VTiSIMScanner::DoSetScanRate(int rateHz)
{
	if (rateHz < minRate_)
		rateHz = minRate_;
	if (rateHz > maxRate_)
		rateHz = maxRate_;

	scanRate_ = rateHz;

	bool scanning;
	int err = DoGetScanning(scanning);
	if (err != DEVICE_OK)
		return err;

	if (scanning)
	{
		err = DoStartStopScan(true);
		if (err != DEVICE_OK)
			return err;
	}

	return DEVICE_OK;
}


int VTiSIMScanner::DoSetScanWidth(int width)
{
	if (width < minWidth_)
		width = minWidth_;
	if (width > maxWidth_)
		width = maxWidth_;

	scanWidth_ = width;

	// Update offset range (and value, if necessary)
	int newMaxOffset = GetMaxOffset();
	if (scanOffset_ > newMaxOffset)
	{
		scanOffset_ = newMaxOffset;
	}
	// change for the negative scan offset  //int err = SetPropertyLimits(g_PropName_ScanOffset, 0, newMaxOffset);
	int err = SetPropertyLimits(g_PropName_ScanOffset, (-1)*newMaxOffset, newMaxOffset);
	char s[MM::MaxStrLength + 1];
	snprintf(s, MM::MaxStrLength, "%d", scanOffset_);
	err = OnPropertyChanged(g_PropName_ScanOffset, s);
	if (err != DEVICE_OK)
		return err;

	bool scanning;
	err = DoGetScanning(scanning);
	if (err != DEVICE_OK)
		return err;

	if (scanning)
	{
		err = DoStartStopScan(true);
		if (err != DEVICE_OK)
			return err;
	}

	return DEVICE_OK;
}


int VTiSIMScanner::DoSetScanOffset(int offset)
{
	// To be able to set the negative scan Offset.
	/* if (offset < 0)
	offset = 0;*/
	if (offset < (-1)*GetMaxOffset())
		offset = (-1)*GetMaxOffset();

	if (offset > GetMaxOffset())
		offset = GetMaxOffset();

	scanOffset_ = offset;

	bool scanning;
	int err = DoGetScanning(scanning);
	if (err != DEVICE_OK)
		return err;

	if (scanning)
	{
		err = DoStartStopScan(true);
		if (err != DEVICE_OK)
			return err;
	}

	return DEVICE_OK;
}

int VTiSIMScanner::DoStartStopScan(bool shouldScan)
{
	float newActualRate = 0.0f;

	if (shouldScan)
	{
		DWORD err = vti_StartScan(VTiHub()->GetScanAndMotorHandle(),
			scanRate_, scanWidth_, scanOffset_);
		if (err != VTI_SUCCESS)
			return err;

		err = vti_GetActualScanRate(VTiHub()->GetScanAndMotorHandle(),
			&newActualRate);

		if (err != VTI_SUCCESS)
			return err;
	}
	else
	{
		DWORD err = vti_StopScan(VTiHub()->GetScanAndMotorHandle());
		if (err != VTI_SUCCESS)
			return err;

		newActualRate = 0.0f;
	}

	if (newActualRate != actualRate_)
	{
		actualRate_ = newActualRate;
		char s[MM::MaxStrLength + 1];
		snprintf(s, MM::MaxStrLength, "%f", static_cast<double>(actualRate_));
		int mmerr = OnPropertyChanged(g_PropName_ActualRate, s);
		if (mmerr != DEVICE_OK)
			return mmerr;
	}

	return DEVICE_OK;
}


int VTiSIMScanner::DoGetScanning(bool& scanning)
{
	BOOL flag;
	DWORD err = vti_IsScanning(VTiHub()->GetScanAndMotorHandle(), &flag);
	if (err != VTI_SUCCESS)
		return err;
	scanning = (flag != FALSE);
	return DEVICE_OK;
}
/*// Ver 2.1.0.0 - FRAP - start
int VTiSIMScanner::PointAndFire(double x, double y, double pulseTime_us)
{
	DWORD err = vti_ActivatePhotobleaching(VTiHub()->GetScanAndMotorHandle(),
			TRUE);
	if (err != VTI_SUCCESS)
			return err;
	return DEVICE_OK;
}
// Ver 2.1.0.0 - FRAP - end*/

VTiSIMPinholeArray::VTiSIMPinholeArray() :
minFinePosition_(0),
	maxFinePosition_(1),
	curFinePosition_(6000),
	backlashCompensation_(100)
{
	memset(pinholePositions_, 0, sizeof(pinholePositions_));

	SetErrorText(VTI_ERR_TIMEOUT_OCCURRED, "Timeout occurred");
	SetErrorText(VTI_ERR_DEVICE_NOT_FOUND, "Device not found");
	SetErrorText(VTI_ERR_NOT_INITIALISED, "Device not initialized");
	SetErrorText(VTI_ERR_NOT_SUPPORTED, "Operation not supported");

	CreateIntegerProperty(g_PropName_Backlash, backlashCompensation_, false,
		new CPropertyAction(this, &VTiSIMPinholeArray::OnBacklashCompensation), true);
	SetPropertyLimits(g_PropName_Backlash, -500, 500);
}


VTiSIMPinholeArray::~VTiSIMPinholeArray()
{
}


int VTiSIMPinholeArray::Initialize()
{
	DWORD vterr = vti_GetMotorRange(VTiHub()->GetScanAndMotorHandle(),
		VTI_MOTOR_PINHOLE_ARRAY, &minFinePosition_, &maxFinePosition_);
	if (vterr != VTI_SUCCESS)
		return vterr;

	int err = DoGetPinholePositions(pinholePositions_);
	if (err != DEVICE_OK)
		return err;

	// Initialize our fine position to that of the 64 um pinhole, which is the
	// default pinhole where we will move to below after properties are set up.
	curFinePosition_ = pinholePositions_[VTI_PINHOLE_64_MICRON];

	err = CreateIntegerProperty(g_PropName_FinePosition, curFinePosition_, false,
		new CPropertyAction(this, &VTiSIMPinholeArray::OnFinePosition));
	if (err != DEVICE_OK)
		return err;
	err = SetPropertyLimits(g_PropName_FinePosition,
		minFinePosition_, maxFinePosition_);
	if (err != DEVICE_OK)
		return err;

	err = CreateIntegerProperty(g_PropName_PinholeSize,
		GetPinholeSizeUmForIndex(VTI_PINHOLE_64_MICRON), false,
		new CPropertyAction(this, &VTiSIMPinholeArray::OnPinholeSize));
	if (err != DEVICE_OK)
		return err;
	for (int i = 0; i < nSizes; ++i)
	{
		char s[MM::MaxStrLength + 1];
		snprintf(s, MM::MaxStrLength, "%d", GetPinholeSizeUmForIndex(i));
		err = AddAllowedValue(g_PropName_PinholeSize, s);
		if (err != DEVICE_OK)
			return err;
	}

	err = DoSetFinePosition(curFinePosition_, backlashCompensation_);
	if (err != DEVICE_OK)
		return err;

	return DEVICE_OK;
}


int VTiSIMPinholeArray::Shutdown()
{
	return DEVICE_OK;
}


void VTiSIMPinholeArray::GetName(char* name) const
{
	CDeviceUtils::CopyLimitedString(name, g_DeviceName_PinholeArray);
}


bool VTiSIMPinholeArray::Busy()
{
	return false;
}


int VTiSIMPinholeArray::OnFinePosition(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set(static_cast<long>(curFinePosition_));
	}
	else if (eAct == MM::AfterSet)
	{
		long v;
		pProp->Get(v);
		if (v == curFinePosition_)
			return DEVICE_OK;
		int err = DoSetFinePosition(v);
		if (err != DEVICE_OK)
			return err;
	}
	return DEVICE_OK;
}


int VTiSIMPinholeArray::OnPinholeSize(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		int sizeUm = GetPinholeSizeUmForIndex(GetNearestPinholeIndex(curFinePosition_));
		pProp->Set(static_cast<long>(sizeUm));
	}
	else if (eAct == MM::AfterSet)
	{
		long v;
		pProp->Get(v);
		int index = GetPinholeSizeIndex(v);
		if (index < 0 || index >= nSizes)
			return DEVICE_ERR; // Shouldn't happen

		int finePos = pinholePositions_[index];
		if (finePos == curFinePosition_)
			return DEVICE_OK;

		int err = DoSetFinePosition(finePos, backlashCompensation_);
		if (err != DEVICE_OK)
			return err;
	}
	return DEVICE_OK;
}


int VTiSIMPinholeArray::OnBacklashCompensation(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set(static_cast<long>(backlashCompensation_));
	}
	else if (eAct == MM::AfterSet)
	{
		long v;
		pProp->Get(v);
		backlashCompensation_ = v;
	}
	return DEVICE_OK;
}


VTiSIMHub* VTiSIMPinholeArray::VTiHub()
{
	return static_cast<VTiSIMHub*>(GetParentHub());
}


int VTiSIMPinholeArray::DoGetPinholePositions(int* positions)
{
	for (int i = 0; i < nSizes; ++i)
	{
		vt_int32 pos;

		VTI_EX_PARAM param;
		memset(&param, 0, sizeof(param)); // Just in case
		param.ParamOption = i;
		param.ArrayBytes = sizeof(vt_int32);
		param.pArray = &pos;

		DWORD err = vti_GetExtendedFeature(VTiHub()->GetScanAndMotorHandle(),
			VTI_FEATURE_GET_PINHOLE_SETTING, &param, sizeof(vt_int32));
		if (err != VTI_SUCCESS)
			return err;

		positions[i] = pos;
	}
	return DEVICE_OK;
}


int VTiSIMPinholeArray::DoSetFinePosition(int position, int backlashComp)
{
	position = ClipFinePositionToMotorRange(position);

	// First movement destination for backlash compensation
	int bcPos = ClipFinePositionToMotorRange(position + backlashComp);

	bool needBacklashComp = true;
	if (bcPos == position)
		needBacklashComp = false;
	if (bcPos > position && curFinePosition_ > bcPos)
		needBacklashComp = false;
	if (bcPos < position && curFinePosition_ < bcPos)
		needBacklashComp = false;

	if (needBacklashComp)
	{
		DWORD err = vti_MoveMotor(VTiHub()->GetScanAndMotorHandle(),
			VTI_MOTOR_PINHOLE_ARRAY, bcPos);
		if (err != VTI_SUCCESS)
			return err;
	}

	DWORD err = vti_MoveMotor(VTiHub()->GetScanAndMotorHandle(),
		VTI_MOTOR_PINHOLE_ARRAY, position);
	if (err != VTI_SUCCESS)
		return err;

	curFinePosition_ = position;

	char s[MM::MaxStrLength + 1];
	snprintf(s, MM::MaxStrLength, "%d", curFinePosition_);
	int mmerr = OnPropertyChanged(g_PropName_FinePosition, s);
	if (mmerr != DEVICE_OK)
		return mmerr;
	snprintf(s, MM::MaxStrLength, "%d",
		GetPinholeSizeUmForIndex(GetNearestPinholeIndex(curFinePosition_)));
	mmerr = OnPropertyChanged(g_PropName_PinholeSize, s);
	if (mmerr != DEVICE_OK)
		return mmerr;

	return DEVICE_OK;
}


int VTiSIMPinholeArray::GetNearestPinholeIndex(int finePosition) const
{
	// There are only several (actually, 7) pinhole positions, so we just do
	// the simplest thing: find the index of the position whose distance from
	// the given fine position is smallest. This avoids making assumptions
	// about the fine positions being monotonous.

	std::vector< std::pair<int, int> > distToIndex;
	for (int i = 0; i < nSizes; ++i)
	{
		int dist = finePosition - pinholePositions_[i];
		if (dist < 0)
			dist = -dist;
		distToIndex.push_back(std::make_pair(dist, i));
	}
	std::sort(distToIndex.begin(), distToIndex.end());
	return distToIndex[0].second;
}


int VTiSIMPinholeArray::GetPinholeSizeUmForIndex(int index) const
{
	switch (index)
	{
	case VTI_PINHOLE_30_MICRON: return 30;
	case VTI_PINHOLE_40_MICRON: return 40;
	case VTI_PINHOLE_50_MICRON: return 50;
	case VTI_PINHOLE_64_MICRON: return 64;
	case VTI_PINHOLE_25_MICRON: return 25;
	case VTI_PINHOLE_15_MICRON: return 15;
	case VTI_PINHOLE_10_MICRON: return 10;
	default: return 0; // Shouldn't happen
	}
}


int VTiSIMPinholeArray::GetPinholeSizeIndex(int sizeUm) const
{
	switch (sizeUm)
	{
	case 30: return VTI_PINHOLE_30_MICRON;
	case 40: return VTI_PINHOLE_40_MICRON;
	case 50: return VTI_PINHOLE_50_MICRON;
	case 64: return VTI_PINHOLE_64_MICRON;
	case 25: return VTI_PINHOLE_25_MICRON;
	case 15: return VTI_PINHOLE_15_MICRON;
	case 10: return VTI_PINHOLE_10_MICRON;
	default: return 0;
	}
}


int VTiSIMPinholeArray::ClipFinePositionToMotorRange(int finePosition) const
{
	if (finePosition < minFinePosition_)
		return minFinePosition_;
	if (finePosition > maxFinePosition_)
		return maxFinePosition_;
	return finePosition;
}

//---------------------------------------------------------
// Dichroic Motor
//---------------------------------------------------------
VTiSIMDichroic::VTiSIMDichroic():
minFinePosition_(1),
	maxFinePosition_(12000)

{
	SetErrorText(VTI_ERR_TIMEOUT_OCCURRED, "Timeout occurred");
	SetErrorText(VTI_ERR_DEVICE_NOT_FOUND, "Device not found");
	SetErrorText(VTI_ERR_NOT_INITIALISED, "Device not initialized");
	SetErrorText(VTI_ERR_NOT_SUPPORTED, "Operation not supported");
	DichroicPositions[0]=5;
	DichroicPositions[1]=2500;
	DichroicPositions[2]=4995;


	int err = CreateIntegerProperty(g_PropName_DichroicPos1, DichroicPositions[0], false,
		new CPropertyAction(this, &VTiSIMDichroic::OnDichroicPos), true);
	err = CreateIntegerProperty(g_PropName_DichroicPos2, DichroicPositions[1], false,
		new CPropertyAction(this, &VTiSIMDichroic::OnDichroicPos), true);
	err = CreateIntegerProperty(g_PropName_DichroicPos3, DichroicPositions[2], false,
		new CPropertyAction(this, &VTiSIMDichroic::OnDichroicPos), true);
	// Cannot get min position and max position here
	/* int vterr = vti_GetMotorRange(VTiHub()->GetScanAndMotorHandle(),
	VTI_MOTOR_PRIMARY_DICHROIC, &minFinePosition_, &maxFinePosition_);
	if (minFinePosition_== 0)
	minFinePosition_= 1;*/
	SetPropertyLimits(g_PropName_DichroicPos1, minFinePosition_, maxFinePosition_);
	SetPropertyLimits(g_PropName_DichroicPos2, minFinePosition_, maxFinePosition_);
	SetPropertyLimits(g_PropName_DichroicPos3, minFinePosition_, maxFinePosition_);
} 


VTiSIMHub* VTiSIMDichroic::VTiHub()
{
	return static_cast<VTiSIMHub*>(GetParentHub());
}


VTiSIMDichroic::~VTiSIMDichroic()
{
}


int VTiSIMDichroic::Initialize()
{

	// Add Labels for motor positions
	// Name
	int err = CreateStringProperty(MM::g_Keyword_Name, g_PropName_DichroicLabel, true);
	if (err != DEVICE_OK)
		return err;

	// Description
	err = CreateStringProperty(MM::g_Keyword_Description, "Dichroic Label", true);
	if (err != DEVICE_OK)
		return err;

	// Dichroic Position
	// create default positions and labels
	const int bufSize = 1024;
	char buf[bufSize];
	for (long i=0; i<nDSizes; i++)
	{
		snprintf(buf, bufSize, "Position-%ld", i+1);
		SetPositionLabel(i, buf);
		AddAllowedValue(g_PropName_DichroicLabel, buf);
	}
	// State
	CPropertyAction* pAct = new CPropertyAction (this, &VTiSIMDichroic::OnState);
	err = CreateIntegerProperty(MM::g_Keyword_State, 0, false, pAct);
	if (err != DEVICE_OK)
		return err;
	err = SetPropertyLimits(MM::g_Keyword_State, 0, nDSizes-1);
	if (err != DEVICE_OK)
		return err;

	// Label
	pAct = new CPropertyAction (this, &CStateBase::OnLabel);
	err = CreateStringProperty(MM::g_Keyword_Label, "", false, pAct);
	if (err != DEVICE_OK)
		return err;

	// Home Motor
	// Do not need to home the motor every time micromanager is started.
	/*DWORD vterr = vti_HomeMotor(VTiHub()->GetScanAndMotorHandle(),
	VTI_MOTOR_PRIMARY_DICHROIC);
	if (vterr != VTI_SUCCESS)
	return vterr;*/
	// Motor Current Position
	DWORD vterr = vti_GetMotorPosition(VTiHub()->GetScanAndMotorHandle(), VTI_MOTOR_PRIMARY_DICHROIC, &curDichroicPosition_);
	if (vterr != VTI_SUCCESS)
		return vterr;
	err = CreateIntegerProperty(g_PropName_DichroicPosition, curDichroicPosition_, false,
		new CPropertyAction(this, &VTiSIMDichroic::OnDichroicPosition));
	if (err != DEVICE_OK)
		return err;
	// Motor Range
	vterr = vti_GetMotorRange(VTiHub()->GetScanAndMotorHandle(),
		VTI_MOTOR_PRIMARY_DICHROIC, &minFinePosition_, &maxFinePosition_);
	if (vterr != VTI_SUCCESS)
		return vterr;
	if (minFinePosition_== 0)
		minFinePosition_= 1;
	err = SetPropertyLimits(g_PropName_DichroicPosition, minFinePosition_, maxFinePosition_);
	if (err != DEVICE_OK)
		return err;
	return DEVICE_OK;
}


int VTiSIMDichroic::Shutdown()
{
	return DEVICE_OK;
}


void VTiSIMDichroic::GetName(char* name) const
{
	CDeviceUtils::CopyLimitedString(name, g_DeviceName_Dichroic);
}


bool VTiSIMDichroic::Busy()
{
	return false;
}


int VTiSIMDichroic::OnDichroicPosition(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set(static_cast<long>(curDichroicPosition_));
	}
	else if (eAct == MM::AfterSet)
	{
		long v;
		pProp->Get(v);
		if (v==0)
			v = 1;
		else if (v>maxFinePosition_)
			v = maxFinePosition_;

		if (v == curDichroicPosition_)
			return DEVICE_OK;
		int err= DoMotorMove(v);
		curDichroicPosition_ = v;
		if (err != VTI_SUCCESS)
			return err;

		char s[MM::MaxStrLength + 1];
		snprintf(s, MM::MaxStrLength, "%d", curDichroicPosition_);
		int mmerr = OnPropertyChanged(g_PropName_DichroicPosition, s);
		if (mmerr != DEVICE_OK)
			return mmerr;
	}
	return DEVICE_OK;
}


int VTiSIMDichroic::OnDichroicPos(MM::PropertyBase* pProp, MM::ActionType eAct)
{ 
	int Pos=0;
	if (pProp->GetName()== g_PropName_DichroicPos1)
		Pos=0;
	else if(pProp->GetName()== g_PropName_DichroicPos2)
		Pos=1;
	else if(pProp->GetName()== g_PropName_DichroicPos3)
		Pos=2;
	if (eAct == MM::BeforeGet)
	{
		pProp->Set(static_cast<long>(DichroicPositions[Pos]));
	}
	else if (eAct == MM::AfterSet)
	{
		long v;
		pProp->Get(v);
		DichroicPositions[Pos]=v;
	}
	return DEVICE_OK;
}


int VTiSIMDichroic::DoMotorMove(int Pos)
{
	DWORD vterr = vti_MoveMotor(VTiHub()->GetScanAndMotorHandle(),
		VTI_MOTOR_PRIMARY_DICHROIC, Pos);
	if (vterr != VTI_SUCCESS)
		return vterr;

	return DEVICE_OK;
}


int VTiSIMDichroic::OnState(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{

	}
	else if (eAct == MM::AfterSet)
	{
		int  v=0;
		long state;
		pProp->Get(state);
		v = DichroicPositions[state];
		if (v > maxFinePosition_)
			v = maxFinePosition_;
		int err= DoMotorMove(v);
		curDichroicPosition_ = v;
		if (err != VTI_SUCCESS)
			return err;
	}
	return DEVICE_OK;
}


//---------------------------------------------------------
// Barrier Filter Motor
//---------------------------------------------------------
VTiSIMBarrierFilter::VTiSIMBarrierFilter():
minFinePosition_(0),
	maxFinePosition_(12000)

{
	SetErrorText(VTI_ERR_TIMEOUT_OCCURRED, "Timeout occurred");
	SetErrorText(VTI_ERR_DEVICE_NOT_FOUND, "Device not found");
	SetErrorText(VTI_ERR_NOT_INITIALISED, "Device not initialized");
	SetErrorText(VTI_ERR_NOT_SUPPORTED, "Operation not supported");

	FilterPositions[0]=100;
	FilterPositions[1]=200;
	FilterPositions[2]=300;
	FilterPositions[3]=400;
	FilterPositions[4]=500;
	FilterPositions[5]=600;

	int err = CreateIntegerProperty(g_PropName_BarrierFilterPos1, FilterPositions[0], false,
		new CPropertyAction(this, &VTiSIMBarrierFilter::OnFilterPos), true);
	err = CreateIntegerProperty(g_PropName_BarrierFilterPos2, FilterPositions[1], false,
		new CPropertyAction(this, &VTiSIMBarrierFilter::OnFilterPos), true);
	err = CreateIntegerProperty(g_PropName_BarrierFilterPos3, FilterPositions[2], false,
		new CPropertyAction(this, &VTiSIMBarrierFilter::OnFilterPos), true);
	err = CreateIntegerProperty(g_PropName_BarrierFilterPos4, FilterPositions[3], false,
		new CPropertyAction(this, &VTiSIMBarrierFilter::OnFilterPos), true);
	err = CreateIntegerProperty(g_PropName_BarrierFilterPos5, FilterPositions[4], false,
		new CPropertyAction(this, &VTiSIMBarrierFilter::OnFilterPos), true);
	err = CreateIntegerProperty(g_PropName_BarrierFilterPos6, FilterPositions[5], false,
		new CPropertyAction(this, &VTiSIMBarrierFilter::OnFilterPos), true);
	// Cannot get min position and max position here
	/*vti_GetMotorRange(VTiHub()->GetScanAndMotorHandle(),
	VTI_MOTOR_BARRIER_FILTER, &minFinePosition_, &maxFinePosition_);
	if (minFinePosition_== 0)
	minFinePosition_= 1;*/

	SetPropertyLimits(g_PropName_BarrierFilterPos1, minFinePosition_, maxFinePosition_);
	SetPropertyLimits(g_PropName_BarrierFilterPos2, minFinePosition_, maxFinePosition_);
	SetPropertyLimits(g_PropName_BarrierFilterPos3, minFinePosition_, maxFinePosition_);
	SetPropertyLimits(g_PropName_BarrierFilterPos4, minFinePosition_, maxFinePosition_);
	SetPropertyLimits(g_PropName_BarrierFilterPos5, minFinePosition_, maxFinePosition_);
	SetPropertyLimits(g_PropName_BarrierFilterPos6, minFinePosition_, maxFinePosition_);
}


VTiSIMHub* VTiSIMBarrierFilter::VTiHub()
{
	return static_cast<VTiSIMHub*>(GetParentHub());
}


VTiSIMBarrierFilter::~VTiSIMBarrierFilter()
{
}


int VTiSIMBarrierFilter::Initialize()
{
	// Home Motor
	// Do not need to home the motor every time micromanager is started.
	/*DWORD vterr = vti_HomeMotor(VTiHub()->GetScanAndMotorHandle(),
	VTI_MOTOR_BARRIER_FILTER);
	if (vterr != VTI_SUCCESS)
	return vterr;*/

	// Motor Current Position
	DWORD vterr = vti_GetMotorPosition(VTiHub()->GetScanAndMotorHandle(), VTI_MOTOR_BARRIER_FILTER, &curFilterPosition_);
	if (vterr != VTI_SUCCESS)
		return vterr;
	int err = CreateIntegerProperty(g_PropName_BarrierFilterPos, curFilterPosition_, false,
		new CPropertyAction(this, &VTiSIMBarrierFilter::OnBarrierFilterPosition));
	if (err != DEVICE_OK)
		return err;
	// Motor Range
	vterr = vti_GetMotorRange(VTiHub()->GetScanAndMotorHandle(),
		VTI_MOTOR_BARRIER_FILTER, &minFinePosition_, &maxFinePosition_);
	if (vterr != VTI_SUCCESS)
		return err;
	//if (minFinePosition_== 0)
	//	minFinePosition_= 1;
	err = SetPropertyLimits(g_PropName_BarrierFilterPos, minFinePosition_, maxFinePosition_);
	if (err != DEVICE_OK)
		return err;
	// Add Labels for motor position

	// create default positions and labels
	const int bufSize = 1024;
	char buf[bufSize];
	for (long i=0; i<nSizes; i++)
	{
		snprintf(buf, bufSize, "Position-%ld", i+1);
		SetPositionLabel(i, buf);
		AddAllowedValue(g_PropName_BarrierFilterLabel, buf);
	}
	// State
	CPropertyAction* pAct = new CPropertyAction (this, &VTiSIMBarrierFilter::OnState);
	err = CreateIntegerProperty(MM::g_Keyword_State, 0, false, pAct);
	if (err != DEVICE_OK)
		return err;
	err = SetPropertyLimits(MM::g_Keyword_State, 0, nSizes-1);
	if (err != DEVICE_OK)
		return err;

	// Label
	pAct = new CPropertyAction (this, &CStateBase::OnLabel);
	err = CreateStringProperty(MM::g_Keyword_Label, "", false, pAct);
	if (err != DEVICE_OK)
		return err;
	return DEVICE_OK;
}


int VTiSIMBarrierFilter::Shutdown()
{
	return DEVICE_OK;
}


void VTiSIMBarrierFilter::GetName(char* name) const
{
	CDeviceUtils::CopyLimitedString(name, g_DeviceName_BarrierFilter);
}


bool VTiSIMBarrierFilter::Busy()
{
	return false;
}


int VTiSIMBarrierFilter::OnBarrierFilterPosition(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set(static_cast<long>(curFilterPosition_));
	}
	else if (eAct == MM::AfterSet)
	{
		long v;
		pProp->Get(v);
		if (v == curFilterPosition_)
			return DEVICE_OK;
		if (v==0)
			v = 1;
		else if (v > maxFinePosition_)
			v = maxFinePosition_;
		int err= DoMotorMove(v);
		curFilterPosition_ = v;
		if (err != VTI_SUCCESS)
			return err;

		char s[MM::MaxStrLength + 1];
		snprintf(s, MM::MaxStrLength, "%d", curFilterPosition_);
		int mmerr = OnPropertyChanged(g_PropName_BarrierFilterPos, s);
		if (mmerr != DEVICE_OK)
			return mmerr;

	}
	return DEVICE_OK;
}


int VTiSIMBarrierFilter::OnFilterPos(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	int Pos=0;
	if (pProp->GetName()== g_PropName_BarrierFilterPos1)
		Pos=0;
	else if(pProp->GetName()== g_PropName_BarrierFilterPos2)
		Pos=1;
	else if(pProp->GetName()== g_PropName_BarrierFilterPos3)
		Pos=2;
	else if(pProp->GetName()== g_PropName_BarrierFilterPos4)
		Pos=3;
	else if(pProp->GetName()== g_PropName_BarrierFilterPos5)
		Pos=4;
	else if(pProp->GetName()== g_PropName_BarrierFilterPos6)
		Pos=5;
	if (eAct == MM::BeforeGet)
	{
		pProp->Set(static_cast<long>(FilterPositions[Pos]));
	}
	else if (eAct == MM::AfterSet)
	{
		long v;
		pProp->Get(v);
		if (v==0)
			v = 1;
		FilterPositions[Pos]=v;
	}

	return DEVICE_OK;
}


int VTiSIMBarrierFilter::DoMotorMove(int Pos)
{
	DWORD err = vti_MoveMotor(VTiHub()->GetScanAndMotorHandle(),
		VTI_MOTOR_BARRIER_FILTER, Pos);
	if (err != VTI_SUCCESS)
		return err;
	return DEVICE_OK;
}


int VTiSIMBarrierFilter::OnState(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{

	}
	else if (eAct == MM::AfterSet)
	{
		int  v=0;
		long state;
		pProp->Get(state);
		v = FilterPositions[state];
		if (v > maxFinePosition_)
			v = maxFinePosition_;
		int err= DoMotorMove(v);
		curFilterPosition_ = v;
		if (err != VTI_SUCCESS)
			return err;
	}
	return DEVICE_OK;
}

// Ver 2.1.0.0 - FRAP - start
VTiSIMFRAP::VTiSIMFRAP():
minFinePosition_(1),
	maxFinePosition_(4095),
	FRAPYRange_(930),
	FRAPXRange_(620),
	FRAPYMin_(1714),
	FRAPXMin_(1653),
	FRAPXOffset_(0), // Ver 2.3.2.0
	FRAPYOffset_(0) // Ver 2.3.2.0
{
	SetErrorText(VTI_ERR_TIMEOUT_OCCURRED, "Timeout occurred");
	SetErrorText(VTI_ERR_DEVICE_NOT_FOUND, "Device not found");
	SetErrorText(VTI_ERR_NOT_INITIALISED, "Device not initialized");
	SetErrorText(VTI_ERR_NOT_SUPPORTED, "Operation not supported");
}

VTiSIMHub* VTiSIMFRAP::VTiHub()
{
	return static_cast<VTiSIMHub*>(GetParentHub());
}

VTiSIMFRAP::~VTiSIMFRAP()
{
}

int VTiSIMFRAP::Initialize()
{		
	char s[MM::MaxStrLength + 1];
	snprintf(s, MM::MaxStrLength, "VTiSIMFRAP: Initialize");
	LogMessage(s);
	FRAPEnable = FALSE;	// Version 2.2.2.0	

	// Ver 2.3.0.0 - start
	int err = CreateIntegerProperty(g_PropVal_FRAPYRange, FRAPYRange_, false,
	new CPropertyAction(this, &VTiSIMFRAP::OnYRange));
	if (err != DEVICE_OK)
		return err;

	err = CreateIntegerProperty(g_PropVal_FRAPYMin, FRAPYMin_, false,
	new CPropertyAction(this, &VTiSIMFRAP::OnYMin));
	if (err != DEVICE_OK)
		return err;

	err = CreateIntegerProperty(g_PropVal_FRAPXRange, FRAPXRange_, false,
	new CPropertyAction(this, &VTiSIMFRAP::OnXRange));
	if (err != DEVICE_OK)
		return err;

	err = CreateIntegerProperty(g_PropVal_FRAPXMin, FRAPXMin_, false,
	new CPropertyAction(this, &VTiSIMFRAP::OnXMin));
	if (err != DEVICE_OK)
		return err;
	// Ver 2.3.0.0 - End

	// Ver 2.3.2.0 - Start
	err = CreateIntegerProperty(g_PropName_FRAPXOffset, FRAPXOffset_, false,
		new CPropertyAction(this, &VTiSIMFRAP::OnXOffset));
	if (err != DEVICE_OK)
		return err;
	int minXOffset;
	minXOffset= -1 * GetMaxXOffset();
	err = SetPropertyLimits(g_PropName_FRAPXOffset, minXOffset, GetMaxXOffset());

	if (err != DEVICE_OK)
		return err;
	// Ver 2.3.2.0 - Start

	// Ver 2.3.2.0 - Start
	err = CreateIntegerProperty(g_PropName_FRAPYOffset, FRAPYOffset_, false,
		new CPropertyAction(this, &VTiSIMFRAP::OnYOffset));
	if (err != DEVICE_OK)
		return err;
	int minYOffset;
	minYOffset= -1 * GetMaxYOffset();
	err = SetPropertyLimits(g_PropName_FRAPYOffset, minYOffset, GetMaxYOffset());

	if (err != DEVICE_OK)
		return err;
	// Ver 2.3.2.0 - Start

	return DEVICE_OK;
}

// Ver 2.3.2.0 - Start
int VTiSIMFRAP::OnXOffset(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set(static_cast<long>(FRAPXOffset_));
	}
	else if (eAct == MM::AfterSet)
	{
		long v;
		pProp->Get(v);
		if (v == FRAPXOffset_)
			return DEVICE_OK;
		return DoSetFRAPXOffset(v);
	}
	return DEVICE_OK;
}
// Ver 2.3.2.0 - End

// Ver 2.3.2.0 - Start
int VTiSIMFRAP::OnYOffset(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set(static_cast<long>(FRAPYOffset_));
	}
	else if (eAct == MM::AfterSet)
	{
		long v;
		pProp->Get(v);
		if (v == FRAPYOffset_)
			return DEVICE_OK;
		return DoSetFRAPYOffset(v);
	}
	return DEVICE_OK;
}
// Ver 2.3.2.0 - End

// Ver 2.3.2.0 - Start
int VTiSIMFRAP::DoSetFRAPXOffset(int XOffset)
{
	// To be able to set the negative scan Offset.
	/* if (offset < 0)
	offset = 0;*/
	if (XOffset < (-1)*GetMaxXOffset())
		XOffset = (-1)*GetMaxXOffset();

	if (XOffset > GetMaxXOffset())
		XOffset = GetMaxXOffset();

	FRAPXOffset_ = XOffset;

	return DEVICE_OK;
}

// Ver 2.3.0.0 - start

// Ver 2.3.2.0 - Start
int VTiSIMFRAP::DoSetFRAPYOffset(int YOffset)
{
	// To be able to set the negative scan Offset.
	/* if (offset < 0)
	offset = 0;*/
	if (YOffset < (-1)*GetMaxYOffset())
		YOffset = (-1)*GetMaxYOffset();

	if (YOffset > GetMaxYOffset())
		YOffset = GetMaxYOffset();

	FRAPYOffset_ = YOffset;

	return DEVICE_OK;
}

// Ver 2.3.0.0 - start


int VTiSIMFRAP::OnYRange(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set(static_cast<long>(FRAPYRange_));
	}
	else if (eAct == MM::AfterSet)
	{
		long v;
		pProp->Get(v);
		if (v == FRAPYRange_)
			return DEVICE_OK;
		return DoSetFRAPYRange(v);
	}
	return DEVICE_OK;
}
// Ver 2.3.0.0 - End

// Ver 2.3.0.0 - start
int VTiSIMFRAP::OnYMin(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set(static_cast<long>(FRAPYMin_));
	}
	else if (eAct == MM::AfterSet)
	{
		long v;
		pProp->Get(v);
		if (v == FRAPYMin_)
			return DEVICE_OK;
		return DoSetFRAPYMin(v);
	}
	return DEVICE_OK;
}
// Ver 2.3.0.0 - End

// Ver 2.3.0.0 - start
int VTiSIMFRAP::OnXRange(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set(static_cast<long>(FRAPXRange_));
	}
	else if (eAct == MM::AfterSet)
	{
		long v;
		pProp->Get(v);
		if (v == FRAPXRange_)
			return DEVICE_OK;
		return DoSetFRAPXRange(v);
	}
	return DEVICE_OK;
}
// Ver 2.3.0.0 - End

// Ver 2.3.0.0 - start
int VTiSIMFRAP::OnXMin(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set(static_cast<long>(FRAPXMin_));
	}
	else if (eAct == MM::AfterSet)
	{
		long v;
		pProp->Get(v);
		if (v == FRAPXMin_)
			return DEVICE_OK;
		return DoSetFRAPXMin(v);
	}
	return DEVICE_OK;
}// Ver 2.3.0.0 - End



// Ver 2.3.0.0 - start
int VTiSIMFRAP::DoSetFRAPYRange(int yRange)
{
	if (yRange < minFinePosition_)
		yRange = minFinePosition_;
	if (yRange > maxFinePosition_)
		yRange = maxFinePosition_;

	FRAPYRange_ = yRange;

	return DEVICE_OK;
}

int VTiSIMFRAP::DoSetFRAPYMin(int yMin)
{
	if (yMin < minFinePosition_)
		yMin = minFinePosition_;
	if (yMin > maxFinePosition_)
		yMin = maxFinePosition_;

	FRAPYMin_ = yMin;

	return DEVICE_OK;
}

int VTiSIMFRAP::DoSetFRAPXRange(int xRange)
{
	if (xRange < minFinePosition_)
		xRange = minFinePosition_;
	if (xRange > maxFinePosition_)
		xRange = maxFinePosition_;

	FRAPXRange_ = xRange;

	return DEVICE_OK;
}

int VTiSIMFRAP::DoSetFRAPXMin(int xMin)
{
	if (xMin < minFinePosition_)
		xMin = minFinePosition_;
	if (xMin > maxFinePosition_)
		xMin = maxFinePosition_;

	FRAPXMin_ = xMin;

	return DEVICE_OK;
}
	// Ver 2.3.0.0 - End

int VTiSIMFRAP::Shutdown()
{
	return DEVICE_OK;
}

void VTiSIMFRAP::GetName(char* name) const
{
	CDeviceUtils::CopyLimitedString(name, g_DeviceName_FRAP);
}

bool VTiSIMFRAP::Busy()
{
	return false;
}

// headers...
int VTiSIMFRAP::PointAndFire(double x, double y, double pulseTime_us)
{
	char s[MM::MaxStrLength + 1];
	// Ver 2.3.0.0 - start
	DWORD err = vti_SetShutter(VTiHub()->GetAOTFHandle(), FALSE);
	if (err != VTI_SUCCESS)
		return err;
	snprintf(s, MM::MaxStrLength, "FRAP_Pointandfire, close shutter");
	LogMessage(s);
	// Ver 2.3.0.0 - End

	SetIlluminationState(FALSE);
	long_x = (long) x;
	snprintf(s, MM::MaxStrLength, "FRAP_Pointandfire, x: %ld",long_x);
	LogMessage(s);

	long_y = (long) y;
	snprintf(s, MM::MaxStrLength, "FRAP_Pointandfire, y: %ld",long_y);
	LogMessage(s);


	// Ver 2.3.2.0 - Start
	snprintf(s, MM::MaxStrLength, "FRAP_Pointandfire, x offset: %ld",FRAPXOffset_);
	LogMessage(s);
	snprintf(s, MM::MaxStrLength, "FRAP_Pointandfire, y offset: %ld",FRAPYOffset_);
	LogMessage(s);

	x = x + FRAPXOffset_;
	if (x >4094)
		{ x = 4094;}
	long_x = (long) x;
	snprintf(s, MM::MaxStrLength, "FRAP_Pointandfire, x (with offset): %ld",long_x);
	LogMessage(s);

	y = y + FRAPYOffset_;
	if (y >4094)
		{ y = 4094;}
	long_y = (long) y;
	snprintf(s, MM::MaxStrLength, "FRAP_Pointandfire, y (with offset): %ld",long_y);
	LogMessage(s);
	// Ver 2.3.2.0 - End


	long_pulseTime_us = (long) pulseTime_us;
	snprintf(s, MM::MaxStrLength, "FRAP_Pointandfire, pulseTime_us: %ld",long_pulseTime_us);
	LogMessage(s);

	// Ver 2.2.1.0 - Start - close shutter when switching between FRAP and Imaging mode
	/*this->SetPosition(x, y);
	DWORD err = vti_ActivatePhotobleaching(VTiHub()->GetScanAndMotorHandle(),TRUE);
	if (err != VTI_SUCCESS)
	return err;*/

	/*DWORD err = vti_SetShutter(VTiHub()->GetAOTFHandle(), FALSE);
	if (err != VTI_SUCCESS)
		return err;*/
	
	// Ver 2.3.1.0 - Start
	VTI_EX_PARAM Param;
	Param.ParamOption = VTI_RECTIFIED_INTENSITY_CONSTANT;
	err = vti_SetExtendedFeature(VTiHub()->GetAOTFHandle(), VTI_FEATURE_RECTIFIED_INTENSITY, &Param, sizeof(Param));
	if (err != VTI_SUCCESS)
	{
		snprintf(s, MM::MaxStrLength, "VTI_RECTIFIED_INTENSITY_CONSTANT-err");
		LogMessage(s);
		return err;
	}
	snprintf(s, MM::MaxStrLength, "VTI_RECTIFIED_INTENSITY_CONSTANT");
	LogMessage(s);
	// Ver 2.3.1.0 - End

	 err = vti_ActivatePhotobleaching(VTiHub()->GetScanAndMotorHandle(),TRUE);
	if (err != VTI_SUCCESS)
	return err;

	this->SetPosition(x, y);

	CDeviceUtils::SleepMs(500);

	// Ver 2.3.1.0 - start
	snprintf(s, MM::MaxStrLength, "Set Position");
	LogMessage(s);
	// Ver 2.3.1.0 - End 

	 err = vti_SetShutter(VTiHub()->GetAOTFHandle(), TRUE);
	if (err != VTI_SUCCESS)
		return err;

	// Ver 2.3.0.0 - Start 
	snprintf(s, MM::MaxStrLength, "FRAP_Pointandfire, open shutter");
	LogMessage(s);
	// Ver 2.3.0.0 - end

	// Ver 2.2.1.0 - End - close shutter when switching between FRAP and Imaging mode

	snprintf(s, MM::MaxStrLength, "FRAP_Activate");
	LogMessage(s);

	SetIlluminationState(TRUE);
	long pulseTime_Ms = (long)(pulseTime_us/1000 +0.5);
	snprintf(s, MM::MaxStrLength, "FRAP_Pointandfire, Start_pulseTime_Ms: %ld",pulseTime_Ms);
	LogMessage(s);

	CDeviceUtils::SleepMs(pulseTime_Ms);
	snprintf(s, MM::MaxStrLength, "FRAP_Pointandfire, End_pulseTime_Ms: %ld",pulseTime_Ms);
	LogMessage(s);
		// Ver 2.3.0.0 - start
	err = vti_SetShutter(VTiHub()->GetAOTFHandle(), FALSE);
	if (err != VTI_SUCCESS)
		return err;
	snprintf(s, MM::MaxStrLength, "FRAP_Pointandfire, close shutter");
	LogMessage(s);
	// Ver 2.3.0.0 - End

	// Ver 2.3.1.0 - Start
	Param.ParamOption = VTI_RECTIFIED_INTENSITY_RAMPED_NORMAL;
	err = vti_SetExtendedFeature(VTiHub()->GetAOTFHandle(), VTI_FEATURE_RECTIFIED_INTENSITY, &Param, sizeof(Param));
	if (err != VTI_SUCCESS)
	{
		snprintf(s, MM::MaxStrLength, "VTI_RECTIFIED_INTENSITY_RAMPED_NORMAL-err");
		LogMessage(s);
		return err;
	}
	snprintf(s, MM::MaxStrLength, "VTI_RECTIFIED_INTENSITY_RAMPED_NORMAL");
	LogMessage(s);
	// Ver 2.3.1.0 - End

	SetIlluminationState(FALSE);
	snprintf(s, MM::MaxStrLength, "FRAP_Pointandfire, end ");
	LogMessage(s);
	return DEVICE_OK;
}

int VTiSIMFRAP::SetSpotInterval(double pulseTime_us)
{
	char s[MM::MaxStrLength + 1];
	long_pulseTime_us = (long) pulseTime_us;
	snprintf(s, MM::MaxStrLength, "FRAP_SetSpotInterval, pulseTime_us: %ld",long_pulseTime_us);
	LogMessage(s);
	return DEVICE_OK;
}
int VTiSIMFRAP::SetPosition(double x, double y)
{
	char s[MM::MaxStrLength + 1];
	VTI_EX_PARAM param;

	snprintf(s, MM::MaxStrLength, "FRAP_SetPosition, double x: %f",x);
	LogMessage(s);
	snprintf(s, MM::MaxStrLength, "FRAP_SetPosition, double y: %f",y);
	LogMessage(s);

	Position_x = (unsigned short)x;
	Position_y = (unsigned short)y;


	DWORD err = vti_SetPhotobleachPoint(VTiHub()->GetScanAndMotorHandle(),Position_x,Position_y);
	if (err != VTI_SUCCESS)
		return err;
	
	// Ver 2.2.1.0 - Start
	snprintf(s, MM::MaxStrLength, "FRAP_SetPosition_set photobleach point");
	LogMessage(s);

	snprintf(s, MM::MaxStrLength, "FRAP_SetPosition_set Galvo X position pre , X: %ld",Position_x);
	LogMessage(s);

	param.pArray = &Position_x;
	param.ArrayBytes = sizeof(vt_int16);
	err = vti_SetExtendedFeature(VTiHub()->GetScanAndMotorHandle(), VTI_FEATURE_GALVO_X_POSITION, &param, sizeof(param));
	if (err != VTI_SUCCESS)
	{
		snprintf(s, MM::MaxStrLength, "FRAP_SetPosition_set Galvo X position, error: %ld",err);
		LogMessage(s);
		return err;
	}
	snprintf(s, MM::MaxStrLength, "FRAP_SetPosition_set Galvo X position post, X: %ld",Position_x);
	LogMessage(s);

	param.pArray = &Position_y;
	param.ArrayBytes = sizeof(vt_int16);
	err = vti_SetExtendedFeature(VTiHub()->GetScanAndMotorHandle(), VTI_FEATURE_GALVO_Y_POSITION, &param, sizeof(param));
	if (err != VTI_SUCCESS)
		return err;

	snprintf(s, MM::MaxStrLength, "FRAP_SetPosition_set Galvo Y position, Y: %ld",Position_y);
	LogMessage(s);
	// Ver 2.2.1.0 - End

	return DEVICE_OK;
}
int VTiSIMFRAP::GetPosition(double& x, double& y)
{
	char s[MM::MaxStrLength + 1];

	//long_x = x; // Ver 2.2.1.0
	//long_y = y; // Ver 2.2.1.0

	// Ver 2.2.1.0 - start
	VTI_EX_PARAM param;
	Position_x = 0;
	Position_y = 0;


	param.pArray = &Position_x;
	param.ArrayBytes = sizeof(vt_int16);

	DWORD err = vti_GetExtendedFeature(VTiHub()->GetScanAndMotorHandle(),VTI_FEATURE_GALVO_X_POSITION, &param, sizeof(param));
	if (err != VTI_SUCCESS)
			return err;
	
	x = Position_x;
	
	snprintf(s, MM::MaxStrLength, "FRAP_GetPosition_Galvo x position, X: %f",x);
	LogMessage(s);

	param.pArray = &Position_y;
	param.ArrayBytes = sizeof(vt_int16);

	err = vti_GetExtendedFeature(VTiHub()->GetScanAndMotorHandle(),VTI_FEATURE_GALVO_Y_POSITION, &param, sizeof(param));
	if (err != VTI_SUCCESS)
			return err;
	
	y = Position_y;
	snprintf(s, MM::MaxStrLength, "FRAP_GetPosition_Galvo y position, Y: %f",y);
	LogMessage(s);

	// Ver 2.2.1.0 - End
	return DEVICE_OK;
}
int VTiSIMFRAP::SetIlluminationState(bool on)
{
	DWORD err;
	char s[MM::MaxStrLength + 1];

	if ( on==true)
	{
		snprintf(s, MM::MaxStrLength, "FRAP_SetIlluminationState, turn ON");
		LogMessage(s);
		err = vti_ActivatePhotobleaching(VTiHub()->GetScanAndMotorHandle(),TRUE);
			if (err != VTI_SUCCESS)
			{
				return err;
			}
		snprintf(s, MM::MaxStrLength, "FRAP_SetIlluminationState, turn ON - success");
		LogMessage(s);
	}
	else
	{
		snprintf(s, MM::MaxStrLength, "FRAP_SetIlluminationState, turn off");
		LogMessage(s);
		err = vti_ActivatePhotobleaching(VTiHub()->GetScanAndMotorHandle(),FALSE);
			if (err != VTI_SUCCESS)
			{
				return err;
			}
		snprintf(s, MM::MaxStrLength, "FRAP_SetIlluminationState, turn off - success");
		LogMessage(s);
	}
	return DEVICE_OK;

	/*DWORD err;
	VTI_EX_PARAM Param;
	char s[MM::MaxStrLength + 1];

	if ( on==true)
	{
		snprintf(s, MM::MaxStrLength, "FRAP_SetIlluminationState, turn ON");
		LogMessage(s);
		
		if (FRAPEnable==false)
		{
			snprintf(s, MM::MaxStrLength, "FRAP_SetIlluminationState, ON");
			LogMessage(s);
			err = vti_ActivatePhotobleaching(VTiHub()->GetScanAndMotorHandle(),TRUE);
			if (err != VTI_SUCCESS)
			{
				return err;
			}
			// Ver 2.2.2.0 - start
			Param.ParamOption = VTI_RECTIFIED_INTENSITY_CONSTANT;
			err = vti_SetExtendedFeature(VTiHub()->GetAOTFHandle(), VTI_FEATURE_RECTIFIED_INTENSITY, &Param, sizeof(Param));
			if (err != VTI_SUCCESS)
			{
				snprintf(s, MM::MaxStrLength, "VTI_RECTIFIED_INTENSITY_CONSTANT-err");
					LogMessage(s);
				return err;
			}
			snprintf(s, MM::MaxStrLength, "VTI_RECTIFIED_INTENSITY_CONSTANT");
			LogMessage(s);
			// Ver 2.2.2.0 - end
			FRAPEnable = true;
		}
		else
		{
			snprintf(s, MM::MaxStrLength, "FRAP_SetIlluminationState, its already on!");
			LogMessage(s);
			return DEVICE_OK;
		}
	}

	if ( on==false)
	{
		snprintf(s, MM::MaxStrLength, "FRAP_SetIlluminationState, turn off");
		LogMessage(s);

		if (FRAPEnable==true)
		{
			snprintf(s, MM::MaxStrLength, "FRAP_SetIlluminationState, OFF");
			LogMessage(s);
			err = vti_ActivatePhotobleaching(VTiHub()->GetScanAndMotorHandle(),FALSE);
			if (err != VTI_SUCCESS)
			{
				return err;
			}
			// Ver 2.2.2.0 - start
			Param.ParamOption = VTI_RECTIFIED_INTENSITY_RAMPED_NORMAL;
			err = vti_SetExtendedFeature(VTiHub()->GetAOTFHandle(), VTI_FEATURE_RECTIFIED_INTENSITY, &Param, sizeof(Param));
			if (err != VTI_SUCCESS)
			{
				snprintf(s, MM::MaxStrLength, "VTI_RECTIFIED_INTENSITY_CONSTANT-err");
				LogMessage(s);
				return err;
			}
			snprintf(s, MM::MaxStrLength, "VTI_RECTIFIED_INTENSITY_RAMPED_NORMAL");
			LogMessage(s);
			// Ver 2.2.2.0 - end
			FRAPEnable = false;
		}
		else
		{
			snprintf(s, MM::MaxStrLength, "FRAP_SetIlluminationState, its already off!");
			LogMessage(s);
			return DEVICE_OK;
		}
	}
	return DEVICE_OK;*/


	/*DWORD err;
	VTI_EX_PARAM Param;
	char s[MM::MaxStrLength + 1];

	BOOL FRAPActive;
	err = vti_IsScanning(VTiHub()->GetScanAndMotorHandle(), &FRAPActive);
	if (err != VTI_SUCCESS)
	{
		return err;
	}
	snprintf(s, MM::MaxStrLength, FRAPEnable?"true" : "false", stdout);
	LogMessage(s);
	if ( on==true)
	{
		snprintf(s, MM::MaxStrLength, "FRAP_SetIlluminationState, turn ON");
		LogMessage(s);

		if (FRAPEnable==false)
		{
			snprintf(s, MM::MaxStrLength, "FRAP_SetIlluminationState, ON");
			LogMessage(s);
			err = vti_ActivatePhotobleaching(VTiHub()->GetScanAndMotorHandle(),TRUE);
			if (err != VTI_SUCCESS)
			{
				return err;
			}
			// Ver 2.2.2.0 - start
			Param.ParamOption = VTI_RECTIFIED_INTENSITY_CONSTANT;
			err = vti_SetExtendedFeature(VTiHub()->GetAOTFHandle(), VTI_FEATURE_RECTIFIED_INTENSITY, &Param, sizeof(Param));
			if (err != VTI_SUCCESS)
			{
				snprintf(s, MM::MaxStrLength, "VTI_RECTIFIED_INTENSITY_CONSTANT-err");
					LogMessage(s);
				return err;
			}
			snprintf(s, MM::MaxStrLength, "VTI_RECTIFIED_INTENSITY_CONSTANT");
			LogMessage(s);
			// Ver 2.2.2.0 - end
			FRAPEnable = true;
		}
		else
		{
			snprintf(s, MM::MaxStrLength, "FRAP_SetIlluminationState, its already on!");
			LogMessage(s);
			return DEVICE_OK;
		}
	}
	else
	{
		if (FRAPEnable==true)
		{
			snprintf(s, MM::MaxStrLength, "FRAP_SetIlluminationState, OFF");
			LogMessage(s);
			err = vti_ActivatePhotobleaching(VTiHub()->GetScanAndMotorHandle(),FALSE);
			if (err != VTI_SUCCESS)
			{
				return err;
			}
			// Ver 2.2.2.0 - start
			Param.ParamOption = VTI_RECTIFIED_INTENSITY_RAMPED_NORMAL;
			err = vti_SetExtendedFeature(VTiHub()->GetAOTFHandle(), VTI_FEATURE_RECTIFIED_INTENSITY, &Param, sizeof(Param));
			if (err != VTI_SUCCESS)
			{
				snprintf(s, MM::MaxStrLength, "VTI_RECTIFIED_INTENSITY_CONSTANT-err");
				LogMessage(s);
				return err;
			}
			snprintf(s, MM::MaxStrLength, "VTI_RECTIFIED_INTENSITY_RAMPED_NORMAL");
			LogMessage(s);
			// Ver 2.2.2.0 - end
			FRAPEnable = false;
		}
		else
		{
			snprintf(s, MM::MaxStrLength, "FRAP_SetIlluminationState, its already off!");
			LogMessage(s);
			return DEVICE_OK;
		}
	}
	

 return DEVICE_OK;*/
}

int VTiSIMFRAP::AddPolygonVertex(int polygonIndex, double x, double y)
{
	/*char s[MM::MaxStrLength + 1];
	long_x = x;
	long_y = y;
	snprintf(s, MM::MaxStrLength, "FRAP_polygonIndex, x: %ld",long_x);
	LogMessage(s);	
	snprintf(s, MM::MaxStrLength, "FRAP_polygonIndex, y: %ld",long_y);	
	LogMessage(s);

	if (polygons_.size() <  (unsigned) (1 + polygonIndex))
   polygons_.resize(polygonIndex + 1);
   polygons_[polygonIndex].first = x;
   polygons_[polygonIndex].second = y;
   return DEVICE_OK;*/
	
	char s[MM::MaxStrLength + 1];
	long_x = (long) x;
	long_y = (long) y;
	snprintf(s, MM::MaxStrLength, "FRAP_polygonIndex, x: %ld",long_x);
	LogMessage(s);	
	snprintf(s, MM::MaxStrLength, "FRAP_polygonIndex, y: %ld",long_y);	
	LogMessage(s);
	snprintf(s, MM::MaxStrLength, "FRAP_polygonIndex, polygonIndex: %ld",polygonIndex);	
	LogMessage(s);
	if (polygons_.size() <  (unsigned) (1 + polygonIndex))
	{
		polygons_.resize(polygonIndex + 1);
	}
	polygons_[polygonIndex].push_back(std::pair<double,double>(x,y));

	

	snprintf(s, MM::MaxStrLength, "FRAP_polygonIndex end");
	LogMessage(s);
	return DEVICE_OK;
}

int VTiSIMFRAP::DeletePolygons()
{
	char s[MM::MaxStrLength + 1];
	snprintf(s, MM::MaxStrLength, "FRAP_DeletePolygons");
	LogMessage(s);
	polygons_.clear();
	return DEVICE_OK;
}
int VTiSIMFRAP::LoadPolygons()
{
	char s[MM::MaxStrLength + 1];
	snprintf(s, MM::MaxStrLength, "FRAP_LoadPolygons");
	LogMessage(s);
	return DEVICE_OK;
}

int VTiSIMFRAP::SetPolygonRepetitions(int repetitions)
{
	char s[MM::MaxStrLength + 1];
	long_repetitions = repetitions;
	snprintf(s, MM::MaxStrLength, "FRAP_SetPolygonRepetitions, repetitions: %ld",long_repetitions);
	LogMessage(s);
	polygonRepetitions_ = repetitions;
	return DEVICE_OK;
}

int VTiSIMFRAP::RunPolygons()
{
	char s[MM::MaxStrLength + 1];
	snprintf(s, MM::MaxStrLength, "FRAP_RunPolygons");
	LogMessage(s);
	for (int j=0; j<polygonRepetitions_; ++j)
	{
		snprintf(s, MM::MaxStrLength, "FRAP_RunPolygons, polygonRepetitions_: %ld",j);
		LogMessage(s);
		 for (int i=0; i< (int) polygons_.size(); ++i)			//polygons_.size() is number of ROI
		{
			snprintf(s, MM::MaxStrLength, "FRAP_RunPolygons, size: %ld",i);
			LogMessage(s);
			FirstVertex = true;	
			for (size_t n = sizeof(polygons_[i]); n--;)
			{
				double Vertex_x = polygons_[i][n].first;
				double Vertex_y = polygons_[i][n].second;
				
				// Check if the Vertex coordinates are in range

				if ( Vertex_x >GALVO_X_MIN && Vertex_x<GALVO_X_MAX && Vertex_y>GALVO_Y_MIN && Vertex_y<GALVO_Y_MAX)
				{
					snprintf(s, MM::MaxStrLength, "FRAP_RunPolygons, n: %ld", (long) n);
					LogMessage(s);

					snprintf(s, MM::MaxStrLength, "FRAP_RunPolygons, x: %f",Vertex_x);
					LogMessage(s);

					snprintf(s, MM::MaxStrLength, "FRAP_RunPolygons, y: %f",Vertex_y);
					LogMessage(s);

					//PointAndFire(x,y,20000);


					// Determine the coordinates of the active region
					if(FirstVertex)
					{
						BleachingRegionLeft = (long) polygons_[i][n].first;
						BleachingRegionRight = (long) polygons_[i][n].first;
						BleachingRegionTop = (long) polygons_[i][n].second;
						BleachingRegionBottom = (long) polygons_[i][n].second;
						FirstVertex = false;
					}
					else
					{
						if(Vertex_x<BleachingRegionLeft)
						{
							BleachingRegionLeft = (long) Vertex_x;
						}

						if(BleachingRegionRight < (long) Vertex_x)
						{
							BleachingRegionRight = (long) Vertex_x;
						}

						if(BleachingRegionTop < (long) Vertex_y)
						{
							BleachingRegionTop = (long) Vertex_y;
						}

						if(Vertex_y < BleachingRegionBottom)
						{
							BleachingRegionBottom = (long) Vertex_y;
						}

					}
				}
				else
				{
					snprintf(s, MM::MaxStrLength, "FRAP_RunPolygons, outside the range");
					LogMessage(s);
				}
			}

			snprintf(s, MM::MaxStrLength, "FRAP_RunPolygons, BleachingRegionLeft: %ld",BleachingRegionLeft);
			LogMessage(s);

			snprintf(s, MM::MaxStrLength, "FRAP_RunPolygons, BleachingRegionRight: %ld",BleachingRegionRight);
			LogMessage(s);

			snprintf(s, MM::MaxStrLength, "FRAP_RunPolygons, BleachingRegionTop: %ld",BleachingRegionTop);
			LogMessage(s);

			snprintf(s, MM::MaxStrLength, "FRAP_RunPolygons, BleachingRegionBottom: %ld",BleachingRegionBottom);
			LogMessage(s);		

			// Ver 2.3.2.0 - Start
			snprintf(s, MM::MaxStrLength, "FRAP_Pointandfire, x offset: %ld",FRAPXOffset_);
			LogMessage(s);
			snprintf(s, MM::MaxStrLength, "FRAP_Pointandfire, y offset: %ld",FRAPYOffset_);
			LogMessage(s);

			BleachingRegionLeft = BleachingRegionLeft + FRAPXOffset_;
			if (BleachingRegionLeft >4094)
				{ BleachingRegionLeft = 4094;}
			BleachingRegionRight = BleachingRegionRight + FRAPXOffset_;
			if (BleachingRegionRight >4094)
				{ BleachingRegionRight = 4094;}
			BleachingRegionTop = BleachingRegionTop + FRAPYOffset_;
			if (BleachingRegionTop >4094)
				{ BleachingRegionTop = 4094;}
			BleachingRegionBottom = BleachingRegionBottom + FRAPYOffset_;
			if (BleachingRegionBottom >4094)
				{ BleachingRegionBottom = 4094;}

			snprintf(s, MM::MaxStrLength, "FRAP_RunPolygons, BleachingRegionLeft: %ld",BleachingRegionLeft);
			LogMessage(s);

			snprintf(s, MM::MaxStrLength, "FRAP_RunPolygons, BleachingRegionRight: %ld",BleachingRegionRight);
			LogMessage(s);

			snprintf(s, MM::MaxStrLength, "FRAP_RunPolygons, BleachingRegionTop: %ld",BleachingRegionTop);
			LogMessage(s);

			snprintf(s, MM::MaxStrLength, "FRAP_RunPolygons, BleachingRegionBottom: %ld",BleachingRegionBottom);
			LogMessage(s);		
			// Ver 2.3.2.0 - End

			// Ver 2.3.0.0 - Start
			DWORD err = vti_SetShutter(VTiHub()->GetAOTFHandle(), FALSE);
			snprintf(s, MM::MaxStrLength, "FRAP_RunPolygons, Close shutter");
			LogMessage(s);

			if (err != VTI_SUCCESS){
				snprintf(s, MM::MaxStrLength, "FRAP_RunPolygons,vti_SetShutter_Fails, open: error: %ld",err);
				LogMessage(s);
				return err;
			}
			// Ver 2.3.0.0 - End

			
			// Ver 2.3.1.0 - Start
			VTI_EX_PARAM Param;
			Param.ParamOption = VTI_RECTIFIED_INTENSITY_CONSTANT;
			err = vti_SetExtendedFeature(VTiHub()->GetAOTFHandle(), VTI_FEATURE_RECTIFIED_INTENSITY, &Param, sizeof(Param));
			if (err != VTI_SUCCESS)
			{
				snprintf(s, MM::MaxStrLength, "VTI_RECTIFIED_INTENSITY_CONSTANT-err");
				LogMessage(s);
				return err;
			}
			snprintf(s, MM::MaxStrLength, "VTI_RECTIFIED_INTENSITY_CONSTANT");
			LogMessage(s);
			// Ver 2.3.1.0 - End

			err = vti_SetPhotobleachRegion(VTiHub()->GetScanAndMotorHandle(),BleachingRegionLeft,BleachingRegionRight,BleachingRegionBottom,BleachingRegionTop);

			if (err != VTI_SUCCESS){
				snprintf(s, MM::MaxStrLength, "FRAP_RunPolygons, vti_SetPhotobleachRegion_Fails: error: %ld",err);
					LogMessage(s);
				return err;
			}
				

			 err = vti_ActivatePhotobleaching(VTiHub()->GetScanAndMotorHandle(),TRUE);

			 	// Ver 2.3.0.0 - Start
			snprintf(s, MM::MaxStrLength, "FRAP_RunPolygons, FRAP Polygon");
			LogMessage(s);
			// Ver 2.3.0.0 - End
			if (err != VTI_SUCCESS){
				snprintf(s, MM::MaxStrLength, "FRAP_RunPolygons, vti_ActivatePhotobleaching_Fails: error: %ld",err);
					LogMessage(s);
				return err;
			}

			err = vti_SetShutter(VTiHub()->GetAOTFHandle(), TRUE);
			// Ver 2.3.0.0 - Start
			snprintf(s, MM::MaxStrLength, "FRAP_RunPolygons, Open shutter");
			LogMessage(s);
			// Ver 2.3.0.0 - End
			if (err != VTI_SUCCESS){
				snprintf(s, MM::MaxStrLength, "FRAP_RunPolygons,vti_SetShutter_Fails, open: error: %ld",err);
				LogMessage(s);
				return err;
			}
			CDeviceUtils::SleepMs(500);

			err = vti_SetShutter(VTiHub()->GetAOTFHandle(), FALSE);
			// Ver 2.3.0.0 - Start
			snprintf(s, MM::MaxStrLength, "FRAP_RunPolygons, Close shutter");
			LogMessage(s);
			// Ver 2.3.0.0 - End

			// Ver 2.3.1.0 - Start
			Param.ParamOption = VTI_RECTIFIED_INTENSITY_RAMPED_NORMAL;
			err = vti_SetExtendedFeature(VTiHub()->GetAOTFHandle(), VTI_FEATURE_RECTIFIED_INTENSITY, &Param, sizeof(Param));
			if (err != VTI_SUCCESS)
			{
				snprintf(s, MM::MaxStrLength, "VTI_RECTIFIED_INTENSITY_RAMPED_NORMAL-err");
				LogMessage(s);
				return err;
			}
			snprintf(s, MM::MaxStrLength, "VTI_RECTIFIED_INTENSITY_RAMPED_NORMAL");
			LogMessage(s);
			// Ver 2.3.1.0 - End

			if (err != VTI_SUCCESS){
				snprintf(s, MM::MaxStrLength, "FRAP_RunPolygons,vti_SetShutter_Fails,close: error: %ld",err);
				LogMessage(s);
				return err;
			}
			 err = vti_ActivatePhotobleaching(VTiHub()->GetScanAndMotorHandle(),FALSE);
			/* err = vti_ActivatePhotobleaching(VTiHub()->GetScanAndMotorHandle(),false);
				if (err != VTI_SUCCESS){
					snprintf(s, MM::MaxStrLength, "FRAP_RunPolygons, vti_ActivatePhotobleaching_Fails: error: %ld",err);
					LogMessage(s);
					return err;
				}
			CDeviceUtils::SleepMs(800);
			snprintf(s, MM::MaxStrLength, "FRAP_RunPolygons, vti_ActivatePhotobleaching - end");
			LogMessage(s);*/

			/* x = polygons_[i].first;
			snprintf(s, MM::MaxStrLength, "FRAP_RunPolygons, 1 x: %f",x);
			LogMessage(s);
			 y = polygons_[i][1].second;
			snprintf(s, MM::MaxStrLength, "FRAP_RunPolygons, 1 y: %f",y);
			LogMessage(s);*/
		}
	}
	return DEVICE_OK;
}
	


int VTiSIMFRAP::RunSequence()
{
   return DEVICE_NOT_YET_IMPLEMENTED;
}
int VTiSIMFRAP::StopSequence()
{
   return DEVICE_NOT_YET_IMPLEMENTED;
}
int VTiSIMFRAP::GetChannel(char* channelName)
{
	char s[MM::MaxStrLength + 1];
	snprintf(s, MM::MaxStrLength, "FRAP_GetChannel, GetChannel: %s", channelName);
	LogMessage(s);
	CDeviceUtils::CopyLimitedString(channelName,"Default");
	return DEVICE_OK;
}
double VTiSIMFRAP::GetXRange()
{
	char s[MM::MaxStrLength + 1];
	snprintf(s, MM::MaxStrLength, "FRAP_GetXRange: %ld",FRAPXRange_);
	LogMessage(s);
	return (double) FRAPXRange_;
}
double VTiSIMFRAP::GetYRange()
{
	// Ver 2.3.0.0 - Start
	char s[MM::MaxStrLength + 1];
	snprintf(s, MM::MaxStrLength, "FRAP_GetYRange: %ld",FRAPYRange_);
	LogMessage(s);
	return (double) FRAPYRange_;
	/*char s[MM::MaxStrLength + 1];
	snprintf(s, MM::MaxStrLength, "FRAP_GetYRange: %ld",GALVO_Y_RANGE);
	LogMessage(s);
	return (double) GALVO_Y_RANGE;*/
	// Ver 2.3.0.0 - End
}
double VTiSIMFRAP::GetXMinimum()
{
	char s[MM::MaxStrLength + 1];
	snprintf(s, MM::MaxStrLength, "FRAP_GetXMinimum: %ld",FRAPXMin_);
	LogMessage(s);
	return (double) FRAPXMin_;
}

double VTiSIMFRAP::GetYMinimum()
{
	char s[MM::MaxStrLength + 1];
	snprintf(s, MM::MaxStrLength, "FRAP_GetYMinimum: %ld",FRAPYMin_);
	LogMessage(s);
	return (double) FRAPYMin_;
}
// Ver 2.1.0.0 - FRAP - End

// Ver 2.2.0.0 - Pifoc - Start

VTiSIMPifoc::VTiSIMPifoc()
    : axisName_("1")
    , stepSizeUm_(0.005)
    , initialized_(false)
    , axisLimitUm_(100.0)
    , invertTravelRange_(false)
	, minTravelRangeUm_(0)
	, maxTravelRangeUm_(10000)
{
	SetErrorText(VTI_ERR_TIMEOUT_OCCURRED, "Timeout occurred");
	SetErrorText(VTI_ERR_DEVICE_NOT_FOUND, "Device not found");
	SetErrorText(VTI_ERR_NOT_INITIALISED, "Device not initialized");
	SetErrorText(VTI_ERR_NOT_SUPPORTED, "Operation not supported");

	// Ver 2.4.0.0 - Start
	PiezoTravelRangeUm = 100.0; 

	CreateIntegerProperty(g_PropVal_PiezoTravelRangeUm, (long) PiezoTravelRangeUm, false,
		new CPropertyAction(this, &VTiSIMPifoc::OnPiezoTravelRangeUm), true);
	
	SetPropertyLimits(g_PropVal_PiezoTravelRangeUm, minTravelRangeUm_, maxTravelRangeUm_);


	// Ver 2.4.0.0 - End

	// Ver 2.4.0.0 - Start
	//CPropertyAction* pAct;

	
	//// Controller name
	//pAct = new CPropertyAction (this, &VTiSIMPifoc::OnControllerName);
	//CreateProperty(g_PropVal_ControllerName, controllerName_.c_str(), MM::String, false, pAct, true);

	//// Axis name
	//pAct = new CPropertyAction (this, &VTiSIMPifoc::OnAxisName);
	//CreateProperty(g_PropVal_AxisName, axisName_.c_str(), MM::String, false, pAct, true);
   
	//// Axis stage type
	//pAct = new CPropertyAction (this, &VTiSIMPifoc::OnStageType);
	//CreateProperty(g_PropVal_StageType, stageType_.c_str(), MM::String, false, pAct, true);
   
	//// Axis homing mode
	//pAct = new CPropertyAction (this, &PIZStage::OnHoming);
	//CreateProperty(/*g_PI_ZStageHoming*/, homingMode_.c_str(), MM::String, false, pAct, true);
   
	// axis limit in um
	//pAct = new CPropertyAction (this, &VTiSIMPifoc::OnAxisLimit);
	//CreateProperty(g_PropVal_PiezoTravelRangeUm, "100.0", MM::Float, false, pAct, true);

	//// axis limit in um
	//pAct = new CPropertyAction (this, &VTiSIMPifoc::OnAxisTravelRange);
	//CreateProperty(g_PropVal_InvertTravelRange, "0.005", MM::Integer, false, pAct, true);

	//// axis limits (assumed symmetrical)
	//pAct = new CPropertyAction (this, &VTiSIMPifoc::OnPosition);
	//CreateProperty(MM::g_Keyword_Position, "0.0", MM::Float, false, pAct);
	//SetPropertyLimits(MM::g_Keyword_Position, 0.0/*-axisLimitUm_*/, axisLimitUm_);
	// Ver 2.4.0.0 - End
}

VTiSIMHub* VTiSIMPifoc::VTiHub()
{
	return static_cast<VTiSIMHub*>(GetParentHub());
}

VTiSIMPifoc::~VTiSIMPifoc()
{
}

// Ver 2.4.0.0 - Start
int VTiSIMPifoc::OnPiezoTravelRangeUm(MM::PropertyBase* pProp, MM::ActionType eAct)
{ 
	if (eAct == MM::BeforeGet)
	{
		pProp->Set(static_cast<double>(PiezoTravelRangeUm));
	}
	else if (eAct == MM::AfterSet)
	{
		double v;
		pProp->Get(v);
		PiezoTravelRangeUm=v;
	}
	return DEVICE_OK;
}
// Ver 2.4.0.0 - End 
int VTiSIMPifoc::Initialize()
{

	// Ver 2.4.0.0 - Start 
	
	/*long InitialPifocPositionNm = 50000;
	double test; // Ver 2.4.0.0

	DWORD err = vti_MovePifoc(VTiHub()->GetAOTFHandle(), InitialPifocPositionNm);

	if (err != VTI_SUCCESS)
		return err;

	err = vti_ConvertNmToInternalPosition(VTiHub()->GetScanAndMotorHandle(), InitialPifocPositionNm, &PositionSteps);
	if (err != VTI_SUCCESS)
		return err;

	err = vti_SetPifocStepsPosition(VTiHub()->GetScanAndMotorHandle(), PositionSteps);

	if (err != VTI_SUCCESS)
		return err;
		
	char s[MM::MaxStrLength + 1];
	snprintf(s, MM::MaxStrLength, "Initialize, steps: %ld",PositionSteps);
	LogMessage(s);
	initialized_ = true;
	return DEVICE_OK;*/

	char s[MM::MaxStrLength + 1];
	long CentrePositionNm;

	snprintf(s, MM::MaxStrLength, "g_PropVal_PiezoTravelRangeUm: %f",PiezoTravelRangeUm);
	LogMessage(s);
	
	CentrePositionNm = (long) PiezoTravelRangeUm * 1000l;
	CentrePositionNm = CentrePositionNm/2;

	snprintf(s, MM::MaxStrLength, "CentrePositionNm: %ld",CentrePositionNm);
	LogMessage(s);
	
	DWORD err = vti_MovePifoc(VTiHub()->GetAOTFHandle(), CentrePositionNm, (vt_int32) PiezoTravelRangeUm);
	if (err != VTI_SUCCESS)
		return err;

	err = vti_ConvertNmToInternalPosition(VTiHub()->GetScanAndMotorHandle(), CentrePositionNm, (vt_int32) PiezoTravelRangeUm, &PositionSteps);
	if (err != VTI_SUCCESS)
		return err;

	snprintf(s, MM::MaxStrLength, "PositionSteps: %ld",PositionSteps);
	LogMessage(s);

	SetPositionSteps(PositionSteps);

	return DEVICE_OK;

	// Ver 2.4.0.0 - End
}

int VTiSIMPifoc::Shutdown()
{
	if (initialized_)
   {
      initialized_ = false;
   }
   return DEVICE_OK;

}

void VTiSIMPifoc::GetName(char* name) const
{
	// Ver 2.4.0.0 - Start
	CDeviceUtils::CopyLimitedString(name, g_DeviceName_Pifoc);
	//CDeviceUtils::CopyLimitedString(name,g_PropVal_ControllerName);
	// Ver 2.4.0.0 - End
}

bool VTiSIMPifoc::Busy()
{
	return false;
}

// Stage API
int VTiSIMPifoc::SetPositionSteps(long steps)
{
	char s[MM::MaxStrLength + 1];
	snprintf(s, MM::MaxStrLength, "SetPositionSteps, steps: %ld",steps);
	LogMessage(s);

	Steps_ = steps;
	// Ver 2.4.0.0 - start
	stepSizeUm_ = PiezoTravelRangeUm/60000;
	
	snprintf(s, MM::MaxStrLength, "SetPositionSteps, PiezoTravelRangeUm: %f",PiezoTravelRangeUm);
	LogMessage(s);
	snprintf(s, MM::MaxStrLength, "SetPositionSteps, steps_: %ld",Steps_);
	LogMessage(s);
	snprintf(s, MM::MaxStrLength, "SetPositionSteps, stepSizeUm_: %f",stepSizeUm_);
	LogMessage(s);
	// Ver 2.4.0.0 - End
   double pos = steps * stepSizeUm_;

  
	snprintf(s, MM::MaxStrLength, "SetPositionSteps, pos: %f",pos);
	LogMessage(s);

   return SetPositionUm(pos);
}

int VTiSIMPifoc::GetPositionSteps(long& steps)
{
	double pos;
	int ret = GetPositionUm(pos);

	char s[MM::MaxStrLength + 1];
	snprintf(s, MM::MaxStrLength, "GetPositionSteps, ret: %ld",ret);
	LogMessage(s);

   if (ret != DEVICE_OK)
      return ret;
   steps = (long) ((pos / stepSizeUm_) + 0.5);
	
   // Ver 2.4.0.0 - start
	steps =  Steps_;
	// Ver 2.4.0.0 - End
	snprintf(s, MM::MaxStrLength, "GetPositionSteps, steps: %ld",steps);
	LogMessage(s);

   return DEVICE_OK;
}

int VTiSIMPifoc::SetPositionUm(double pos)
{
	double Temp;
	int PositionNM = 0;
    if (invertTravelRange_)
    {
       // pos = axisLimitUm_ - pos;	// Ver 2.4.0.0
		pos = (double)PiezoTravelRangeUm - pos;
    }

	char s[MM::MaxStrLength + 1];
	snprintf(s, MM::MaxStrLength, "SetPositionUm, pos: %f",pos);
	LogMessage(s);
	
	if(pos <= 0.0)
	{
		return DEVICE_OK;
	}

	// Ver 2.4.0.0 - Start
	/*if(pos > 100.0)
	{
		pos = 100.0;
	}*/

	if (pos > PiezoTravelRangeUm)
	{
		snprintf(s, MM::MaxStrLength, "SetPositionUm, PiezoTravelRangeUm: %f",PiezoTravelRangeUm);
		LogMessage(s);
		snprintf(s, MM::MaxStrLength, "Pos > PiezoTravelRangeUm");
		LogMessage(s);
		return DEVICE_OK;
	}
	// Ver 2.4.0.0 - End


	Temp = pos*1000;	// Convert um into nm.
	PositionNM = (int) (Temp + 0.5);

	snprintf(s, MM::MaxStrLength, "SetPositionUm, Temp: %f",Temp);
	LogMessage(s);

	snprintf(s, MM::MaxStrLength, "SetPositionUm, PositionNM: %ld",PositionNM);
	LogMessage(s);

	// Ver 2.4.0.0 - Start
	//DWORD err = vti_MovePifoc(VTiHub()->GetAOTFHandle(), PositionNM);
	//if (err != VTI_SUCCESS)
	//	return err;
	DWORD err = vti_MovePifoc(VTiHub()->GetAOTFHandle(), PositionNM, (vt_int32) PiezoTravelRangeUm);
	if (err != VTI_SUCCESS)
		return err;

	// Ver 2.4.0.0 - End
	 err = vti_ConvertNmToInternalPosition(VTiHub()->GetScanAndMotorHandle(), PositionNM, (vt_int32) PiezoTravelRangeUm, &PositionSteps);
	if (err != VTI_SUCCESS)
		return err;
	
	snprintf(s, MM::MaxStrLength, "SetPositionUm, PositionSteps: %ld",PositionSteps);
	LogMessage(s);

	// Ver 2.4.0.0 - 2. Start
	//SetPositionSteps(PositionSteps);
	Steps_ = PositionSteps;

	/*err = vti_SetPifocStepsPosition(VTiHub()->GetScanAndMotorHandle(), PositionSteps);

	if (err != VTI_SUCCESS)
		return err;*/
	// Ver 2.4.0.0 - 2. End
	return DEVICE_OK;
}
int VTiSIMPifoc::GetPositionUm(double& pos)
{

	vt_int32 ConvertedNmPosition;
	char s[MM::MaxStrLength + 1];
	
	// Ver 2.4.0.0 - 2. Start
	/*VTI_EX_PARAM param;
	PositionSteps = 0;
	param.pArray = &PositionSteps;
	param.ArrayBytes = sizeof(vt_int16);


	DWORD err = vti_GetExtendedFeature(VTiHub()->GetScanAndMotorHandle(),
			VTI_FEATURE_PIFOC_POSITION, &param, sizeof(param));
	if (err != VTI_SUCCESS)
			return err;*/
	//long steps;
	//int err_ = GetPositionSteps(steps);
	snprintf(s, MM::MaxStrLength, "GetPositionUm, Steps_: %ld",Steps_);
	LogMessage(s);
	PositionSteps = (unsigned short) Steps_;
	// Ver 2.4.0.0 - 2. End
	

	// Ver 2.4.0.0 - Start
	snprintf(s, MM::MaxStrLength, "GetPositionUm, PositionSteps: %ld",PositionSteps);
	LogMessage(s);
	snprintf(s, MM::MaxStrLength, "GetPositionUm, PiezoTravelRangeUm: %f",PiezoTravelRangeUm);
	LogMessage(s);

	// Ver 2.4.0.0 - End

	DWORD err = vti_ConvertInternalPositionToNm(VTiHub()->GetScanAndMotorHandle(), &ConvertedNmPosition, PositionSteps, (vt_int32) PiezoTravelRangeUm);
	if (err != VTI_SUCCESS)
		return err;


	pos = ConvertedNmPosition/1000.0;
	// Ver 2.4.0.0 - Start
	//if (pos >100000)
	if (pos > PiezoTravelRangeUm)
	{
		return DEVICE_OK;
	}
	// Ver 2.4.0.0 - End
	snprintf(s, MM::MaxStrLength, "GetPositionUm, pre - ConvertedNmPosition: %ld",ConvertedNmPosition);
	LogMessage(s);

	snprintf(s, MM::MaxStrLength, "GetPositionUm, pre - pos: %f",pos);
	LogMessage(s);

    if (invertTravelRange_)
    {
        //pos = axisLimitUm_ - pos;// Ver 2.4.0.0
		pos = PiezoTravelRangeUm - pos;// Ver 2.4.0.0
    }
	snprintf(s, MM::MaxStrLength, "GetPositionUm, pos - invertTravelRange_: %s", invertTravelRange_ ? "true" : "false");
	LogMessage(s);
	snprintf(s, MM::MaxStrLength, "GetPositionUm, pos - pos: %f",pos);
	LogMessage(s);
	return DEVICE_OK;
}
int VTiSIMPifoc::SetOrigin()
{
	char s[MM::MaxStrLength + 1];
	snprintf(s, MM::MaxStrLength, "SetOrigin()");
	LogMessage(s);

   return DEVICE_UNSUPPORTED_COMMAND;
}

int VTiSIMPifoc::GetLimits(double& min, double& max)
{
	char s[MM::MaxStrLength + 1];

    min = 0;

	snprintf(s, MM::MaxStrLength, "GetLimits, min: %f",min);
	LogMessage(s);

    //max = axisLimitUm_;	// Ver 2.4.0.0
	max = PiezoTravelRangeUm;	// Ver 2.4.0.0

	snprintf(s, MM::MaxStrLength, "GetLimits, max: %f",max);
	LogMessage(s);

    return DEVICE_OK;
}

// Action handlers

int VTiSIMPifoc::OnControllerName(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(controllerName_.c_str());
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Get(controllerName_);
   }

   return DEVICE_OK;
}



int VTiSIMPifoc::OnAxisName(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(axisName_.c_str());
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Get(axisName_);
   }

   return DEVICE_OK;
}

int VTiSIMPifoc::OnStepSizeUm(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(stepSizeUm_);
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Get(stepSizeUm_);
   }

   return DEVICE_OK;
}

int VTiSIMPifoc::OnAxisLimit(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
     // pProp->Set(axisLimitUm_);		// Ver 2.4.0.0
	   pProp->Set(PiezoTravelRangeUm);	// Ver 2.4.0.0
	  //SetPropertyLimits(MM::g_Keyword_Position, 0.0/*-axisLimitUm_*/, axisLimitUm_);	// Ver 2.4.0.0
	   SetPropertyLimits(MM::g_Keyword_Position, 0.0/*-axisLimitUm_*/, PiezoTravelRangeUm);	// Ver 2.4.0.0
   }
   else if (eAct == MM::AfterSet)
   {
      //pProp->Get(axisLimitUm_);	// Ver 2.4.0.0
	   pProp->Get(PiezoTravelRangeUm);
   }

   return DEVICE_OK;
}

int VTiSIMPifoc::OnAxisTravelRange(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(long (invertTravelRange_ ? 1 : 0));
   }
   else if (eAct == MM::AfterSet)
   {
      long value;
      pProp->Get(value);
      invertTravelRange_ = (value != 0);
   }

   return DEVICE_OK;
}

int VTiSIMPifoc::OnPosition(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
	   if (!initialized_)
	   {
		   pProp->Set(0.0);
		   return DEVICE_OK;
	   }
      double pos;
      int ret = GetPositionUm(pos);
      if (ret != DEVICE_OK)
         return ret;

      pProp->Set(pos);
   }
   else if (eAct == MM::AfterSet)
   {
 	   if (!initialized_)
	   {
		   return DEVICE_OK;
	   }
     double pos;
      pProp->Get(pos);
      int ret = SetPositionUm(pos);
      if (ret != DEVICE_OK)
         return ret;

   }

   return DEVICE_OK;
}

int VTiSIMPifoc::OnStageType(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(stageType_.c_str());
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Get(stageType_);
   }

   return DEVICE_OK;
}

int VTiSIMPifoc::OnHoming(MM::PropertyBase* pProp, MM::ActionType eAct)
{
  /*  if (NULL == ctrl_)
    {
        return DEVICE_ERR;
    }*/
    if (eAct == MM::BeforeGet)
    {
        pProp->Set(0.0);
    }
    else if (eAct == MM::AfterSet)
    {
        std::string homingMode;
        pProp->Get(homingMode);
       // int ret = ctrl_->Home( axisName_, homingMode );
       // if (ret != DEVICE_OK)
       //     return ret;
    }

    return DEVICE_OK;
}

int VTiSIMPifoc::OnVelocity(MM::PropertyBase* /* pProp */, MM::ActionType /* eAct */)
{
    ////if (NULL == ctrl_)
    ////{
    ////    return DEVICE_ERR;
    ////}
    //if (eAct == MM::BeforeGet)
    //{
    //    double velocity = 0.0;
    //    if (ctrl_->qVEL(axisName_, &velocity))
    //        pProp->Set(velocity);
    //    else
    //        pProp->Set(0.0);
    //}
    //else if (eAct == MM::AfterSet)
    //{
    //    double velocity = 0.0;
    //    pProp->Get(velocity);
    //    if (!ctrl_->VEL( axisName_, &velocity ))
    //        return ctrl_->GetTranslatedError();
    //}

    return DEVICE_OK;
}

// Ver 2.2.0.0 - Pifoc - End

