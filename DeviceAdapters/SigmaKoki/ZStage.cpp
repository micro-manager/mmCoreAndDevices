///////////////////////////////////////////////////////////////////////////////
// FILE:          ZStage.cpp
// PROJECT:       Micro-Manager 2.0
// SUBSYSTEM:     DeviceAdapters
//  
//-----------------------------------------------------------------------------
// DESCRIPTION:   SIGMA-KOKI device adapter 2.0
//                
// AUTHOR   :    Hiroki Kibata, Abed Toufik  Release Date :  05/02/2023
//
// COPYRIGHT:     SIGMA KOKI CO.,LTD, Tokyo, 2023
#include "ZStage.h"

using namespace std;
// cont variable to be used  proprietes
const char* g_ZStageDeviceName = "ZStage";
const char* g_ZStageChannel = "Channel";
const char* g_ZStageControlMode = "Control Mode";
const char* g_ZStageOpenControl = "OPEN";
const char* g_ZStageCloseControl = "CLOSE";
const char* g_ZStageStepSizeUm = "StepSize (micron)";
const char* g_ZStageFullStepSize = "FullStepSize (micron)";
const char* g_ZStageDivision = "Division of Controller";
const char* g_ZStageSpeed = "Speed (micron/sec)";
const char* g_ZinfoDivision = "Division Information";
const char* g_ZStageController = "Name of Controller";
const char* g_ZfineMode = "[CLOSE/OPEN] Setting, Please set to CLOSE mode in Memory Switch";
const char* g_PulseRateInfoZ = "Please set the correct PULSE_RATE for each axis in memory switches";

/// <summary>
/// Constructor
/// </summary>
ZStage::ZStage() :
	SigmaBase(this),
	model_(SHOT2),
	channel_("1"),
	controlMode_("OPEN"),
	fullstepSizeZum_(2.0),
	stepSizeZum_(0.1),
	divisionZ_("20"),
	speed_(1000),
	answerTimeoutMs_(500),
	stepsZ_(0.0),
	PlsRateZ(1)
{
	InitializeDefaultErrorMessages();

	// Create pre-initialization properties
	// Name
	CreateProperty(MM::g_Keyword_Name, g_ZStageDeviceName, MM::String, true);
	// Description
	CreateProperty(MM::g_Keyword_Description, "SIGMA-KOKI ZStage adapter", MM::String, true);
	// Port
	CPropertyAction* pAct = new CPropertyAction(this, &ZStage::OnPort);
	CreateProperty(MM::g_Keyword_Port, "Undefined", MM::String, false, pAct, true);
}

/// <summary>
/// Destructor
/// </summary>
ZStage::~ZStage() 
{
	Shutdown();
}

/// <summary>
/// Busy status
/// </summary>
/// <returns></returns>
bool ZStage::Busy()
{
	// First Clear serial port from previous stuff------------------------------------------------------------------------------------------------
	int ret = ClearPort();
	if (ret != DEVICE_OK)
		return false;
	// Send command // Recieve Busy/Ready data----------------------------------------------------------------------------------------------------
	string answer = "";
	ret = SendRecieve("!:",answer);
	if (ret != DEVICE_OK)
		return false;
	// Receive check (busy or ready)--------------------------------------------------------------------------------------------------------------
	if (model_ == HSC3|| model_== SHRC3)
	{
		string::size_type index = answer.find("1");
		if (index == string::npos)
		{
			return false;	/*ready*/
		}
		else
		{
			return true;	/*busy*/
		}
	}
	else if (answer.substr(0, 1).compare("B") == 0)
	{
		return true;	/*busy*/
	}
	return false;	/*ready*/
}

/// <summary>
/// Get Name
/// </summary>
/// <param name="pszName"></param>
void ZStage::GetName(char* pszName) const
{
	CDeviceUtils::CopyLimitedString(pszName, g_ZStageDeviceName);
}

/// <summary>
/// Initialize
/// </summary>
/// <returns></returns>
int ZStage::Initialize() 
{
	if (initialized_) { return DEVICE_OK; }
	core_ = GetCoreCallback();

	int ret = SetDeviceModel();
	if (ret != DEVICE_OK) { return ERR_XYSTEGE_DEVICE_UNRECOGNIZABLE; }

	// Channel
	CPropertyAction* pAct = new CPropertyAction(this, &ZStage::OnChannel);
	ret = CreateProperty(g_ZStageChannel, "1", MM::String, false, pAct);
	// Assign Chanels number 
	CreateChanelProp();


	// stages full step Z 
	// NOTE: In the case of the piezo-controller (FINE-01, FINE-503), No Step stages.
	if (model_ == SHOT1 || model_ == SHOT2 || model_ == SHOT4 || model_ == HSC3 || model_ == SHRC3)
	{
		pAct = new CPropertyAction(this, &ZStage::OnFullStepSize);
		ret = CreateProperty(g_ZStageFullStepSize, "2", MM::String, false, pAct);
		if (ret != DEVICE_OK)
			return ret;
		AddAllowedValue(g_ZStageFullStepSize, "0.2");
		AddAllowedValue(g_ZStageFullStepSize, "1");
		AddAllowedValue(g_ZStageFullStepSize, "2");
		AddAllowedValue(g_ZStageFullStepSize, "4");
		AddAllowedValue(g_ZStageFullStepSize, "20");
	}

	bool ifdisabled = true;  // enable/disable Division Setting in proprietes Available on (SHOT Series) / disable the others

	switch (model_)
	{
	case ZStage::SHOT1:
		ifdisabled = false;
		
		break;
	case ZStage::SHOT2:case ZStage::SHOT4:
		// control mode prop
		pAct = new CPropertyAction(this, &ZStage::OnControlModel);
		ret = CreateProperty(g_ZStageControlMode, controlMode_.c_str(), MM::String, false, pAct);
		if (ret != DEVICE_OK)
			return ret;
		AddAllowedValue(g_ZStageControlMode, g_ZStageOpenControl);
		AddAllowedValue(g_ZStageControlMode, g_ZStageCloseControl);
		ifdisabled = false;
		break;

	case ZStage::FINE1:case ZStage::FINE3:
		ifdisabled = true;
		stepSizeZum_ = 0.010;
		CreateProperty("Control Mode Info", g_ZfineMode, MM::String, true);
		break;

	case ZStage::HSC3:
		ifdisabled = true;	

		// Pulse Rate Information 
		CreateProperty("Stage Pulse Rate Info", g_PulseRateInfoZ, MM::String, true);
		// Pulse rate 
		pAct = new CPropertyAction(this, &ZStage::OnPulseRateZ);
		ret = CreateProperty("StageZ Pulse Rate", to_string(PlsRateZ).c_str() , MM::String, true, pAct);
		if (ret != DEVICE_OK)
			return ret;
		break;

	case ZStage::SHRC3:


		ifdisabled = false;
		break;

	default:
		break;
	}

	// Axis Division Z 
	// NOTE: In the case of the piezo-controller (FINE-01, FINE-503), No division setting.
	if (model_ == SHOT1 || model_ == SHOT2 || model_ == SHOT4 || model_ == HSC3 || model_ == SHRC3)
	{
		pAct = new CPropertyAction(this, &ZStage::OnDivision);
		ret = CreateProperty(g_ZStageDivision, "20", MM::String, ifdisabled, pAct);
		if (ret != DEVICE_OK)
			return ret;
		CreateDivisionProp();
	}

	// Read only step Size
	pAct = new CPropertyAction(this, &ZStage::OnStepSizeZ);
	ret = CreateProperty(g_ZStageStepSizeUm, to_string(stepSizeZum_).c_str(), MM::String, true, pAct);
	if (ret != DEVICE_OK)
		return ret;

	// Speed (micron/sec)
	// NOTE: In the case of the piezo-controller (FINE-01, FINE-503), speed can not be set.
	if (model_ == SHOT1 || model_ == SHOT2 || model_ == SHOT4 || model_ == HSC3 || model_ == SHRC3)
	{
		pAct = new CPropertyAction(this, &ZStage::OnSpeed);
		ret = CreateProperty(g_ZStageSpeed, to_string(speed_).c_str(), MM::Integer, false, pAct);
		if (ret != DEVICE_OK)
			return ret;
		SetPropertyLimits(g_ZStageSpeed, 1, 2000);
	}

	// Division Information
	CreateProperty("Z Division Information", g_ZinfoDivision, MM::String, true);

	// controller name 
	CreateProperty("Name of Controller(Z)", g_ZStageController, MM::String, true);

	ret = UpdateStatus();
	if (ret != DEVICE_OK)
		return ret;

	initialized_ = true;
	return DEVICE_OK;
}

/// <summary>
/// Shut down 
/// </summary>
/// <returns></returns>
int ZStage::Shutdown()
{
	if (initialized_)
	{
		initialized_ = false;
	}
	return DEVICE_OK;
}

/// <summary>
/// Get position in Um 
/// </summary>
/// <param name="pos"></param>
/// <returns></returns>
int ZStage::GetPositionUm(double& pos)
{
	// get position of Z in steps
	int ret = UpdatePositionZ();
	
	if (ret != DEVICE_OK)
		return ret;

	// Extract position to um  
	if (model_ == FINE1 || model_ == FINE3)
	{
		pos = (stepsZ_ / 1000); //nano -> micron 
	}
	else if (model_ == HSC3)
	{
		pos =(stepsZ_ * 0.01);
	}
	else
	{
		if (controlMode_ == g_ZStageOpenControl)
		{
			pos = (double)(stepsZ_ * stepSizeZum_);	//pulse -> micron
		}
		else if (controlMode_ == g_ZStageCloseControl)
		{
			pos = (double)(stepsZ_ * 0.01);	//pulse -> micron
		}
	}

	return DEVICE_OK;
}

/// <summary>
/// Get the position in steps
/// </summary>
/// <param name="steps"></param>
/// <returns></returns>
int ZStage::GetPositionSteps(long& steps)
{
	// get position of Z in steps
	int ret = UpdatePositionZ();
	if (ret != DEVICE_OK)
		return ret;

	steps = stepsZ_;

	return DEVICE_OK;
}

/// <summary>
/// Set Position in um unit 
/// </summary>
/// <param name="pos"></param>
/// <returns></returns>
int ZStage::SetPositionUm(double pos)
{
	// Set position Um (Absolue drive )
	int ret = DriveCommadProcessZ(pos);
	if (ret != DEVICE_OK)
		return ret;

	return DEVICE_OK;

}

/// <summary>
/// Home set 
/// </summary>
/// <returns></returns>
int ZStage::SetOrigin()
{
	// First Clear serial port from previous stuff-------------------------------------------------------
	int ret = ClearPort();
	if (ret != DEVICE_OK)
	{return ret;}
	

	// Create command-------------------------------------------------------
	string cmd = "R:";
	
	if (model_ == HSC3)
	{
		for (int i = 1; i <= 3; i++)
		{
			if (i == stoi(channel_)) { cmd += "1"; }
			else { cmd += "0"; }
			if (i != 3) { cmd += ","; }
		}
	}
	else
	{
		cmd += channel_;
	}
	
	// Send command <common>-------------------------------------------------------
	ret = SendCheckRecievedOK(cmd);
	if (ret != DEVICE_OK)
		return ret;

	return DEVICE_OK;
}

/// <summary>
/// Not supported command
/// <returns></returns>
int ZStage::GetLimits(double& min, double& max)
{
	return DEVICE_UNSUPPORTED_COMMAND;
}

/// Action Handlers //////////////////////////////////////////////////////////////////
/// 
/// On Port
/// All acctions of proprietes handlers 
/// <returns></returns>
int ZStage::OnPort(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set(port_.c_str());
	}
	else if (eAct == MM::AfterSet)
	{
		if (initialized_)
		{
			// revert
			pProp->Set(port_.c_str());
			return ERR_PORT_CHANGE_FORBIDDEN;
		}

		pProp->Get(port_);
	}

	return DEVICE_OK;
}

/// <summary>
/// Set Channels for all controller 
/// <param name="eAct"></param>
/// <returns></returns>
int ZStage::OnChannel(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set(channel_.c_str());
	}
	else if (eAct == MM::AfterSet)
	{
		pProp->Get(channel_);
	}

	return DEVICE_OK;
}

/// <summary>
/// set/get full step size for all controller 
/// <param name="pProp"></param>
/// <param name="eAct"></param>
/// <returns></returns>
int ZStage::OnFullStepSize(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set(fullstepSizeZum_);
	}
	else if (eAct == MM::AfterSet)
	{
		pProp->Get(fullstepSizeZum_);
	}

	if (model_ == SHOT2 || model_ == SHOT4)
	{
		if (controlMode_ == g_ZStageOpenControl)
		{
			//CreateResolutionList(fullstepSizeZum_);
		}
		else if (controlMode_ == g_ZStageCloseControl)
		{
			ClearAllowedValues(g_ZStageStepSizeUm);
			AddAllowedValue(g_ZStageStepSizeUm, "0.100");
			AddAllowedValue(g_ZStageStepSizeUm, "0.500");
		}
	}
	return DEVICE_OK;
}


/// <summary>
/// REad only step size of Z
/// </summary>
/// <param name="pProp"></param>
/// <param name="eAct"></param>
/// <returns></returns>
int ZStage::OnStepSizeZ(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		if (model_==SHOT2 || model_ == SHOT4 || model_ == SHRC3)
		{
			pProp->Set(stepSizeZum_);
			
		}
	}
	return DEVICE_OK;
}



/// <summary>
/// Get Set Step Size (division)
/// </summary>
/// <param name="pProp"></param>
/// <param name="eAct"></param>
/// <returns></returns>
int ZStage::OnDivision(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	int ret;
	if (eAct == MM::BeforeGet)
	{

		if (model_ == HSC3)
		{
			divisionZ_ = "40";
			stepSizeZum_ = fullstepSizeZum_ / stod(divisionZ_);
			stepSizeZum_ = 0.01;
		}
		
		pProp->Set(divisionZ_.c_str());

		if (model_ == SHOT2 || model_ == SHOT4 || model_ == SHRC3 )
		{
			ret = SetDivision(stoi(divisionZ_));
			if (ret != DEVICE_OK)
				return ERR_ZSTAGE_STEPSIZE_FAILED;
			double d = fullstepSizeZum_ / stod(divisionZ_);
			stepSizeZum_ = d;
		}
		if (model_ == SHOT1)
		{
			stepSizeZum_ = fullstepSizeZum_ / stod(divisionZ_);
		}
		
	}
	else if (eAct == MM::AfterSet)
	{
		if (model_ == SHOT2 || model_ == SHOT4)
		{
			if (controlMode_ == g_ZStageCloseControl)
			{
				double pos;
				GetPositionUm(pos);
				if (pos != 0.0)
				{
					return ERR_ZSTAGE_SET_RESOLUTION_CLOSE_CONTROL_FAILED;
				}
			}
		}

		pProp->Get(divisionZ_);

		if (model_ == SHOT2 || model_ == SHOT4 || model_ == SHRC3)
		{

			ret = SetDivision(stoi(divisionZ_));
			if (ret != DEVICE_OK)
				return ERR_ZSTAGE_STEPSIZE_FAILED;
			double d = fullstepSizeZum_ / stod(divisionZ_);
			stepSizeZum_ = d;
		}
		if (model_ == SHOT1)
		{
			stepSizeZum_ = fullstepSizeZum_ / stod(divisionZ_);
		}
	}

	return DEVICE_OK;
}

/// <summary>
/// Set get speed 
/// </summary>
/// <param name="pProp"></param>
/// <param name="eAct"></param>
/// <returns></returns>
int ZStage::OnSpeed(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	int ret;
	if (eAct == MM::BeforeGet)
	{
		pProp->Set((long)speed_);

		ret = SetSpeed(speed_);
		if (ret != DEVICE_OK)
			return ERR_ZSTAGE_SPEED_FAILED;
	}
	else if (eAct == MM::AfterSet)
	{
		long n;
		pProp->Get(n);
		speed_ = (int)n;

		ret = SetSpeed(speed_);
		if (ret != DEVICE_OK)
			return ERR_ZSTAGE_SPEED_FAILED;
	}

	return DEVICE_OK;
}
/// <summary>
/// 
/// </summary>
/// <param name="pProp"></param>
/// <param name="eAct"></param>
/// <returns></returns>
int ZStage::OnControlModel(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set(controlMode_.c_str());
	}
	else if (eAct == MM::AfterSet)
	{
		pProp->Get(controlMode_);

		if (controlMode_ == g_ZStageOpenControl)
		{
			CreateDivisionProp();
		}
		else if (controlMode_ == g_ZStageCloseControl)
		{
			ClearAllowedValues(g_ZStageStepSizeUm);
			AddAllowedValue(g_ZStageStepSizeUm, "0.100");
			AddAllowedValue(g_ZStageStepSizeUm, "0.500");
		}
	}

	return DEVICE_OK;
}
int ZStage::OnPulseRateZ(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		// Pulse Rate Calculclation in HSC-103 controller 
		// PULSE RATE = (FULL_STEP / DIVISION)* 1000 /0.1
		const int UM_TO_NM_RATE = 1000;
		const double NM_TO_PULSERATE_RATE = 0.1;
		if (model_ == HSC3)
		{
			PlsRateZ = ((fullstepSizeZum_/40) * UM_TO_NM_RATE) / NM_TO_PULSERATE_RATE;
		}
		pProp->Set(long(PlsRateZ));
	}

	return DEVICE_OK;
}
/// <summary>
/// Set device model
/// </summary>
/// <param name="model"></param>
/// <returns></returns>
int ZStage::SetDeviceModel()
{
	//First Clear serial port from previous stuff----------------------------------------------------------------------------------------------------
	int ret = ClearPort();
	if (ret != DEVICE_OK)
		return ret;

	// Send status command and recieve data----------------------------------------------------------------------------------------------------------
	string data = "";
	ret = SendRecieve("Q:", data);
	if (ret != DEVICE_OK)
		return ret;

	const int SHOT1_STATUS_STRING_LENGTH = 16;
	const int SHOT2_STATUS_STRING_LENGTH = 27;
	const int FINE3_STATUS_STRING_LENGTH = 38;
	const int SHOT4_STATUS_STRING_LENGTH = 49;



	if (data.length() == SHOT1_STATUS_STRING_LENGTH)
	{
		//Read the controller model
		string answer = "";
		ret = SendRecieve("?:N", answer);
		if (ret != DEVICE_OK)
			return ret;

		if (answer== "FINE-01r  ")
		{
			model_ = FINE1;	// FINE-01
			g_ZinfoDivision = "No division for fine controllers";
			g_ZStageController = "Fine-01";
		}
		else 
		{
			model_ = SHOT1;	// GIP-101, TAF-C01
			g_ZinfoDivision = "Please set the division from dip switch";
			g_ZStageController = "GIP-101/TAF-C01";
		}
	}
	else if (data.length() == FINE3_STATUS_STRING_LENGTH)
	{
		model_ = FINE3;		// FINE-503
		g_ZinfoDivision = "No division for fine controllers";
		g_ZStageController = "Fine-503";
	}
	else if (data.length() == SHOT2_STATUS_STRING_LENGTH)
	{
		model_ = SHOT2;		// SHOT-302GS, SHOT-702
		g_ZinfoDivision = "Set Division Of Shot-2Axis";
		g_ZStageController = "SHOT-302GS/SHOT-702";
	}
	else if (data.length() == SHOT4_STATUS_STRING_LENGTH)
	{
		model_ = SHOT4;		// SHOT-304GS
		g_ZinfoDivision = "Set Division Of Shot-304";
		g_ZStageController = "SHOT-304GS";
		channel_ = "3";
	}
	else
	{
		//Get product name
		string name = "unknown";
		ret = SendRecieve("?:N", name);
		if (ret != DEVICE_OK) { return ret; }

		// remove white spaces 
		name.erase(std::remove_if(name.begin(), name.end(), isspace), name.end());

		if (strcmp(name.c_str(), "HSC-103") == 0)
		{
			model_ = HSC3;			// HSC-103
			g_ZinfoDivision = "No division setting for HSC controller|fixed to 40";
			g_ZStageController = "HSC-103";
			channel_ = "3";
		}
		else if (strcmp(name.c_str(), "SHRC-203") == 0)
		{
			model_ = SHRC3;	// SHRC-203
			g_ZinfoDivision = "Set Division Of SHRC-203";
			g_ZStageController = "SHRC-203";
			channel_ = "3";
			string mode = "";
			//MODE:HOST (to be confirmed)
			//FMT:HIT
			ret = SendRecieve("?:FMT", mode);
			if (ret != DEVICE_OK) { return ret; }

			if (mode != "HIT")
			{
				SendCheckRecievedOK("FMT:HIT");
				if (ret != DEVICE_OK) { return ret; }
			}
		}
		
	}

	return DEVICE_OK;
}

/// <summary>
/// Creating a Division list in prop.
/// </summary>
/// <param name="fullstep"></param>
void ZStage::CreateDivisionProp()
{
	// Clear all allowed value declared before.--------------------------------------------------------------------------------------------
	ClearAllowedValues(g_ZStageDivision);
	AddAllowedDivisionPropXY(g_ZStageDivision);  // Added 9ŒŽ12“ú2022   t.abed
}

/// <summary>
/// Add division prop for Z (reduce size)
/// </summary>
/// <param name="div"></param>
void ZStage::AddAllowedDivisionPropXY(const char* div)
{
	switch (model_)
	{
	case ZStage::SHOT1:case ZStage::SHOT2:case ZStage::SHOT4:
		// Division Z
		AddAllowedValue(div, "1");
		AddAllowedValue(div, "2");
		AddAllowedValue(div, "4");
		AddAllowedValue(div, "5");
		AddAllowedValue(div, "8");
		AddAllowedValue(div, "10");
		AddAllowedValue(div, "20");
		AddAllowedValue(div, "25");
		AddAllowedValue(div, "40");
		AddAllowedValue(div, "50");
		AddAllowedValue(div, "80");
		AddAllowedValue(div, "100");
		AddAllowedValue(div, "125");
		AddAllowedValue(div, "200");
		AddAllowedValue(div, "250");
		break;
	case ZStage::FINE1:case ZStage::FINE3:
		AddAllowedValue(div, "200");
		break;
	case ZStage::HSC3:
		AddAllowedValue(div, "40");
		break;
	case ZStage::SHRC3:
		for (int i = 1; i <= 6; i++)    // Add from 1 to 6
		{
			AddAllowedValue(div, to_string(i).c_str());
		}
		AddAllowedValue(div, "8");
		AddAllowedValue(div, "12");
		AddAllowedValue(div, "25");
		for (int i = 1; i < 60; i++) // Add from 10 to 60
		{
			i += 9;
			AddAllowedValue(div, to_string(i).c_str());
		}
		AddAllowedValue(div, "80");
		AddAllowedValue(div, "120");
		AddAllowedValue(div, "125");
		AddAllowedValue(div, "250");
		for (int i = 1; i < 400; i++) // add from 100 to 400
		{
			i += 99;
			AddAllowedValue(div, to_string(i).c_str());
		}
		for (int i = 600; i <= 1000; i++) // add from 600 to 1000
		{
			int k = i;
			AddAllowedValue(div, to_string(k).c_str());
			i = i + 199;
		}
		for (int i = 1000; i <= 8000; i++) // add from 1000 to 8000
		{
			int l = i;
			AddAllowedValue(div, to_string(l).c_str());
			i = i + i - 1;
		}
		break;
	default:
		break;
	}

}

/// <summary>
/// Create allowed channel in prop
/// </summary>
void ZStage::CreateChanelProp()
{
	if (model_ == SHOT1 || model_ == FINE1)
	{
		AddAllowedValue(g_ZStageChannel, "1");
	}
	else if (model_ == FINE3 || model_ == HSC3 || model_ == SHRC3)
	{
		for (int i = 1; i <= 3; i++)
		{
			AddAllowedValue(g_ZStageChannel, to_string(i).c_str());
		}
	}
	else if (model_ == SHOT2)
	{
		for (int i = 1; i <= 2; i++)
		{
			AddAllowedValue(g_ZStageChannel, to_string(i).c_str());
		}
	}
	else
	{
		for (int i = 1; i <= 4; i++)
		{
			AddAllowedValue(g_ZStageChannel, to_string(i).c_str());
		}
	}
}

/// <summary>
/// Set Speed 
/// </summary>
/// <param name="val"></param>
/// <returns></returns>
int ZStage::SetSpeed(int val)
{
	//First Clear serial port from previous stuff
	int ret = ClearPort();
	if (ret != DEVICE_OK)
		return ret;

	long pulse_fast;
	long pulse_slow = 0;

	// Setting the speed pulse. 
	pulse_fast = (long)(val / stepSizeZum_);

	//Speed(F) [PPS]
	if (pulse_fast < 1 || pulse_fast > 500000) { return ERR_ZSTAGE_SPEED_FAILED; }

	//Speed(S) [PPS]
	pulse_slow = GetSlowSpeedPulse(pulse_fast);

	// Create command
	// NOTE: Acceleration and Deceleration time fixed to 100 msec.
	string cmd;
	if (model_ == HSC3|| model_==SHRC3)
	{
		cmd = "D:" + channel_ + "," + to_string(pulse_slow) + "," + to_string(pulse_fast) + ",100";
	}
	else
	{
		cmd = "D:" +channel_ + "S" + to_string(pulse_slow) + "F" + to_string(pulse_fast) + "R100";
	}

	// Send command
	ret = SendCheckRecievedOK(cmd);
	if (ret != DEVICE_OK)
		return ret;
	return DEVICE_OK;
}

int ZStage::UpdatePositionZ()
{
	//First Clear serial port from previous stuff--------------------------------------------------------------------------------------------------------
	int ret = ClearPort();
	if (ret != DEVICE_OK)
		return ret;
	// Send command & Recieve position data--------------------------------------------------------------------------------------------------------------
	string status = "";
	ret = SendRecieve("Q:", status);
	if (ret != DEVICE_OK)
		return ret;
	// delete spaces 
	status.erase(std::remove_if(status.begin(), status.end(), ::isspace), status.end());

	// split status by comma ----------------------------------------------------------------------------------------------------------------------------
	// ','-separated array
	vector<string> result = split(status, ',');

	int AxisZ = stoi(channel_) - 1;
	// get position in steps depends controller ---------------------------------------------------------------------------------------------------------
	switch (model_)
	{
	case ZStage::SHOT1:
		stepsZ_ = atol(result[AxisZ].c_str());
		break;
	case ZStage::SHOT2:
		stepsZ_ = atol(result[AxisZ].c_str());
		break;
	case ZStage::SHOT4:
		stepsZ_ = atol(result[AxisZ].c_str());
		break;
	case ZStage::FINE1:
		stepsZ_ = atol(result[AxisZ].c_str());
		break;
	case ZStage::FINE3:
		stepsZ_ = atol(result[AxisZ].c_str());
		break;
	case ZStage::HSC3:
		stepsZ_ = atol(result[AxisZ].c_str());
		break;
	case ZStage::SHRC3:
		stepsZ_ = atol(result[AxisZ].c_str());
		break;
	default:
		stepsZ_ = 0.0;
		break;
	}
	return DEVICE_OK;
}

/// <summary>
/// Drive Command  (Absolue Drive only)
/// </summary>
/// <param name="position"></param>
/// <returns></returns>
int ZStage::DriveCommadProcessZ(double position)
{
	// First Clear serial port from previous stuff-------------------------------------------------------------------------------------------------------
	int ret = ClearPort();
	if (ret != DEVICE_OK)
		return ret;

	// Setting the driving direction----------------------------------------------------------------------------------------------------
	string sign;
	if (position >= 0) { sign = '+'; }
	else { sign = '-'; }

	// Create command parameters----------------------------------------------------------------------------------------------------
	string cmd = "";
	long pos_pulse = 0;
	long pos_nano = 0;

	// Create command depends controller------------------------------------------------------------------------------------------------
	switch (model_)
	{
	case ZStage::SHOT1:
		pos_pulse = (long)(position / stepSizeZum_); //micron -> pulse
		cmd = "A:" + channel_ + sign + "P" + to_string(abs(pos_pulse));
		break;
	case ZStage::SHOT2:case ZStage::SHOT4:
		if (controlMode_ == g_ZStageOpenControl)
		{
			pos_pulse = (long)(position / stepSizeZum_); //micron -> pulse
			cmd = "A:" + channel_ + sign + "P" + to_string(abs(pos_pulse));
		}
		else if (controlMode_ == g_ZStageCloseControl)
		{
			double d = abs(position);
			char pos_micro[256] = { '\0' };
			sprintf(pos_micro, "%.*f", 2, d); //micron -> micron
			cmd = "A:" + channel_ + sign + "P" + pos_micro;
		}
		break;
	case ZStage::FINE1:case ZStage::FINE3:
		pos_nano = (long)(position * 1000);	//micron -> nano
		cmd = "A:" + channel_ + sign + "P" + to_string(abs(pos_nano));
		break;
	case ZStage::HSC3:case ZStage::SHRC3:

		if (model_ == HSC3) {
			pos_pulse = (long)(position / 0.01); //micron -> pulse
		}
		else {
			pos_pulse = (long)(position / stepSizeZum_); //micron -> pulse
		}
		
		cmd = "A:";
		for (int i = 1; i <= 3; i++)
		{
			if (i == stoi(channel_))
			{
				cmd += sign + to_string(abs(pos_pulse));
			}

			if (i != 3)
			{
				cmd += ",";
			}
		}
		//cmd = "A:,," + to_string(abs(pos_pulse));
		break;
	default:
		break;
	}
	// Send command parameters// Recieve data---------------------------------------------------------------

	ret = SendCheckRecievedOK(cmd);
	if (ret != DEVICE_OK)
		return ret;
	switch (model_)
	{
	case ZStage::SHOT1:case ZStage::SHOT2:case ZStage::SHOT4:case ZStage::FINE1:case ZStage::FINE3:
		ret = SendCheckRecievedOK("G:");
		if (ret != DEVICE_OK)
			return ret;
		break;
	case ZStage::HSC3:case ZStage::SHRC3:
		break;
	default:
		break;
	}
	

	return DEVICE_OK;
}

/// <summary>
/// Set division for Z stage 
/// </summary>
/// <param name="division"></param>
/// <returns></returns>
int ZStage::SetDivision(int division)
{
	//First Clear serial port from previous stuff-----------------------------------------------------------------------------------------------------
	int ret = ClearPort();
	if (ret != DEVICE_OK)
		return ret;
	// Create divsion set command---------------------------------------------------------------------------------------------------------------------

	if (model_ == SHRC3)
	{
		if (channel_ == "1") { ret = SendCheckRecievedOK("S:" + to_string(division) + ",,"); }
		else if (channel_ == "2") { ret = SendCheckRecievedOK("S:," + to_string(division) + ","); }
		else if (channel_ == "3") { ret = SendCheckRecievedOK("S:,," + to_string(division)); }
		if (ret != DEVICE_OK)
			return ret;
	}
	else 
	{
		ret = SendCheckRecievedOK("S:" + channel_ + to_string(division));
		if (ret != DEVICE_OK)
			return ret;
	}

	return DEVICE_OK;
}


/// <summary>
/// Getting the value of a slow-speed pulse <SHOT-controller>
/// </summary>
/// <param name="pls_vxy"></param>
/// <returns></returns>
long ZStage::GetSlowSpeedPulse(long pls_vxy)
{
	long pls_slow;

	if (pls_vxy <= 50000) { pls_slow = pls_vxy / 2; }
	else { pls_slow = 30000; }

	return pls_slow;
}

