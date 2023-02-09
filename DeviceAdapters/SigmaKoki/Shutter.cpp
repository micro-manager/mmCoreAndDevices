///////////////////////////////////////////////////////////////////////////////
// FILE:          Shutter.cpp
// PROJECT:       Micro-Manager 2.0
// SUBSYSTEM:     DeviceAdapters
//  
//-----------------------------------------------------------------------------
// DESCRIPTION:   SIGMA-KOKI device adapter 2.0
//                
// AUTHOR   :    Hiroki Kibata, Abed Toufik  Release Date :  05/02/2023
//
// COPYRIGHT:     SIGMA KOKI CO.,LTD, Tokyo, 2023
#include "Shutter.h"
#include <iostream>

const char* g_ShutterDeviceName_C2B1 = "C2B-1";
const char* g_ShutterDeviceName_C2B2 = "C2B-2";
const char* g_ShutterDeviceName_C4B1 = "C4B-1";
const char* g_ShutterDeviceName_C4B2 = "C4B-2";
const char* g_ShutterDeviceName_C4B3 = "C4B-3";
const char* g_ShutterDeviceName_C4B4 = "C4B-4";

const char* g_ShutterModel = "ShutterModel";
const char* g_ShutterModel_BSH = "BSH/SSH-R";
const char* g_ShutterModel_BSH2 = "BSH2/SSH-25RA";
const char* g_ShutterModel_SSHS = "SSH-S";
const char* g_ShutterModel_SHPS = "SHPS";


Shutter::Shutter(const char* name, int channel) :
	SigmaBase(this),
	name_(name),
	channel_(channel),
	defUser1_("USER1"),
	defUser2_("USER2"),
	defUser3_("USER3"),
	changedTime_(0.0),
	model_(C2B)
{
	InitializeDefaultErrorMessages();

	// create pre-initialization properties
	// ------------------------------------

	// Name
	CreateProperty(MM::g_Keyword_Name, name_.c_str(), MM::String, true);     // should be fixed "name_"

	// Description
	CreateProperty(MM::g_Keyword_Description, "SIGMA-KOKI Shutter adapter", MM::String, true);

	// Port
	CPropertyAction* pAct = new CPropertyAction(this, &Shutter::OnPort);
	CreateProperty(MM::g_Keyword_Port, "Undefined", MM::String, false, pAct, true);
	EnableDelay();

	UpdateStatus();
}

Shutter::~Shutter()
{
	Shutdown();
}
/// <summary>
/// Busy 
/// </summary>
/// <returns></returns>
bool Shutter::Busy()
{
	MM::MMTime interval = GetCurrentMMTime() - changedTime_;
	MM::MMTime delay(GetDelayMs() * 1000.0);
	double tt = GetDelayMs();
	cout << "On busy" << endl;
	cout << "Get Delay (MS) = " << tt << endl;
	cout << "Get Delay value = " << tt*1000 << endl;


	if (interval < delay)
		return true;
	else
		return false;
}

/// <summary>
/// Get Name
/// </summary>
/// <param name="pszName"></param>
void Shutter::GetName(char* pszName) const
{
	CDeviceUtils::CopyLimitedString(pszName, name_.c_str());
}

/// <summary>
/// Initialization
/// </summary>
/// <returns></returns>
int Shutter::Initialize()
{
	
	if (initialized_)
		return DEVICE_OK;

	core_ = GetCoreCallback();
	
	// Set the device ID
	int ret = SetDeviceID(model_);
	if (ret != DEVICE_OK)
		return ERR_SHUTTER_DEVICE_UNRECOGNIZABLE;

	
	// Set the proprietes command system of controller 
	switch (model_)
	{
	case Shutter::C2B:// Set the command system of controller. 'SSH-C2B' controller only.
		ret = SetCommandSystem();
		if (ret != DEVICE_OK)
			return ERR_SHUTTER_COMMANDSYSTEM_FAILED;
		break;
	case Shutter::C4B:// Set the interrupt packet state. 'SSH-C4B' controller only
		ret = SetInterruptPacketState();
		if (ret != DEVICE_OK)
			return ERR_SHUTTER_INTERRUPT_FAILED;
		break;
	default:
		break;
	}

	// state propriety 
	CPropertyAction* pAct = new CPropertyAction(this, &Shutter::OnState);
	ret = CreateProperty(MM::g_Keyword_State, "0", MM::String, false, pAct);
	if (ret != DEVICE_OK)
		return ret;
	AddAllowedValue(MM::g_Keyword_State, "0");
	AddAllowedValue(MM::g_Keyword_State, "1");

	// Delay propriety
	pAct = new CPropertyAction(this, &Shutter::OnDelay);
	ret = CreateProperty(MM::g_Keyword_Delay, "0.0", MM::Float, false, pAct);
	if (ret != DEVICE_OK)
		return ret;


	// Set the delay time of controller side. Delay time is 0.0 seconds.
	if (model_ == C2B)
	{
		ret = SetControllerDelay(0.0);
		if (ret != DEVICE_OK)
			return ERR_SHUTTER_DELAY_FAILED;
	}

	

	// Model propriety
	pAct = new CPropertyAction(this, &Shutter::OnModel);
	string s = "";
	switch (model_)
	{
	case Shutter::C2B:
		modelType_ = g_ShutterModel_BSH2;
		ret = CreateProperty(g_ShutterModel, g_ShutterModel_BSH2, MM::String, false, pAct);
		if (ret != DEVICE_OK)
			return ret;
		AddAllowedValue(g_ShutterModel, g_ShutterModel_BSH);
		AddAllowedValue(g_ShutterModel, g_ShutterModel_SSHS);
		AddAllowedValue(g_ShutterModel, g_ShutterModel_SHPS);
		AddAllowedValue(g_ShutterModel, g_ShutterModel_BSH2);
		for (int i = 1; i <= 3; i++)
		{   // assign shutter name to s (propriety)
			ret = GetShutterName(s, 4 + i);
			if (ret != DEVICE_OK)
				return ERR_SHUTTER_MODELNAME_FAILED;
			switch (i)
			{
			case 1:
				defUser1_ = s;
				break;
			case 2:
				defUser2_ = s;
				break;
			case 3:
				defUser3_ = s;
				break;
			}
			AddAllowedValue(g_ShutterModel, s.c_str());
		}
		break;
	case Shutter::C4B:
		modelType_ = g_ShutterModel_BSH;
		ret = CreateProperty(g_ShutterModel, g_ShutterModel_BSH, MM::String, false, pAct);
		if (ret != DEVICE_OK)
			return ret;
		AddAllowedValue(g_ShutterModel, g_ShutterModel_BSH);
		AddAllowedValue(g_ShutterModel, g_ShutterModel_SSHS);
		break;
	default:
		break;
	}

	// Set the action mode.
	ret = SetShutterActionMode();
	if (ret != DEVICE_OK)
		return ERR_SHUTTER_ACTIONMODE_FAILED;

	ret = UpdateStatus();
	if (ret != DEVICE_OK)
		return ret;

	// Set timer for the Busy signal
	changedTime_ = GetCurrentMMTime();
	initialized_ = true;

	return DEVICE_OK;
}

/// <summary>
/// ShutDown 
/// </summary>
/// <returns></returns>
int Shutter::Shutdown()
{
	if (initialized_)
	{
		initialized_ = false;
	}
	return DEVICE_OK;
}
/// <summary>
/// Open Set 
/// </summary>
/// <param name="open"></param>
/// <returns></returns>

int Shutter::SetOpen(bool open)
{
	long pos;
	if (open)
	{
		pos = 1; //open
	}
	else
	{
		pos = 0; //close
	}
	return SetProperty(MM::g_Keyword_State, CDeviceUtils::ConvertToString(pos));
}

/// <summary>
/// Get Open 
/// </summary>
/// <param name="open"></param>
/// <returns></returns>
int Shutter::GetOpen(bool& open)
{
	char buf[MM::MaxStrLength];
	int ret = GetProperty(MM::g_Keyword_State, buf);
	if (ret != DEVICE_OK)
		return ret;
	long pos = atol(buf);
	pos == 1 ? open = true : open = false;

	return DEVICE_OK;
}

/// <summary>
/// Fire Command Not supported 
/// </summary>
/// <param name="deltaT"></param>
/// <returns></returns>
int Shutter::Fire(double deltaT)
{
	return DEVICE_UNSUPPORTED_COMMAND;
}




///////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////// Action handlers //////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////


/// <summary>
/// Action Handler On State 
/// </summary>
/// <param name="pProp"></param>
/// <param name="eAct"></param>
/// <returns></returns>
int Shutter::OnState(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	bool open;
	int ret;
	// Set timer for the Busy signal
	
	double tt = GetDelayMs();
	cout << "On state" << endl;
	cout << "Get Delay (MS) = " << tt << endl;
	if (eAct == MM::BeforeGet)
	{
		ret = GetShutterPosition(open);
		if (ret != DEVICE_OK)
			return ret;
		if (open)
		{
			pProp->Set("1");
		}
		else
		{
			pProp->Set("0");
		}
	}
	else if (eAct == MM::AfterSet)
	{
		changedTime_ = GetCurrentMMTime();
		string state = "";
		pProp->Get(state);

		if (state == "1")
		{
			ret = GetShutterPosition(open); //Check the status(Open?) of the shutter before run.
			if (ret != DEVICE_OK)
				return ret;
			if (!open)
			{
				return SetShutterPosition(true); //open
				
			}
		}
		else if (state == "0")
		{
			ret = GetShutterPosition(open); //Check the status(Close?) of the shutter before run.
			if (ret != DEVICE_OK)
				return ret;
			if (open)
			{
				return SetShutterPosition(false); //close
			}
		}
		else
		{
			return ERR_SHUTTER_STATE_FAILED;
		}
	}

	return DEVICE_OK;
}
/// <summary>
/// Action On POrt
/// </summary>
/// <param name="pProp"></param>
/// <param name="eAct"></param>
/// <returns></returns>
int Shutter::OnPort(MM::PropertyBase* pProp, MM::ActionType eAct)
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
/// On Delay
/// </summary>
/// <param name="pProp"></param>
/// <param name="eAct"></param>
/// <returns></returns>
int Shutter::OnDelay(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set(this->GetDelayMs());
	}
	else if (eAct == MM::AfterSet)
	{
		double delay;
		pProp->Get(delay);
		this->SetDelayMs(delay);
	}

	return DEVICE_OK;
}

/// <summary>
/// On Model
/// </summary>
/// <param name="pProp"></param>
/// <param name="eAct"></param>
/// <returns></returns>
int Shutter::OnModel(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	bool open;
	int ret = GetShutterPosition(open);
	if (ret != DEVICE_OK)
		return ERR_SHUTTER_STATE_FAILED;

	if (eAct == MM::BeforeGet)
	{
		if (!open)
		{
			pProp->Set(modelType_.c_str());
			ret = SetShutterModel(modelType_);
			if (ret != DEVICE_OK)
				return ERR_SHUTTER_MODEL_FAILED;
		}
	}
	else if (eAct == MM::AfterSet)
	{
		if (!open)
		{
			pProp->Get(modelType_);
			ret = SetShutterModel(modelType_);
			if (ret != DEVICE_OK)
				return ERR_SHUTTER_MODEL_FAILED;
		}
	}

	return DEVICE_OK;
}

/// <summary>
/// Send an open/close 
/// </summary>
/// <param name="state"></param>
/// <returns></returns>
int Shutter::SetShutterPosition(bool state)
{
	// First Clear serial port from previous stuff
	int ret = ClearPort();
	if (ret != DEVICE_OK)
		return ret;

	// Create command
	string cmd;
	switch (model_)
	{
	case Shutter::C4B:
		if (state)
		{
			cmd = "SH " + to_string(channel_) + ",1"; //open
		}
		else
		{
			cmd = "SH " + to_string(channel_) + ",0"; //close
		}
		break;
	case Shutter::C2B:
		if (state)
		{
			cmd = "OPEN:" + to_string(channel_); //open
		}
		else
		{
			cmd = "CLOSE:" + to_string(channel_); //close
		}
		break;
	}

	// Send command and Recieve data
	string answer = "";
	ret = SendRecieve(cmd, answer);
	if (ret != DEVICE_OK)
		return ret;

	if (answer == "A" || answer == "S")
	{
		return DEVICE_OK;
	}
	else
	{
		return ERR_SHUTTER_CONTROL_FAILED;
	}
}

/// <summary>
/// Check the state of the shutter.
/// </summary>
/// <param name="state"></param>
/// <returns></returns>
int Shutter::GetShutterPosition(bool& state)
{
	// First Clear serial port from previous stuff
	int ret = ClearPort();
	if (ret != DEVICE_OK)
		return ret;

	// Create command
	string cmd;
	switch (model_)
	{
	case Shutter::C4B:
		cmd = "GH " + to_string(channel_);
		break;
	case Shutter::C2B:
		cmd = "OPEN?" + to_string(channel_);
		break;
	}

	// Send command Recieve data
	string answer = "";

	ret = SendRecieve(cmd, answer);
	if (ret != DEVICE_OK)
		return ret;

	string buf = "";
	switch (model_)
	{
	case Shutter::C4B:
		if (answer == "B")
		{
			return ERR_SHUTTER_STATE_FAILED;
		}
		buf = answer.substr(2, 1);
		if (buf == "1") //open
		{
			state = true;
		}
		else if (buf == "0") //close
		{
			state = false;
		}
		break;
	case Shutter::C2B:
		if (answer == "C" || answer == "P")
		{
			return ERR_SHUTTER_STATE_FAILED;
		}
		buf = answer.substr(4, 1);
		if (buf == "O") //open
		{
			state = true;
		}
		else if (buf == "C") //close
		{
			state = false;
		}
		break;
	}

	return DEVICE_OK;
}

/// <summary>
/// Setting the Model_ of device model
/// </summary>
/// <param name="id"></param>
/// <returns></returns>
int Shutter::SetDeviceID(ShutterModel& model)
{

	if (name_ == g_ShutterDeviceName_C4B1)
	{
		model = Shutter::C4B;
	}
	else if (name_ == g_ShutterDeviceName_C4B2)
	{
		model = Shutter::C4B;
	}
	else if (name_ == g_ShutterDeviceName_C4B3)
	{
		model = Shutter::C4B;
	}
	else if (name_ == g_ShutterDeviceName_C4B4)
	{
		model = Shutter::C4B;
	}
	else if (name_ == g_ShutterDeviceName_C2B1)
	{
		model = Shutter::C2B;
	}
	else if (name_ == g_ShutterDeviceName_C2B2)
	{
		model = Shutter::C2B;
	}
	else
	{
		return false;
	}

	return DEVICE_OK;
}

/// <summary>
/// Setting the command system of controller.
// NOTE: 'SSH-C2B' controller only is valid.
// NOTE: It is controlled by the 'SSH-C2B command-system'. 'SSH-C4B command-system' is not used.
/// </summary>
/// <returns></returns>
int Shutter::SetCommandSystem()
{
	if (model_ != C2B)
	{
		return false;
	}

	// First Clear serial port from previous stuff
	int ret = ClearPort();
	if (ret != DEVICE_OK)
		return ret;

	// Send command// Recieve data
	string answer = "";
	ret = SendRecieve("SC 1", answer);
	if (ret != DEVICE_OK)
		return ret;

	if (answer == "S")
	{
		return DEVICE_OK;
	}
	else
	{
		return ERR_SHUTTER_COMMANDSYSTEM_FAILED;
	}
}

/// <summary>
/// Setting the shutter action mode.
/// NOTE: It is controlled by the 'BULB' mode. 'TIMER' mode is not used.
/// </summary>
/// <returns></returns>
int Shutter::SetShutterActionMode()
{
	// First Clear serial port from previous stuff
	int ret = ClearPort();
	if (ret != DEVICE_OK)
		return ret;

	// Create command
	string cmd;
	switch (model_)
	{
	case Shutter::C2B:
		cmd = "MODE:" + to_string(channel_) + ",B";
		break;
	case Shutter::C4B:
		cmd = "SM " + to_string(channel_) + ",0";
		break;
	default:
		break;
	}
	
	// Send command// Recieve data
	string answer = "";
	ret = SendRecieve(cmd, answer);
	if (ret != DEVICE_OK)
		return ret;

	if (answer == "A" || answer == "S")
	{
		return DEVICE_OK;
	}
	else
	{
		return ERR_SHUTTER_ACTIONMODE_FAILED;
	}
}

/// <summary>
/// Setting the shutter models.
/// </summary>
/// <param name="model"></param>
/// <returns></returns>
int Shutter::SetShutterModel(const std::string model)
{
	// First Clear serial port from previous stuff
	int ret = ClearPort();
	if (ret != DEVICE_OK)
		return ret;

	// Create command
	string cmd;
	if (model == g_ShutterModel_BSH) // BSH/SSH-R
	{
		switch (model_)
		{
		case Shutter::C4B:
			cmd = "PC " + to_string(channel_) + ",100";
			break;
		case Shutter::C2B:
			cmd = "SEL:" + to_string(channel_) + ",1";
			break;
		}
	}
	else if (model == g_ShutterModel_SSHS) // SSH-S
	{
		switch (model_)
		{
		case Shutter::C4B:
			cmd = "PC " + to_string(channel_) + ",002";
			break;
		case Shutter::C2B:
			cmd = "SEL:" + to_string(channel_) + ",2";
			break;
		}
	}
	else if (model == g_ShutterModel_SHPS) // SHPS
	{
		cmd = "SEL:" + to_string(channel_) + ",3";
	}
	else if (model == g_ShutterModel_BSH2) // BSH2/SSH-25RA
	{
		cmd = "SEL:" + to_string(channel_) + ",4";
	}
	else if (model == defUser1_) // User-defined model 1
	{
		cmd = "SEL:" + to_string(channel_) + ",5";
	}
	else if (model == defUser2_) // User-defined model 2
	{
		cmd = "SEL:" + to_string(channel_) + ",6";
	}
	else if (model == defUser3_) // User-defined model 3
	{
		cmd = "SEL:" + to_string(channel_) + ",7";
	}

	// Send command // Recieve data
	string answer = "";
	ret = SendRecieve(cmd, answer);
	if (ret != DEVICE_OK)
		return ret;

	if (answer == "A" || answer == "S")
	{
		return DEVICE_OK;
	}
	else
	{
		return ERR_SHUTTER_MODEL_FAILED;
	}
}

/// <summary>
/// Get the shutter model name.
/// NOTE: 'SSH-C2B' controller only is valid.
/// <returns></returns>
int Shutter::GetShutterName(std::string& name, int index)
{
	if (model_ != Shutter::C2B)
	{
		return false;
	}

	// First Clear serial port from previous stuff
	int ret = ClearPort();
	if (ret != DEVICE_OK)
		return ret;

	// Create command
	string cmd;
	cmd = "NAME?" + to_string(index);

	// Send command // Recieve data
	string answer = "";
	ret = SendRecieve(cmd, answer);
	if (ret != DEVICE_OK)
		return ret;

	if (answer == "C" || answer == "P")
	{
		return ERR_SHUTTER_MODELNAME_FAILED;
	}
	else
	{
		name = answer.substr(5, 7);
		while (*name.rbegin() == ' ')
		{
			name.erase(name.size() - 1);
		}

		return DEVICE_OK;
	}
}

/// <summary>
///  Setting the interrupt packet state.
/// NOTE: 'SSH-C4B' controller only is valid.
/// NOTE: Interrupt packet is not used.
/// <returns></returns>
int Shutter::SetInterruptPacketState()
{
	if (model_ != Shutter::C4B)
	{
		return false;
	}

	// First Clear serial port from previous stuff
	int ret = ClearPort();
	if (ret != DEVICE_OK)
		return ret;

	// Send command ---> 'SP 0' is the non-use mode. 'SP 1' is the use mode.
	// Recieve data
	string answer = "";
	ret = SendRecieve("SP 0", answer);
	if (ret != DEVICE_OK)
		return ret;

	if (answer == "A")
	{
		return DEVICE_OK;
	}
	else
	{
		return ERR_SHUTTER_INTERRUPT_FAILED;
	}
}

/// <summary>
/// Setting the delay time of the controller side.
/// NOTE: 'SSH-C2B' controller only is valid.
/// NOTE: In the case of 0.0 seconds is no delay time.
/// <returns></returns>
int Shutter::SetControllerDelay(double val)
{
	if (model_ !=Shutter::C2B)
	{
		return false;
	}

	if (val < 0.0 || val > 999.9)
	{
		return ERR_SHUTTER_DELAY_FAILED;
	}

	// First Clear serial port from previous stuff
	int ret = ClearPort();
	if (ret != DEVICE_OK)
		return ret;

	// Create command
	ostringstream c_val;
	c_val << val;
	string cmd = "DLY:" + to_string(channel_) + "," + c_val.str();

	// Send command// Recieve data
	string answer = "";
	ret = SendRecieve(cmd, answer);
	if (ret != DEVICE_OK)
		return ret;

	if (answer == "S")
	{
		return DEVICE_OK;
	}
	else
	{
		return ERR_SHUTTER_DELAY_FAILED;
	}
}
