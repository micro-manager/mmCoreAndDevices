/*
 * Project: ASIStage Device Adapter
 * License/Copyright: BSD 3-clause, see license.txt
 * Maintainers: Brandon Simpson (brandon@asiimaging.com)
 *              Jon Daniels (jon@asiimaging.com)
 */

#include "ASICRIF.h"

CRIF::CRIF() :
	ASIBase(this, ""), // LX-4000 Prefix Unknown
	justCalibrated_(false),
	axis_("Z"),
	stepSizeUm_(0.1),
	waitAfterLock_(3000)
{
	InitializeDefaultErrorMessages();

	SetErrorText(ERR_NOT_CALIBRATED, "CRIF is not calibrated.  Try focusing close to a coverslip and selecting 'Calibrate'");
	SetErrorText(ERR_UNRECOGNIZED_ANSWER, "The ASI controller said something incomprehensible");
	SetErrorText(ERR_NOT_LOCKED, "The CRIF failed to lock");

	// create pre-initialization properties
	// ------------------------------------

	// Name
	CreateProperty(MM::g_Keyword_Name, g_CRIFDeviceName, MM::String, true);

	// Description
	CreateProperty(MM::g_Keyword_Description, g_CRIFDeviceDescription, MM::String, true);

	// Port
	CPropertyAction* pAct = new CPropertyAction(this, &CRIF::OnPort);
	CreateProperty(MM::g_Keyword_Port, "Undefined", MM::String, false, pAct, true);
}

CRIF::~CRIF()
{
	initialized_ = false;
}

void CRIF::GetName(char* pszName) const
{
	CDeviceUtils::CopyLimitedString(pszName, g_CRIFDeviceName);
}


int CRIF::Initialize()
{
	core_ = GetCoreCallback();

	if (initialized_)
	{
		return DEVICE_OK;
	}

	// check status first (test for communication protocol)
	int ret = CheckDeviceStatus();
	if (ret != DEVICE_OK)
	{
		return ret;
	}

	CPropertyAction* pAct = new CPropertyAction(this, &CRIF::OnFocus);
	CreateProperty(g_CRIFState, "Undefined", MM::String, false, pAct);

	// Add values (TODO: check manual)
	AddAllowedValue(g_CRIFState, g_CRIF_I);
	AddAllowedValue(g_CRIFState, g_CRIF_L);
	AddAllowedValue(g_CRIFState, g_CRIF_Cal);
	AddAllowedValue(g_CRIFState, g_CRIF_G);
	AddAllowedValue(g_CRIFState, g_CRIF_B);
	AddAllowedValue(g_CRIFState, g_CRIF_k);
	AddAllowedValue(g_CRIFState, g_CRIF_K);
	AddAllowedValue(g_CRIFState, g_CRIF_O);

	pAct = new CPropertyAction(this, &CRIF::OnWaitAfterLock);
	CreateProperty("Wait ms after Lock", "3000", MM::Integer, false, pAct);

	pAct = new CPropertyAction(this, &CRIF::OnVersion);
	CreateProperty("Version", "", MM::String, true, pAct);

	pAct = new CPropertyAction(this, &CRIF::OnCompileDate);
	CreateProperty("CompileDate", "", MM::String, true, pAct);
	UpdateProperty("CompileDate");

	// get the date of the firmware
	char compile_date[MM::MaxStrLength];
	if (GetProperty("CompileDate", compile_date) == DEVICE_OK)
	{
		compileDay_ = ExtractCompileDay(compile_date);
	}

	// if really old firmware then don't get build name
	// build name is really just for diagnostic purposes anyway
	// I think it was present before 2010 but this is easy way
	if (compileDay_ >= ConvertDay(2010, 1, 1))
	{
		pAct = new CPropertyAction(this, &CRIF::OnBuildName);
		CreateProperty("BuildName", "", MM::String, true, pAct);
		UpdateProperty("BuildName");
	}

	initialized_ = true;
	return DEVICE_OK;
}

int CRIF::Shutdown()
{
	initialized_ = false;
	return DEVICE_OK;
}

bool CRIF::Busy()
{
	//TODO implement
	return false;
}

// TODO: See if this can be implemented for the CRIF
int CRIF::GetOffset(double& offset)
{
	offset = 0;
	return DEVICE_OK;
}

// TODO: See if this can be implemented for the CRIF
int CRIF::SetOffset(double /* offset */)
{
	return DEVICE_OK;
}

int CRIF::GetFocusState(std::string& focusState)
{
	// empty the Rx serial buffer before sending command
	ClearPort();

	const char* command = "LOCK X?"; // Requests single char lock state description
	std::string answer;

	// query command
	int ret = QueryCommand(command, answer);
	if (ret != DEVICE_OK)
	{
		return ERR_UNRECOGNIZED_ANSWER;
	}

	// translate response to one of our globals (see page 6 of CRIF manual)
	char test = answer.c_str()[3];
	switch (test) {
		case 'I':
			focusState = g_CRIF_I;
			break;
		case 'L':
			focusState = g_CRIF_L;
			break;
		case '1':
		case '2':
		case '3':
			focusState = g_CRIF_Cal;
			break;
		case 'G':
			focusState = g_CRIF_G;
			break;
		case 'B':
			focusState = g_CRIF_B;
			break;
		case 'k':
			focusState = g_CRIF_k;
			break;
		case 'K':
			focusState = g_CRIF_K;
			break;
		case 'E':
			focusState = g_CRIF_E;
			break;
		case 'O':
			focusState = g_CRIF_O;
			break;
		default:
			return ERR_UNRECOGNIZED_ANSWER;
	}
	return DEVICE_OK;
}

int CRIF::SetFocusState(std::string focusState)
{
	std::string currentState;
	int ret = GetFocusState(currentState);
	if (ret != DEVICE_OK)
	{
		return ret;
	}

	if (focusState == g_CRIF_I || focusState == g_CRIF_O)
	{
		// Unlock and switch off laser:
		ret = SetContinuousFocusing(false);
		if (ret != DEVICE_OK)
		{
			return ret;
		}
	}

	/*
	else if (focusState == g_CRIF_O) // we want the laser off and discard calibration (start anew)
	{
	   if (currentState == g_CRIF_K)
	   { // unlock first
		  ret = SetContinuousFocusing(false);
		  if (ret != DEVICE_OK)
			 return ret;
	   }
	   if (currentState == g_CRIF_G || currentState == g_CRIF_B || currentState == g_CRIF_L)
	   {
		  const char* command = "LK Z";
		  // query command and wait for acknowledgement
		  int ret = QueryCommandACK(command);
		  if (ret != DEVICE_OK)
			 return ret;
	   }
	   if (currentState == g_CRIF_L) // we need to advance the state once more.  Wait a bit for calibration to finish)
	   {
		  CDeviceUtils::SleepMs(1000); // ms
		  const char* command = "LK Z";
		  // query command and wait for acknowledgement
		  int ret = QueryCommandACK(command);
		  if (ret != DEVICE_OK)
			 return ret;
	   }
	}
	*/

	else if (focusState == g_CRIF_L)
	{
		if ((currentState == g_CRIF_I) || currentState == g_CRIF_O)
		{
			const char* command = "LK Z";
			// query command and wait for acknowledgement
			ret = QueryCommandACK(command);
			if (ret != DEVICE_OK)
			{
				return ret;
			}
		}
	}
	else if (focusState == g_CRIF_Cal)
	{
		const char* command = "LK Z";
		if (currentState == g_CRIF_B || currentState == g_CRIF_O)
		{
			// query command and wait for acknowledgement
			ret = QueryCommandACK(command);
			if (ret != DEVICE_OK)
			{
				return ret;
			}
			ret = GetFocusState(currentState);
			if (ret != DEVICE_OK)
			{
				return ret;
			}
		}
		if (currentState == g_CRIF_I) // Idle, first switch on laser
		{
			// query command and wait for acknowledgement
			ret = QueryCommandACK(command);
			if (ret != DEVICE_OK)
			{
				return ret;
			}
			ret = GetFocusState(currentState);
			if (ret != DEVICE_OK)
			{
				return ret;
			}
		}
		if (currentState == g_CRIF_L)
		{
			// query command and wait for acknowledgement
			ret = QueryCommandACK(command);
			if (ret != DEVICE_OK)
			{
				return ret;
			}
		}

		// now wait for the lock to occur
		MM::MMTime startTime = GetCurrentMMTime();
		MM::MMTime wait(3, 0);
		bool cont = false;
		std::string finalState;
		do {
			CDeviceUtils::SleepMs(250);
			GetFocusState(finalState);
			cont = (startTime - GetCurrentMMTime()) < wait;
		} while (finalState != g_CRIF_G && finalState != g_CRIF_B && cont);

		justCalibrated_ = true; // we need this to know whether this is the first time we lock
	}

	else if ((focusState == g_CRIF_K) || (focusState == g_CRIF_k))
	{
		// only try a lock when we are good
		if ((currentState == g_CRIF_G) || (currentState == g_CRIF_O))
		{
			ret = SetContinuousFocusing(true);
			if (ret != DEVICE_OK)
			{
				return ret;
			}
		}
		else if (!((currentState == g_CRIF_k) || currentState == g_CRIF_K))
		{
			// tell the user that we first need to calibrate before starting a lock
			return ERR_NOT_CALIBRATED;
		}
	}

	return DEVICE_OK;
}

bool CRIF::IsContinuousFocusLocked()
{
	std::string focusState;
	int ret = GetFocusState(focusState);
	if (ret != DEVICE_OK)
	{
		return false;
	}

	if (focusState == g_CRIF_K)
	{
		return true;
	}

	return false;
}

int CRIF::SetContinuousFocusing(bool state)
{
	// empty the Rx serial buffer before sending command
	ClearPort();

	std::string command;
	if (state)
	{
		// TODO: check that the system has been calibrated and can be locked!
		if (justCalibrated_)
		{
			command = "LK";
		}
		else
		{
			command = "RL"; // Turns on laser and initiated lock state using previously saved reference
		}
	}
	else
	{
		command = "UL X"; // Turns off laser and unlocks
	}
	std::string answer;
	// query command
	int ret = QueryCommand(command.c_str(), answer);
	if (ret != DEVICE_OK)
	{
		return ret;
	}

	// The controller only acknowledges receipt of the command
	if (answer.substr(0, 2) != ":A")
	{
		return ERR_UNRECOGNIZED_ANSWER;
	}

	justCalibrated_ = false;

	return DEVICE_OK;
}

int CRIF::GetContinuousFocusing(bool& state)
{
	std::string focusState;
	int ret = GetFocusState(focusState);
	if (ret != DEVICE_OK)
	{
		return ret;
	}

	if (focusState == g_CRIF_K)
	{
		state = true;
	}
	else
	{
		state = false;
	}

	return DEVICE_OK;
}

int CRIF::FullFocus()
{
	double pos;
	int ret = GetPositionUm(pos);
	if (ret != DEVICE_OK)
	{
		return ret;
	}
	ret = SetContinuousFocusing(true);
	if (ret != DEVICE_OK)
	{
		return ret;
	}

	MM::MMTime startTime = GetCurrentMMTime();
	MM::MMTime wait(3, 0);
	while (!IsContinuousFocusLocked() && ((GetCurrentMMTime() - startTime) < wait))
	{
		CDeviceUtils::SleepMs(25);
	}

	CDeviceUtils::SleepMs(waitAfterLock_);

	if (!IsContinuousFocusLocked())
	{
		SetContinuousFocusing(false);
		SetPositionUm(pos);
		return ERR_NOT_LOCKED;
	}

	return SetContinuousFocusing(false);
}

int CRIF::IncrementalFocus()
{
	return FullFocus();
}

int CRIF::GetLastFocusScore(double& score)
{
	// empty the Rx serial buffer before sending command
	ClearPort();

	score = 0;
	const char* command = "LOCK Y?"; // Requests present value of the PSD signal as shown on LCD panel
	std::string answer;
	// query command
	int ret = QueryCommand(command, answer);
	if (ret != DEVICE_OK)
	{
		return ret;
	}

	score = atof(answer.substr(2).c_str());
	if (score == 0)
	{
		return ERR_UNRECOGNIZED_ANSWER;
	}

	return DEVICE_OK;
}

int CRIF::SetPositionUm(double pos)
{
	// empty the Rx serial buffer before sending command
	ClearPort();

	std::ostringstream command;
	command << std::fixed << "M " << axis_ << "=" << pos / stepSizeUm_; // in 10th of micros

	std::string answer;
	// query the device
	int ret = QueryCommand(command.str().c_str(), answer);
	if (ret != DEVICE_OK)
	{
		return ret;
	}

	if (answer.substr(0, 2).compare(":A") == 0 || answer.substr(1, 2).compare(":A") == 0)
	{
		return DEVICE_OK;
	}
	// deal with error later
	else if (answer.substr(0, 2).compare(":N") == 0 && answer.length() > 2)
	{
		int errNo = atoi(answer.substr(4).c_str());
		return ERR_OFFSET + errNo;
	}

	return ERR_UNRECOGNIZED_ANSWER;
}

int CRIF::GetPositionUm(double& pos)
{
	// empty the Rx serial buffer before sending command
	ClearPort();

	std::ostringstream command;
	command << "W " << axis_;
	std::string answer;

	// query command
	int ret = QueryCommand(command.str().c_str(), answer);
	if (ret != DEVICE_OK)
	{
		return ret;
	}

	if (answer.length() > 2 && answer.substr(0, 2).compare(":N") == 0)
	{
		int errNo = atoi(answer.substr(2).c_str());
		return ERR_OFFSET + errNo;
	}
	else if (answer.length() > 0)
	{
		char head[64];
		float zz;
		char iBuf[256];
		strcpy(iBuf, answer.c_str());
		sscanf(iBuf, "%s %f\r\n", head, &zz);

		pos = zz * stepSizeUm_;

		return DEVICE_OK;
	}

	return ERR_UNRECOGNIZED_ANSWER;
}

/////////////////////////////////////////////////////////////////////////////////
//// Action handlers
/////////////////////////////////////////////////////////////////////////////////

int CRIF::OnPort(MM::PropertyBase* pProp, MM::ActionType eAct)
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

int CRIF::OnFocus(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		int ret = GetFocusState(focusState_);
		if (ret != DEVICE_OK)
		{
			return ret;
		}
		pProp->Set(focusState_.c_str());
	}
	else if (eAct == MM::AfterSet)
	{
		pProp->Get(focusState_);
		int ret = SetFocusState(focusState_);
		if (ret != DEVICE_OK)
		{
			return ret;
		}
	}
	return DEVICE_OK;
}

int CRIF::OnWaitAfterLock(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set(waitAfterLock_);
	}
	else if (eAct == MM::AfterSet)
	{
		pProp->Get(waitAfterLock_);
	}
	return DEVICE_OK;
}
