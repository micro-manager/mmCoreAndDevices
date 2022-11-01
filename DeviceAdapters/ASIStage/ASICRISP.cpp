/*
 * Project: ASIStage Device Adapter
 * License/Copyright: BSD 3-clause, see license.txt
 * Maintainers: Brandon Simpson (brandon@asiimaging.com)
 *              Jon Daniels (jon@asiimaging.com)
 */

#include "ASICRISP.h"

CRISP::CRISP() :
	ASIBase(this, ""),
	axis_("Z"),
	waitAfterLock_(1000),
	answerTimeoutMs_(1000)
{
	InitializeDefaultErrorMessages();

	SetErrorText(ERR_NOT_CALIBRATED, "CRISP is not calibrated.  Try focusing close to a coverslip and selecting 'Calibrate'");
	SetErrorText(ERR_UNRECOGNIZED_ANSWER, "The ASI controller said something incomprehensible");
	SetErrorText(ERR_NOT_LOCKED, "The CRISP failed to lock");

	// create pre-initialization properties
	// ------------------------------------

	// Name
	CreateProperty(MM::g_Keyword_Name, g_CRISPDeviceName, MM::String, true);

	// Description
	CreateProperty(MM::g_Keyword_Description, g_CRISPDeviceDescription, MM::String, true);

	// Port
	CPropertyAction* pAct = new CPropertyAction(this, &CRISP::OnPort);
	CreateProperty(MM::g_Keyword_Port, "Undefined", MM::String, false, pAct, true);

	// Axis
	pAct = new CPropertyAction(this, &CRISP::OnAxis);
	CreateProperty("Axis", "Z", MM::String, false, pAct, true);
	AddAllowedValue("Axis", "Z");
	AddAllowedValue("Axis", "P");
	AddAllowedValue("Axis", "F");
}

CRISP::~CRISP()
{
	initialized_ = false;
}

void CRISP::GetName(char* pszName) const
{
	CDeviceUtils::CopyLimitedString(pszName, g_CRISPDeviceName);
}

bool CRISP::SupportsDeviceDetection(void)
{
	return true;
}

MM::DeviceDetectionStatus CRISP::DetectDevice(void)
{
	return ASICheckSerialPort(*this, *GetCoreCallback(), port_, answerTimeoutMs_);
}

int CRISP::Initialize()
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

	// Read-only "AxisLetter" property, axis_ is set using a pre-init property named "Axis".
	CreateProperty("AxisLetter", axis_.c_str(), MM::String, true);
	
	CPropertyAction* pAct = new CPropertyAction(this, &CRISP::OnVersion);
	CreateProperty("Version", "", MM::String, true, pAct);

	pAct = new CPropertyAction(this, &CRISP::OnCompileDate);
	CreateProperty("CompileDate", "", MM::String, true, pAct);
	UpdateProperty("CompileDate");

	// get the date of the firmware
	char compile_date[MM::MaxStrLength];
	if (GetProperty("CompileDate", compile_date) == DEVICE_OK)
	{
		compileDay_ = ExtractCompileDay(compile_date);
	}

	// get the firmware version data
	char version[MM::MaxStrLength];
	if (GetProperty("Version", version) == DEVICE_OK)
	{
		versionData_ = ExtractVersionData(std::string(version));
	}

	// if really old firmware then don't get build name
	// build name is really just for diagnostic purposes anyway
	// I think it was present before 2010 but this is easy way
	if (compileDay_ >= ConvertDay(2010, 1, 1))
	{
		pAct = new CPropertyAction(this, &CRISP::OnBuildName);
		CreateProperty("BuildName", "", MM::String, true, pAct);
		UpdateProperty("BuildName");
	}

	pAct = new CPropertyAction(this, &CRISP::OnFocus);
	CreateProperty(g_CRISPState, "Undefined", MM::String, false, pAct);

	// Add values (TODO: check manual)
	AddAllowedValue(g_CRISPState, g_CRISP_I);
	AddAllowedValue(g_CRISPState, g_CRISP_R);
	AddAllowedValue(g_CRISPState, g_CRISP_D);
	AddAllowedValue(g_CRISPState, g_CRISP_K);
	AddAllowedValue(g_CRISPState, g_CRISP_F);
	AddAllowedValue(g_CRISPState, g_CRISP_N);
	AddAllowedValue(g_CRISPState, g_CRISP_E);
	AddAllowedValue(g_CRISPState, g_CRISP_G);
	AddAllowedValue(g_CRISPState, g_CRISP_f);
	AddAllowedValue(g_CRISPState, g_CRISP_C);
	AddAllowedValue(g_CRISPState, g_CRISP_B);
	AddAllowedValue(g_CRISPState, g_CRISP_SG);
	AddAllowedValue(g_CRISPState, g_CRISP_RFO);
	AddAllowedValue(g_CRISPState, g_CRISP_SSZ);

	pAct = new CPropertyAction(this, &CRISP::OnWaitAfterLock);
	CreateProperty("Wait ms after Lock", "1000", MM::Integer, false, pAct);

	pAct = new CPropertyAction(this, &CRISP::OnNA);
	CreateProperty("Objective NA", "0.8", MM::Float, false, pAct);
	SetPropertyLimits("Objective NA", 0, 1.65);

	pAct = new CPropertyAction(this, &CRISP::OnLockRange);
	CreateProperty("Max Lock Range(mm)", "0.05", MM::Float, false, pAct);

	pAct = new CPropertyAction(this, &CRISP::OnCalGain);
	CreateProperty("Calibration Gain", "0.05", MM::Integer, false, pAct);

	pAct = new CPropertyAction(this, &CRISP::OnLEDIntensity);
	CreateProperty("LED Intensity", "50", MM::Integer, false, pAct);
	SetPropertyLimits("LED Intensity", 0, 100);

	pAct = new CPropertyAction(this, &CRISP::OnGainMultiplier);
	CreateProperty("GainMultiplier", "10", MM::Integer, false, pAct);
	SetPropertyLimits("GainMultiplier", 1, 100);

	pAct = new CPropertyAction(this, &CRISP::OnNumAvg);
	CreateProperty("Number of Averages", "1", MM::Integer, false, pAct);
	SetPropertyLimits("Number of Averages", 0, 8);

	pAct = new CPropertyAction(this, &CRISP::OnOffset);
	CreateProperty(g_CRISPOffsetPropertyName, "", MM::Integer, true, pAct);
	UpdateProperty(g_CRISPOffsetPropertyName);

	pAct = new CPropertyAction(this, &CRISP::OnState);
	CreateProperty(g_CRISPStatePropertyName, "", MM::String, true, pAct);
	UpdateProperty(g_CRISPStatePropertyName);

	// not sure exactly when Gary made these firmware changes, but they were there by start of 2015
	if (compileDay_ >= ConvertDay(2015, 1, 1))
	{
		pAct = new CPropertyAction(this, &CRISP::OnNumSkips);
		CreateProperty("Number of Skips", "0", MM::Integer, false, pAct);
		SetPropertyLimits("Number of Skips", 0, 100);
		UpdateProperty("Number of Skips");

		pAct = new CPropertyAction(this, &CRISP::OnInFocusRange);
		CreateProperty("In Focus Range(um)", "0.1", MM::Float, false, pAct);
		UpdateProperty("In Focus Range(um)");
	}

	const char* fc = "Obtain Focus Curve";
	pAct = new CPropertyAction(this, &CRISP::OnFocusCurve);
	CreateProperty(fc, " ", MM::String, false, pAct);
	AddAllowedValue(fc, " ");
	AddAllowedValue(fc, "Do it");

	for (long i = 0; i < SIZE_OF_FC_ARRAY; i++)
	{
		std::ostringstream os("");
		os << "Focus Curve Data" << i;
		CPropertyActionEx* pActEx = new CPropertyActionEx(this, &CRISP::OnFocusCurveData, i);
		CreateProperty(os.str().c_str(), "", MM::String, true, pActEx);
	}

	pAct = new CPropertyAction(this, &CRISP::OnSNR);
	CreateProperty("Signal Noise Ratio", "", MM::Float, true, pAct);

	pAct = new CPropertyAction(this, &CRISP::OnLogAmpAGC);
	CreateProperty("LogAmpAGC", "", MM::Integer, true, pAct);

	// use faster serial commands with new versions of the firmware
	if (versionData_.isVersionAtLeast(9, 2, 'o'))
	{
		// These commands use LK T? and LK Y? => ":A 0 \r\n"
		pAct = new CPropertyAction(this, &CRISP::OnDitherError);
		CreateProperty(g_CRISPDitherErrorPropertyName, "", MM::Integer, true, pAct);
		UpdateProperty(g_CRISPDitherErrorPropertyName);

		pAct = new CPropertyAction(this, &CRISP::OnSum);
		CreateProperty(g_CRISPSumPropertyName, "", MM::Integer, true, pAct);
		UpdateProperty(g_CRISPSumPropertyName);
	}
	else
	{
		// These commands use EXTRA X? => "I    9    0 \r\n"
		pAct = new CPropertyAction(this, &CRISP::OnDitherErrorLegacy);
		CreateProperty(g_CRISPDitherErrorPropertyName, "", MM::Integer, true, pAct);
		UpdateProperty(g_CRISPDitherErrorPropertyName);

		pAct = new CPropertyAction(this, &CRISP::OnSumLegacy);
		CreateProperty(g_CRISPSumPropertyName, "", MM::Integer, true, pAct);
		UpdateProperty(g_CRISPSumPropertyName);
	}

	return DEVICE_OK;
}

int CRISP::Shutdown()
{
	initialized_ = false;
	return DEVICE_OK;
}

bool CRISP::Busy()
{
	// TODO: implement this feature (if it makes sense!)
	return false;
}

/**
 * Note that offset is not in um but arbitrary (integer) numbers
 */
int CRISP::GetOffset(double& offset)
{
	float val;
	int ret = GetValue("LK Z?", val);
	if (ret != DEVICE_OK)
	{
		return ret;
	}

	int v = (int)val;
	offset = (double)v;

	return DEVICE_OK;
}

/**
 * Note that offset is not in um but arbitrary (integer) numbers
 */
int CRISP::SetOffset(double offset)
{
	std::ostringstream os;
	os << "LK Z=" << std::fixed << (int)offset;
	return SetCommand(os.str().c_str());
}

int CRISP::GetFocusState(std::string& focusState)
{
	// empty the Rx serial buffer before sending command
	ClearPort();

	const char* command = "LK X?"; // Requests single char lock state description
	std::string answer;
	// query command
	int ret = QueryCommand(command, answer);
	if (ret != DEVICE_OK)
	{
		return ERR_UNRECOGNIZED_ANSWER;
	}

	// translate response to one of our globals (see CRISP manual)
	char test = answer.c_str()[3];
	switch (test)
	{
		case 'I': focusState = g_CRISP_I; break;
		case 'R': focusState = g_CRISP_R; break;
		case 'D': focusState = g_CRISP_D; break;
		case 'K': focusState = g_CRISP_K; break;  // trying to lock, goes to F when locked
		case 'F': focusState = g_CRISP_F; break;  // this is read-only state
		case 'N': focusState = g_CRISP_N; break;
		case 'E': focusState = g_CRISP_E; break;
		case 'G': focusState = g_CRISP_G; break;
		case 'H':
		case 'C': focusState = g_CRISP_Cal; break;
		case 'o':
		case 'l': focusState = g_CRISP_RFO; break;
		case 'f': focusState = g_CRISP_f; break;
		case '1':
		case '2':
		case '3':
		case '4':
		case '5':
		case 'g':
		case 'h':
		case 'i':
		case 'j':
		case 't': focusState = g_CRISP_Cal; break;
		case 'B': focusState = g_CRISP_B; break;
		case 'a':
		case 'b':
		case 'c':
		case 'd':
		case 'e': focusState = g_CRISP_C; break;
		default:  focusState = g_CRISP_Unknown; break;
	}

	return DEVICE_OK;
}

int CRISP::SetFocusState(std::string focusState)
{
	std::string currentState;
	int ret = GetFocusState(currentState);
	if (ret != DEVICE_OK)
	{
		return ret;
	}

	if (focusState == currentState)
	{
		return DEVICE_OK;
	}

	return ForceSetFocusState(focusState);
}

int CRISP::ForceSetFocusState(std::string focusState)
{
	std::string currentState;
	int ret = GetFocusState(currentState);
	if (ret != DEVICE_OK)
	{
		return ret;
	}

	if (focusState == currentState)
	{
		return DEVICE_OK;
	}

	if (focusState == g_CRISP_I)
	{
		// Idle (switch off LED)
		const char* command = "LK F=79";
		return SetCommand(command);
	}
	else if (focusState == g_CRISP_R)
	{
		// Unlock
		const char* command = "LK F=85";
		return SetCommand(command);
	}
	else if (focusState == g_CRISP_K)
	{
		// Lock
		const char* command = "LK F=83";
		return SetCommand(command);
	}
	else if (focusState == g_CRISP_G)
	{
		// Log-Amp Calibration
		const char* command = "LK F=72";
		return SetCommand(command);
	}
	else if (focusState == g_CRISP_SG)
	{
		// gain_Cal Calibration
		const char* command = "LK F=67";
		return SetCommand(command);
	}
	else if (focusState == g_CRISP_f)
	{
		// Dither
		const char* command = "LK F=102";
		return SetCommand(command);
	}
	else if (focusState == g_CRISP_RFO)
	{
		// Reset focus offset
		const char* command = "LK F=111";
		return SetCommand(command);
	}
	else if (focusState == g_CRISP_SSZ)
	{
		// Reset focus offset
		const char* command = "SS Z";
		return SetCommand(command);
	}

	return DEVICE_OK;
}

bool CRISP::IsContinuousFocusLocked()
{
	std::string focusState;
	int ret = GetFocusState(focusState);
	if (ret != DEVICE_OK)
	{
		return false;
	}
	return (focusState == g_CRISP_K);
}

int CRISP::SetContinuousFocusing(bool state)
{
	bool focusingOn;
	int ret = GetContinuousFocusing(focusingOn);
	if (ret != DEVICE_OK)
	{
		return ret;
	}
	if (focusingOn && !state)
	{
		// was on, turning off
		return ForceSetFocusState(g_CRISP_R);
	}
	else if (!focusingOn && state)
	{
		// was off, turning on
		if (focusState_ == g_CRISP_R)
		{
			return ForceSetFocusState(g_CRISP_K);
		}
		else
		{
			// need to move to ready state, then turn on
			ret = ForceSetFocusState(g_CRISP_R);
			if (ret != DEVICE_OK)
			{
				return ret;
			}
			return ForceSetFocusState(g_CRISP_K);
		}
	}
	// if was already in state requested we don't need to do anything
	return DEVICE_OK;
}

int CRISP::GetContinuousFocusing(bool& state)
{
	std::string focusState;
	int ret = GetFocusState(focusState);
	if (ret != DEVICE_OK)
	{
		return ret;
	}

	state = ((focusState == g_CRISP_K) || (focusState == g_CRISP_F));
	return DEVICE_OK;
}

// Does a "one-shot" autofocus: locks and then unlocks again
int CRISP::FullFocus()
{
	int ret = SetContinuousFocusing(true);
	if (ret != DEVICE_OK)
	{
		return ret;
	}

	MM::MMTime startTime = GetCurrentMMTime();
	MM::MMTime wait(0, waitAfterLock_ * 1000);
	while (!IsContinuousFocusLocked() && ((GetCurrentMMTime() - startTime) < wait))
	{
		CDeviceUtils::SleepMs(25);
	}

	CDeviceUtils::SleepMs(waitAfterLock_);

	if (!IsContinuousFocusLocked())
	{
		SetContinuousFocusing(false);
		return ERR_NOT_LOCKED;
	}

	return SetContinuousFocusing(false);
}

int CRISP::IncrementalFocus()
{
	return FullFocus();
}

int CRISP::GetLastFocusScore(double& score)
{
	// empty the Rx serial buffer before sending command
	ClearPort();

	score = 0;
	const char* command = "LK Y?"; // Requests present value of the focus error as shown on LCD panel
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

int CRISP::GetCurrentFocusScore(double& score)
{
	return GetLastFocusScore(score);
}

int CRISP::GetValue(std::string cmd, float& val)
{
	std::string answer;
	// query command
	int ret = QueryCommand(cmd.c_str(), answer);
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
		size_t index = 0;
		while (!isdigit(answer[index]) && (index < answer.length()))
		{
			index++;
		}

		if (index >= answer.length())
		{
			return ERR_UNRECOGNIZED_ANSWER;
		}

		val = (float)atof((answer.substr(index)).c_str());

		return DEVICE_OK;
	}

	return ERR_UNRECOGNIZED_ANSWER;
}

int CRISP::SetCommand(std::string cmd)
{
	std::string answer;
	// query command
	int ret = QueryCommand(cmd.c_str(), answer);
	if (ret != DEVICE_OK)
	{
		return ret;
	}

	if (answer.length() > 2 && answer.substr(0, 2).compare(":N") == 0)
	{
		int errNo = atoi(answer.substr(2).c_str());
		return ERR_OFFSET + errNo;
	}

	if (answer.substr(0, 2) == ":A")
	{
		return DEVICE_OK;
	}

	return ERR_UNRECOGNIZED_ANSWER;
}

/////////////////////////////////////////////////////////////////////////////////
//// Action handlers
/////////////////////////////////////////////////////////////////////////////////

int CRISP::OnPort(MM::PropertyBase* pProp, MM::ActionType eAct)
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

int CRISP::OnFocus(MM::PropertyBase* pProp, MM::ActionType eAct)
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

int CRISP::OnWaitAfterLock(MM::PropertyBase* pProp, MM::ActionType eAct)
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

int CRISP::OnNA(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		float na;
		int ret = GetValue("LR Y?", na);
		if (ret != DEVICE_OK)
		{
			return ret;
		}
		pProp->Set(na);
	}
	else if (eAct == MM::AfterSet)
	{
		long na;
		pProp->Get(na);
		std::ostringstream command;
		command << std::fixed << "LR Y=" << na;

		return SetCommand(command.str());
	}
	return DEVICE_OK;
}

int CRISP::OnCalGain(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		float calGain;
		int ret = GetValue("LR X?", calGain);
		if (ret != DEVICE_OK)
		{
			return ret;
		}
		pProp->Set(calGain);
	}
	else if (eAct == MM::AfterSet)
	{
		double lr;
		pProp->Get(lr);
		std::ostringstream command;
		command << std::fixed << "LR X=" << (int)lr;

		return SetCommand(command.str());
	}

	return DEVICE_OK;
}

int CRISP::OnLockRange(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		float lockRange;
		int ret = GetValue("LR Z?", lockRange);
		if (ret != DEVICE_OK)
		{
			return ret;
		}
		pProp->Set(lockRange);
	}
	else if (eAct == MM::AfterSet)
	{
		double lr;
		pProp->Get(lr);
		std::ostringstream command;
		command << std::fixed << "LR Z=" << lr;

		return SetCommand(command.str());
	}
	return DEVICE_OK;
}

int CRISP::OnNumAvg(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		float numAvg;
		int ret = GetValue("RT F?", numAvg);
		if (ret != DEVICE_OK)
		{
			return ret;
		}
		pProp->Set(numAvg);
	}
	else if (eAct == MM::AfterSet)
	{
		long nr;
		pProp->Get(nr);
		std::ostringstream command;
		command << std::fixed << "RT F=" << nr;

		return SetCommand(command.str());
	}
	return DEVICE_OK;
}

int CRISP::OnGainMultiplier(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		float gainMultiplier;
		std::string command = "LR T?";
		int ret = GetValue(command.c_str(), gainMultiplier);
		if (ret != DEVICE_OK)
		{
			return ret;
		}
		pProp->Set(gainMultiplier);
	}
	else if (eAct == MM::AfterSet)
	{
		long nr;
		pProp->Get(nr);
		std::ostringstream command;
		command << std::fixed << "LR T=" << nr;

		return SetCommand(command.str());
	}
	return DEVICE_OK;
}

int CRISP::OnLEDIntensity(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		float ledIntensity;
		int ret = GetValue("UL X?", ledIntensity);
		if (ret != DEVICE_OK)
		{
			return ret;
		}
		pProp->Set(ledIntensity);
	}
	else if (eAct == MM::AfterSet)
	{
		long ledIntensity;
		pProp->Get(ledIntensity);
		std::ostringstream command;
		command << std::fixed << "UL X=" << ledIntensity;

		return SetCommand(command.str());
	}
	return DEVICE_OK;
}

int CRISP::OnFocusCurve(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set(" ");
	}
	else if (eAct == MM::AfterSet)
	{
		std::string val;
		pProp->Get(val);
		if (val == "Do it")
		{
			std::string answer;
			int ret = QueryCommand("LK F=97", answer);
			if (ret != DEVICE_OK)
			{
				return ret;
			}

			// We will time out while getting these data, so do not throw an error
			// Also, the total length will be about 3500 chars, since MM::MaxStrlength is 1024, we
			// need at least 4 strings.
			int index = 0;
			focusCurveData_[index] = "";
			bool done = false;
			// the GetSerialAnswer call will likely take more than 500ms, the likely timeout for the port set by the user
			// instead, wait for a total of ??? seconds
			MM::MMTime startTime = GetCurrentMMTime();
			MM::MMTime wait(10, 0);
			bool cont = true;
			while (cont && !done && index < SIZE_OF_FC_ARRAY)
			{
				ret = GetSerialAnswer(port_.c_str(), "\r\n", answer);
				if (answer == "end")
				{
					done = true;
				}
				else
				{
					focusCurveData_[index] += answer + "\r\n";
					if (focusCurveData_[index].length() > (MM::MaxStrLength - 40))
					{
						index++;
						if (index < SIZE_OF_FC_ARRAY)
						{
							focusCurveData_[index] = "";
						}
					}
				}
				cont = (GetCurrentMMTime() - startTime) < wait;
			}
		}
	}
	return DEVICE_OK;
}

int CRISP::OnFocusCurveData(MM::PropertyBase* pProp, MM::ActionType eAct, long index)
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set(focusCurveData_[index].c_str());
	}
	return DEVICE_OK;
}

int CRISP::OnAxis(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set(axis_.c_str());
	}
	else if (eAct == MM::AfterSet)
	{
		pProp->Get(axis_);
	}
	return DEVICE_OK;
}

int CRISP::OnSNR(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		float snr;
		std::string command = "EXTRA Y?";
		int ret = GetValue(command.c_str(), snr);
		if (ret != DEVICE_OK)
		{
			return ret;
		}
		pProp->Set(snr);
	}
	return DEVICE_OK;
}

int CRISP::OnDitherError(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		float sum;
		int ret = GetValue("LK Y?", sum);
		if (ret != DEVICE_OK)
		{
			return ret;
		}
		pProp->Set(sum);
	}
	else if (eAct == MM::AfterSet)
	{
		long nr;
		pProp->Get(nr);
		std::ostringstream command;
		command << std::fixed << "LK Y = " << nr;

		return SetCommand(command.str());
	}
	return DEVICE_OK;
}

int CRISP::OnLogAmpAGC(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		float val;
		int ret = GetValue("AL X?", val);
		if (ret != DEVICE_OK)
		{
			return ret;
		}
		pProp->Set(val);
	}
	return DEVICE_OK;
}

int CRISP::OnNumSkips(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		float numSkips;
		int ret = GetValue("UL Y?", numSkips);
		if (ret != DEVICE_OK)
		{
			return ret;
		}
		pProp->Set(numSkips);
	}
	else if (eAct == MM::AfterSet)
	{
		long nr;
		pProp->Get(nr);
		std::ostringstream command;
		command << std::fixed << "UL Y=" << nr;

		return SetCommand(command.str());
	}
	return DEVICE_OK;
}

int CRISP::OnInFocusRange(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		float focusRange;
		int ret = GetValue("AL Z?", focusRange);
		if (ret != DEVICE_OK)
		{
			return ret;
		}
		pProp->Set(focusRange * 1000);
	}
	else if (eAct == MM::AfterSet)
	{
		double lr;
		pProp->Get(lr);
		std::ostringstream command;
		command << std::fixed << "AL Z=" << lr / 1000;

		return SetCommand(command.str());
	}
	return DEVICE_OK;
}

int CRISP::OnSum(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		float sum;
		int ret = GetValue("LK T?", sum);
		if (ret != DEVICE_OK)
		{
			return ret;
		}
		pProp->Set(sum);
	}
	else if (eAct == MM::AfterSet)
	{
		long nr;
		pProp->Get(nr);
		std::ostringstream command;
		command << std::fixed << "LK T=" << nr;

		return SetCommand(command.str());
	}
	return DEVICE_OK;
}

int CRISP::OnOffset(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		double offset;
		int ret = GetOffset(offset);
		if (ret != DEVICE_OK)
		{
			return ret;
		}

		if (!pProp->Set(offset))
		{
			return DEVICE_INVALID_PROPERTY_VALUE;
		}
	}
	return DEVICE_OK;
}

int CRISP::OnState(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		std::string answer;
		int ret = QueryCommand("LK X?", answer);
		if (ret != DEVICE_OK)
		{
			return ret;
		}

		const char *state = &answer[answer.size()-1];
		if (!pProp->Set(state))
		{
			return DEVICE_INVALID_PROPERTY_VALUE;
		}
	}
	return DEVICE_OK;
}

// Provide support for MS2000 firmware < 9.2o
int CRISP::OnDitherErrorLegacy(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		std::string answer;
		int ret = QueryCommand("EXTRA X?", answer);
		if (ret != DEVICE_OK)
		{
			return ret;
		}

		std::istringstream is(answer);
		std::string ditherError;
		for (int i = 0; i < 3; i++)
		{
			is >> ditherError; // 3rd "is" is error
		}

		if (!pProp->Set(ditherError.c_str())) {
			return DEVICE_INVALID_PROPERTY_VALUE;
		}
	}
	return DEVICE_OK;
}

// Provide support for MS2000 firmware < 9.2o
int CRISP::OnSumLegacy(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		std::string answer;
		int ret = QueryCommand("EXTRA X?", answer);
		if (ret != DEVICE_OK)
		{
			return ret;
		}

		std::istringstream is(answer);
		std::string sum;
		for (int i = 0; i < 2; i++)
		{
			is >> sum; // 2nd "is" is sum
		}

		if (!pProp->Set(sum.c_str())) {
			return DEVICE_INVALID_PROPERTY_VALUE;
		}
	}
	return DEVICE_OK;
}
