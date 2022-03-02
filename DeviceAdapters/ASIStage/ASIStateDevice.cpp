/*
 * Project: ASIStage Device Adapter
 * License/Copyright: BSD 3-clause, see license.txt
 * Maintainers: Brandon Simpson (brandon@asiimaging.com)
 *              Jon Daniels (jon@asiimaging.com)
 */

#include "ASIStateDevice.h"

StateDevice::StateDevice() :
	ASIBase(this, ""), // LX-4000 Prefix Unknown
	numPos_(4),
	axis_("F"),
	position_(0),
	answerTimeoutMs_(1000)
{
	InitializeDefaultErrorMessages();

	// create pre-initialization properties
	// ------------------------------------

	// Name
	CreateProperty(MM::g_Keyword_Name, g_StateDeviceName, MM::String, true);

	// Description
	CreateProperty(MM::g_Keyword_Description, g_StateDeviceDescription, MM::String, true);

	// Port
	CPropertyAction* pAct = new CPropertyAction(this, &StateDevice::OnPort);
	CreateProperty(MM::g_Keyword_Port, "Undefined", MM::String, false, pAct, true);

	// Number of positions needs to be specified beforehand (later firmware allows querying)
	pAct = new CPropertyAction(this, &StateDevice::OnNumPositions);
	CreateProperty("NumPositions", "4", MM::Integer, false, pAct, true);

	// Axis names
	pAct = new CPropertyAction(this, &StateDevice::OnAxis);
	CreateProperty("Axis", "F", MM::String, false, pAct, true);
	AddAllowedValue("Axis", "F");
	AddAllowedValue("Axis", "T");
	AddAllowedValue("Axis", "Z");
}

StateDevice::~StateDevice()
{
	Shutdown();
}

void StateDevice::GetName(char* Name) const
{
	CDeviceUtils::CopyLimitedString(Name, g_StateDeviceName);
}

bool StateDevice::SupportsDeviceDetection(void)
{
	return true;
}

MM::DeviceDetectionStatus StateDevice::DetectDevice(void)
{
	return ASICheckSerialPort(*this, *GetCoreCallback(), port_, answerTimeoutMs_);
}

int StateDevice::Initialize()
{
	core_ = GetCoreCallback();

	CPropertyAction* pAct = new CPropertyAction(this, &StateDevice::OnVersion);
	CreateProperty("Version", "", MM::String, true, pAct);

	pAct = new CPropertyAction(this, &StateDevice::OnCompileDate);
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
		pAct = new CPropertyAction(this, &StateDevice::OnBuildName);
		CreateProperty("BuildName", "", MM::String, true, pAct);
		UpdateProperty("BuildName");
	}

	// state
	pAct = new CPropertyAction(this, &StateDevice::OnState);
	int ret = CreateProperty(MM::g_Keyword_State, "0", MM::Integer, false, pAct);
	if (ret != DEVICE_OK)
	{
		return ret;
	}

	char pos[3];
	for (int i = 0; i < numPos_; i++)
	{
		sprintf(pos, "%d", i);
		AddAllowedValue(MM::g_Keyword_State, pos);
	}

	// label
	pAct = new CPropertyAction(this, &CStateBase::OnLabel);
	ret = CreateProperty(MM::g_Keyword_Label, "", MM::String, false, pAct);
	if (ret != DEVICE_OK)
	{
		return ret;
	}

	char state[11];
	for (int i = 0; i < numPos_; i++)
	{
		sprintf(state, "Position-%d", i);
		SetPositionLabel(i, state);
	}

	// get current position
	ret = UpdateCurrentPosition(); // updates position_
	if (ret != DEVICE_OK)
	{
		return ret;
	}

	ret = UpdateStatus();
	if (ret != DEVICE_OK)
	{
		return ret;
	}

	initialized_ = true;
	return DEVICE_OK;
}

int StateDevice::Shutdown()
{
	if (initialized_)
	{
		initialized_ = false;
	}
	return DEVICE_OK;
}

bool StateDevice::Busy()
{
	// empty the Rx serial buffer before sending command
	ClearPort();

	const char* command = "/";
	std::string answer;

	// query command
	int ret = QueryCommand(command, answer);
	if (ret != DEVICE_OK)
	{
		return false;
	}

	if (answer.length() >= 1)
	{
		if (answer.substr(0, 1) == "B")
		{
			return true;
		}
		else if (answer.substr(0, 1) == "N")
		{
			return false;
		}
		else
		{
			return false;
		}
	}
	return false;
}

int StateDevice::UpdateCurrentPosition()
{
	// find out what position we are currently in
	std::ostringstream os;
	os << "W " << axis_;
	std::string answer;

	// query command
	int ret = QueryCommand(os.str().c_str(), answer);
	if (ret != DEVICE_OK)
	{
		return ret;
	}

	if (answer.substr(0, 2).compare(":N") == 0 && answer.length() > 2)
	{
		int errNo = atoi(answer.substr(2, 4).c_str());
		return ERR_OFFSET + errNo;
	}

	if (answer.substr(0, 2) == ":A")
	{
		position_ = (long)atoi(answer.substr(3, 2).c_str()) - 1;
	}
	else
	{
		return ERR_UNRECOGNIZED_ANSWER;
	}

	return DEVICE_OK;
}

/////////////////////////////////////////////////////////////////////////////////
//// Action handlers
/////////////////////////////////////////////////////////////////////////////////

int StateDevice::OnNumPositions(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set(numPos_);
	}
	else if (eAct == MM::AfterSet)
	{
		pProp->Get(numPos_);
	}
	return DEVICE_OK;
}

int StateDevice::OnAxis(MM::PropertyBase* pProp, MM::ActionType eAct)
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

int StateDevice::OnPort(MM::PropertyBase* pProp, MM::ActionType eAct)
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

int StateDevice::OnState(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set(position_);
	}
	else if (eAct == MM::AfterSet)
	{
		long position;
		pProp->Get(position);

		std::ostringstream os;
		os << "M " << axis_ << "=" << position + 1;
		std::string answer;

		// query command
		int ret = QueryCommand(os.str().c_str(), answer);
		if (ret != DEVICE_OK)
		{
			return ret;
		}

		if (answer.substr(0, 2).compare(":N") == 0 && answer.length() > 2)
		{
			int errNo = atoi(answer.substr(2, 4).c_str());
			return ERR_OFFSET + errNo;
		}

		if (answer.substr(0, 2) == ":A")
		{
			position_ = position;
		}
		else
		{
			return ERR_UNRECOGNIZED_ANSWER;
		}
	}
	return DEVICE_OK;
}
