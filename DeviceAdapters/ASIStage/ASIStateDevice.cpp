/*
 * Project: ASIStage Device Adapter
 * License/Copyright: BSD 3-clause, see license.txt
 * Maintainers: Brandon Simpson (brandon@asiimaging.com)
 *              Jon Daniels (jon@asiimaging.com)
 */

#include "ASIStateDevice.h"

StateDevice::StateDevice() :
	ASIBase(this, ""),
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

void StateDevice::GetName(char* name) const
{
	CDeviceUtils::CopyLimitedString(name, g_StateDeviceName);
}

bool StateDevice::SupportsDeviceDetection()
{
	return true;
}

MM::DeviceDetectionStatus StateDevice::DetectDevice()
{
	return ASIDetectDevice(*this, *GetCoreCallback(), port_, answerTimeoutMs_);
}

int StateDevice::Initialize()
{
	core_ = GetCoreCallback();

   int ret = GetVersion(firmwareVersion_);
	if (ret != DEVICE_OK)
       return ret;
	CPropertyAction* pAct = new CPropertyAction(this, &StateDevice::OnVersion);
	CreateProperty("Version", firmwareVersion_.c_str(), MM::String, true, pAct);

	// get the firmware version data from cached value
	version_ = Version::ParseString(firmwareVersion_);

	ret = GetCompileDate(firmwareDate_);
	if (ret != DEVICE_OK)
	{
		return ret;
	}
	pAct = new CPropertyAction(this, &StateDevice::OnCompileDate);
	CreateProperty("CompileDate", "", MM::String, true, pAct);

	// if really old firmware then don't get build name
	// build name is really just for diagnostic purposes anyway
	// I think it was present before 2010 but this is easy way

	// previously compared against compile date (2010, 1, 1)
	if (version_ >= Version(8, 8, 'a')) {
		ret = GetBuildName(firmwareBuild_);
		if (ret != DEVICE_OK)
		{
			return ret;
		}
		pAct = new CPropertyAction(this, &StateDevice::OnBuildName);
		CreateProperty("BuildName", "", MM::String, true, pAct);
	}

	// state
	pAct = new CPropertyAction(this, &StateDevice::OnState);
	ret = CreateProperty(MM::g_Keyword_State, "0", MM::Integer, false, pAct);
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

	std::string answer;
	int ret = QueryCommand("/", answer);
	if (ret != DEVICE_OK)
	{
		return false;
	}

	return !answer.empty() && answer.front() == 'B';
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

	else if (answer.length() > 2 && answer.compare(0, 2, ":N") == 0)
	{
		int errNo = atoi(answer.substr(2, 4).c_str());
		return ERR_OFFSET + errNo;
	}

	if (answer.compare(0, 2, ":A") == 0)
	{
		position_ = (long)atoi(answer.substr(3, 2).c_str()) - 1;
	}
	else
	{
		return ERR_UNRECOGNIZED_ANSWER;
	}

	return DEVICE_OK;
}

// Action handlers

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

		else if (answer.length() > 2 && answer.compare(0, 2, ":N") == 0)
		{
			int errNo = atoi(answer.substr(2, 4).c_str());
			return ERR_OFFSET + errNo;
		}

		if (answer.compare(0, 2, ":A") == 0)
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
