/*
 * Project: ASIStage Device Adapter
 * License/Copyright: BSD 3-clause, see license.txt
 * Maintainers: Brandon Simpson (brandon@asiimaging.com)
 *              Jon Daniels (jon@asiimaging.com)
 */

#include "ASITurret.h"

AZ100Turret::AZ100Turret() :
	ASIBase(this, ""),
	numPos_(4),
	position_(0)
{
	InitializeDefaultErrorMessages();

	// create pre-initialization properties
	// ------------------------------------

	// Name
	CreateProperty(MM::g_Keyword_Name, g_AZ100TurretName, MM::String, true);

	// Description
	CreateProperty(MM::g_Keyword_Description, g_AZ100TurretDescription, MM::String, true);

	// Port
	CPropertyAction* pAct = new CPropertyAction(this, &AZ100Turret::OnPort);
	CreateProperty(MM::g_Keyword_Port, "Undefined", MM::String, false, pAct, true);
}

AZ100Turret::~AZ100Turret()
{
	Shutdown();
}

void AZ100Turret::GetName(char* name) const
{
	CDeviceUtils::CopyLimitedString(name, g_AZ100TurretName);
}

int AZ100Turret::Initialize()
{
	core_ = GetCoreCallback();

	CPropertyAction* pAct = new CPropertyAction(this, &AZ100Turret::OnState);
	int ret = CreateProperty(MM::g_Keyword_State, "0", MM::Integer, false, pAct);
	if (ret != DEVICE_OK)
	{
		return ret;
	}

	AddAllowedValue(MM::g_Keyword_State, "0");
	AddAllowedValue(MM::g_Keyword_State, "1");
	AddAllowedValue(MM::g_Keyword_State, "2");
	AddAllowedValue(MM::g_Keyword_State, "3");

	// label
	pAct = new CPropertyAction(this, &CStateBase::OnLabel);
	ret = CreateProperty(MM::g_Keyword_Label, "", MM::String, false, pAct);
	if (ret != DEVICE_OK)
	{
		return ret;
	}

	SetPositionLabel(0, "Position-1");
	SetPositionLabel(1, "Position-2");
	SetPositionLabel(2, "Position-3");
	SetPositionLabel(3, "Position-4");

	ret = UpdateStatus();
	if (ret != DEVICE_OK)
	{
		return ret;
	}

   ret = GetVersion(firmwareVersion_);
   if (ret != DEVICE_OK)
       return ret;
	pAct = new CPropertyAction(this, &AZ100Turret::OnVersion);
	CreateProperty("Version", firmwareVersion_.c_str(), MM::String, true, pAct);

	// get the firmware version data from cached value
	version_ = Version::ParseString(firmwareVersion_);

	ret = GetCompileDate(firmwareDate_);
	if (ret != DEVICE_OK)
	{
		return ret;
	}
	pAct = new CPropertyAction(this, &AZ100Turret::OnCompileDate);
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
		pAct = new CPropertyAction(this, &AZ100Turret::OnBuildName);
		CreateProperty("BuildName", "", MM::String, true, pAct);
	}

	initialized_ = true;
	return DEVICE_OK;
}

int AZ100Turret::Shutdown()
{
	if (initialized_)
	{
		initialized_ = false;
	}
	return DEVICE_OK;
}

bool AZ100Turret::Busy()
{
	// empty the Rx serial buffer before sending command
	ClearPort();

	std::string answer;
	int ret = QueryCommand("RS F", answer);
	if (ret != DEVICE_OK)
	{
		return false;
	}

	if (answer.length() >= 1)
	{
		int status = atoi(answer.substr(2).c_str());
		if (status & 1)
		{
			return true;
		}
		return false;
	}
	return false;
}

// Action handlers

int AZ100Turret::OnPort(MM::PropertyBase* pProp, MM::ActionType eAct)
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

int AZ100Turret::OnState(MM::PropertyBase* pProp, MM::ActionType eAct)
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
		os << "MTUR X=" << position + 1;
		std::string answer;

		// query command
		int ret = QueryCommand(os.str().c_str(), answer);
		if (ret != DEVICE_OK)
		{
			return ret;
		}

		if (answer.length() > 2 && answer.compare(0, 2, ":N") == 0)
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
