///////////////////////////////////////////////////////////////////////////////
// FILE:          Stage.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Stage
//
// AUTHOR:        Athabasca Witschi (contact@zaber.com)

// COPYRIGHT:     Zaber Technologies Inc., 2014

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

#ifdef WIN32
#pragma warning(disable: 4355)
#endif

#include "Stage.h"

const char* g_StageName = "Stage";
const char* g_StageDescription = "Zaber Stage";

using namespace std;

Stage::Stage() :
	ZaberBase(this),
	deviceAddress_(1),
	axisNumber_(1),
	lockstepGroup_(0),
	stepSizeUm_(0.15625),
	convFactor_(1.6384), // not very informative name
	cmdPrefix_("/"),
	resolution_(64),
	motorSteps_(200),
	linearMotion_(2.0)
{
	this->LogMessage("Stage::Stage\n", true);

	InitializeDefaultErrorMessages();
	ZaberBase::setErrorMessages([&](auto code, auto message) { this->SetErrorText(code, message); });

	// Pre-initialization properties
	CreateProperty(MM::g_Keyword_Name, g_StageName, MM::String, true);

	CreateProperty(MM::g_Keyword_Description, "Zaber stage driver adapter", MM::String, true);

	CPropertyAction* pAct = new CPropertyAction (this, &Stage::OnPort);
	CreateProperty("Zaber Serial Port", port_.c_str(), MM::String, false, pAct, true);

	pAct = new CPropertyAction (this, &Stage::OnDeviceAddress);
	CreateIntegerProperty("Controller Device Number", deviceAddress_, false, pAct, true);
	SetPropertyLimits("Controller Device Number", 1, 99);

	pAct = new CPropertyAction(this, &Stage::OnAxisNumber);
	CreateIntegerProperty("Axis Number", axisNumber_, false, pAct, true);
	SetPropertyLimits("Axis Number", 1, 9);

	pAct = new CPropertyAction(this, &Stage::OnLockstepGroup);
	CreateIntegerProperty("Lockstep Group", lockstepGroup_, false, pAct, true);
	SetPropertyLimits("Lockstep Group", 0, 3);

	pAct = new CPropertyAction(this, &Stage::OnMotorSteps);
	CreateIntegerProperty("Motor Steps Per Rev", motorSteps_, false, pAct, true);

	pAct = new CPropertyAction(this, &Stage::OnLinearMotion);
	CreateFloatProperty("Linear Motion Per Motor Rev [mm]", linearMotion_, false, pAct, true);
}

Stage::~Stage()
{
	this->LogMessage("Stage::~Stage\n", true);
	Shutdown();
}

///////////////////////////////////////////////////////////////////////////////
// Stage & Device API methods
///////////////////////////////////////////////////////////////////////////////

void Stage::GetName(char* name) const
{
	CDeviceUtils::CopyLimitedString(name, g_StageName);
}

int Stage::Initialize()
{
	if (initialized_) return DEVICE_OK;

	core_ = GetCoreCallback();

	this->LogMessage("Stage::Initialize\n", true);

	// Activate any recently changed peripherals.
	auto ret = ActivatePeripheralsIfNeeded(deviceAddress_);
	if (ret != DEVICE_OK)
	{
		LogMessage("Peripheral activation check failed.\n", true);
		return ret;
	}

	// Calculate step size.
	ret = GetSetting(deviceAddress_, axisNumber_, "resolution", resolution_);
	if (ret != DEVICE_OK)
	{
		return ret;
	}
	stepSizeUm_ = ((double)linearMotion_/(double)motorSteps_)*(1/(double)resolution_)*1000;

	CPropertyAction* pAct;
	// Initialize Speed (in mm/s)
	pAct = new CPropertyAction (this, &Stage::OnSpeed);
	ret = CreateFloatProperty("Speed [mm/s]", 0.0, false, pAct);
	if (ret != DEVICE_OK)
	{
		return ret;
	}

	// Initialize Acceleration (in m/s²)
	pAct = new CPropertyAction (this, &Stage::OnAccel);
	ret = CreateFloatProperty("Acceleration [m/s^2]", 0.0, false, pAct);
	if (ret != DEVICE_OK)
	{
		return ret;
	}

	ret = UpdateStatus();
	if (ret != DEVICE_OK)
	{
		return ret;
	}

	if (ret == DEVICE_OK)
	{
		initialized_ = true;
	}

	return ret;
}

int Stage::Shutdown()
{
	this->LogMessage("Stage::Shutdown\n", true);
	if (initialized_)
	{
		initialized_ = false;
	}
	return DEVICE_OK;
}

bool Stage::Busy()
{
	this->LogMessage("Stage::Busy\n", true);
	return IsBusy(deviceAddress_);
}

int Stage::GetPositionUm(double& pos)
{
	this->LogMessage("Stage::GetPositionUm\n", true);

	long steps;
	int ret =  GetSetting(deviceAddress_, axisNumber_, "pos", steps);
	if (ret != DEVICE_OK)
	{
		return ret;
	}
	pos = steps * stepSizeUm_;
	return DEVICE_OK;
}

int Stage::GetPositionSteps(long& steps)
{
	this->LogMessage("Stage::GetPositionSteps\n", true);
	return GetSetting(deviceAddress_, axisNumber_, "pos", steps);
}

int Stage::SetPositionUm(double pos)
{
	this->LogMessage("Stage::SetPositionUm\n", true);
	long steps = nint(pos/stepSizeUm_);
	return SetPositionSteps(steps);
}

int Stage::SetRelativePositionUm(double d)
{
	this->LogMessage("Stage::SetRelativePositionUm\n", true);
	long steps = nint(d/stepSizeUm_);
	return SetRelativePositionSteps(steps);
}

int Stage::SetPositionSteps(long steps)
{
	this->LogMessage("Stage::SetPositionSteps\n", true);
	bool lockstep = lockstepGroup_ > 0;
	long axis = lockstep ? lockstepGroup_ : axisNumber_;
	return SendMoveCommand(deviceAddress_, axis, "abs", steps, lockstep);
}

int Stage::SetRelativePositionSteps(long steps)
{
	this->LogMessage("Stage::SetRelativePositionSteps\n", true);
	bool lockstep = lockstepGroup_ > 0;
	long axis = lockstep ? lockstepGroup_ : axisNumber_;
	return SendMoveCommand(deviceAddress_, axis, "rel", steps, lockstep);
}

int Stage::Move(double velocity)
{
	this->LogMessage("Stage::Move\n", true);
	// convert velocity from mm/s to Zaber data value
	long velData = nint(velocity*convFactor_*1000/stepSizeUm_);
	if (lockstepGroup_ > 0)
	{
		return SendMoveCommand(deviceAddress_, lockstepGroup_, "vel", velData, true);
	}
	else
	{
		return SendMoveCommand(deviceAddress_, axisNumber_, "vel", velData);
	}
}

int Stage::Stop()
{
	this->LogMessage("Stage::Stop\n", true);
	return ZaberBase::Stop(deviceAddress_, lockstepGroup_);
}

int Stage::Home()
{
	this->LogMessage("Stage::Home\n", true);
	//TODO try tools findrange first?
	if (lockstepGroup_ > 0)
	{
		ostringstream cmd;
		cmd << "lockstep " << lockstepGroup_ << " home";
		return SendAndPollUntilIdle(deviceAddress_, 0, cmd.str());
	}
	else
	{
		return SendAndPollUntilIdle(deviceAddress_, axisNumber_, "home");
	}
}

int Stage::SetAdapterOriginUm(double /*d*/)
{
	this->LogMessage("Stage::SetAdapterOriginUm\n", true);
	return DEVICE_UNSUPPORTED_COMMAND;
}

int Stage::SetOrigin()
{
	this->LogMessage("Stage::SetOrigin\n", true);
	return DEVICE_UNSUPPORTED_COMMAND;
}

int Stage::GetLimits(double& lower, double& upper)
{
	this->LogMessage("Stage::GetLimits\n", true);

	long min, max;
	int ret = ZaberBase::GetLimits(deviceAddress_, axisNumber_, min, max);
	if (ret != DEVICE_OK)
	{
		return ret;
	}
	lower = (double)min;
	upper = (double)max;
	return DEVICE_OK;
}

///////////////////////////////////////////////////////////////////////////////
// Action handlers
// Handle changes and updates to property values.
///////////////////////////////////////////////////////////////////////////////

int Stage::OnPort (MM::PropertyBase* pProp, MM::ActionType eAct)
{
	ostringstream os;
	os << "Stage::OnPort(" << pProp << ", " << eAct << ")\n";
	this->LogMessage(os.str().c_str(), false);

	if (eAct == MM::BeforeGet)
	{
		pProp->Set(port_.c_str());
	}
	else if (eAct == MM::AfterSet)
	{
		if (initialized_)
		{
			resetConnection();
		}

		pProp->Get(port_);
	}
	return DEVICE_OK;
}

int Stage::OnDeviceAddress (MM::PropertyBase* pProp, MM::ActionType eAct)
{
	this->LogMessage("Stage::OnDeviceAddress\n", true);

	if (eAct == MM::AfterSet)
	{
		pProp->Get(deviceAddress_);

		ostringstream cmdPrefix;
		cmdPrefix << "/" << deviceAddress_ << " ";
		cmdPrefix_ = cmdPrefix.str();
	}
	else if (eAct == MM::BeforeGet)
	{
		pProp->Set(deviceAddress_);
	}
	return DEVICE_OK;
}

int Stage::OnAxisNumber (MM::PropertyBase* pProp, MM::ActionType eAct)
{
	this->LogMessage("Stage::OnAxisNumber\n", true);

	if (eAct == MM::BeforeGet)
	{
		pProp->Set(axisNumber_);
	}
	else if (eAct == MM::AfterSet)
	{
		pProp->Get(axisNumber_);
	}
	return DEVICE_OK;
}

int Stage::OnLockstepGroup (MM::PropertyBase* pProp, MM::ActionType eAct)
{
	this->LogMessage("Stage::OnLockstepGroup\n", true);

	if (eAct == MM::BeforeGet)
	{
		pProp->Set(lockstepGroup_);
	}
	else if (eAct == MM::AfterSet)
	{
		pProp->Get(lockstepGroup_);
	}
	return DEVICE_OK;
}

int Stage::OnSpeed (MM::PropertyBase* pProp, MM::ActionType eAct)
{
	this->LogMessage("Stage::OnSpeed\n", true);

	if (eAct == MM::BeforeGet)
	{
		long speedData;
		int ret = GetSetting(deviceAddress_, axisNumber_, "maxspeed", speedData);
		if (ret != DEVICE_OK)
		{
			return ret;
		}

		// convert to mm/s
		double speed = (speedData/convFactor_)*stepSizeUm_/1000;
		pProp->Set(speed);
	}
	else if (eAct == MM::AfterSet)
	{
		double speed;
		pProp->Get(speed);

		// convert to data
		long speedData = nint(speed*convFactor_*1000/stepSizeUm_);
		if (speedData == 0 && speed != 0) speedData = 1; // Avoid clipping to 0.

		int ret = SetSetting(deviceAddress_, axisNumber_, "maxspeed", speedData);
		if (ret != DEVICE_OK)
		{
			return ret;
		}
	}
	return DEVICE_OK;
}

int Stage::OnAccel (MM::PropertyBase* pProp, MM::ActionType eAct)
{
	this->LogMessage("Stage::OnAccel\n", true);

	if (eAct == MM::BeforeGet)
	{
		long accelData;
		int ret = GetSetting(deviceAddress_, axisNumber_, "accel", accelData);
		if (ret != DEVICE_OK)
		{
			return ret;
		}

		// convert to m/s²
		double accel = (accelData*10/convFactor_)*stepSizeUm_/1000;
		pProp->Set(accel);
	}
	else if (eAct == MM::AfterSet)
	{
		double accel;
		pProp->Get(accel);

		// convert to data
		long accelData = nint(accel*convFactor_*100/(stepSizeUm_));
		if (accelData == 0 && accel != 0) accelData = 1; // Only set accel to 0 if user intended it.

		int ret = SetSetting(deviceAddress_, axisNumber_, "accel", accelData);
		if (ret != DEVICE_OK)
		{
			return ret;
		}
	}
	return DEVICE_OK;
}

int Stage::OnMotorSteps(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	this->LogMessage("Stage::OnMotorSteps\n", true);

	if (eAct == MM::BeforeGet)
	{
		pProp->Set(motorSteps_);
	}
	else if (eAct == MM::AfterSet)
	{
		pProp->Get(motorSteps_);
	}
	return DEVICE_OK;
}

int Stage::OnLinearMotion(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	this->LogMessage("Stage::OnLinearMotion\n", true);

	if (eAct == MM::BeforeGet)
	{
		pProp->Set(linearMotion_);
	}
	else if (eAct == MM::AfterSet)
	{
		pProp->Get(linearMotion_);
	}
	return DEVICE_OK;
}
