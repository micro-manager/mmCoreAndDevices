///////////////////////////////////////////////////////////////////////////////
// FILE:          ObjectiveChanger.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Device adapter for Zaber's X-MOR objective changer.
//
// AUTHOR:        Martin Zak (contact@zaber.com)

// COPYRIGHT:     Zaber Technologies, 2023

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

#include "ObjectiveChanger.h"

const char* g_ObjectiveChangerName = "ObjectiveChanger";
const char* g_ObjectiveChangerDescription = "Zaber Objective Changer device adapter";

using namespace std;

ObjectiveChanger::ObjectiveChanger()
	: ZaberBase(this)
	, xMorAddress_(1)
	, xLdaAddress_(2)
	, numPositions_(0)
	, currentObjective_(1)
	, focusOffset_(0)
	, changedTime_(0.0)
{
	this->LogMessage("ObjectiveChanger::ObjectiveChanger\n", true);

	InitializeDefaultErrorMessages();
	ZaberBase::setErrorMessages([&](auto code, auto message) { this->SetErrorText(code, message); });

	EnableDelay(); // signals that the delay setting will be used

	// Pre-initialization properties
	CreateProperty(MM::g_Keyword_Name, g_ObjectiveChangerName, MM::String, true);

	CreateProperty(MM::g_Keyword_Description, g_ObjectiveChangerDescription, MM::String, true);

	CPropertyAction* pAct = new CPropertyAction(this, &ObjectiveChanger::PortGetSet);
	CreateProperty("Zaber Serial Port", port_.c_str(), MM::String, false, pAct, true);

	pAct = new CPropertyAction(this, &ObjectiveChanger::XMorAddressGetSet);
	CreateIntegerProperty("Objective Changer Device Number", xMorAddress_, false, pAct, true);
	SetPropertyLimits("Objective Changer Device Number", 1, 99);

	pAct = new CPropertyAction(this, &ObjectiveChanger::XLdaAddressGetSet);
	CreateIntegerProperty("Focus Stage Device Number", xLdaAddress_, false, pAct, true);
	SetPropertyLimits("Focus Stage Device Number", 1, 99);
}


ObjectiveChanger::~ObjectiveChanger()
{
	this->LogMessage("ObjectiveChanger::~ObjectiveChanger\n", true);
	Shutdown();
}


///////////////////////////////////////////////////////////////////////////////
// Stage & Device API methods
///////////////////////////////////////////////////////////////////////////////

void ObjectiveChanger::GetName(char* name) const
{
	CDeviceUtils::CopyLimitedString(name, g_ObjectiveChangerDescription);
}


int ObjectiveChanger::Initialize()
{
	if (initialized_)
	{
		return DEVICE_OK;
	}

	core_ = GetCoreCallback();

	this->LogMessage("ObjectiveChanger::Initialize\n", true);

	auto ret = handleException([=]() {
		ensureConnected();
		if (!this->changer_.getFocusAxis().isHomed()) {
			this->changer_.change(1);
		}
		});
	if (ret != DEVICE_OK)
	{
		this->LogMessage("Attempt to detect and home objective changer failed.\n", true);
		return ret;
	}

	// Get the number of positions and the current position.
	long index = -1;
	ret = GetRotaryIndexedDeviceInfo(xMorAddress_, 0, numPositions_, index);
	if (ret != DEVICE_OK)
	{
		this->LogMessage("Attempt to detect objective changer state and number of positions failed.\n", true);
		return ret;
	}

	CreateIntegerProperty("Number of Positions", numPositions_, true, 0, false);

	auto pAct = new CPropertyAction(this, &ObjectiveChanger::FocusOffsetGetSet);
	CreateFloatProperty("Objective Focus [mm]", focusOffset_, false, pAct);

	pAct = new CPropertyAction(this, &ObjectiveChanger::PositionGetSet);
	CreateIntegerProperty(MM::g_Keyword_State, index, false, pAct, false);

	pAct = new CPropertyAction(this, &ObjectiveChanger::DelayGetSet);
	ret = CreateProperty(MM::g_Keyword_Delay, "0.0", MM::Float, false, pAct);
	if (ret != DEVICE_OK)
	{
		return ret;
	}

	pAct = new CPropertyAction(this, &CStateBase::OnLabel);
	ret = CreateProperty(MM::g_Keyword_Label, "", MM::String, false, pAct);
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
		return DEVICE_OK;
	}
	else
	{
		return ret;
	}
}


int ObjectiveChanger::Shutdown()
{
	this->LogMessage("ObjectiveChanger::Shutdown\n", true);

	if (initialized_)
	{
		initialized_ = false;
	}

	return DEVICE_OK;
}


bool ObjectiveChanger::Busy()
{
	this->LogMessage("ObjectiveChanger::Busy\n", true);

	MM::MMTime interval = GetCurrentMMTime() - changedTime_;
	MM::MMTime delay(GetDelayMs() * 1000.0);

	if (interval < delay) {
		return true;
	}
	else
	{
		return IsBusy(xMorAddress_) || IsBusy(xLdaAddress_);
	}
}


int ObjectiveChanger::GetPositionLabel(long pos, char* label) const
{
	if (DEVICE_OK != CStateDeviceBase<ObjectiveChanger>::GetPositionLabel(pos, label))
	{
		std::stringstream labelStr("Objective ");
		labelStr << pos + 1;
		CDeviceUtils::CopyLimitedString(label, labelStr.str().c_str());
	}

	return DEVICE_OK;
}


///////////////////////////////////////////////////////////////////////////////
// Action handlers
// Handle changes and updates to property values.
///////////////////////////////////////////////////////////////////////////////

int ObjectiveChanger::DelayGetSet(MM::PropertyBase* pProp, MM::ActionType eAct)
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

int ObjectiveChanger::PortGetSet(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	ostringstream os;
	os << "ObjectiveChanger::PortGetSet(" << pProp << ", " << eAct << ")\n";
	this->LogMessage(os.str().c_str(), false);

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


int ObjectiveChanger::XMorAddressGetSet(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	this->LogMessage("ObjectiveChanger::XMorAddressGetSet\n", true);

	if (eAct == MM::AfterSet)
	{
		if (initialized_)
		{
			resetConnection();
		}

		pProp->Get(xMorAddress_);
	}
	else if (eAct == MM::BeforeGet)
	{
		pProp->Set(xMorAddress_);
	}

	return DEVICE_OK;
}


int ObjectiveChanger::XLdaAddressGetSet(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	this->LogMessage("ObjectiveChanger::XLdaAddressGetSet\n", true);

	if (eAct == MM::AfterSet)
	{
		if (initialized_)
		{
			resetConnection();
		}

		pProp->Get(xLdaAddress_);
	}
	else if (eAct == MM::BeforeGet)
	{
		pProp->Set(xLdaAddress_);
	}

	return DEVICE_OK;
}


int ObjectiveChanger::PositionGetSet(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	this->LogMessage("ObjectiveChanger::PositionGetSet\n", true);

	if (eAct == MM::BeforeGet)
	{
		if (initialized_)
		{
			int ret = GetSetting(xMorAddress_, 0, "motion.index.num", currentObjective_);
			if (ret != DEVICE_OK)
			{
				return ret;
			}
		}

		// MM uses 0-based indices for states, but the Zaber index
		// numbers position indices starting at 1.
		pProp->Set(currentObjective_ - 1);
	}
	else if (eAct == MM::AfterSet)
	{
		long indexToSet;
		pProp->Get(indexToSet);

		if (initialized_)
		{
			if ((indexToSet >= 0) && (indexToSet < numPositions_))
			{
				return setObjective(indexToSet + 1, false);
			}
			else
			{
				this->LogMessage("Requested position is outside the legal range.\n", true);
				return DEVICE_UNKNOWN_POSITION;
			}
		}
	}

	return DEVICE_OK;
}

int ObjectiveChanger::FocusOffsetGetSet(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	this->LogMessage("ObjectiveChanger::FocusOffsetGetSet\n", true);

	if (eAct == MM::AfterSet)
	{
		pProp->Get(focusOffset_);
	}
	else if (eAct == MM::BeforeGet)
	{
		pProp->Set(focusOffset_);
		return setObjective(currentObjective_, true);
	}

	return DEVICE_OK;
}

int ObjectiveChanger::setObjective(long objective, bool applyOffset) {
	return handleException([=]() {
		ensureConnected();
		zmlbase::Measurement offset;
		if (applyOffset) {
			offset = zmlbase::Measurement(focusOffset_ * ObjectiveChanger_::xLdaNativePerMm);
		}
		this->changer_.change(objective, offset);
		currentObjective_ = objective;
		changedTime_ = GetCurrentMMTime();
		});
}

void ObjectiveChanger::onNewConnection() {
	ZaberBase::onNewConnection();
	changer_ = zmlmi::ObjectiveChanger::find(*this->connection_, static_cast<int>(xMorAddress_), static_cast<int>(xLdaAddress_));
}
