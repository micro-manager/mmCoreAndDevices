///////////////////////////////////////////////////////////////////////////////
// FILE:          WdiAutofocus.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Device adapter for WDI autofocus.
//
// AUTHOR:        Martin Zak (contact@zaber.com)

// COPYRIGHT:     Zaber Technologies, 2024

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

#include "WdiAutofocus.h"

const char* g_WdiAutofocusName = "WdiAutofocus";
const char* g_WdiAutofocusDescription = "Zaber WDI Autofocus device adapter";
const char* g_Msg_AF_MOVEMENT_FAILED = "The movement has failed. Ensure that the autofocus is in range and the focus axis is within the limits.";

using namespace std;

WdiAutofocus::WdiAutofocus() :
	ZaberBase(this),
	focusAddress_(1),
	focusAxis_(1),
	limitMin_(0.0),
	limitMax_(0.0)
{
	this->LogMessage("WdiAutofocus::WdiAutofocus\n", true);

	InitializeDefaultErrorMessages();
	ZaberBase::setErrorMessages([&](auto code, auto message) { this->SetErrorText(code, message); });
	this->SetErrorText(ERR_MOVEMENT_FAILED, g_Msg_AF_MOVEMENT_FAILED);

	CreateProperty(MM::g_Keyword_Name, g_WdiAutofocusName, MM::String, true);
	CreateProperty(MM::g_Keyword_Description, g_WdiAutofocusDescription, MM::String, true);

	CPropertyAction* pAct = new CPropertyAction(this, &WdiAutofocus::PortGetSet);
	CreateProperty("Zaber Serial Port", port_.c_str(), MM::String, false, pAct, true);

	pAct = new CPropertyAction(this, &WdiAutofocus::FocusAddressGetSet);
	CreateIntegerProperty("Focus Stage Device Number", focusAddress_, false, pAct, true);
	SetPropertyLimits("Focus Stage Device Number", 1, 99);

	pAct = new CPropertyAction(this, &WdiAutofocus::FocusAxisGetSet);
	CreateIntegerProperty("Focus Stage Axis Number", focusAxis_, false, pAct, true);
	SetPropertyLimits("Focus Stage Axis Number", 1, 99);
}

WdiAutofocus::~WdiAutofocus()
{
	this->LogMessage("WdiAutofocus::~WdiAutofocus\n", true);
	Shutdown();
}

void WdiAutofocus::GetName(char* name) const
{
	CDeviceUtils::CopyLimitedString(name, g_WdiAutofocusDescription);
}

int WdiAutofocus::Initialize()
{
	if (initialized_)
	{
		return DEVICE_OK;
	}

	core_ = GetCoreCallback();

	this->LogMessage("WdiAutofocus::Initialize\n", true);

	auto pAct = new CPropertyAction(this, &WdiAutofocus::LimitMinGetSet);
	CreateFloatProperty("Limit Min [mm]", limitMin_, false, pAct);
	pAct = new CPropertyAction(this, &WdiAutofocus::LimitMaxGetSet);
	CreateFloatProperty("Limit Max [mm]", limitMax_, false, pAct);

	auto ret = UpdateStatus();
	if (ret == DEVICE_OK)
	{
		initialized_ = true;
	}

	return ret;
}

int WdiAutofocus::Shutdown()
{
	this->LogMessage("WdiAutofocus::Shutdown\n", true);

	if (initialized_)
	{
		initialized_ = false;
	}

	return DEVICE_OK;
}

bool WdiAutofocus::Busy()
{
	this->LogMessage("WdiAutofocus::Busy\n", true);

	bool busy = false;
	auto ret = handleException([&]() {
		ensureConnected();
		busy = axis_.isBusy();
	});
	return ret == DEVICE_OK && busy;
}

int WdiAutofocus::PortGetSet(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	ostringstream os;
	os << "WdiAutofocus::PortGetSet(" << pProp << ", " << eAct << ")\n";
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

int WdiAutofocus::FocusAddressGetSet(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	this->LogMessage("WdiAutofocus::FocusAddressGetSet\n", true);

	if (eAct == MM::AfterSet)
	{
		if (initialized_)
		{
			resetConnection();
		}

		pProp->Get(focusAddress_);
	}
	else if (eAct == MM::BeforeGet)
	{
		pProp->Set(focusAddress_);
	}

	return DEVICE_OK;
}

int WdiAutofocus::FocusAxisGetSet(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	this->LogMessage("WdiAutofocus::FocusAxisGetSet\n", true);

	if (eAct == MM::AfterSet)
	{
		if (initialized_)
		{
			resetConnection();
		}

		pProp->Get(focusAxis_);
	}
	else if (eAct == MM::BeforeGet)
	{
		pProp->Set(focusAxis_);
	}

	return DEVICE_OK;
}

int WdiAutofocus::LimitMinGetSet(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	this->LogMessage("ObjectiveChanger::LimitMinGetSet\n", true);
	return LimitGetSet(pProp, eAct, limitMin_, "motion.tracking.limit.min");
}

int WdiAutofocus::LimitMaxGetSet(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	this->LogMessage("ObjectiveChanger::LimitMaxGetSet\n", true);
	return LimitGetSet(pProp, eAct, limitMax_, "motion.tracking.limit.max");
}

int WdiAutofocus::LimitGetSet(MM::PropertyBase* pProp, MM::ActionType eAct, double& limit, const char* setting)
{
	if (eAct == MM::AfterSet)
	{
		double newLimit;
		pProp->Get(newLimit);
		bool update = limit != newLimit;
		limit = newLimit;

		if (update) {
			return handleException([&]() {
				ensureConnected();
				axis_.getSettings().set(setting, limit * WdiAutofocus_::xLdaNativePerMm);
			});
		}
	}
	else if (eAct == MM::BeforeGet)
	{
		if (initialized_)
		{
			int ret = handleException([&]() {
				ensureConnected();
				limit = axis_.getSettings().get(setting) / WdiAutofocus_::xLdaNativePerMm;
			});
			if (ret != DEVICE_OK)
			{
				return ret;
			}
		}

		pProp->Set(limit);
	}

	return DEVICE_OK;
}

int WdiAutofocus::FullFocus() {
	this->LogMessage("WdiAutofocus::FullFocus\n", true);

	return handleException([&]() {
		ensureConnected();

		axis_.genericCommand("move track once");
		axis_.waitUntilIdle();
	});
}

int WdiAutofocus::IncrementalFocus() {
	this->LogMessage("WdiAutofocus::IncrementalFocus\n", true);

	return FullFocus();
}

int WdiAutofocus::GetLastFocusScore(double& score) {
	this->LogMessage("WdiAutofocus::GetLastFocusScore\n", true);

	return GetCurrentFocusScore(score);
}

int WdiAutofocus::GetCurrentFocusScore(double& score) {
	this->LogMessage("WdiAutofocus::GetCurrentFocusScore\n", true);

	score = 0.0;
	return handleException([&]() {
		ensureConnected();

		auto reply = axis_.getDevice().genericCommand("io get ai 1");
		score = stod(reply.getData());
	});
}

int WdiAutofocus::GetOffset(double& offset) {
	this->LogMessage("WdiAutofocus::GetOffset\n", true);
	offset = 0.0;
	return DEVICE_OK;
}
int WdiAutofocus::SetOffset(double offset) {
	this->LogMessage("WdiAutofocus::SetOffset\n", true);
	if (offset != 0.0) {
		return DEVICE_UNSUPPORTED_COMMAND;
	}
	return DEVICE_OK;
}

int WdiAutofocus::SetContinuousFocusing(bool state) {
	this->LogMessage("WdiAutofocus::SetContinuousFocusing\n", true);

	return handleException([&]() {
		ensureConnected();

		if (state) {
			axis_.genericCommand("move track");
		} else {
			axis_.stop();
		}
	});
}

int WdiAutofocus::GetContinuousFocusing(bool& state) {
	this->LogMessage("WdiAutofocus::GetContinuousFocusing\n", true);
	return handleException([&]() {
		ensureConnected();
		state = axis_.isBusy();
	});
}

bool WdiAutofocus::IsContinuousFocusLocked() {
	this->LogMessage("WdiAutofocus::IsContinuousFocusLocked\n", true);

	bool locked;
	auto ret = handleException([&]() {
		ensureConnected();
		locked = axis_.getSettings().get("motion.tracking.settle.tolerance.met") > 0;
	});
	return ret == DEVICE_OK && locked;
}

void WdiAutofocus::onNewConnection() {
	ZaberBase::onNewConnection();
	axis_ = connection_->getDevice(focusAddress_).getAxis(focusAxis_);
}
