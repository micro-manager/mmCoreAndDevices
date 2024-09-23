#pragma once
///////////////////////////////////////////////////////////////////////////////
// FILE:          WdiAutofocus.h
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

#include "Zaber.h"

extern const char* g_WdiAutofocusName;
extern const char* g_WdiAutofocusDescription;

namespace WdiAutofocus_ {
	const double xLdaNativePerMm = 1000000.0;
}

class WdiAutofocus : public CAutoFocusBase<WdiAutofocus>, public ZaberBase
{
public:
	WdiAutofocus();
	~WdiAutofocus();

	// Device API
	// ----------
	int Initialize() override;
	int Shutdown() override;
	void GetName(char* name) const override;
	bool Busy() override;

	// AutoFocus API
	// ----------
	bool IsContinuousFocusLocked() override;
	int FullFocus() override;
	int IncrementalFocus() override;
	int GetLastFocusScore(double& score) override;
	int GetCurrentFocusScore(double& score) override;
	int GetOffset(double& offset) override;
	int SetOffset(double offset) override;
	int SetContinuousFocusing(bool state) override;
	int GetContinuousFocusing(bool& state) override;

	// ZaberBase class overrides
	// ----------------
	void onNewConnection() override;

	// Properties
	// ----------------
	int PortGetSet(MM::PropertyBase* pProp, MM::ActionType eAct);
	int FocusAddressGetSet(MM::PropertyBase* pProp, MM::ActionType eAct);
	int FocusAxisGetSet(MM::PropertyBase* pProp, MM::ActionType eAct);
	int LimitMinGetSet(MM::PropertyBase* pProp, MM::ActionType eAct);
	int LimitMaxGetSet(MM::PropertyBase* pProp, MM::ActionType eAct);
	int LimitGetSet(MM::PropertyBase* pProp, MM::ActionType eAct, double& limit, const char* setting);

private:
	long focusAddress_;
	long focusAxis_;
	zml::Axis axis_;

	double limitMin_;
	double limitMax_;
};
