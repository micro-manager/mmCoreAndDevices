///////////////////////////////////////////////////////////////////////////////
// FILE:          Stage.h
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

#ifndef _STAGE_H_
#define _STAGE_H_

#include "Zaber.h"

extern const char* g_StageName;
extern const char* g_StageDescription;

class Stage : public CStageBase<Stage>, public ZaberBase
{
public:
	Stage();
	~Stage();

	// Device API
	// ----------
	int Initialize() override;
	int Shutdown() override;
	void GetName(char* name) const override;
	bool Busy() override;

	// Stage API
	// ---------
	int GetPositionUm(double& pos) override;
	int GetPositionSteps(long& steps) override;
	int SetPositionUm(double pos) override;
	int SetRelativePositionUm(double d) override;
	int SetPositionSteps(long steps) override;
	int SetRelativePositionSteps(long steps); // not in the base class
	int Move(double velocity) override;
	int Stop() override;
	int Home() override;
	int SetAdapterOriginUm(double d) override;
	int SetOrigin() override;
	int GetLimits(double& lower, double& upper) override;

	int IsStageSequenceable(bool& isSequenceable) const override { isSequenceable = false; return DEVICE_OK; }
	bool IsContinuousFocusDrive() const override { return false; }

	// action interface
	// ----------------
	int OnPort          (MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnDeviceAddress (MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnAxisNumber    (MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnLockstepGroup (MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnMotorSteps    (MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnLinearMotion  (MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnSpeed         (MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnAccel         (MM::PropertyBase* pProp, MM::ActionType eAct);

private:
	long deviceAddress_;
	long axisNumber_;
	long lockstepGroup_;
	double stepSizeUm_;
	double convFactor_; // not very informative name
	std::string cmdPrefix_;
	long resolution_;
	long motorSteps_;
	double linearMotion_;
};

#endif //_STAGE_H_
