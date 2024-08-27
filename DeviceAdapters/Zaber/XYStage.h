///////////////////////////////////////////////////////////////////////////////
// FILE:          XYStage.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   XYStage Device Adapter
//
// AUTHOR:        David Goosen & Athabasca Witschi (contact@zaber.com)
//
// COPYRIGHT:     Zaber Technologies Inc., 2017
//
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

#ifndef _XYSTAGE_H_
#define _XYSTAGE_H_

#include "Zaber.h"

extern const char* g_XYStageName;
extern const char* g_XYStageDescription;

class XYStage : public CXYStageBase<XYStage>, public ZaberBase
{
public:
	XYStage();
	~XYStage();

	// Device API
	// ----------
	int Initialize() override;
	int Shutdown() override;
	void GetName(char* name) const override;
	bool Busy() override;

	// XYStage API
	// -----------
	int GetLimitsUm(double& xMin, double& xMax, double& yMin, double& yMax) override;
	int Move(double vx, double vy) override;
	int SetPositionSteps(long x, long y) override;
	int GetPositionSteps(long& x, long& y) override;
	int SetRelativePositionSteps(long x, long y) override;
	int Home() override;
	int Stop() override;
	int SetOrigin() override;
	int GetStepLimits(long& xMin, long& xMax, long& yMin, long& yMax) override;
	double GetStepSizeXUm() override { return stepSizeXUm_; }
	double GetStepSizeYUm() override { return stepSizeYUm_; }

	int IsXYStageSequenceable(bool& isSequenceable) const override { isSequenceable = false; return DEVICE_OK; }

	// action interface
	// ----------------
	int OnPort           (MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnAxisX          (MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnAxisY          (MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnLockstepGroupX (MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnLockstepGroupY (MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnMotorStepsX    (MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnMotorStepsY    (MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnLinearMotionX  (MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnLinearMotionY  (MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnSpeedX         (MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnSpeedY         (MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnAccelX         (MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnAccelY         (MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnDeviceAddress  (MM::PropertyBase* pProp, MM::ActionType eAct); // Single controller
	int OnDeviceAddressY (MM::PropertyBase* pProp, MM::ActionType eAct); // Composite XY (two controllers)

private:
	int SendXYMoveCommand(std::string type, long x, long y);
	int OnSpeed(long address, long axis, MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnAccel(long address, long axis, MM::PropertyBase* pProp, MM::ActionType eAct);
	void GetOrientation(bool& mirrorX, bool& mirrorY);

	inline bool IsSingleController() const
	{
		return (deviceAddressX_ == deviceAddressY_);
	}

	long deviceAddressX_;
	long deviceAddressY_;
	bool deviceAddressYInitialized_;
	bool rangeMeasured_;
	int homingTimeoutMs_;
	double stepSizeXUm_;
	double stepSizeYUm_;
	double convFactor_; // not very informative name
	long axisX_;
	long axisY_;
	long lockstepGroupX_;
	long lockstepGroupY_;
	long resolutionX_;
	long resolutionY_;
	long motorStepsX_;
	long motorStepsY_;
	double linearMotionX_;
	double linearMotionY_;
};

#endif //_XYSTAGE_H_
