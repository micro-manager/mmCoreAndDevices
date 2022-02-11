///////////////////////////////////////////////////////////////////////////////
// FILE:          ASIDacXYStage.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   ASI DAC XYStage device adapter
//
// COPYRIGHT:     Applied Scientific Instrumentation, Eugene OR
//
// LICENSE:       This file is distributed under the BSD license.
//
//                This file is distributed in the hope that it will be useful,
//                but WITHOUT ANY WARRANTY; without even the implied warranty
//                of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
//
//                IN NO EVENT SHALL THE COPYRIGHT OWNER OR
//                CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
//                INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES.
//
// AUTHOR:        Brandon Simpson (brandon@asiimaging.com) 2/2022
//
// BASED ON:      ASIStage.cpp and others
//

#ifndef _ASIDACXYStage_H_
#define _ASIDACXYStage_H_

#include "ASIPeripheralBase.h"
#include "MMDevice.h"
#include "DeviceBase.h"

class CDACXYStage : public ASIPeripheralBase<CXYStageBase, CDACXYStage>
{
public:
	CDACXYStage(const char* name);
	~CDACXYStage() { }

	// Device API
	// ----------
	int Initialize();
	bool Busy() { return false; }

	// DAC API
	// ----------
	// Note: added axisLetter to deal with multiple axes
	int SetGateOpen(bool open, std::string axisLetter);
	int GetGateOpen(bool& open, std::string axisLetter);

	// XYStage API
	// -----------
	double GetStepSizeXUm() { return 1.0; }
	double GetStepSizeYUm() { return 1.0; }
	int GetPositionSteps(long& x, long& y);
	int SetPositionSteps(long x, long y);
	int SetRelativePositionSteps(long x, long y);
	int GetStepLimits(long& xMin, long& xMax, long& yMin, long& yMax);
	int SetOrigin();
	int SetXOrigin();
	int SetYOrigin();
	int Stop();
	int Home();
	int SetHome();
	int Move(double vx, double vy);

	int IsXYStageSequenceable(bool& isSequenceable) const { isSequenceable = ttl_trigger_enabled_; return DEVICE_OK; }
	int GetXYStageSequenceMaxLength(long& nrEvents) const { nrEvents = ring_buffer_capacity_; return DEVICE_OK; }

	int StartXYStageSequence();
	int StopXYStageSequence();
	int ClearXYStageSequence();
	int AddToXYStageSequence(double positionX, double positionY);
	int SendXYStageSequence();

	// not implemented yet
	int GetLimitsUm(double& /*xMin*/, double& /*xMax*/, double& /*yMin*/, double& /*yMax*/) { return DEVICE_UNSUPPORTED_COMMAND; }

	// action interface
	// ----------------
	int OnSaveCardSettings(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnRefreshProperties(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnConversionFactorX(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnConversionFactorY(MM::PropertyBase* pProp, MM::ActionType eAct);

	// joystick properties
	int OnJoystickFastSpeed(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnJoystickSlowSpeed(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnJoystickMirror(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnJoystickRotate(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnJoystickEnableDisable(MM::PropertyBase* pProp, MM::ActionType eAct);

	// ring buffer properties
	int OnRBDelayBetweenPoints(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnRBMode(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnRBTrigger(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnRBRunning(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnUseSequence(MM::PropertyBase* pProp, MM::ActionType eAct);

	// signal DAC properties
	int OnDACModeGeneric(MM::PropertyBase* pProp, MM::ActionType eAct, std::string axisLetter);
	int OnDACGateGeneric(MM::PropertyBase* pProp, MM::ActionType eAct, std::string axisLetter);
	int OnCutoffFreqGeneric(MM::PropertyBase* pProp, MM::ActionType eAct, std::string axisLetter);
	int OnDACModeX(MM::PropertyBase* pProp, MM::ActionType eAct) { return OnDACModeGeneric(pProp, eAct, axisLetterX_); }
	int OnDACModeY(MM::PropertyBase* pProp, MM::ActionType eAct) { return OnDACModeGeneric(pProp, eAct, axisLetterY_); }
	int OnDACGateX(MM::PropertyBase* pProp, MM::ActionType eAct) { return OnDACGateGeneric(pProp, eAct, axisLetterX_); }
	int OnDACGateY(MM::PropertyBase* pProp, MM::ActionType eAct) { return OnDACGateGeneric(pProp, eAct, axisLetterY_); }
	int OnCutoffFreqX(MM::PropertyBase* pProp, MM::ActionType eAct) { return OnCutoffFreqGeneric(pProp, eAct, axisLetterX_); }
	int OnCutoffFreqY(MM::PropertyBase* pProp, MM::ActionType eAct) { return OnCutoffFreqGeneric(pProp, eAct, axisLetterY_); }

	// single axis properties
	int OnSAAdvancedX(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnSAAdvancedY(MM::PropertyBase* pProp, MM::ActionType eAct);

	int OnSAAmplitudeGeneric(MM::PropertyBase* pProp, MM::ActionType eAct, std::string axisLetter);
	int OnSAOffsetGeneric(MM::PropertyBase* pProp, MM::ActionType eAct, std::string axisLetter);
	int OnSAPeriodGeneric(MM::PropertyBase* pProp, MM::ActionType eAct, std::string axisLetter);
	int OnSAModeGeneric(MM::PropertyBase* pProp, MM::ActionType eAct, std::string axisLetter);
	int OnSAPatternGeneric(MM::PropertyBase* pProp, MM::ActionType eAct, std::string axisLetter);
	int OnSAClkSrcGeneric(MM::PropertyBase* pProp, MM::ActionType eAct, std::string axisLetter);
	int OnSAClkPolGeneric(MM::PropertyBase* pProp, MM::ActionType eAct, std::string axisLetter);
	int OnSAPatternByteGeneric(MM::PropertyBase* pProp, MM::ActionType eAct, std::string axisLetter);
	int OnSATTLOutGeneric(MM::PropertyBase* pProp, MM::ActionType eAct, std::string axisLetter);
	int OnSATTLPolGeneric(MM::PropertyBase* pProp, MM::ActionType eAct, std::string axisLetter);

	int OnSAAmplitudeX(MM::PropertyBase* pProp, MM::ActionType eAct) { return OnSAAmplitudeGeneric(pProp, eAct, axisLetterX_); }
	int OnSAAmplitudeY(MM::PropertyBase* pProp, MM::ActionType eAct) { return OnSAAmplitudeGeneric(pProp, eAct, axisLetterY_); }
	int OnSAOffsetX(MM::PropertyBase* pProp, MM::ActionType eAct) { return OnSAOffsetGeneric(pProp, eAct, axisLetterX_); }
	int OnSAOffsetY(MM::PropertyBase* pProp, MM::ActionType eAct) { return OnSAOffsetGeneric(pProp, eAct, axisLetterY_); }
	int OnSAPeriodX(MM::PropertyBase* pProp, MM::ActionType eAct) { return OnSAPeriodGeneric(pProp, eAct, axisLetterX_); }
	int OnSAPeriodY(MM::PropertyBase* pProp, MM::ActionType eAct) { return OnSAPeriodGeneric(pProp, eAct, axisLetterY_); }
	int OnSAModeX(MM::PropertyBase* pProp, MM::ActionType eAct) { return OnSAModeGeneric(pProp, eAct, axisLetterX_); }
	int OnSAModeY(MM::PropertyBase* pProp, MM::ActionType eAct) { return OnSAModeGeneric(pProp, eAct, axisLetterY_); }
	int OnSAPatternX(MM::PropertyBase* pProp, MM::ActionType eAct) { return OnSAPatternGeneric(pProp, eAct, axisLetterX_); }
	int OnSAPatternY(MM::PropertyBase* pProp, MM::ActionType eAct) { return OnSAPatternGeneric(pProp, eAct, axisLetterY_); }
	int OnSAClkSrcX(MM::PropertyBase* pProp, MM::ActionType eAct) { return OnSAClkSrcGeneric(pProp, eAct, axisLetterX_); }
	int OnSAClkSrcY(MM::PropertyBase* pProp, MM::ActionType eAct) { return OnSAClkSrcGeneric(pProp, eAct, axisLetterY_); }
	int OnSAClkPolX(MM::PropertyBase* pProp, MM::ActionType eAct) { return OnSAClkPolGeneric(pProp, eAct, axisLetterX_); }
	int OnSAClkPolY(MM::PropertyBase* pProp, MM::ActionType eAct) { return OnSAClkPolGeneric(pProp, eAct, axisLetterY_); }
	int OnSAPatternByteX(MM::PropertyBase* pProp, MM::ActionType eAct) { return OnSAPatternByteGeneric(pProp, eAct, axisLetterX_); }
	int OnSAPatternByteY(MM::PropertyBase* pProp, MM::ActionType eAct) { return OnSAPatternByteGeneric(pProp, eAct, axisLetterY_); }
	int OnSATTLOutX(MM::PropertyBase* pProp, MM::ActionType eAct) { return OnSATTLOutGeneric(pProp, eAct, axisLetterX_); }
	int OnSATTLOutY(MM::PropertyBase* pProp, MM::ActionType eAct) { return OnSATTLOutGeneric(pProp, eAct, axisLetterY_); }
	int OnSATTLPolX(MM::PropertyBase* pProp, MM::ActionType eAct) { return OnSATTLPolGeneric(pProp, eAct, axisLetterX_); }
	int OnSATTLPolY(MM::PropertyBase* pProp, MM::ActionType eAct) { return OnSATTLPolGeneric(pProp, eAct, axisLetterY_); }

	// ttl
	int OnTTLin(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnTTLout(MM::PropertyBase* pProp, MM::ActionType eAct);

private:
	std::string axisLetterX_;
	std::string axisLetterY_;
	double maxvoltsX_; // X axis limits
	double minvoltsX_;
	double maxvoltsY_; // Y axis limits
	double minvoltsY_;
	bool ring_buffer_supported_;
	long ring_buffer_capacity_;
	bool ttl_trigger_supported_;
	bool ttl_trigger_enabled_;
	std::vector<double> sequenceX_;
	std::vector<double> sequenceY_;
	
	// microns to millivolts conversion factor
	long umToMvX_;
	long umToMvY_;

	// private helper functions
	int OnSaveJoystickSettings();

	// DAC helpers
	int GetMaxVolts(double& volts, std::string axisLetter);
	int GetMinVolts(double& volts, std::string axisLetter);
};

#endif // _ASIDACXYStage_H_
