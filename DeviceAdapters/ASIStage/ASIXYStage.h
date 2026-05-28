/*
 * Project: ASIStage Device Adapter
 * License/Copyright: BSD 3-clause, see license.txt
 * Maintainers: Brandon Simpson (brandon@asiimaging.com)
 *              Jon Daniels (jon@asiimaging.com)
 */

#pragma once

#include "ASIBase.h"

class XYStage : public CXYStageBase<XYStage>, public ASIBase
{
public:
	XYStage();
	~XYStage();

	// Device API
	int Initialize();
	int Shutdown();

	void GetName(char* name) const;
	bool Busy();

	// so far, only the XYStage attempts to get the controller status on initialization, so
	// that's where the device detection is going for now
	bool SupportsDeviceDetection();
	MM::DeviceDetectionStatus DetectDevice();

	// XYStage API
	int SetPositionSteps(long x, long y);
	int SetRelativePositionSteps(long x, long y);
	int GetPositionSteps(long& x, long& y);
	int Home();
	int Stop();
	int SetOrigin();
	int Calibrate();
	int Calibrate1();
	int GetLimitsUm(double& xMin, double& xMax, double& yMin, double& yMax);
	int GetStepLimits(long& xMin, long& xMax, long& yMin, long& yMax);
	double GetStepSizeXUm() { return stepSizeXUm_; }
	double GetStepSizeYUm() { return stepSizeYUm_; }
	int IsXYStageSequenceable(bool& isSequenceable) const { isSequenceable = false; return DEVICE_OK; }

	// action interface
	int OnPort(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnStepSizeX(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnStepSizeY(MM::PropertyBase* pProp, MM::ActionType eAct);

private:
	int OnAcceleration(MM::PropertyBase* pProp, MM::ActionType eAct);
	int GetAcceleration(long& acceleration);
	int OnBacklash(MM::PropertyBase* pProp, MM::ActionType eAct);
	int GetBacklash(double& backlash);
	int OnFinishError(MM::PropertyBase* pProp, MM::ActionType eAct);
	int GetFinishError(double& finishError);
	int OnError(MM::PropertyBase* pProp, MM::ActionType eAct);
	int GetError(double& error);
	int OnOverShoot(MM::PropertyBase* pProp, MM::ActionType eAct);
	int GetOverShoot(double& overShoot);
	int OnWait(MM::PropertyBase* pProp, MM::ActionType eAct);
	int GetWaitCycles(long& waitCycles);
	int OnSpeed(MM::PropertyBase* pProp, MM::ActionType eAct);
	int GetSpeed(double& speed_);
	int GetMaxSpeed(char* maxSpeedStr);
	int OnMotorCtrl(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnNrMoveRepetitions(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnJSMirror(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnJSSwapXY(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnJSFastSpeed(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnJSSlowSpeed(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnSerialCommand(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnSerialResponse(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnSerialCommandOnlySendChanged(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnAdvancedProperties(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnKIntegral(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnKProportional(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnKDerivative(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnAAlign(MM::PropertyBase* pProp, MM::ActionType eAct);
	int GetPositionStepsSingle(char axis, long& steps);
	int SetAxisDirection();
	bool HasCommand(const std::string& command);
	void Wait();
	static std::string EscapeControlCharacters(const std::string& v);
	static std::string UnescapeControlCharacters(const std::string& v0);
	int OnVectorGeneric(MM::PropertyBase* pProp, MM::ActionType eAct, const std::string& axisLetter);
	int OnVectorX(MM::PropertyBase* pProp, MM::ActionType eAct) { return OnVectorGeneric(pProp, eAct, axisletterX_); }
	int OnVectorY(MM::PropertyBase* pProp, MM::ActionType eAct) { return OnVectorGeneric(pProp, eAct, axisletterY_); }


	std::string axisletterX_ = "X";
	std::string axisletterY_ = "Y";
	double stepSizeXUm_= 0.0;
	double stepSizeYUm_ = 0.0;
	double maxSpeed_ = 7.5;
	double ASISerialUnit_ = 10.0; // this variable converts the floating point number provided by ASI (in 10ths of microns) into a long
	bool motorOn_ = true;
	int joyStickSpeedFast_ = 60;
	int joyStickSpeedSlow_ = 5;
	bool joyStickMirror_ = false;
	long nrMoveRepetitions_ = 0;
	double answerTimeoutMs_ = 1000;
	bool stopSignal_ = false;
	bool serialOnlySendChanged_ = true; // if true the serial command is only sent when it has changed
	std::string manualSerialAnswer_{}; // last answer received when the SerialCommand property was used
	bool advancedPropsEnabled_ = false;
	// cached properties
	long acceleration_ = 0;
	long waitCycles_ = 0;
	double speed_ = 0;
	double backlash_ = 0;
	double error_ = 0;
	double finishError_ = 0;
	double overShoot_ = 0;
};
