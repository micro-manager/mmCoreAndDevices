/*
 * Project: ASIStage Device Adapter
 * License/Copyright: BSD 3-clause, see license.txt
 * Maintainers: Brandon Simpson (brandon@asiimaging.com)
 *              Jon Daniels (jon@asiimaging.com)
 */

#ifndef _ASIXYSTAGE_H_
#define _ASIXYSTAGE_H_

#include "ASIBase.h"

class XYStage : public CXYStageBase<XYStage>, public ASIBase
{
public:
	XYStage();
	~XYStage();

	// Device API
	// ----------
	int Initialize();
	int Shutdown();

	void GetName(char* pszName) const;
	bool Busy();

	// so far, only the XYStage attempts to get the controller status on initialization, so
	// that's where the device detection is going for now
	bool SupportsDeviceDetection(void);
	MM::DeviceDetectionStatus DetectDevice(void);

	// XYStage API
	// -----------
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
	// ----------------
	int OnPort(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnStepSizeX(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnStepSizeY(MM::PropertyBase* pProp, MM::ActionType eAct);

private:
	int OnAcceleration(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnBacklash(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnFinishError(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnError(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnOverShoot(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnWait(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnSpeed(MM::PropertyBase* pProp, MM::ActionType eAct);
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
	bool hasCommand(std::string command);
	void Wait();
	static std::string EscapeControlCharacters(const std::string v);
	static std::string UnescapeControlCharacters(const std::string v0);
	int OnVectorGeneric(MM::PropertyBase* pProp, MM::ActionType eAct, std::string axisLetter);
	int OnVectorX(MM::PropertyBase* pProp, MM::ActionType eAct) { return OnVectorGeneric(pProp, eAct, axisletterX_); }
	int OnVectorY(MM::PropertyBase* pProp, MM::ActionType eAct) { return OnVectorGeneric(pProp, eAct, axisletterY_); }


	double stepSizeXUm_;
	double stepSizeYUm_;
	double maxSpeed_;
	double ASISerialUnit_; // this variable converts the floating point number provided by ASI (in 10ths of microns) into a long
	bool motorOn_;
	int joyStickSpeedFast_;
	int joyStickSpeedSlow_;
	bool joyStickMirror_;
	long nrMoveRepetitions_;
	double answerTimeoutMs_;
	bool stopSignal_;
	bool serialOnlySendChanged_; // if true the serial command is only sent when it has changed
	std::string manualSerialAnswer_; // last answer received when the SerialCommand property was used
	bool advancedPropsEnabled_;
	std::string axisletterX_;
	std::string axisletterY_;
};

#endif // _ASIXYSTAGE_H_
