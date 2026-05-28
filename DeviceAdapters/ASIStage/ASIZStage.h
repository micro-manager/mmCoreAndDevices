/*
 * Project: ASIStage Device Adapter
 * License/Copyright: BSD 3-clause, see license.txt
 * Maintainers: Brandon Simpson (brandon@asiimaging.com)
 *              Jon Daniels (jon@asiimaging.com)
 */

#pragma once

#include "ASIBase.h"

class ZStage : public CStageBase<ZStage>, public ASIBase
{
public:
	ZStage();
	~ZStage();

	// Device API
	int Initialize();
	int Shutdown();

	void GetName(char* name) const;
	bool Busy();
	bool SupportsDeviceDetection();
	MM::DeviceDetectionStatus DetectDevice();

	// Stage API
	int SetPositionUm(double pos);
	int GetPositionUm(double& pos);
	int SetRelativePositionUm(double d);
	int SetPositionSteps(long steps);
	int GetPositionSteps(long& steps);
	int SetOrigin();
	int Calibrate();
	int GetLimits(double& min, double& max);

	bool IsContinuousFocusDrive() const { return false; }

	// action interface
	int OnPort(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnAxis(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnSequence(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnFastSequence(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnRingBufferSize(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnLinearSequenceTimeout(MM::PropertyBase* pProp, MM::ActionType eAct);

	// Sequence functions
	int IsStageSequenceable(bool& isSequenceable) const { isSequenceable = sequenceable_; return DEVICE_OK; }
	int GetStageSequenceMaxLength(long& nrEvents) const { nrEvents = nrEvents_; return DEVICE_OK; }
	int StartStageSequence();
	int StopStageSequence();
	int ClearStageSequence();
	int AddToStageSequence(double position);
	int SendStageSequence();

	// Linear sequence
	int IsStageLinearSequenceable(bool& isSequenceable) const
	{
		isSequenceable = sequenceable_ && supportsLinearSequence_; return DEVICE_OK;
	}
	int SetStageLinearSequence(double dZ_um, long nSlices);

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
	int GetWait(long& waitCycles);
	int OnSpeed(MM::PropertyBase* pProp, MM::ActionType eAct);
	int GetSpeed(double& speed);
	int GetMaxSpeed(char* maxSpeedStr);
	int OnMotorCtrl(MM::PropertyBase* pProp, MM::ActionType eAct);
	bool HasRingBuffer() const;
	int GetControllerInfo();
	bool HasCommand(const std::string& command);
	int OnVector(MM::PropertyBase* pProp, MM::ActionType eAct);

	std::vector<double> sequence_{};
	std::string axis_ = "Z";
	unsigned int axisNr_ = 4;
	double stepSizeUm_ = 0.1;
	double answerTimeoutMs_ = 1000;
	bool sequenceable_ = false;
	bool runningFastSequence_ = false;
	bool hasRingBuffer_ = false;
	long nrEvents_ = 0;
	long curSteps_ = 0;
	double maxSpeed_ = 7.5;
	bool motorOn_ = true;
	bool supportsLinearSequence_ = false;
	double linearSequenceIntervalUm_ = 0.0;
	long linearSequenceLength_ = 0;
	long linearSequenceTimeoutMs_ = 10000;
	// cached properties
	double speed_ = 0;
	long waitCycles_ = 0;
	double backlash_ = 0;
	double error_ = 0;
	long acceleration_ = 0;
	double finishError_ = 0;
	double overShoot_ = 0;
};
