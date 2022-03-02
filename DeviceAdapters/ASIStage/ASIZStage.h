/*
 * Project: ASIStage Device Adapter
 * License/Copyright: BSD 3-clause, see license.txt
 * Maintainers: Brandon Simpson (brandon@asiimaging.com)
 *              Jon Daniels (jon@asiimaging.com)
 */

#ifndef _ASIZSTAGE_H_
#define _ASIZSTAGE_H_

#include "ASIBase.h"

class ZStage : public CStageBase<ZStage>, public ASIBase
{
public:
	ZStage();
	~ZStage();

	// Device API
	// ----------
	int Initialize();
	int Shutdown();

	void GetName(char* name) const;
	bool Busy();
	bool SupportsDeviceDetection(void);
	MM::DeviceDetectionStatus DetectDevice(void);

	// Stage API
	// ---------
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
	// ----------------
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
	int OnBacklash(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnFinishError(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnError(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnOverShoot(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnWait(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnSpeed(MM::PropertyBase* pProp, MM::ActionType eAct);
	int GetMaxSpeed(char* maxSpeedStr);
	int OnMotorCtrl(MM::PropertyBase* pProp, MM::ActionType eAct);
	bool HasRingBuffer();
	int GetControllerInfo();
	//int ExecuteCommand(const std::string& cmd, std::string& response);
	//int Autofocus(long param);
	//int GetResolution(double& res);
	bool hasCommand(std::string commnand);
	int OnVector(MM::PropertyBase* pProp, MM::ActionType eAct);

	std::vector<double> sequence_;
	std::string axis_;
	unsigned int axisNr_;
	double stepSizeUm_;
	double answerTimeoutMs_;
	bool sequenceable_;
	bool runningFastSequence_;
	bool hasRingBuffer_;
	long nrEvents_;
	long curSteps_;
	double maxSpeed_;
	bool motorOn_;
	bool supportsLinearSequence_;
	double linearSequenceIntervalUm_;
	long linearSequenceLength_;
	long linearSequenceTimeoutMs_;
};

#endif // _ASIZSTAGE_H_
