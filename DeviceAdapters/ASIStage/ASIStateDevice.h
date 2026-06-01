/*
 * Project: ASIStage Device Adapter
 * License/Copyright: BSD 3-clause, see license.txt
 * Maintainers: Brandon Simpson (brandon@asiimaging.com)
 *              Jon Daniels (jon@asiimaging.com)
 */

#pragma once

#include "ASIBase.h"

class StateDevice : public CStateDeviceBase<StateDevice>, public ASIBase {
public:
	StateDevice();
	~StateDevice();

	// MMDevice API
	bool Busy();
	void GetName(char* name) const;
	unsigned long GetNumberOfPositions() const { return numPos_; }

	int Initialize();
	int Shutdown();
	bool SupportsDeviceDetection();
	MM::DeviceDetectionStatus DetectDevice();

	int OnState(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnPort(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnNumPositions(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnAxis(MM::PropertyBase* pProp, MM::ActionType eAct);

private:
	int UpdateCurrentPosition();

	std::string axis_ = "F";
	long numPos_ = 4;
	long position_ = 0;
	double answerTimeoutMs_ = 1000;
};
