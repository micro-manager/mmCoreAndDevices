/*
 * Project: ASIStage Device Adapter
 * License/Copyright: BSD 3-clause, see license.txt
 * Maintainers: Brandon Simpson (brandon@asiimaging.com)
 *              Jon Daniels (jon@asiimaging.com)
 */

#ifndef _ASISTATEDEVICE_H_
#define _ASISTATEDEVICE_H_

#include "ASIBase.h"

class StateDevice : public CStateDeviceBase<StateDevice>, public ASIBase
{
public:
	StateDevice();
	~StateDevice();

	// MMDevice API
	bool Busy();
	void GetName(char* pszName) const;
	unsigned long GetNumberOfPositions() const { return numPos_; }

	int Initialize();
	int Shutdown();
	bool SupportsDeviceDetection(void);
	MM::DeviceDetectionStatus DetectDevice(void);

	int OnState(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnPort(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnNumPositions(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnAxis(MM::PropertyBase* pProp, MM::ActionType eAct);

private:
	long numPos_;
	std::string axis_;
	long position_;
	double answerTimeoutMs_;

	int UpdateCurrentPosition();
};

#endif // _ASISTATEDEVICE_H_
