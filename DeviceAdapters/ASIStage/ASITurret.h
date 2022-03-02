/*
 * Project: ASIStage Device Adapter
 * License/Copyright: BSD 3-clause, see license.txt
 * Maintainers: Brandon Simpson (brandon@asiimaging.com)
 *              Jon Daniels (jon@asiimaging.com)
 */

#ifndef _ASITURRET_H_
#define _ASITURRET_H_

#include "ASIBase.h"

class AZ100Turret : public CStateDeviceBase<AZ100Turret>, public ASIBase
{
public:
	AZ100Turret();
	~AZ100Turret();

	// MMDevice API
	bool Busy();
	void GetName(char* pszName) const;
	unsigned long GetNumberOfPositions() const { return numPos_; }

	int Initialize();
	int Shutdown();

	int OnState(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnPort(MM::PropertyBase* pProp, MM::ActionType eAct);

private:
	long numPos_;
	MM::MMTime changedTime_;
	long position_;
};

#endif // _ASITURRET_H_
