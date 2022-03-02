/*
 * Project: ASIStage Device Adapter
 * License/Copyright: BSD 3-clause, see license.txt
 * Maintainers: Brandon Simpson (brandon@asiimaging.com)
 *              Jon Daniels (jon@asiimaging.com)
 */

#ifndef _ASILED_H_
#define _ASILED_H_

#include "ASIBase.h"

class LED : public CShutterBase<LED>, public ASIBase
{
public:
	LED();
	~LED();

	int Initialize();
	int Shutdown();

	bool SupportsDeviceDetection(void);
	MM::DeviceDetectionStatus DetectDevice(void);

	void GetName(char* pszName) const;
	bool Busy();

	// Shutter API
	int SetOpen(bool open = true);
	int GetOpen(bool& open);
	int Fire(double deltaT);

	// action interface
	// int OnState(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnIntensity(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnPort(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnChannel(MM::PropertyBase* pProp, MM::ActionType eAct);

private:
	int IsOpen(bool* open); // queries the device rather than using a cached value
	int CurrentIntensity(long* intensity); // queries the device rather than using a cached value
	bool open_;
	long intensity_;
	std::string name_;
	int answerTimeoutMs_;
	long channel_;
	char channelAxisChar_;
	bool hasDLED_;
};

#endif // _ASILED_H_
