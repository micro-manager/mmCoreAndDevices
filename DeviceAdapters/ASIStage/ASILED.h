/*
 * Project: ASIStage Device Adapter
 * License/Copyright: BSD 3-clause, see license.txt
 * Maintainers: Brandon Simpson (brandon@asiimaging.com)
 *              Jon Daniels (jon@asiimaging.com)
 */

#pragma once

#include "ASIBase.h"

class LED : public CShutterBase<LED>, public ASIBase {
public:
	LED();
	~LED();

	int Initialize();
	int Shutdown();

	bool SupportsDeviceDetection();
	MM::DeviceDetectionStatus DetectDevice();

	void GetName(char* name) const;
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

	std::string name_ = "LED";
	long channel_ = 0;
	char channelAxisChar_ = 'X';

	bool open_ = false;
	bool hasDLED_ = false;

	long intensity_ = 20;
	int answerTimeoutMs_ = 1000;
};
