#pragma once

#include "OxxiusCombinerHub.h"

#include "DeviceBase.h"
#include <cstdlib>
#include <string>
#include <map>

//////////////////////////////////////////////////////////////////////////////
//
// Device adaptaters for "shutter" source in Combiner
//
//////////////////////////////////////////////////////////////////////////////

class OxxiusShutter : public CShutterBase<OxxiusShutter>
{
public:
	OxxiusShutter(const char* nameAndChannel);
	~OxxiusShutter();

	// MMDevice API
	// ------------
	int Initialize();
	int Shutdown();

	void GetName(char* pszName) const;
	bool Busy();

	// Shutter API
	int SetOpen(bool openCommand = true);
	int GetOpen(bool& isOpen);
	int Fire(double /*deltaT*/) { return DEVICE_UNSUPPORTED_COMMAND; }

	// Action Interface
	// ----------------
	int OnState(MM::PropertyBase* pProp, MM::ActionType eAct);

private:
	bool initialized_;
	std::string name_;

	OxxiusCombinerHub* parentHub_;
	bool isOpen_;
	unsigned int channel_;

	MM::MMTime changedTime_;
};
