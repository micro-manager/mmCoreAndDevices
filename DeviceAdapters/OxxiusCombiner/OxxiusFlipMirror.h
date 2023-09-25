#pragma once

#include "OxxiusCombinerHub.h"

#include "../../MMDevice/MMDevice.h"
#include "../../MMDevice/DeviceBase.h"
#include "../../MMDevice/ModuleInterface.h"
#include <cstdlib>
#include <string>
#include <map>

class OxxiusFlipMirror : public CStateDeviceBase<OxxiusFlipMirror>
{
public:
	OxxiusFlipMirror(const char* name);
	~OxxiusFlipMirror();

	// MMDevice API
	// ------------
	int Initialize();
	int Shutdown();

	void GetName(char* pszName) const;
	bool Busy();
	unsigned long GetNumberOfPositions() const { return 2; };
	
	// Action Interface
	// ----------------
	int OnSwitchPos(MM::PropertyBase* pProp, MM::ActionType eAct);


private:
	bool initialized_;
	std::string nameF_;
	unsigned int slot_;
	MM::Core* core_;

	OxxiusCombinerHub* parentHub_;

	unsigned long numPos_;
};
