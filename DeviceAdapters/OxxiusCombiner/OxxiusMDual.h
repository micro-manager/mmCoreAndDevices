#pragma once

#include "OxxiusCombinerHub.h"

#include "DeviceBase.h"
#include <cstdlib>
#include <string>
#include <map>

class OxxiusMDual : public CGenericBase<OxxiusMDual>
{
public:
	OxxiusMDual(const char* name);
	~OxxiusMDual();

	// MMDevice API
	// ------------
	int Initialize();
	int Shutdown();

	void GetName(char* pszName) const;
	bool Busy();

	// Action Interface
	// ----------------
	int OnSetRatio(MM::PropertyBase* pProp, MM::ActionType eAct);

private:
	bool initialized_;
	std::string name_;
	std::string slot_;
	MM::Core* core_;

	OxxiusCombinerHub* parentHub_;
};
