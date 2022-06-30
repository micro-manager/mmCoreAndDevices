///////////////////////////////////////////////////////////////////////////////
// FILE:          PM100x.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters - Thorlabs PM100x adapter
//-----------------------------------------------------------------------------
// DESCRIPTION:   This device adapter interfaces with Thorlabs light power meters
//
//                
// AUTHOR:        Nico Stuurman, Altos Labs, 2022
//
// COPYRIGHT:     Altos Labs Inc., 2022
// LICENSE:       This file is distributed under the BSD license.
//                License text is included with the source distribution.
//
//                This file is distributed in the hope that it will be useful,
//                but WITHOUT ANY WARRANTY; without even the implied warranty
//                of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
//
//                IN NO EVENT SHALL THE COPYRIGHT OWNER OR
//                CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
//                INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES.


#ifndef _PM100_H_
#define _PM100_H_

#include "MMDevice.h"
#include "DeviceBase.h"
#include "TLPM.h"

//////////////////////////////////////////////////////////////////////////////
// Error codes
//
#define ERR_UNKNOWN_MODE         102
#define ERR_UNKNOWN_POSITION     103
#define ERR_NO_PM_CONNECTED      104
#define ERR_PM_NOT_CONNECTED     105

class PM100 : public CGenericBase<PM100>
{
public:
	PM100();
	~PM100();

	// MMDevice API
	// ------------
	int Initialize();
	int Shutdown();

	void GetName(char* pszName) const;
	bool Busy();

	// action interface
	// ----------------
	int OnValue(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnRawValue(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnRawUnit(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnPMName(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnWavelength (MM::PropertyBase* pProp, MM::ActionType eAct);

	int OnAutoRange(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnPowerRange(MM::PropertyBase* pProp, MM::ActionType eAct);
private:
	int FindPowerMeters(std::vector<std::string>& deviceNames);

	ViSession   instrHdl_;
	bool initialized_;
	MM::MMTime changedTime_;
	std::string deviceName_;
};

#endif _PM100_H_