#pragma once
///////////////////////////////////////////////////////////////////////////////
// FILE:          ObjectiveChanger.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Device adapter for Zaber's X-MOR objective changer.
//
// AUTHOR:        Martin Zak (contact@zaber.com)

// COPYRIGHT:     Zaber Technologies, 2023

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

#include "Zaber.h"

extern const char* g_ObjectiveChangerName;
extern const char* g_ObjectiveChangerDescription;

namespace ObjectiveChanger_ {
	const double xLdaNativePerMm = 1000000.0;
}

class ObjectiveChanger : public CStateDeviceBase<ObjectiveChanger>, public ZaberBase
{
public:
	ObjectiveChanger();
	~ObjectiveChanger();

	// Device API
	// ----------
	int Initialize();
	int Shutdown();
	void GetName(char* name) const;
	bool Busy();

	// Stage API
	// ---------
	unsigned long GetNumberOfPositions() const
	{
		return numPositions_;
	}

	// Base class overrides
	// ----------------
	virtual int GetPositionLabel(long pos, char* label) const;

	// ZaverBase class overrides
	// ----------------
	virtual void onNewConnection();

	// Properties
	// ----------------
	int DelayGetSet(MM::PropertyBase* pProp, MM::ActionType eAct);
	int PortGetSet(MM::PropertyBase* pProp, MM::ActionType eAct);
	int XMorAddressGetSet(MM::PropertyBase* pProp, MM::ActionType eAct);
	int XLdaAddressGetSet(MM::PropertyBase* pProp, MM::ActionType eAct);
	int PositionGetSet(MM::PropertyBase* pProp, MM::ActionType eAct);
	int FocusOffsetGetSet(MM::PropertyBase* pProp, MM::ActionType eAct);

	int setObjective(long objective, bool applyOffset);
private:
	long xMorAddress_;
	long xLdaAddress_;
	long numPositions_;
	double focusOffset_;
	long currentObjective_;
	MM::MMTime changedTime_;

	zmlmi::ObjectiveChanger changer_;
};
