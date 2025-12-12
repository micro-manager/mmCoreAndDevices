///////////////////////////////////////////////////////////////////////////////
// FILE:          AMF_Hub.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Hub for AMF Devices
//                 
// AUTHOR:        Lars Kool, Institut Pierre-Gilles de Gennes, Paris, France
//
// YEAR:          2025
//                
// VERSION:       0.1
//
// LICENSE:       This file is distributed under the BSD license.
//                License text is included with the source distribution.
//
//                This file is distributed in the hope that it will be useful,
//                but WITHOUT ANY WARRANTY; without even the implied warranty
//                of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
//
//                IN NO EVENT SHALL THE COPYRIGHT OWNER OR
//                CONTRIBUTORS BE   LIABLE FOR ANY DIRECT, INDIRECT,
//                INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES.
//
//LAST UPDATE:    04.04.2025 LK

#ifndef _AMF_HUB_H_
#define _AMF_HUB_H_

#include "DeviceBase.h"

///////////////////////////////////////////////////////////////////////////////
// AMF Hub API
///////////////////////////////////////////////////////////////////////////////

class AMF_Hub : public HubBase<AMF_Hub> {
public:
	AMF_Hub();
	~AMF_Hub() {}

	// Device API
	int Initialize();
	int Shutdown() { return DEVICE_OK;  }
	void GetName(char* pName) const;
	bool Busy() { return false; }

	// Hub API
	int DetectInstalledDevices();

	// Action Handlers
	int OnRVMCount(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnLSPCount(MM::PropertyBase* pProp, MM::ActionType eAct);

private:
	bool initialized_ = false;
	long nRVM_ = 0;
	long nLSP_ = 0;
};

#endif // _AMF_HUB_H_