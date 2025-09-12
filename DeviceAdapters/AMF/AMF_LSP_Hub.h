///////////////////////////////////////////////////////////////////////////////
// FILE:          AMF_LSP.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Hub for AMF LSP pump.
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
//LAST UPDATE:    09.09.2025 LK

#ifndef _AMF_LSP_HUB_H_
#define _AMF_LSP_HUB_H_

#include "DeviceBase.h"

///////////////////////////////////////////////////////////////////////////////
// AMF LSP Hub
///////////////////////////////////////////////////////////////////////////////

class AMF_LSP_Hub : public HubBase<AMF_LSP_Hub> {
public:
	AMF_LSP_Hub();
	~AMF_LSP_Hub() {}

	// Device API
	int Initialize();
	int Shutdown() { return DEVICE_OK;  }
	void GetName(char* pName) const;
	bool Busy() { return false; }

	// Hub API
	int DetectInstalledDevices();

	// Action Handlers
	int OnPort(MM::PropertyBase* pAct, MM::ActionType eAct);

	// AMF LSP API
	int GetPort(std::string& port);
	
private:
	bool initialized_ = false;
	std::string port_;
};


#endif // _AMF_LSP_HUB_H_
