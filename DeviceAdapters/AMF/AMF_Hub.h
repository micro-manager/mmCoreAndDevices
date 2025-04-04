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

const char* g_AMF_Baud = "9600";
const char* g_AMF_Parity = "None";
const char* g_AMF_StopBits = "1";
const char* g_AMF_EOL = "\r";

const char* g_AMF_Hub_Name = "AMF Hub";
const char* g_AMF_RVM_Name = "AMF RVM";
const char* g_AMF_LSP_Name = "AMF LSP";
const char* g_AMF_Test_Name = "AMF Test";

const char* AMF_START = "/";
const char* AMF_END = "R";
const char* AMF_TERM = "\r";
const char AMF_ACK = 0;
const char AMF_NACK = 1;

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