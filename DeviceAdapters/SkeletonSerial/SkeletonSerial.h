///////////////////////////////////////////////////////////////////////////////
// FILE:          SkeletonDevice.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Skeleton adapter for a general device
//                
// AUTHOR:        Kyle M. Douglass, https://kylemdouglass.com
//
// VERSION:       0.0.0
//
// FIRMWARE:      xxx
//                
// COPYRIGHT:     ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland
//                Laboratory of Experimental Biophysics (LEB), 2025
//

#pragma once

#include "DeviceBase.h"

class SkeletonSerial : public CGenericBase<SkeletonSerial>
{
	//	Unofficial device adapter for lasers from MPB Communications Inc.

public:
	SkeletonSerial();
	~SkeletonSerial();

	// MMDevice API
	int Initialize();
	int Shutdown();
	void GetName(char* name) const;
	bool Busy() { return false; };

	// Pre-init action handlers
	int OnPort(MM::PropertyBase* pProp, MM::ActionType eAct);

	// Action handlers
	int OnMessageChange(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnResponseChange(MM::PropertyBase* pProp, MM::ActionType eAct);

private:
	// MM API
	bool initialized_;

	// Pre-init device properties
	std::string port_;

	// MM Properties
	int GeneratePreInitProperties();
	int GenerateReadOnlyProperties();
	int GenerateControlledProperties();

	// Serial communications
	std::string buffer_;
	std::string msg_;
	const std::string CMD_TERM_ = "\n";
	const std::string ANS_TERM_ = "\r\n";

	int PurgeBuffer();
	int QueryDevice(std::string msg);
	int ReceiveMsg();
	int SendMsg(std::string msg);

	// Error codes
	const int ERR_PORT_CHANGE_FORBIDDEN = 101;
};
