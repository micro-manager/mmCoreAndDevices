///////////////////////////////////////////////////////////////////////////////
// FILE:          NIDAQWaveforms.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Generates analog waveforms on a NI-DAQ.
//                
// AUTHOR:        Kyle M. Douglass, https://kylemdouglass.com
//
// VERSION:       0.0.0
//
// FIRMWARE:      xxx
//                
// COPYRIGHT:     ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland
//                Laboratory of Experimental Biophysics (LEB), 2026
//

#pragma once

#include "DeviceBase.h"

#include <memory>
#include <string>
#include <vector>

class IDAQDevice;

class NIDAQWaveforms : public CGenericBase<NIDAQWaveforms>
{
public:
	NIDAQWaveforms();
	~NIDAQWaveforms();

	// MMDevice API
	int Initialize();
	int Shutdown();
	void GetName(char* name) const;
	bool Busy() { return false; };

	// Action handlers
	int OnDevice(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnChannelEnabled(MM::PropertyBase* pProp, MM::ActionType eAct);

private:
	bool initialized_;
	std::string deviceName_;
	std::vector<std::string> availableChannels_;
	std::vector<std::string> enabledChannels_;
	std::unique_ptr<IDAQDevice> daq_;
};
