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

private:
	// MM API
	bool initialized_;
};
