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

class SkeletonDevice : public CGenericBase<SkeletonDevice>
{
public:
	SkeletonDevice();
	~SkeletonDevice();

	// MMDevice API
	int Initialize();
	int Shutdown();
	void GetName(char* name) const;
	bool Busy() { return false; };

private:
	// MM API
	bool initialized_;
};
