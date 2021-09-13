///////////////////////////////////////////////////////////////////////////////
// FILE:          WieneckeSinske.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Wienecke & Sinske Stage Controller Driver
//                XY Stage
//             
//
// AUTHOR:        S3L GmbH, info@s3l.de, www.s3l.de,  11/21/2017
// COPYRIGHT:     S3L GmbH, Rosdorf, 2017
// LICENSE:       This library is free software; you can redistribute it and/or
//                modify it under the terms of the GNU Lesser General Public
//                License as published by the Free Software Foundation.
//                
//                You should have received a copy of the GNU Lesser General Public
//                License along with the source distribution; if not, write to
//                the Free Software Foundation, Inc., 59 Temple Place, Suite 330,
//                Boston, MA  02111-1307  USA
//
//                This file is distributed in the hope that it will be useful,
//                but WITHOUT ANY WARRANTY; without even the implied warranty
//                of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
//
//                IN NO EVENT SHALL THE COPYRIGHT OWNER OR
//                CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
//                INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES.  
//

#ifdef WIN32
#include <windows.h>
#else
#include <arpa/inet.h>
#endif
#include "FixSnprintf.h"

#include "WieneckeSinske.h"
#include <string>
#include <math.h>
#include "ModuleInterface.h"
#include "DeviceUtils.h"
#include "DeviceBase.h"
#include "ZPiezoCanDevice.h"
#include "ZPiezoWSDevice.h"
#include "XYStageDevice.h"

#include <sstream>


///////////////////////////////////////////////////////////////////////////////

using namespace std;


///////////////////////////////////////////////////////////////////////////////
// Devices in this adapter.  
// The device name needs to be a class name in this file

const char* g_XYStageDeviceDeviceName = "XYStage Piezo CAN Controller";
const char* g_ZPiezoCANDeviceName = "Z Piezo CAN Controller";
const char* g_ZPiezoWSDeviceName = "Z Piezo Controller";

///////////////////////////////////////////////////////////////////////////////
// Exported MMDevice API
///////////////////////////////////////////////////////////////////////////////

MODULE_API void InitializeModuleData()
{
	RegisterDevice(g_XYStageDeviceDeviceName, MM::XYStageDevice,  "Wienecke & Sinske WSB PiezoDrive CAN");
	RegisterDevice(g_ZPiezoCANDeviceName, MM::StageDevice,  "Wienecke & Sinske WSB ZPiezo CAN");
	RegisterDevice(g_ZPiezoWSDeviceName, MM::StageDevice,  "Wienecke & Sinske WSB ZPiezo");
}

MODULE_API MM::Device* CreateDevice(const char* deviceName)                  
{                                                                            
	if (deviceName == 0)                                                      
		return 0;

	if (strcmp(deviceName, g_XYStageDeviceDeviceName) == 0)
	{
		XYStageDevice* pXYStageDevice = new XYStageDevice();
		return pXYStageDevice;
	}
	else if (strcmp(deviceName, g_ZPiezoCANDeviceName) == 0)
	{
		ZPiezoCANDevice* pZPiezoCANDevice = new ZPiezoCANDevice();
		return pZPiezoCANDevice;
	}
	else if (strcmp(deviceName, g_ZPiezoWSDeviceName) == 0)
	{
		ZPiezoWSDevice* pZPiezoWSDevice = new ZPiezoWSDevice();
		return pZPiezoWSDevice;
	}

	return 0;
}

MODULE_API void DeleteDevice(MM::Device* pDevice)                            
{                                                                            
	delete pDevice;                                                           
}







