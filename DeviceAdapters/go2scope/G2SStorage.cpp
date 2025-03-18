///////////////////////////////////////////////////////////////////////////////
// FILE:          G2SStorage.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Go2Scope devices. Includes the experimental StorageDevice
//
// AUTHOR:        Nenad Amodaj <nenad@go2scope.com>
//						Milos Jovanovic <milos@tehnocad.rs>
//
// COPYRIGHT:     Nenad Amodaj, Chan Zuckerberg Initiative, 2024
//
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
//
// NOTE:          Storage Device development is supported in part by
//                Chan Zuckerberg Initiative (CZI)
// 
///////////////////////////////////////////////////////////////////////////////
#include "G2SStorage.h"
#include "ModuleInterface.h"
#include "G2SBigTiffStorage.h"
#include "AcqZarrStorage.h"


///////////////////////////////////////////////////////////////////////////////
// Exported MMDevice API
///////////////////////////////////////////////////////////////////////////////
MODULE_API void InitializeModuleData()
{
   RegisterDevice(g_AcqZarrStorage, MM::StorageDevice, "Zarr storage based on Acquire");
	RegisterDevice(g_BigTiffStorage, MM::StorageDevice, "BigTIFF storage based on Go2Scope");
}

MODULE_API MM::Device* CreateDevice(const char* deviceName)
{
   if(deviceName == 0)
      return 0;

   if(strcmp(deviceName, g_AcqZarrStorage) == 0)
		return new AcqZarrStorage();
	if(strcmp(deviceName, g_BigTiffStorage) == 0)
		return new G2SBigTiffStorage();

   return 0;
}

MODULE_API void DeleteDevice(MM::Device* pDevice)
{
   delete pDevice;
}
