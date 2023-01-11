/*
 * Project: Light Sheet Device Manager
 * License: BSD 3-clause, see license.txt
 * Author: Brandon Simpson (brandon@asiimaging.com)
 * Copyright (c) 2022, Applied Scientific Instrumentation
 */

#ifndef _LIGHTSHEET_MANAGER_H_
#define _LIGHTSHEET_MANAGER_H_

#include "MMDevice.h"
#include "DeviceBase.h"
#include "LightSheetDeviceManager.h"

// version number
const char* const gVersionNumber = "0.1.0";
const char* const gVersionNumberPropertyName = "Version";

// device name and description
const char* const gDeviceName = "LightSheetDeviceManager";
const char* const gDeviceDescription = "Maps logical devices to physical devices for the LightSheetManager plugin.";

// properties
const char* const gImagingCameraPropertyName = "ImagingCamera";
const char* const gIllumBeamPropertyName = "IllumBeam";

// property prefixes
const std::string gIllumPrefix = "Illum";
const std::string gImagingPrefix = "Imaging";

// pre-init properties
const char* const gMicroscopeGeometry = "MicroscopeGeometry";
const char* const gSimultaneousCameras = "SimultaneousCameras";
const char* const gIlluminationPaths = "IlluminationPaths";
const char* const gLightSheetType = "LightSheetType";
const char* const gImagingPaths = "ImagingPaths";

// types of light sheets
const char* const gLightSheetTypeStatic = "Static";
const char* const gLightSheetTypeScanned = "Scanned";

// no device selected
const char* const gUndefined = "Undefined";

#endif // _LIGHTSHEET_MANAGER_H_
