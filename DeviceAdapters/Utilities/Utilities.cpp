///////////////////////////////////////////////////////////////////////////////
// FILE:          Utilities.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Various 'Meta-Devices' that add to or combine functionality of 
//                physcial devices.
//
// AUTHOR:        Nico Stuurman, nico@cmp.ucsf.edu, 11/07/2008
//                DAXYStage by Ed Simmon, 11/28/2011
//                Nico Stuurman, nstuurman@altoslabs.com, 4/22/2022
// COPYRIGHT:     University of California, San Francisco, 2008
//                2015-2016, Open Imaging, Inc.
//                Altos Labs, 2022
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

#ifdef _WIN32
// Prevent windows.h from defining min and max macros,
// which clash with std::min and std::max.
#define NOMINMAX
#endif

#include "Utilities.h"

#include "ModuleInterface.h"

#include <algorithm>


const char* g_Undefined = "Undefined";
const char* g_NoDevice = "None";
const char* g_DeviceNameMultiShutter = "Multi Shutter";
const char* g_DeviceNameComboXYStage = "Combo XY Stage";
const char* g_DeviceNameMultiCamera = "Multi Camera";
const char* g_DeviceNameMultiStage = "Multi Stage";
const char* g_DeviceNameSingleAxisStage = "Single Axis Stage";
const char* g_DeviceNameDAShutter = "DA Shutter";
const char* g_DeviceNameDAMonochromator = "DA Monochromator";
const char* g_DeviceNameDAZStage = "DA Z Stage";
const char* g_DeviceNameDAXYStage = "DA XY Stage";
const char* g_DeviceNamePropertyShutter = "Property Shutter";
const char* g_DeviceNameDATTLStateDevice = "DA TTL State Device";
const char* g_DeviceNameDAGalvoDevice = "DA Galvo";
const char* g_DeviceNameMultiDAStateDevice = "Multi DA State Device";
const char* g_DeviceNameAutoFocusStage = "AutoFocus Stage";
const char* g_DeviceNameStateDeviceShutter = "State Device Shutter";
const char* g_DeviceNameSerialDTRShutter = "Serial port DTR Shutter";

const char* g_PropertyMinUm = "Stage Low Position(um)";
const char* g_PropertyMaxUm = "Stage High Position(um)";
const char* g_SyncNow = "Sync positions now";

const char* g_normalLogicString = "Normal";
const char* g_invertedLogicString = "Inverted";
const char* g_InvertLogic = "Invert Logic";
const char* g_TTLVoltage = "TTL Voltage";
const char* g_3_3 = "3.3";
const char* g_5_0 = "5.0";



///////////////////////////////////////////////////////////////////////////////
// Exported MMDevice API
///////////////////////////////////////////////////////////////////////////////
MODULE_API void InitializeModuleData()
{
   RegisterDevice(g_DeviceNameMultiShutter, MM::ShutterDevice, "Combine multiple physical shutters into a single logical shutter");
   RegisterDevice(g_DeviceNameMultiCamera, MM::CameraDevice, "Combine multiple physical cameras into a single logical camera");
   RegisterDevice(g_DeviceNameMultiStage, MM::StageDevice, "Combine multiple physical 1D stages into a single logical 1D stage");
   RegisterDevice(g_DeviceNameComboXYStage, MM::XYStageDevice, "Combine two single-axis stages into an XY stage");
   RegisterDevice(g_DeviceNameSingleAxisStage, MM::StageDevice, "Use single axis of XY stage as a logical 1D stage");
   RegisterDevice(g_DeviceNameDAShutter, MM::ShutterDevice, "DA used as a shutter");
   RegisterDevice(g_DeviceNameDAMonochromator, MM::ShutterDevice, "DA used to control a monochromator");
   RegisterDevice(g_DeviceNameDAZStage, MM::StageDevice, "DA-controlled Z-stage");
   RegisterDevice(g_DeviceNameDAXYStage, MM::XYStageDevice, "DA-controlled XY-stage");
   RegisterDevice(g_DeviceNameDATTLStateDevice, MM::StateDevice, "Several DAs as a TTL state device");
   RegisterDevice(g_DeviceNameDAGalvoDevice, MM::GalvoDevice, "Two DAs operating a Galvo pair");
   RegisterDevice(g_DeviceNameMultiDAStateDevice, MM::StateDevice, "Several DAs as a single state device allowing digital masking");
   RegisterDevice(g_DeviceNameAutoFocusStage, MM::StageDevice, "AutoFocus offset acting as a Z-stage");
   RegisterDevice(g_DeviceNameStateDeviceShutter, MM::ShutterDevice, "State device used as a shutter");
   RegisterDevice(g_DeviceNamePropertyShutter, MM::ShutterDevice, "Any device property used as a shutter");
   RegisterDevice(g_DeviceNameSerialDTRShutter, MM::ShutterDevice, "Serial port DTR used as a shutter");
}

MODULE_API MM::Device* CreateDevice(const char* deviceName)                  
{
   if (deviceName == 0)
      return 0;

   if (strcmp(deviceName, g_DeviceNameMultiShutter) == 0) { 
      return new MultiShutter();
   } else if (strcmp(deviceName, g_DeviceNameMultiCamera) == 0) { 
      return new MultiCamera();
   } else if (strcmp(deviceName, g_DeviceNameMultiStage) == 0) {
      return new MultiStage();
   } else if (strcmp(deviceName, g_DeviceNameComboXYStage) == 0) {
      return new ComboXYStage();
   } else if (strcmp(deviceName, g_DeviceNameSingleAxisStage) == 0) {
      return new SingleAxisStage();
   } else if (strcmp(deviceName, g_DeviceNameDAShutter) == 0) { 
      return new DAShutter();
   } else if (strcmp(deviceName, g_DeviceNameDAMonochromator) == 0) {
      return new DAMonochromator();
   } else if (strcmp(deviceName, g_DeviceNameDAZStage) == 0) { 
      return new DAZStage();
   } else if (strcmp(deviceName, g_DeviceNameDAXYStage) == 0) { 
      return new DAXYStage();
   } else if (strcmp(deviceName, g_DeviceNameDATTLStateDevice) == 0) {
      return new DATTLStateDevice();
   } else if (strcmp(deviceName, g_DeviceNameDAGalvoDevice) == 0) {
      return new DAGalvo();
   } else if (strcmp(deviceName, g_DeviceNameMultiDAStateDevice) == 0) {
      return new MultiDAStateDevice();
   } else if (strcmp(deviceName, g_DeviceNameAutoFocusStage) == 0) { 
      return new AutoFocusStage();
   } else if (strcmp(deviceName, g_DeviceNameStateDeviceShutter) == 0) {
      return new StateDeviceShutter();
   } else if (strcmp(deviceName, g_DeviceNamePropertyShutter) == 0) {
      return new PropertyShutter();
   } else if (strcmp(deviceName, g_DeviceNameSerialDTRShutter) == 0) {
      return new SerialDTRShutter();
   }

   return 0;
}

MODULE_API void DeleteDevice(MM::Device* pDevice)                            
{                                                                            
   delete pDevice;                                                           
}

