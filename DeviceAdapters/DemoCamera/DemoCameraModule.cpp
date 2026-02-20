///////////////////////////////////////////////////////////////////////////////
// FILE:          DemoCameraModule.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Module initialization and device factory for DemoCamera adapter
//
// AUTHOR:        Nenad Amodaj, nenad@amodaj.com, 06/08/2005
//
// COPYRIGHT:     University of California, San Francisco, 2006
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

#include "DemoCamera.h"
#include "ModuleInterface.h"
#include <cstring>

// External names used by the rest of the system
// to load particular device from the "DemoCamera.dll" library
extern const char* g_CameraDeviceName;
extern const char* g_WheelDeviceName;
extern const char* g_StateDeviceName;
extern const char* g_LightPathDeviceName;
extern const char* g_ObjectiveDeviceName;
extern const char* g_StageDeviceName;
extern const char* g_XYStageDeviceName;
extern const char* g_AutoFocusDeviceName;
extern const char* g_ShutterDeviceName;
extern const char* g_DADeviceName;
extern const char* g_DA2DeviceName;
extern const char* g_GalvoDeviceName;
extern const char* g_MagnifierDeviceName;
extern const char* g_PressurePumpDeviceName;
extern const char* g_VolumetricPumpDeviceName;
extern const char* g_HubDeviceName;

///////////////////////////////////////////////////////////////////////////////
// Exported MMDevice API
///////////////////////////////////////////////////////////////////////////////

MODULE_API void InitializeModuleData()
{
   RegisterDevice(g_CameraDeviceName, MM::CameraDevice, "Demo camera");
   RegisterDevice(g_WheelDeviceName, MM::StateDevice, "Demo filter wheel");
   RegisterDevice(g_StateDeviceName, MM::StateDevice, "Demo State Device");
   RegisterDevice(g_ObjectiveDeviceName, MM::StateDevice, "Demo objective turret");
   RegisterDevice(g_StageDeviceName, MM::StageDevice, "Demo stage");
   RegisterDevice(g_XYStageDeviceName, MM::XYStageDevice, "Demo XY stage");
   RegisterDevice(g_LightPathDeviceName, MM::StateDevice, "Demo light path");
   RegisterDevice(g_AutoFocusDeviceName, MM::AutoFocusDevice, "Demo auto focus");
   RegisterDevice(g_ShutterDeviceName, MM::ShutterDevice, "Demo shutter");
   RegisterDevice(g_DADeviceName, MM::SignalIODevice, "Demo DA");
   RegisterDevice(g_DA2DeviceName, MM::SignalIODevice, "Demo DA-2");
   RegisterDevice(g_MagnifierDeviceName, MM::MagnifierDevice, "Demo Optovar");
   RegisterDevice(g_GalvoDeviceName, MM::GalvoDevice, "Demo Galvo");
   RegisterDevice(g_PressurePumpDeviceName, MM::PressurePumpDevice, "Demo Pressure Pump");
   RegisterDevice(g_VolumetricPumpDeviceName, MM::VolumetricPumpDevice, "Demo Volumetric Pump");
   RegisterDevice("TransposeProcessor", MM::ImageProcessorDevice, "TransposeProcessor");
   RegisterDevice("ImageFlipX", MM::ImageProcessorDevice, "ImageFlipX");
   RegisterDevice("ImageFlipY", MM::ImageProcessorDevice, "ImageFlipY");
   RegisterDevice("MedianFilter", MM::ImageProcessorDevice, "MedianFilter");
   RegisterDevice(g_HubDeviceName, MM::HubDevice, "DHub");
}

MODULE_API MM::Device* CreateDevice(const char* deviceName)
{
   if (deviceName == 0)
      return 0;

   // decide which device class to create based on the deviceName parameter
   if (strcmp(deviceName, g_CameraDeviceName) == 0)
   {
      // create camera
      return new CDemoCamera();
   }
   else if (strcmp(deviceName, g_WheelDeviceName) == 0)
   {
      // create filter wheel
      return new CDemoFilterWheel();
   }
   else if (strcmp(deviceName, g_ObjectiveDeviceName) == 0)
   {
      // create objective turret
      return new CDemoObjectiveTurret();
   }
   else if (strcmp(deviceName, g_StateDeviceName) == 0)
   {
      // create state device
      return new CDemoStateDevice();
   }
   else if (strcmp(deviceName, g_StageDeviceName) == 0)
   {
      // create stage
      return new CDemoStage();
   }
   else if (strcmp(deviceName, g_XYStageDeviceName) == 0)
   {
      // create stage
      return new CDemoXYStage();
   }
   else if (strcmp(deviceName, g_LightPathDeviceName) == 0)
   {
      // create light path
      return new CDemoLightPath();
   }
   else if (strcmp(deviceName, g_ShutterDeviceName) == 0)
   {
      // create shutter
      return new DemoShutter();
   }
   else if (strcmp(deviceName, g_DADeviceName) == 0)
   {
      // create DA
      return new DemoDA(0);
   }
   else if (strcmp(deviceName, g_DA2DeviceName) == 0)
   {
      // create DA
      return new DemoDA(1);
   }
   else if (strcmp(deviceName, g_AutoFocusDeviceName) == 0)
   {
      // create autoFocus
      return new DemoAutoFocus();
   }
   else if (strcmp(deviceName, g_MagnifierDeviceName) == 0)
   {
      // create Optovar
      return new DemoMagnifier();
   }
   else if (strcmp(deviceName, g_GalvoDeviceName) == 0)
   {
      // create Galvo
      return new DemoGalvo();
   }
   else if(strcmp(deviceName, "TransposeProcessor") == 0)
   {
      return new TransposeProcessor();
   }
   else if(strcmp(deviceName, "ImageFlipX") == 0)
   {
      return new ImageFlipX();
   }
   else if(strcmp(deviceName, "ImageFlipY") == 0)
   {
      return new ImageFlipY();
   }
   else if(strcmp(deviceName, "MedianFilter") == 0)
   {
      return new MedianFilter();
   }
   else if (strcmp(deviceName, g_PressurePumpDeviceName) == 0)
   {
      return new DemoPressurePump();
   }
   else if (strcmp(deviceName, g_VolumetricPumpDeviceName) == 0)
   {
      return new DemoVolumetricPump();
   }
   else if (strcmp(deviceName, g_HubDeviceName) == 0)
   {
	  return new DemoHub();
   }

   // ...supplied name not recognized
   return 0;
}

MODULE_API void DeleteDevice(MM::Device* pDevice)
{
   delete pDevice;
}
