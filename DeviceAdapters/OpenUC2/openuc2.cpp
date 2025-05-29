#include "openUC2.h"
#include "UC2Hub.h"
#include "XYStage.h"
#include "ZStage.h"
#include "Shutter.h"
#include "ModuleInterface.h"
#include <cstring>

// Module API: registers devices for Micro-Manager

MODULE_API void InitializeModuleData()
{
   RegisterDevice(g_HubName,     MM::HubDevice,    "openUC2 hub device");
   RegisterDevice(g_XYStageName, MM::XYStageDevice,"XY Stage for openUC2");
   RegisterDevice(g_ZStageName,  MM::StageDevice,  "Z Stage for openUC2");
   RegisterDevice(g_ShutterName, MM::ShutterDevice,"LED/Laser Shutter for openUC2");
}

MODULE_API MM::Device* CreateDevice(const char* deviceName)
{
   if (!deviceName)
      return 0;

   if (strcmp(deviceName, g_HubName) == 0)
      return new UC2Hub();
   else if (strcmp(deviceName, g_XYStageName) == 0)
      return new XYStage();
   else if (strcmp(deviceName, g_ZStageName) == 0)
      return new ZStage();
   else if (strcmp(deviceName, g_ShutterName) == 0)
      return new UC2Shutter();

   return 0;
}

MODULE_API void DeleteDevice(MM::Device* pDevice)
{
   delete pDevice;
}
