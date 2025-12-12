#include "TeensyPulseGenerator.h"
#include "CameraPulser.h"

///////////////////////////////////////////////////////////////////////////////
// Exported MMDevice API
///////////////////////////////////////////////////////////////////////////////

const char* g_PulseGenerator = "TeensyPulseGenerator";
const char* g_CameraPulser = "TeensySendsPulsesToCamera";

MODULE_API void InitializeModuleData()
{
   RegisterDevice(g_PulseGenerator, MM::GenericDevice, "Teensy Pulse Generator");
   RegisterDevice(g_CameraPulser, MM::CameraDevice, "CameraPulser");
}

MODULE_API MM::Device* CreateDevice(const char* deviceName)
{
   if (deviceName == 0)
       return 0;

   if (strcmp(deviceName, g_PulseGenerator) == 0)
   {
       return new TeensyPulseGenerator();
   } 
   else if (strcmp(deviceName, g_CameraPulser) == 0)
   {
      return new CameraPulser;
   }

   return 0;
}

MODULE_API void DeleteDevice(MM::Device* pDevice)
{
    delete pDevice;
}
