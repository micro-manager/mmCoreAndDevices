#include "PureFocus.h"
#include "ModuleInterface.h"

const char* g_PureFocusDeviceName = "PureFocus";
const char* g_PureFocusDeviceDescription = "Prior Scientific PureFocus Autofocus System";

const char* g_Stepper = "Stepper Drive";
const char* g_Piezo = "Piezo Drive";
const char* g_Measure = "Measure";


///////////////////////////////////////////////////////////////////////////////
// Exported MMDevice API
///////////////////////////////////////////////////////////////////////////////

MODULE_API void InitializeModuleData()
{
   RegisterDevice(g_PureFocusDeviceName, MM::HubDevice, g_PureFocusDeviceDescription);
   RegisterDevice(g_PureFocusAutoFocusDeviceName, MM::AutoFocusDevice, g_PureFocusAutoFocusDescription);
   RegisterDevice(g_PureFocusOffsetDeviceName, MM::StageDevice, g_PureFocusOffsetDescription);
}

MODULE_API MM::Device* CreateDevice(const char* deviceName)
{
   if (deviceName == 0)
      return 0;

   if (strcmp(deviceName, g_PureFocusDeviceName) == 0)
   {
      return new PureFocusHub();
   }
   else if (strcmp(deviceName, g_PureFocusAutoFocusDeviceName) == 0)
   {
       return new PureFocusAutoFocus();
   }
   else if (strcmp(deviceName, g_PureFocusOffsetDeviceName) == 0)
   {
       return new PureFocusOffset();
   }

   return 0;
}

MODULE_API void DeleteDevice(MM::Device* pDevice)
{
   delete pDevice;
}
