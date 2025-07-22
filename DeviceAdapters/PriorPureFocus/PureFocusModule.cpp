#include "PureFocus.h"
#include "ModuleInterface.h"

// External names used by the rest of the system
// to load particular device from the "PriorPureFocus.dll" library
const char* g_PureFocusDevice = "Prior PureFocus";

// Device name variables
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
   RegisterDevice(g_PureFocusDevice, MM::AutoFocusDevice, g_PureFocusDeviceDescription);
}

MODULE_API MM::Device* CreateDevice(const char* deviceName)
{
   if (deviceName == 0)
      return 0;

   if (strcmp(deviceName, g_PureFocusDevice) == 0)
   {
      return new PureFocusHub();
   }

   return 0;
}

MODULE_API void DeleteDevice(MM::Device* pDevice)
{
   delete pDevice;
}
