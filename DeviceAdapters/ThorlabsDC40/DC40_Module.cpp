#include "ModuleInterface.h"
#include "DC40.h"

#include "include/TLDC.h"

const char* g_DeviceDC40Name = "Thorlabs DC40";

MODULE_API void InitializeModuleData()
{
   RegisterDevice("DC40", MM::ShutterDevice, "DC40 4.0 A LED Driver");
}

/*---------------------------------------------------------------------------
 Creates and returns a device specified by the device name.
---------------------------------------------------------------------------*/
MODULE_API MM::Device* CreateDevice(const char* deviceName)
{
    return new DC40(deviceName);
}

/*---------------------------------------------------------------------------
 Deletes a device pointed by pDevice.
---------------------------------------------------------------------------*/
MODULE_API void DeleteDevice(MM::Device* pDevice)
{
    delete pDevice;
}
