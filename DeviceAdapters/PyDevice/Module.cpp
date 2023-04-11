#include <ModuleInterface.h>
#include "PyDevice.h"

const char* g_GenericDeviceName = "Generic Python device";

// Entry points for loading the dll and creating devices.
MODULE_API void InitializeModuleData()
{
    RegisterDevice(g_GenericDeviceName, MM::GenericDevice, "Generic micro-manager device that is implemented by a Python script");
}

MODULE_API MM::Device* CreateDevice(const char* deviceName)
{
    if (!deviceName)
        return nullptr;

    // decide which device class to create based on the deviceName parameter
    if (strcmp(deviceName, g_GenericDeviceName) == 0)
    {
        return new CPyGenericDevice();
    }

    // ...supplied name not recognized
    return nullptr;
}

MODULE_API void DeleteDevice(MM::Device* pDevice)
{
    delete pDevice;
}
