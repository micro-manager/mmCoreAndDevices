#include "pch.h"
#include "PyDevice.h"

// Entry points for loading the dll and creating devices.
MODULE_API void InitializeModuleData()
{
    RegisterDevice(CPyHub::g_adapterName, MM::HubDevice, "Runs a Python script that constructs one or more device objects");
//    RegisterDevice(CPyGenericDevice::g_adapterName, MM::GenericDevice, "Generic micro-manager device that is implemented by a Python script");
//    RegisterDevice(CPyCamera::g_adapterName, MM::CameraDevice, "Camera device that obtains images from a Python script");
}


/**
 * @brief Creates a hub device of MM device wrapper for a Python object
 * @param deviceName "PyHub" or "{hubname}:{objectname}"
 * @return newly created device, or nullptr if device name is not found
*/
MODULE_API MM::Device* CreateDevice(const char* deviceName)
{
    if (!deviceName)
        return nullptr;

    if (strcmp(deviceName, CPyHub::g_adapterName) == 0)
        return new CPyHub();

    // else, we are (re)creating a MM Device wrapper for an existing Python object. To locate the object, we must first locate the hub
    return new CPyGenericDevice(deviceName);
}

MODULE_API void DeleteDevice(MM::Device* pDevice)
{
    delete pDevice;
}
