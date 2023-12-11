#include "pch.h"
#include "PyDevice.h"
#include "PyCamera.h"
#include "PyStage.h"

// Entry points for loading the dll and creating devices.
MODULE_API void InitializeModuleData()
{
    RegisterDevice(CPyHub::g_adapterName, MM::HubDevice, "Runs a Python script that constructs one or more device objects");
}


/**
 * @brief Creates a hub device of MM device wrapper for a Python object
 * @param deviceName "PyHub" or "{devicetype}:{objectname}"
 * @return newly created device, or nullptr if device name is not found
*/
MODULE_API MM::Device* CreateDevice(const char* id)
{
    if (!id)
        return nullptr;

    auto path = string(id);

    if (path == CPyHub::g_adapterName)
        return new CPyHub();

    // else, we are (re)creating a MM Device wrapper for an existing Python object. To locate the object, we must first extract the object type
    
    string deviceType;
    string deviceName;
    if (!CPyHub::SplitId(id, deviceType, deviceName))
        return nullptr; // invalid id

    if (deviceType == "Device")
        return new CPyGenericDevice(id);
    if (deviceType == "Camera")
        return new CPyCamera(id);
    if (deviceType == "Stage")
        return new CPyStage(id);
    if (deviceType == "XYStage")
        return new CPyXYStage(id);
    return nullptr;
}


MODULE_API void DeleteDevice(MM::Device* pDevice)
{
    delete pDevice;
}
