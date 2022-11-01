/*
 * Project: Light Sheet Device Manager
 * License: BSD 3-clause, see license.txt
 * Author: Brandon Simpson (brandon@asiimaging.com)
 * Copyright (c) 2022, Applied Scientific Instrumentation
 */

#include "LightSheetManager.h"

MODULE_API void InitializeModuleData()
{
    RegisterDevice(gDeviceName, MM::GenericDevice, gDeviceDescription);
}

MODULE_API MM::Device* CreateDevice(const char* deviceName)
{
    if (deviceName == 0)
    {
        return 0;
    }
    if (strcmp(deviceName, gDeviceName) == 0)
    {
        return new LightSheetDeviceManager();
    }
    else
    {
        return 0;
    }
}

MODULE_API void DeleteDevice(MM::Device* pDevice)
{
    delete pDevice;
}