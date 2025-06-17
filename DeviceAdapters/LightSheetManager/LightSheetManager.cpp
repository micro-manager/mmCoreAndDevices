/*
 * Project: Light Sheet Device Manager
 * License: BSD 3-clause, see license.txt
 * Author: Brandon Simpson (brandon@asiimaging.com)
 * Copyright (c) 2022, Applied Scientific Instrumentation
 */

#include "LightSheetManager.h"

MODULE_API void InitializeModuleData() {
    RegisterDevice(gDeviceName, MM::GenericDevice, gDeviceDescription);
}

MODULE_API MM::Device* CreateDevice(const char* deviceName)
{
    if (deviceName == nullptr) {
        return nullptr;
    }
    if (std::string(deviceName) == gDeviceName) {
        return new LightSheetDeviceManager();
    }
    return nullptr;
}

MODULE_API void DeleteDevice(MM::Device* pDevice) {
    delete pDevice;
}
