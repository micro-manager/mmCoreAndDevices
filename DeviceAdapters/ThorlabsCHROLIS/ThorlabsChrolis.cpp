#include "ThorlabsChrolis.h"
#include "ModuleInterface.h"

#include <string>

MODULE_API void InitializeModuleData() {
    RegisterDevice(CHROLIS_SHUTTER_NAME, // deviceName: model identifier and default device label
        MM::ShutterDevice, 
        "Thorlabs CHROLIS Shutter"); // description
}

MODULE_API MM::Device* CreateDevice(char const* name) {
    if (!name)
        return nullptr;

    if (name == std::string(CHROLIS_SHUTTER_NAME))
        return new ChrolisShutter();

    return nullptr;
}


MODULE_API void DeleteDevice(MM::Device* device) {
    delete device;
}