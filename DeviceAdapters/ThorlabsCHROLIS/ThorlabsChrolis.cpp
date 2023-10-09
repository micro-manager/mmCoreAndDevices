#include "ThorlabsChrolis.h"
#include "ModuleInterface.h"

#include <string>

MODULE_API void InitializeModuleData() {
    RegisterDevice(CHROLIS_HUB_NAME, // deviceName: model identifier and default device label
        MM::HubDevice, 
        "Thorlabs CHROLIS Hub"); // description
    RegisterDevice(CHROLIS_SHUTTER_NAME,
        MM::ShutterDevice,
        "Thorlabs CHROLIS Shutter"); 
    RegisterDevice(CHROLIS_STATE_NAME,
        MM::StateDevice,
        "Thorlabs CHROLIS Enable State");
    RegisterDevice(CHROLIS_GENERIC_DEVICE_NAME,
        MM::GenericDevice,
        "Thorlabs CHROLIS Power Control");
}

MODULE_API MM::Device* CreateDevice(char const* name) {
    if (!name)
        return nullptr;

    if (name == std::string(CHROLIS_HUB_NAME))
        return new ChrolisHub();

    if (name == std::string(CHROLIS_SHUTTER_NAME))
        return new ChrolisShutter();

    if (name == std::string(CHROLIS_STATE_NAME))
        return new ChrolisStateDevice();

    if (name == std::string(CHROLIS_GENERIC_DEVICE_NAME))
        return new ChrolisPowerControl();

    return nullptr;
}


MODULE_API void DeleteDevice(MM::Device* device) {
    delete device;
}
//Hub Methods
ChrolisHub::ChrolisHub() :
    initialized_(false),
    busy_(false)
{
    CreateHubIDProperty();
    chrolisDeviceInstance_ = new ThorlabsChrolisDeviceWrapper();
}

int ChrolisHub::DetectInstalledDevices()
{
    ClearInstalledDevices();

    // make sure this method is called before we look for available devices
    InitializeModuleData();

    char hubName[MM::MaxStrLength];
    GetName(hubName); // this device name
    for (unsigned i = 0; i < GetNumberOfDevices(); i++)
    {
        char deviceName[MM::MaxStrLength];
        bool success = GetDeviceName(i, deviceName, MM::MaxStrLength);
        if (success && (strcmp(hubName, deviceName) != 0))
        {
            MM::Device* pDev = CreateDevice(deviceName);
            AddInstalledDevice(pDev);
        }
    }
    return DEVICE_OK;
}

int ChrolisHub::Initialize()
{
    auto err = static_cast<ThorlabsChrolisDeviceWrapper*>(chrolisDeviceInstance_)->InitializeDevice();
    if (err != 0)
    {
        return DEVICE_ERR;
    }
    initialized_ = true;
    return DEVICE_OK;
}

int ChrolisHub::Shutdown()
{
    if (initialized_)
    {
        auto err = static_cast<ThorlabsChrolisDeviceWrapper*>(chrolisDeviceInstance_)->ShutdownDevice();
        if (err != 0)
        {
            return DEVICE_ERR;
        }
        initialized_ = false;
    }
    return DEVICE_OK;
}

void ChrolisHub::GetName(char* name) const
{
    CDeviceUtils::CopyLimitedString(name, CHROLIS_SHUTTER_NAME);
}

bool ChrolisHub::Busy()
{
    return false;
}

bool ChrolisHub::IsInitialized()
{
    return initialized_;
}

void* ChrolisHub::GetChrolisDeviceInstance()
{
    return chrolisDeviceInstance_;
}

//Chrolis Shutter Methods
int ChrolisShutter::Initialize()
{
    return DEVICE_OK;
}

int ChrolisShutter::Shutdown()
{
    return DEVICE_OK;
}

void ChrolisShutter::GetName(char* name) const
{
    CDeviceUtils::CopyLimitedString(name, CHROLIS_SHUTTER_NAME);
}

bool ChrolisShutter::Busy()
{
    return false;
}

int ChrolisShutter::SetOpen(bool open)
{
    ChrolisHub* pHub = static_cast<ChrolisHub*>(GetParentHub());
    if (!pHub || !pHub->IsInitialized())
    {
        return DEVICE_ERR; // TODO Add custom error messages
    }
    ThorlabsChrolisDeviceWrapper* wrapperInstance = static_cast<ThorlabsChrolisDeviceWrapper*>(pHub->GetChrolisDeviceInstance());
    if (!wrapperInstance->IsDeviceConnected())
    {
        return DEVICE_ERR;
    }
    auto err = wrapperInstance->SetShutterState(open);
    if (err != 0)
    {
        return DEVICE_ERR;
    }

    return DEVICE_OK;
}

int ChrolisShutter::GetOpen(bool& open)
{
    ChrolisHub* pHub = static_cast<ChrolisHub*>(GetParentHub());
    if (!pHub || !pHub->IsInitialized())
    {
        return DEVICE_ERR; // TODO Add custom error messages
    }
    ThorlabsChrolisDeviceWrapper* wrapperInstance = static_cast<ThorlabsChrolisDeviceWrapper*>(pHub->GetChrolisDeviceInstance());
    if (!wrapperInstance->IsDeviceConnected())
    {
        return DEVICE_ERR;
    }
    wrapperInstance->GetShutterState(open);
    return DEVICE_OK;
}


//Chrolis State Device Methods
int ChrolisStateDevice::Initialize()
{
    return DEVICE_OK;
}

int ChrolisStateDevice::Shutdown()
{
    return DEVICE_OK;
}

void ChrolisStateDevice::GetName(char* name) const
{
    CDeviceUtils::CopyLimitedString(name, CHROLIS_SHUTTER_NAME);
}

bool ChrolisStateDevice::Busy()
{
    return false;
}

int ChrolisStateDevice::OnState(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    return DEVICE_OK;
}

int ChrolisStateDevice::OnDelay(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    return DEVICE_OK;
}


//Chrolis Power Control (Genric Device) Methods
int ChrolisPowerControl::Initialize()
{
    return DEVICE_OK;
}

int ChrolisPowerControl::Shutdown()
{
    return DEVICE_OK;
}

void ChrolisPowerControl::GetName(char* name) const
{
    CDeviceUtils::CopyLimitedString(name, CHROLIS_SHUTTER_NAME);
}

bool ChrolisPowerControl::Busy()
{
    return false;
}