#include "ThorlabsChrolis.h"
#include "ModuleInterface.h"

#include <string>

/*
* Questions for Mark
* 1- Pre-init properties
* 2- Sequencer action for properties and utilizing the sequencer
* 3- Slider bars?
* 4- Non-default property states for state devices
*/

/*TODO
* Set states of properties based on current LED states
* Properties for device ID and stuff
* Error handling on device control methods
* custom errors and messages
* logs for errors
*/

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
    if (!initialized_)
    {

        auto err = static_cast<ThorlabsChrolisDeviceWrapper*>(chrolisDeviceInstance_)->InitializeDevice();
        if (err != 0)
        {
            LogMessage("Error in CHROLIS Initialization");
            return DEVICE_ERR;
        }
        initialized_ = true;
    }
    return DEVICE_OK;
}

int ChrolisHub::Shutdown()
{
    if (initialized_)
    {
        auto err = static_cast<ThorlabsChrolisDeviceWrapper*>(chrolisDeviceInstance_)->ShutdownDevice();
        if (err != 0)
        {
            LogMessage("Error shutting down device");
            return DEVICE_ERR;
        }
        initialized_ = false;
    }
    return DEVICE_OK;
}

void ChrolisHub::GetName(char* name) const
{
    CDeviceUtils::CopyLimitedString(name, CHROLIS_HUB_NAME);
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
//TODO test on property change with label for shutter state
ChrolisShutter::ChrolisShutter()
{
    InitializeDefaultErrorMessages();
    //SetErrorText(ERR_UNKNOWN_POSITION, "Requested position not available in this device");
    //EnableDelay(); // signals that the delay setting will be used
    CreateHubIDProperty();
}

int ChrolisShutter::Initialize()
{
    ChrolisHub* pHub = static_cast<ChrolisHub*>(GetParentHub());
    if (pHub)
    {
        char hubLabel[MM::MaxStrLength];
        pHub->GetLabel(hubLabel);
        SetParentID(hubLabel); // for backward comp.
    }
    else
        LogMessage("No Hub");

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
ChrolisStateDevice::ChrolisStateDevice() :
    numPos_(6),
    curLedState_(0)
{
    InitializeDefaultErrorMessages();
    //SetErrorText(ERR_UNKNOWN_POSITION, "Requested position not available in this device");
    //EnableDelay(); // signals that the delay setting will be used
    CreateHubIDProperty();
}

int ChrolisStateDevice::Initialize()
{
    ChrolisHub* pHub = static_cast<ChrolisHub*>(GetParentHub());
    if (pHub)
    {
        char hubLabel[MM::MaxStrLength];
        pHub->GetLabel(hubLabel);
        SetParentID(hubLabel); // for backward comp.
    }
    else
        LogMessage("No Hub");

    // create default positions and labels
    const int bufSize = 1024;
    char buf[bufSize];
    for (long i = 0; i < numPos_; i++)
    {
        snprintf(buf, bufSize, "-%ld", i);
        SetPositionLabel(i, buf);
    }

    //State Property
    CPropertyAction* pAct = new CPropertyAction(this, &ChrolisStateDevice::OnState);
    auto err = CreateIntegerProperty(MM::g_Keyword_State, 0, false, pAct);
    if (err != DEVICE_OK)
        return err;

    return DEVICE_OK;
}

int ChrolisStateDevice::Shutdown()
{
    return DEVICE_OK;
}

void ChrolisStateDevice::GetName(char* name) const
{
    CDeviceUtils::CopyLimitedString(name, CHROLIS_STATE_NAME);
}

bool ChrolisStateDevice::Busy()
{
    return false;
}

int ChrolisStateDevice::OnState(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        pProp->Set(curLedState_);
        // nothing to do, let the caller to use cached property
    }
    else if (eAct == MM::AfterSet)
    {
        long pos;
        pProp->Get(pos);
        if (pos >= numPos_ || pos < 0)
        {
            pProp->Set(curLedState_); // revert
            return ERR_UNKNOWN_LED_STATE;
        }
        // Do something with the incoming state info  
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

        ViBoolean curStates[6];
        wrapperInstance->GetLEDEnableStates(curStates);
        for (int i = 0; i < 6; i++)
        {
            curStates[i] = false;
        }
        curStates[pos] = true;
        int err = wrapperInstance->SetLEDEnableStates(curStates);
        if (err != 0)
        {
            return DEVICE_ERR;
        }

        curLedState_ = pos;
        return DEVICE_OK;
    }
    else if (eAct == MM::IsSequenceable)
    {}
    return DEVICE_OK;
}

int ChrolisStateDevice::OnDelay(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    return DEVICE_OK;
}


//Chrolis Power Control (Genric Device) Methods
ChrolisPowerControl::ChrolisPowerControl() : 
    ledMaxPower_(100), ledMinPower_(0), led1Power_(0), led2Power_(0), led3Power_(0), led4Power_(0), led5Power_(0), led6Power_(0)
{
    InitializeDefaultErrorMessages();
    //SetErrorText(ERR_UNKNOWN_POSITION, "Requested position not available in this device");
    //EnableDelay(); // signals that the delay setting will be used
    CreateHubIDProperty();
}


int ChrolisPowerControl::Initialize()
{
    ChrolisHub* pHub = static_cast<ChrolisHub*>(GetParentHub());
    if (pHub)
    {
        char hubLabel[MM::MaxStrLength];
        pHub->GetLabel(hubLabel);
        SetParentID(hubLabel); // for backward comp.
    }
    else
        LogMessage("No Hub");

    //Properties for power control
    CPropertyAction* pAct = new CPropertyAction(this, &ChrolisPowerControl::OnPowerChange);
    auto err = CreateFloatProperty("LED 1", 0, false, pAct);
    SetPropertyLimits("LED 1 Power", ledMinPower_, ledMaxPower_);
    if (err != 0)
    {
        return DEVICE_ERR;
        LogMessage("Error with property set in power control");
    }

    return DEVICE_OK;
}

int ChrolisPowerControl::Shutdown()
{
    return DEVICE_OK;
}

void ChrolisPowerControl::GetName(char* name) const
{
    CDeviceUtils::CopyLimitedString(name, CHROLIS_GENERIC_DEVICE_NAME);
}

bool ChrolisPowerControl::Busy()
{
    return false;
}

int ChrolisPowerControl::OnPowerChange(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        pProp->Set((long)led1Power_);
        // nothing to do, let the caller to use cached property
    }
    else if (eAct == MM::AfterSet)
    {
        double val;
        pProp->Get(val);
        if (val > ledMaxPower_ || val < ledMinPower_)
        {
            pProp->Set((long)led1Power_); // revert
            return ERR_UNKNOWN_LED_STATE;
        }
        // Do something with the incoming state info  
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

        ViInt16 states[] = {0,(int)val,(int)val,(int)val,(int)val,0};
        wrapperInstance->SetLEDPowerStates(states);
        led1Power_ = (int)val;


        return DEVICE_OK;
    }
    else if (eAct == MM::IsSequenceable)
    {
    }
    return DEVICE_OK;
}

