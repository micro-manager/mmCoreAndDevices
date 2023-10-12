#include "ThorlabsChrolis.h"
#include "ModuleInterface.h"

#include <string>
#include <regex>
using namespace std;

/*TODO
* Set states of properties based on current LED states - x
* Properties for device ID and stuff - x
* Error handling on device control methods
* custom errors and messages
* logs for errors
* Remove HubID Property 
* Integer property range 0 to 1 for each LED on off - x
* Is sequencable property for each device check arduino implementation
* No need for sequence stuff in CHROLIS. Should check if breakout box needs to be enabled in software
* no need for event callbacks in UI for triggering
* Keep LED control in State Device - x
* Maybe keep triggering in Generic if that gets implemented 
* pre-init properties in constructor
* set error text in constructor
* enumerate in constructor 
* no logging in constructor
* store error in device instance
* after constructor, needs to be safe to call shutdown or destructor
* device specific properties in the hub
* state device allowed values added individually for drop down
* leave state as text box - integer property
* put wavelength in property name
* handle cases for initialization failing
* Julian tester
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
        "Thorlabs CHROLIS LED Control");
    //RegisterDevice(CHROLIS_GENERIC_DEVICE_NAME,
    //    MM::GenericDevice,
    //    "Thorlabs CHROLIS Power Control");
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

    //if (name == std::string(CHROLIS_GENERIC_DEVICE_NAME))
    //    return new ChrolisPowerControl();

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

        ViChar sNum[256]; 
        static_cast<ThorlabsChrolisDeviceWrapper*>(chrolisDeviceInstance_)->GetSerialNumber(sNum);
        err = CreateStringProperty("Serial Number", sNum, true);
        if (err != 0)
        {
            LogMessage("Error with property set in hub control");
            return DEVICE_ERR;
        }

        ViChar manfName[256];
        static_cast<ThorlabsChrolisDeviceWrapper*>(chrolisDeviceInstance_)->GetManufacturerName(manfName);
        err = CreateStringProperty("Manufacturer Name", manfName, true);
        if (err != 0)
        {
            LogMessage("Error with property set in hub control");
            return DEVICE_ERR;
        }

        std::string wavelengthList = "";
        ViUInt16 wavelengths[6];
        err = static_cast<ThorlabsChrolisDeviceWrapper*>(chrolisDeviceInstance_)->GetLEDWavelengths(wavelengths);
        if (err != 0)
        {
            LogMessage("Error with property set in hub control");
            return DEVICE_ERR;
        }
        for (int i = 0; i < 6; i ++)
        {
            wavelengthList += std::to_string(wavelengths[i]);
            if (i != 5)
            {
                wavelengthList += ", ";
            }
        }
        err = CreateStringProperty("Available Wavelengths", wavelengthList.c_str(), true);
        if (err != 0)
        {
            LogMessage("Error with property set in hub control");
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

    ThorlabsChrolisDeviceWrapper* wrapperInstance = static_cast<ThorlabsChrolisDeviceWrapper*>(pHub->GetChrolisDeviceInstance());
    if (wrapperInstance->IsDeviceConnected())
    {
        auto err = wrapperInstance->SetShutterState(false);
        if (err != 0)
        {
            LogMessage("Could not close shutter on it");
        }
    }

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
    numPos_(6),curLedState_(0), ledMaxPower_(100), ledMinPower_(0), 
    led1Power_(0), led2Power_(0), led3Power_(0), led4Power_(0), led5Power_(0), led6Power_(0),
    led1State_(false), led2State_(false), led3State_(false), led4State_(false), led5State_(false), led6State_(false)
{
    InitializeDefaultErrorMessages();
    //SetErrorText(ERR_UNKNOWN_POSITION, "Requested position not available in this device");
    //EnableDelay(); // signals that the delay setting will be used
    CreateHubIDProperty();
}

int ChrolisStateDevice::Initialize() //TODO: Initialized property?
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
    int err;

    ThorlabsChrolisDeviceWrapper* wrapperInstance = static_cast<ThorlabsChrolisDeviceWrapper*>(pHub->GetChrolisDeviceInstance());
    if (wrapperInstance->IsDeviceConnected())
    {
        err = wrapperInstance->GetLEDEnableStates(led1State_, led2State_, led3State_, led4State_, led5State_, led6State_);
        err = wrapperInstance->GetLEDPowerStates(led1Power_, led2Power_, led3Power_, led4Power_, led5Power_, led6Power_);
        curLedState_ =
            ((static_cast<uint8_t>(led1State_) << 0) | (static_cast<uint8_t>(led2State_) << 1) | (static_cast<uint8_t>(led3State_) << 2)
                | (static_cast<uint8_t>(led4State_) << 3) | (static_cast<uint8_t>(led5State_) << 4) | (static_cast<uint8_t>(led6State_) << 5));
    }

    //State Property
    CPropertyAction* pAct = new CPropertyAction(this, &ChrolisStateDevice::OnState);
    err = CreateIntegerProperty(MM::g_Keyword_State, curLedState_, false, pAct);
    if (err != DEVICE_OK)
        return err;


    ////Properties for power control
    pAct = new CPropertyAction(this, &ChrolisStateDevice::OnPowerChange);
    err = CreateIntegerProperty("LED 1 Power", led1Power_, false, pAct);
    if (err != 0)
    {
        LogMessage("Error with property set in power control");
        return DEVICE_ERR;
    }
    SetPropertyLimits("LED 1 Power", ledMinPower_, ledMaxPower_);

    pAct = new CPropertyAction(this, &ChrolisStateDevice::OnPowerChange);
    err = CreateIntegerProperty("LED 2 Power", led2Power_, false, pAct);
    SetPropertyLimits("LED 2 Power", ledMinPower_, ledMaxPower_);
    if (err != 0)
    {
        return DEVICE_ERR;
        LogMessage("Error with property set in state control");
    }

    pAct = new CPropertyAction(this, &ChrolisStateDevice::OnPowerChange);
    err = CreateIntegerProperty("LED 3 Power", led3Power_, false, pAct);
    SetPropertyLimits("LED 3 Power", ledMinPower_, ledMaxPower_);
    if (err != 0)
    {
        return DEVICE_ERR;
        LogMessage("Error with property set in state control");
    }

    pAct = new CPropertyAction(this, &ChrolisStateDevice::OnPowerChange);
    err = CreateIntegerProperty("LED 4 Power", led4Power_, false, pAct);
    SetPropertyLimits("LED 4 Power", ledMinPower_, ledMaxPower_);
    if (err != 0)
    {
        return DEVICE_ERR;
        LogMessage("Error with property set in state control");
    }

    pAct = new CPropertyAction(this, &ChrolisStateDevice::OnPowerChange);
    err = CreateIntegerProperty("LED 5 Power", led5Power_, false, pAct);
    SetPropertyLimits("LED 5 Power", ledMinPower_, ledMaxPower_);
    if (err != 0)
    {
        return DEVICE_ERR;
        LogMessage("Error with property set in state control");
    }

    pAct = new CPropertyAction(this, &ChrolisStateDevice::OnPowerChange);
    err = CreateIntegerProperty("LED 6 Power", led6Power_, false, pAct);
    SetPropertyLimits("LED 6 Power", ledMinPower_, ledMaxPower_);
    if (err != 0)
    {
        return DEVICE_ERR;
        LogMessage("Error with property set in state control");
    }


    //Properties for state control
    pAct = new CPropertyAction(this, &ChrolisStateDevice::OnEnableStateChange);
    err = CreateIntegerProperty("LED Enable State 1", led1State_, false, pAct);
    if (err != 0)
    {
        LogMessage("Error with property set in state control");
        return DEVICE_ERR;
    }
    SetPropertyLimits("LED Enable State 1", 0, 1);

    pAct = new CPropertyAction(this, &ChrolisStateDevice::OnEnableStateChange);
    err = CreateIntegerProperty("LED Enable State 2", led2State_, false, pAct);
    if (err != 0)
    {
        LogMessage("Error with property set in state control");
        return DEVICE_ERR;
    }
    SetPropertyLimits("LED Enable State 2", 0, 1);

    pAct = new CPropertyAction(this, &ChrolisStateDevice::OnEnableStateChange);
    err = CreateIntegerProperty("LED Enable State 3", led3State_, false, pAct);
    if (err != 0)
    {
        LogMessage("Error with property set in state control");
        return DEVICE_ERR;
    }
    SetPropertyLimits("LED Enable State 3", 0, 1);

    pAct = new CPropertyAction(this, &ChrolisStateDevice::OnEnableStateChange);
    err = CreateIntegerProperty("LED Enable State 4", led4State_, false, pAct);
    if (err != 0)
    {
        LogMessage("Error with property set in state control");
        return DEVICE_ERR;
    }
    SetPropertyLimits("LED Enable State 4", 0, 1);

    pAct = new CPropertyAction(this, &ChrolisStateDevice::OnEnableStateChange);
    err = CreateIntegerProperty("LED Enable State 5", led5State_, false, pAct);
    if (err != 0)
    {
        LogMessage("Error with property set in state control");
        return DEVICE_ERR;
    }
    SetPropertyLimits("LED Enable State 5", 0, 1);

    pAct = new CPropertyAction(this, &ChrolisStateDevice::OnEnableStateChange);
    err = CreateIntegerProperty("LED Enable State 6", led6Power_, false, pAct);
    if (err != 0)
    {
        LogMessage("Error with property set in state control");
        return DEVICE_ERR;
    }
    SetPropertyLimits("LED Enable State 6", 0, 1);

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

int ChrolisStateDevice::OnDelay(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    return DEVICE_OK;
}

//OnPropertyChanged("AsyncPropertyFollower", leaderValue.c_str());
//Update onState to use binary value
//single state changes should modify the single led, verify that it was changed, update global state long, onchange OnState
// Update states of properties based on initial values
//Error check hardware calls and revert properties if something fails
int ChrolisStateDevice::OnState(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        pProp->Set((long)curLedState_);
        // nothing to do, let the caller to use cached property
    }
    else if (eAct == MM::AfterSet)
    {
        long val;
        pProp->Get(val);
        if (val >= pow(2, numPos_) || val < 0)
        {
            pProp->Set((long)curLedState_); // revert
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

        ViBoolean newStates[6]
        {
            static_cast<bool>(val & (1 << 0)),
            static_cast<bool>(val & (1 << 1)),
            static_cast<bool>(val & (1 << 2)),
            static_cast<bool>(val & (1 << 3)),
            static_cast<bool>(val & (1 << 4)),
            static_cast<bool>(val & (1 << 5))
        };
        int err = wrapperInstance->SetLEDEnableStates(newStates);
        if (err != 0)
        {
            return DEVICE_ERR;
        }

        led1State_ = static_cast<bool>(val & (1 << 0));
        led2State_ = static_cast<bool>(val & (1 << 1));
        led3State_ = static_cast<bool>(val & (1 << 2));
        led4State_ = static_cast<bool>(val & (1 << 3));
        led5State_ = static_cast<bool>(val & (1 << 4));
        led6State_ = static_cast<bool>(val & (1 << 5));

        OnPropertiesChanged();

        curLedState_ = val;
        return DEVICE_OK;
    }
    return DEVICE_OK;
}

int ChrolisStateDevice::OnEnableStateChange(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    ViPBoolean ledBeingControlled;
    int numFromName = -1;
    std::string searchString = pProp->GetName();
    regex regexp("[-+]?([0-9]*\.[0-9]+|[0-9]+)");
    std::smatch sm;
    regex_search(searchString, sm, regexp);

    //The names for the LED's should contain only a single number representing the LED
    //Use this to set the power
    if (sm.size() > 0)
    {
        if (sm[0].str().length() > 0)
        {
            numFromName = stoi(sm[0].str());
        }
    }
    else
    {
        LogMessage("Regex match failed");
        return DEVICE_ERR;
    }

    switch (numFromName)
    {
    case 1:
        ledBeingControlled = &led1State_;
        break;
    case 2:
        ledBeingControlled = &led2State_;
        break;
    case 3:
        ledBeingControlled = &led3State_;
        break;
    case 4:
        ledBeingControlled = &led4State_;
        break;
    case 5:
        ledBeingControlled = &led5State_;
        break;
    case 6:
        ledBeingControlled = &led6State_;
        break;
    default:
        return DEVICE_ERR;
        break;
    }

    if (eAct == MM::BeforeGet)
    {
        pProp->Set((long)*ledBeingControlled);
        // nothing to do, let the caller to use cached property
    }
    else if (eAct == MM::AfterSet)
    {
        double val;
        pProp->Get(val);
        if (val > ledMaxPower_ || val < ledMinPower_)
        {
            pProp->Set((long)*ledBeingControlled); // revert
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

        wrapperInstance->SetSingleLEDEnableState(numFromName-1, (ViBoolean)val);
        *ledBeingControlled = (ViBoolean)val;
        curLedState_ = 
            ((static_cast<uint8_t>(led1State_) << 0) | (static_cast<uint8_t>(led2State_) << 1) | (static_cast<uint8_t>(led3State_) << 2) 
                | (static_cast<uint8_t>(led4State_) << 3) | (static_cast<uint8_t>(led5State_) << 4) | (static_cast<uint8_t>(led6State_) << 5));
        OnPropertiesChanged();
        return DEVICE_OK;
    }

    return DEVICE_OK;
}

int ChrolisStateDevice::OnPowerChange(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    ViPUInt16 ledBeingControlled;
    int numFromName = -1;
    std::string searchString = pProp->GetName();
    regex regexp("[-+]?([0-9]*\.[0-9]+|[0-9]+)");    
    std::smatch sm;
    regex_search(searchString, sm, regexp);

    //The names for the LED's should contain only a single number representing the LED
    //Use this to set the power
    if (sm.size() > 0)
    {
        if (sm[0].str().length() > 0)
        {
            numFromName = stoi(sm[0].str());
        }
    }
    else
    {
        LogMessage("Regex match failed");
        return DEVICE_ERR;
    }

    switch (numFromName)
    {
    case 1:
        ledBeingControlled = &led1Power_;
        break;
    case 2:
        ledBeingControlled = &led2Power_;
        break;
    case 3:
        ledBeingControlled = &led3Power_;
        break;
    case 4:
        ledBeingControlled = &led4Power_;
        break;
    case 5:
        ledBeingControlled = &led5Power_;
        break;
    case 6:
        ledBeingControlled = &led6Power_;
        break;
    default:
        return DEVICE_ERR;
        break;
    }

    if (eAct == MM::BeforeGet)
    {
        pProp->Set((long)*ledBeingControlled);
        // nothing to do, let the caller to use cached property
    }
    else if (eAct == MM::AfterSet)
    {
        double val;
        pProp->Get(val);
        if (val > ledMaxPower_ || val < ledMinPower_)
        {
            pProp->Set((long)*ledBeingControlled); // revert
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

        wrapperInstance->SetSingleLEDPowerState(numFromName-1, val);
        *ledBeingControlled = (int)val;
        OnPropertiesChanged();

        return DEVICE_OK;
    }
    return DEVICE_OK;
}


//Chrolis Power Control (Genric Device) Methods
//ChrolisPowerControl::ChrolisPowerControl() : 
//    ledMaxPower_(100), ledMinPower_(0), led1Power_(0), led2Power_(0), led3Power_(0), led4Power_(0), led5Power_(0), led6Power_(0)
//{
//    InitializeDefaultErrorMessages();
//    //SetErrorText(ERR_UNKNOWN_POSITION, "Requested position not available in this device");
//    //EnableDelay(); // signals that the delay setting will be used
//    CreateHubIDProperty();
//}
//

//int ChrolisPowerControl::Initialize()
//{
//    ChrolisHub* pHub = static_cast<ChrolisHub*>(GetParentHub());
//    if (pHub)
//    {
//        char hubLabel[MM::MaxStrLength];
//        pHub->GetLabel(hubLabel);
//        SetParentID(hubLabel); // for backward comp.
//    }
//    else
//        LogMessage("No Hub");
//
//    //Properties for power control
//    CPropertyAction* pAct = new CPropertyAction(this, &ChrolisPowerControl::OnPowerChange);
//    auto err = CreateFloatProperty("LED 1", 0, false, pAct);
//    SetPropertyLimits("LED 1 Power", ledMinPower_, ledMaxPower_);
//    if (err != 0)
//    {
//        return DEVICE_ERR;
//        LogMessage("Error with property set in power control");
//    }
//
//    return DEVICE_OK;
//}
//
//int ChrolisPowerControl::Shutdown()
//{
//    return DEVICE_OK;
//}
//
//void ChrolisPowerControl::GetName(char* name) const
//{
//    CDeviceUtils::CopyLimitedString(name, CHROLIS_GENERIC_DEVICE_NAME);
//}
//
//bool ChrolisPowerControl::Busy()
//{
//    return false;
//}


