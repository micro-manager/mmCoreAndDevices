#include "ThorlabsChrolis.h"
#include "ModuleInterface.h"

#include <string>
#include <regex>

/*TODO
* Set states of properties based on current LED states - x
* Properties for device ID and stuff - x
* Error handling on device control methods - x
* custom errors and messages
* logs for errors - x
* Remove HubID Property -x
* Integer property range 0 to 1 for each LED on off - x
* Is sequencable property for each device check arduino implementation
* No need for sequence stuff in CHROLIS. Should check if breakout box needs to be enabled in software
* no need for event callbacks in UI for triggering
* Keep LED control in State Device - x
* Maybe keep triggering in Generic if that gets implemented 
* pre-init properties in constructor - x
* set error text in constructor -x
* enumerate in constructor 
* no logging in constructor
* store error in device instance
* after constructor, needs to be safe to call shutdown or destructor- x
* device specific properties in the hub - x
* state device allowed values added individually for drop down
* leave state as text box - integer property - X
* put wavelength in property name - X
* handle cases for initialization failing - x 
* Verify LED's all turned off with Shutter button -x
* Shutter off in case of Device Status LLG open -x
* Can a message be displayed in popup box without a return code?
* Check if lock is needed for multi threading -x
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

    return nullptr;
}


MODULE_API void DeleteDevice(MM::Device* device) {
    delete device;
}
//Hub Methods
ChrolisHub::ChrolisHub() :
    initialized_(false),
    busy_(false),
    threadRunning_(false), 
    deviceStatusMessage_("No Error")
{
    //Custom Errors
    SetErrorText(ERR_HUB_NOT_AVAILABLE, "Hub is not available");
    SetErrorText(ERR_CHROLIS_NOT_AVAIL, "CHROLIS Device is not available");
    SetErrorText(ERR_IMPROPER_SET, "Error setting property value. Value will be reset");
    SetErrorText(ERR_PARAM_NOT_VALID, "Value passed to property was out of bounds.");
    SetErrorText(ERR_NO_AVAIL_DEVICES, "No available devices were found on the system.");
    
    //VISA Errors
    SetErrorText(ERR_INSUF_INFO, "Insufficient location information of the device or the resource is not present on the system");
    SetErrorText(ERR_UNKOWN_HW_STATE, "Unknown Hardware State");
    SetErrorText(ERR_VAL_OVERFLOW, "Parameter Value Overflow");


    //CHROLIS Device Specific Errors. Located in TL6WL.h
    SetErrorText(INSTR_RUNTIME_ERROR, "CHROLIS Instrument Runtime Error");
    SetErrorText(INSTR_REM_INTER_ERROR, "CHROLIS Instrument Internal Error");
    SetErrorText(INSTR_AUTHENTICATION_ERROR, "CHROLIS Instrument Authentication Error");
    SetErrorText(INSTR_PARAM_ERROR, "CHROLIS Invalid Parameter Error");
    SetErrorText(INSTR_HW_ERROR, "CHROLIS Instrument Hardware Error. Please verify that no hardware faults are present.");
    SetErrorText(INSTR_PARAM_CHNG_ERROR, "CHROLIS Instrument Parameter Error");
    SetErrorText(INSTR_INTERNAL_TX_ERR, "CHROLIS Instrument Internal Command Sending Error");
    SetErrorText(INSTR_INTERNAL_RX_ERR, "CHROLIS Instrument Internal Command Receiving Error");
    SetErrorText(INSTR_INVAL_MODE_ERR, "CHROLIS Instrument Invalid Mode Error");
    SetErrorText(INSTR_SERVICE_ERR, "CHROLIS Instrument Service Error");

    chrolisDeviceInstance_ = new ThorlabsChrolisDeviceWrapper();
    atomic_init(&currentDeviceStatusCode_, 0);
    atomic_init(&threadRunning_, false);

    std::vector<std::string> serialNumbers;
    static_cast<ThorlabsChrolisDeviceWrapper*>(chrolisDeviceInstance_)->GetAvailableSerialNumbers(serialNumbers);

    CreateStringProperty("Serial Number", "DEFAULT", false, 0, true);
    for (int i = 0; i < serialNumbers.size(); i++)
    {
        AddAllowedValue("Serial Number", serialNumbers[i].c_str());
    }
}

int ChrolisHub::DetectInstalledDevices()
{
    ClearInstalledDevices();
    InitializeModuleData();// make sure this method is called before we look for available devices

    char hubName[MM::MaxStrLength];
    GetName(hubName);
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
        char buf[MM::MaxStrLength];
        int ret = GetProperty("Serial Number", buf);       

        auto err = static_cast<ThorlabsChrolisDeviceWrapper*>(chrolisDeviceInstance_)->InitializeDevice(buf);
        if (err != 0)
        {
            LogMessage("Error in CHROLIS Initialization");
            return err;
        }

        ViChar sNum[TL6WL_LONG_STRING_SIZE];
        static_cast<ThorlabsChrolisDeviceWrapper*>(chrolisDeviceInstance_)->GetSerialNumber(sNum);
        err = CreateStringProperty("Device Serial Number", sNum, true);
        if (err != 0)
        {
            LogMessage("Error with property set in hub initialize");
            return DEVICE_ERR;
        }

        ViChar manfName[TL6WL_LONG_STRING_SIZE];
        static_cast<ThorlabsChrolisDeviceWrapper*>(chrolisDeviceInstance_)->GetManufacturerName(manfName);
        err = CreateStringProperty("Manufacturer Name", manfName, true);
        if (err != 0)
        {
            LogMessage("Error with property set in hub initialize");
            return DEVICE_ERR;
        }

        std::string wavelengthList = "";
        ViUInt16 wavelengths[6];
        err = static_cast<ThorlabsChrolisDeviceWrapper*>(chrolisDeviceInstance_)->GetLEDWavelengths(wavelengths);
        if (err != 0)
        {
            LogMessage("Unable to get wavelengths from device");
            return err;
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
            LogMessage("Error with property set in hub initialize");
            return DEVICE_ERR;
        }

        err = CreateStringProperty("Device Status", deviceStatusMessage_.c_str(), true);
        if (err != 0)
        {
            LogMessage("Error with property set in hub initialize");
            return DEVICE_ERR;
        }

        threadRunning_.store(true);
        updateThread_ = std::thread(&ChrolisHub::StatusChangedPollingThread, this);
        initialized_ = true;
    }
    return DEVICE_OK;
}

int ChrolisHub::Shutdown()
{
    if (threadRunning_.load())
    {
        threadRunning_.store(false); // TODO make this atmoic or mutex
        updateThread_.join();
    }

    if (initialized_ || static_cast<ThorlabsChrolisDeviceWrapper*>(chrolisDeviceInstance_)->IsDeviceConnected())
    {
        auto err = static_cast<ThorlabsChrolisDeviceWrapper*>(chrolisDeviceInstance_)->ShutdownDevice();
        if (err != 0)
        {
            LogMessage("Error shutting down device");
            return DEVICE_ERR;
        }
        initialized_ = false;
        delete chrolisDeviceInstance_;
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

void ChrolisHub::StatusChangedPollingThread()
{
    bool statusChangedFlag = false;
    ViUInt32 tempStatus = 0;
    std::string message = "";
    ViBoolean tempEnableStates[6];
    while (threadRunning_.load())
    {
        ChrolisHub* pHub = static_cast<ChrolisHub*>(GetParentHub());
        if (!pHub || !pHub->IsInitialized())
        {
            LogMessage("Hub not available");
            threadRunning_.store(false);
            continue;
        }

        ThorlabsChrolisDeviceWrapper* wrapperInstance = static_cast<ThorlabsChrolisDeviceWrapper*>(pHub->GetChrolisDeviceInstance());
        if (wrapperInstance->IsDeviceConnected())
        {
            message = "";
            auto err = wrapperInstance->GetDeviceStatus(tempStatus);
            if (err != 0)
            {
                LogMessage("Error Getting Status");
                threadRunning_.store(false);
                continue;
            }
            if (currentDeviceStatusCode_.load() != tempStatus)
            {
                statusChangedFlag = true;
            }
            currentDeviceStatusCode_.store(tempStatus);
            if (currentDeviceStatusCode_.load() == 0)
            {
                message += "No Error";
            }
            else
            {
                if ((currentDeviceStatusCode_.load() & (1 << 0)) >= 1)
                {
                    message += "Box is Open";
                }
                if ((currentDeviceStatusCode_.load() & (1 << 1)) >= 1)
                {
                    if (message.length() > 0)
                    {
                        message += ", ";
                    }
                    message += "LLG not Connected";
                }
                if ((currentDeviceStatusCode_.load() & (1 << 2)) >= 1)
                {
                    if (message.length() > 0)
                    {
                        message += ", ";
                    }
                    message += "Interlock is Open";
                }
                if ((currentDeviceStatusCode_.load() & (1 << 3)) >= 1)
                {
                    if (message.length() > 0)
                    {
                        message += ", ";
                    }
                    message += "Using Default Adjustment";
                }
                if ((currentDeviceStatusCode_.load() & (1 << 4)) >= 1)
                {
                    if (message.length() > 0)
                    {
                        message += ", ";
                    }
                    message += "Box Overheated";
                }
                if ((currentDeviceStatusCode_.load() & (1 << 5)) >= 1)
                {
                    if (message.length() > 0)
                    {
                        message += ", ";
                    }
                    message += "LED Overheated";
                }
                if ((currentDeviceStatusCode_.load() & (1 << 6)) >= 1)
                {
                    if (message.length() > 0)
                    {
                        message += ", ";
                    }
                    message += "Invalid Box Setup";
                }
                if (message.length() == 0)
                {
                    message = "Unknown Status";
                }
            }
            if (statusChangedFlag)
            {
                if (currentDeviceStatusCode_.load() != 0)
                {
                    wrapperInstance->VerifyLEDEnableStatesWithLock();
                    if (wrapperInstance->GetLEDEnableStates(tempEnableStates[0], 
                        tempEnableStates[1], tempEnableStates[2], tempEnableStates[3], tempEnableStates[4], tempEnableStates[5]) != 0)
                    {
                        LogMessage("Error getting info from chrolis");
                    }
                    else 
                    {
                        std::ostringstream os;
                        os << tempEnableStates[0];
                        pHub->GetDevice(CHROLIS_STATE_NAME)->SetProperty("LED Enable State 1", os.str().c_str());

                        os.clear();
                        os << tempEnableStates[1];
                        pHub->GetDevice(CHROLIS_STATE_NAME)->SetProperty("LED Enable State 2", os.str().c_str());

                        os.clear();
                        os << tempEnableStates[2];
                        pHub->GetDevice(CHROLIS_STATE_NAME)->SetProperty("LED Enable State 3", os.str().c_str());

                        os.clear();
                        os << tempEnableStates[3];
                        pHub->GetDevice(CHROLIS_STATE_NAME)->SetProperty("LED Enable State 4", os.str().c_str());

                        os.clear();
                        os << tempEnableStates[4];
                        pHub->GetDevice(CHROLIS_STATE_NAME)->SetProperty("LED Enable State 5", os.str().c_str());

                        os.clear();
                        os << tempEnableStates[5];
                        pHub->GetDevice(CHROLIS_STATE_NAME)->SetProperty("LED Enable State 6", os.str().c_str());

                        os.clear();
                        os << ((static_cast<uint8_t>(tempEnableStates[0]) << 0) | (static_cast<uint8_t>(tempEnableStates[1]) << 1) | (static_cast<uint8_t>(tempEnableStates[2]) << 2)
                            | (static_cast<uint8_t>(tempEnableStates[3]) << 3) | (static_cast<uint8_t>(tempEnableStates[4]) << 4 | (static_cast<uint8_t>(tempEnableStates[5]) << 5)));
                        pHub->GetDevice(CHROLIS_STATE_NAME)->SetProperty("State", os.str().c_str());

                    }
                }
                statusChangedFlag = false;
            }
            OnPropertyChanged("Device Status", message.c_str());
        }
        Sleep(500);
    }
}

//Chrolis Shutter Methods
ChrolisShutter::ChrolisShutter()
{
    InitializeDefaultErrorMessages();
}

int ChrolisShutter::Initialize()
{
    ChrolisHub* pHub = static_cast<ChrolisHub*>(GetParentHub());
    if (pHub)
    {
    }
    else
        LogMessage("No Hub");

    ThorlabsChrolisDeviceWrapper* wrapperInstance = static_cast<ThorlabsChrolisDeviceWrapper*>(pHub->GetChrolisDeviceInstance());
    if (wrapperInstance->IsDeviceConnected())
    {
        auto err = wrapperInstance->SetShutterState(false);
        //return error but reset if needed
        if (err != 0)
        {
            LogMessage("Could not close shutter on init");
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
        LogMessage("Hub not available");
        return ERR_HUB_NOT_AVAILABLE;
    }
    ThorlabsChrolisDeviceWrapper* wrapperInstance = static_cast<ThorlabsChrolisDeviceWrapper*>(pHub->GetChrolisDeviceInstance());
    if (!wrapperInstance->IsDeviceConnected())
    {
        LogMessage("CHROLIS not available");
        return ERR_CHROLIS_NOT_AVAIL;
    }
    auto err = wrapperInstance->SetShutterState(open);
    if (err != 0)
    {
        LogMessage("Error setting shutter state");
        return err;
    }

    return DEVICE_OK;
}

int ChrolisShutter::GetOpen(bool& open)
{
    ChrolisHub* pHub = static_cast<ChrolisHub*>(GetParentHub());
    if (!pHub || !pHub->IsInitialized())
    {
        LogMessage("Hub not available");
        return ERR_HUB_NOT_AVAILABLE;
    }
    ThorlabsChrolisDeviceWrapper* wrapperInstance = static_cast<ThorlabsChrolisDeviceWrapper*>(pHub->GetChrolisDeviceInstance());
    if (!wrapperInstance->IsDeviceConnected())
    {
        LogMessage("CHROLIS not available");
        return ERR_CHROLIS_NOT_AVAIL;
    }
    wrapperInstance->GetShutterState(open);
    return DEVICE_OK;
}


//Chrolis State Device Methods
ChrolisStateDevice::ChrolisStateDevice() :
    numPos_(6), ledMaxPower_(100), ledMinPower_(0), 
    led1Power_(0), led2Power_(0), led3Power_(0), led4Power_(0), led5Power_(0), led6Power_(0),
    led1State_(false), led2State_(false), led3State_(false), led4State_(false), led5State_(false), led6State_(false)
{
    InitializeDefaultErrorMessages();
    //SetErrorText(ERR_UNKNOWN_POSITION, "Requested position not available in this device");
    //EnableDelay(); // signals that the delay setting will be used
}

int ChrolisStateDevice::Initialize()
{
    ChrolisHub* pHub = static_cast<ChrolisHub*>(GetParentHub());
    if (pHub)
    {
    }
    else
    {
        LogMessage("Hub not available");
    }

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
    uint32_t tmpLedState = 0;
    if (wrapperInstance->IsDeviceConnected())
    {
        err = wrapperInstance->GetLEDEnableStates(led1State_, led2State_, led3State_, led4State_, led5State_, led6State_);
        err = wrapperInstance->GetLEDPowerStates(led1Power_, led2Power_, led3Power_, led4Power_, led5Power_, led6Power_);
        tmpLedState =
            ((static_cast<uint8_t>(led1State_) << 0) | (static_cast<uint8_t>(led2State_) << 1) | (static_cast<uint8_t>(led3State_) << 2)
                | (static_cast<uint8_t>(led4State_) << 3) | (static_cast<uint8_t>(led5State_) << 4) | (static_cast<uint8_t>(led6State_) << 5));
    }

    //State Property
    CPropertyAction* pAct = new CPropertyAction(this, &ChrolisStateDevice::OnState);
    err = CreateIntegerProperty(MM::g_Keyword_State, tmpLedState, false, pAct);
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


// Get set process
// Get: pull instances of hub and chrolis, get the latest led states and set local vars to these states, if error use stored vals. 
//      This ensures UI is always updated with the current chrolis vals when possible 
// Set: use local stored vals as a fallback if the instances cannot be retrieved, set the val in the wrapper, wrapper takes care of hardware verification. 
//      In the event of an error, leave property unset and let OnChange handle update. The get uses the current instance so this would keep values synced
int ChrolisStateDevice::OnState(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        ChrolisHub* pHub = static_cast<ChrolisHub*>(GetParentHub());
        if (!pHub || !pHub->IsInitialized())
        {
            LogMessage("Hub not available");
        }
        ThorlabsChrolisDeviceWrapper* wrapperInstance = static_cast<ThorlabsChrolisDeviceWrapper*>(pHub->GetChrolisDeviceInstance());
        if (!wrapperInstance->IsDeviceConnected())
        {
            LogMessage("CHROLIS not available");
        }
        if (wrapperInstance->GetLEDEnableStates(led1State_, led2State_, led3State_, led4State_, led5State_, led6State_) != 0)
        {
            LogMessage("Error getting info from chrolis");
        }

        pProp->Set((long)((static_cast<uint8_t>(led1State_) << 0) | (static_cast<uint8_t>(led2State_) << 1) | (static_cast<uint8_t>(led3State_) << 2)
            | (static_cast<uint8_t>(led4State_) << 3) | (static_cast<uint8_t>(led5State_) << 4) | (static_cast<uint8_t>(led6State_) << 5)));
    }
    else if (eAct == MM::AfterSet)
    {
        //temp state from last set used as fallback
        uint8_t currentLEDState = ((static_cast<uint8_t>(led1State_) << 0) | (static_cast<uint8_t>(led2State_) << 1) | (static_cast<uint8_t>(led3State_) << 2)
            | (static_cast<uint8_t>(led4State_) << 3) | (static_cast<uint8_t>(led5State_) << 4) | (static_cast<uint8_t>(led6State_) << 5));

        //Get the current instances for hub and chrolis
        //In the event of error do not set the property. Set old value. Updated values will be pulled from getters if possible
        ChrolisHub* pHub = static_cast<ChrolisHub*>(GetParentHub());
        if (!pHub || !pHub->IsInitialized())
        {
            LogMessage("Hub not available");
            std::ostringstream os;
            os << currentLEDState;
            OnPropertyChanged(pProp->GetName().c_str(), os.str().c_str());
            return ERR_HUB_NOT_AVAILABLE;
        }
        ThorlabsChrolisDeviceWrapper* wrapperInstance = static_cast<ThorlabsChrolisDeviceWrapper*>(pHub->GetChrolisDeviceInstance());
        if (!wrapperInstance->IsDeviceConnected())
        {
            LogMessage("CHROLIS not available");
            std::ostringstream os;
            os << currentLEDState;
            OnPropertyChanged(pProp->GetName().c_str(), os.str().c_str());
            return ERR_CHROLIS_NOT_AVAIL;
        }


        long val; // incoming value from user
        pProp->Get(val);
        if (val >= pow(2, numPos_) || val < 0)
        {
            LogMessage("Requested state out of bounds");
            std::ostringstream os;
            os << currentLEDState;
            OnPropertyChanged(pProp->GetName().c_str(), os.str().c_str());
            return ERR_PARAM_NOT_VALID;
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
            //Do not set the property in the case of this error. Let the property change handle it. 
            //This will cover error where LED failed to set but chrolis is still ok
            LogMessage("Error Setting LED state");
            if (err != ERR_CHROLIS_NOT_AVAIL)
            {
                if (wrapperInstance->GetLEDEnableStates(led1State_, led2State_, led3State_, led4State_, led5State_, led6State_) != 0)
                {
                    LogMessage("Error getting info from chrolis");
                }
                currentLEDState = ((static_cast<uint8_t>(led1State_) << 0) | (static_cast<uint8_t>(led2State_) << 1) | (static_cast<uint8_t>(led3State_) << 2)
                    | (static_cast<uint8_t>(led4State_) << 3) | (static_cast<uint8_t>(led5State_) << 4) | (static_cast<uint8_t>(led6State_) << 5));

                std::ostringstream os;
                os << currentLEDState;
                OnPropertyChanged(pProp->GetName().c_str(), os.str().c_str());
            }

            return err;
        }

        std::ostringstream os;
        os << val;
        OnPropertyChanged(pProp->GetName().c_str(), os.str().c_str());

        //pProp->Set((long)val);
        
        //Probably don't need these but leaving for now
        //led1State_ = static_cast<bool>(val & (1 << 0));
        //led2State_ = static_cast<bool>(val & (1 << 1));
        //led3State_ = static_cast<bool>(val & (1 << 2));
        //led4State_ = static_cast<bool>(val & (1 << 3));
        //led5State_ = static_cast<bool>(val & (1 << 4));
        //led6State_ = static_cast<bool>(val & (1 << 5));

        //delete newStates;
        return DEVICE_OK;
    }
    return DEVICE_OK;
}

//On properties change only way to update range of property
int ChrolisStateDevice::OnEnableStateChange(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    ViPBoolean ledBeingControlled;
    int numFromName = -1;
    std::string searchString = pProp->GetName();
    std::regex regexp("[-+]?([0-9]*\.[0-9]+|[0-9]+)");
    std::smatch sm;
    std::regex_search(searchString, sm, regexp);

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
        LogMessage("Error selecting LED state");
        return DEVICE_ERR;
        break;
    }

    if (eAct == MM::BeforeGet)
    {
        ChrolisHub* pHub = static_cast<ChrolisHub*>(GetParentHub());
        if (!pHub || !pHub->IsInitialized())
        {
            LogMessage("Hub not available");
        }
        ThorlabsChrolisDeviceWrapper* wrapperInstance = static_cast<ThorlabsChrolisDeviceWrapper*>(pHub->GetChrolisDeviceInstance());
        if (!wrapperInstance->IsDeviceConnected())
        {
            LogMessage("CHROLIS not available");
        }
        if (wrapperInstance->GetSingleLEDEnableState(numFromName-1, *ledBeingControlled) != 0)
        {
            LogMessage("Error getting info from chrolis");
        }
        pProp->Set((long)*ledBeingControlled);
    }
    else if (eAct == MM::AfterSet)
    {
        double val;
        pProp->Get(val);

        ChrolisHub* pHub = static_cast<ChrolisHub*>(GetParentHub());
        if (!pHub || !pHub->IsInitialized())
        {
            LogMessage("Hub not available");
            std::ostringstream os;
            os << *ledBeingControlled;
            OnPropertyChanged(pProp->GetName().c_str(), os.str().c_str());
            return ERR_HUB_NOT_AVAILABLE;
        }
        ThorlabsChrolisDeviceWrapper* wrapperInstance = static_cast<ThorlabsChrolisDeviceWrapper*>(pHub->GetChrolisDeviceInstance());
        if (!wrapperInstance->IsDeviceConnected())
        {
            LogMessage("CHROLIS not available");
            std::ostringstream os;
            os << *ledBeingControlled;
            OnPropertyChanged(pProp->GetName().c_str(), os.str().c_str());
            return ERR_CHROLIS_NOT_AVAIL;
        }

        int err = wrapperInstance->SetSingleLEDEnableState(numFromName-1, (ViBoolean)val);
        if (err != 0)
        {
            LogMessage("Error Setting LED state");
            wrapperInstance->GetSingleLEDEnableState(numFromName - 1, *ledBeingControlled);
            std::ostringstream os;
            os << *ledBeingControlled;
            OnPropertyChanged(pProp->GetName().c_str(), os.str().c_str());
            return err;
        }

        *ledBeingControlled = (ViBoolean)val;
        std::ostringstream os;
        os << *ledBeingControlled;
        OnPropertyChanged(pProp->GetName().c_str(), os.str().c_str());
        return DEVICE_OK;
    }

    return DEVICE_OK;
}

int ChrolisStateDevice::OnPowerChange(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    ViPUInt16 ledBeingControlled;
    int numFromName = -1;
    std::string searchString = pProp->GetName();
    std::regex regexp("[-+]?([0-9]*\.[0-9]+|[0-9]+)");    
    std::smatch sm;
    std::regex_search(searchString, sm, regexp);

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
        LogMessage("Error selecting LED state");
        return DEVICE_ERR;
        break;
    }

    if (eAct == MM::BeforeGet)
    {
        ChrolisHub* pHub = static_cast<ChrolisHub*>(GetParentHub());
        if (!pHub || !pHub->IsInitialized())
        {
            LogMessage("Hub not available");
        }
        ThorlabsChrolisDeviceWrapper* wrapperInstance = static_cast<ThorlabsChrolisDeviceWrapper*>(pHub->GetChrolisDeviceInstance());
        if (!wrapperInstance->IsDeviceConnected())
        {
            LogMessage("CHROLIS not available");
        }
        if (wrapperInstance->GetSingleLEDPowerState(numFromName - 1, *ledBeingControlled) != 0)
        {
            LogMessage("Error getting info from chrolis");
        }
        pProp->Set((long)*ledBeingControlled);
    }
    else if (eAct == MM::AfterSet)
    {
        double val;
        pProp->Get(val);

        ChrolisHub* pHub = static_cast<ChrolisHub*>(GetParentHub());
        if (!pHub || !pHub->IsInitialized())
        {
            LogMessage("Hub not available");
            std::ostringstream os;
            os << *ledBeingControlled;
            OnPropertyChanged(pProp->GetName().c_str(), os.str().c_str());
            return ERR_HUB_NOT_AVAILABLE;
        }
        ThorlabsChrolisDeviceWrapper* wrapperInstance = static_cast<ThorlabsChrolisDeviceWrapper*>(pHub->GetChrolisDeviceInstance());
        if (!wrapperInstance->IsDeviceConnected())
        {
            LogMessage("CHROLIS not available");
            std::ostringstream os;
            os << *ledBeingControlled;
            OnPropertyChanged(pProp->GetName().c_str(), os.str().c_str());
            return ERR_CHROLIS_NOT_AVAIL;
        }

        int err = wrapperInstance->SetSingleLEDPowerState(numFromName - 1, (ViUInt16)val);
        if (err != 0)
        {
            LogMessage("Error Setting LED state");
            wrapperInstance->GetSingleLEDPowerState(numFromName - 1, *ledBeingControlled);
            std::ostringstream os;
            os << *ledBeingControlled;
            OnPropertyChanged(pProp->GetName().c_str(), os.str().c_str());
            return err;
        }

        *ledBeingControlled = (ViUInt16)val;
        std::ostringstream os;
        os << *ledBeingControlled;
        OnPropertyChanged(pProp->GetName().c_str(), os.str().c_str());
        return DEVICE_OK;
    }
    return DEVICE_OK;
}

