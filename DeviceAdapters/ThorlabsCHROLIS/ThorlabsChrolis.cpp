#include "ThorlabsChrolis.h"
#include "ModuleInterface.h"

#include <string>
#include <regex>

MODULE_API void InitializeModuleData() {
    RegisterDevice(CHROLIS_HUB_NAME,
        MM::HubDevice, 
        "Thorlabs CHROLIS Hub");
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
    threadRunning_(false), 
    deviceStatusMessage_("No Error")
{
    for (const auto& errMessage : ErrorMessages())
    {
        SetErrorText(errMessage.first, errMessage.second.c_str());
    }

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

int ChrolisHub::Initialize() // only gets called once
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
    for (int i = 0; i < 6; i++)
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
    return DEVICE_OK;
}

int ChrolisHub::Shutdown()
{
    if (threadRunning_.load())
    {
        threadRunning_.store(false); // TODO make this atmoic or mutex
        updateThread_.join();
    }

    if (static_cast<ThorlabsChrolisDeviceWrapper*>(chrolisDeviceInstance_)->IsDeviceConnected())
    {
        auto err = static_cast<ThorlabsChrolisDeviceWrapper*>(chrolisDeviceInstance_)->ShutdownDevice();
        if (err != 0)
        {
            LogMessage("Error shutting down device");
            return DEVICE_ERR;
        }
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

void* ChrolisHub::GetChrolisDeviceInstance()
{
    return chrolisDeviceInstance_;
}

void ChrolisHub::StatusChangedPollingThread()
{
    ViUInt32 tempStatus = 0;
    std::string message;
    ViBoolean tempEnableStates[6];
    while (threadRunning_.load())
    {
        ThorlabsChrolisDeviceWrapper* wrapperInstance = static_cast<ThorlabsChrolisDeviceWrapper*>(GetChrolisDeviceInstance());
        if (wrapperInstance->IsDeviceConnected())
        {
            message.clear();
            auto err = wrapperInstance->GetDeviceStatus(tempStatus);
            if (err != 0)
            {
                LogMessage("Error Getting Status");
                threadRunning_.store(false);
                return;
            }
            const auto curStatus = currentDeviceStatusCode_.load();
            const bool statusChanged = curStatus != tempStatus;
            currentDeviceStatusCode_.store(tempStatus);
            if (curStatus == 0)
            {
                message += "No Error";
            }
            else
            {
                if ((curStatus & (1 << 0)) >= 1)
                {
                    message += "Box is Open";
                }
                if ((curStatus & (1 << 1)) >= 1)
                {
                    if (message.length() > 0)
                    {
                        message += ", ";
                    }
                    message += "LLG not Connected";
                }
                if ((curStatus & (1 << 2)) >= 1)
                {
                    if (message.length() > 0)
                    {
                        message += ", ";
                    }
                    message += "Interlock is Open";
                }
                if ((curStatus & (1 << 3)) >= 1)
                {
                    if (message.length() > 0)
                    {
                        message += ", ";
                    }
                    message += "Using Default Adjustment";
                }
                if ((curStatus & (1 << 4)) >= 1)
                {
                    if (message.length() > 0)
                    {
                        message += ", ";
                    }
                    message += "Box Overheated";
                }
                if ((curStatus & (1 << 5)) >= 1)
                {
                    if (message.length() > 0)
                    {
                        message += ", ";
                    }
                    message += "LED Overheated";
                }
                if ((curStatus & (1 << 6)) >= 1)
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
            if (statusChanged)
            {
                if (curStatus != 0)
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
                        GetDevice(CHROLIS_STATE_NAME)->SetProperty("LED Enable State 1", os.str().c_str());

                        os.clear();
                        os << tempEnableStates[1];
                        GetDevice(CHROLIS_STATE_NAME)->SetProperty("LED Enable State 2", os.str().c_str());

                        os.clear();
                        os << tempEnableStates[2];
                        GetDevice(CHROLIS_STATE_NAME)->SetProperty("LED Enable State 3", os.str().c_str());

                        os.clear();
                        os << tempEnableStates[3];
                        GetDevice(CHROLIS_STATE_NAME)->SetProperty("LED Enable State 4", os.str().c_str());

                        os.clear();
                        os << tempEnableStates[4];
                        GetDevice(CHROLIS_STATE_NAME)->SetProperty("LED Enable State 5", os.str().c_str());

                        os.clear();
                        os << tempEnableStates[5];
                        GetDevice(CHROLIS_STATE_NAME)->SetProperty("LED Enable State 6", os.str().c_str());

                        os.clear();
                        os << ((static_cast<uint8_t>(tempEnableStates[0]) << 0) | (static_cast<uint8_t>(tempEnableStates[1]) << 1) | (static_cast<uint8_t>(tempEnableStates[2]) << 2)
                            | (static_cast<uint8_t>(tempEnableStates[3]) << 3) | (static_cast<uint8_t>(tempEnableStates[4]) << 4 | (static_cast<uint8_t>(tempEnableStates[5]) << 5)));
                        GetDevice(CHROLIS_STATE_NAME)->SetProperty("State", os.str().c_str());

                    }
                }
            }
            OnPropertyChanged("Device Status", message.c_str());
        }
        Sleep(500);
    }
}

void ChrolisHub::SetShutterCallback(std::function<void(int, int)> function)
{
    shutterCallback_ = function;
}

void ChrolisHub::SetStateCallback(std::function<void(int, int)> function)
{
    stateCallback_ = function;
}

//Chrolis Shutter Methods
ChrolisShutter::ChrolisShutter()
{
    for (const auto& errMessage : ErrorMessages())
    {
        SetErrorText(errMessage.first, errMessage.second.c_str());
    }
    InitializeDefaultErrorMessages();
}

int ChrolisShutter::Initialize()
{
    ChrolisHub* pHub = static_cast<ChrolisHub*>(GetParentHub());
    if (pHub)
    {
    }
    else
    {
        LogMessage("No Hub");
        return ERR_HUB_NOT_AVAILABLE;
    }

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

    pHub->SetShutterCallback([this](int ledNum, int state) 
        {
        });

    return DEVICE_OK;
}

int ChrolisShutter::Shutdown()
{
    ChrolisHub* pHub = static_cast<ChrolisHub*>(GetParentHub());
    if (pHub)
    {
        pHub->SetShutterCallback([](int , int) {});
    }
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
    if (!pHub)
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
    if (!pHub)
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
    numPos_(6), ledMaxBrightness_(1000), ledMinBrightness_(0),
    led1Brightness_(0), led2Brightness_(0), led3Brightness_(0), led4Brightness_(0), led5Brightness_(0), led6Brightness_(0),
    led1State_(false), led2State_(false), led3State_(false), led4State_(false), led5State_(false), led6State_(false)
{
    for (const auto& errMessage : ErrorMessages())
    {
        SetErrorText(errMessage.first, errMessage.second.c_str());
    }
    InitializeDefaultErrorMessages();
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
        err = wrapperInstance->GetLEDBrightnessStates(led1Brightness_, led2Brightness_, led3Brightness_, led4Brightness_, led5Brightness_, led6Brightness_);
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
    err = CreateIntegerProperty("LED 1 Power", led1Brightness_, false, pAct);
    if (err != 0)
    {
        LogMessage("Error with property set in power control");
        return DEVICE_ERR;
    }
    SetPropertyLimits("LED 1 Power", ledMinBrightness_, ledMaxBrightness_);

    pAct = new CPropertyAction(this, &ChrolisStateDevice::OnPowerChange);
    err = CreateIntegerProperty("LED 2 Power", led2Brightness_, false, pAct);
    SetPropertyLimits("LED 2 Power", ledMinBrightness_, ledMaxBrightness_);
    if (err != 0)
    {
        return DEVICE_ERR;
        LogMessage("Error with property set in state control");
    }

    pAct = new CPropertyAction(this, &ChrolisStateDevice::OnPowerChange);
    err = CreateIntegerProperty("LED 3 Power", led3Brightness_, false, pAct);
    SetPropertyLimits("LED 3 Power", ledMinBrightness_, ledMaxBrightness_);
    if (err != 0)
    {
        return DEVICE_ERR;
        LogMessage("Error with property set in state control");
    }

    pAct = new CPropertyAction(this, &ChrolisStateDevice::OnPowerChange);
    err = CreateIntegerProperty("LED 4 Power", led4Brightness_, false, pAct);
    SetPropertyLimits("LED 4 Power", ledMinBrightness_, ledMaxBrightness_);
    if (err != 0)
    {
        return DEVICE_ERR;
        LogMessage("Error with property set in state control");
    }

    pAct = new CPropertyAction(this, &ChrolisStateDevice::OnPowerChange);
    err = CreateIntegerProperty("LED 5 Power", led5Brightness_, false, pAct);
    SetPropertyLimits("LED 5 Power", ledMinBrightness_, ledMaxBrightness_);
    if (err != 0)
    {
        return DEVICE_ERR;
        LogMessage("Error with property set in state control");
    }

    pAct = new CPropertyAction(this, &ChrolisStateDevice::OnPowerChange);
    err = CreateIntegerProperty("LED 6 Power", led6Brightness_, false, pAct);
    SetPropertyLimits("LED 6 Power", ledMinBrightness_, ledMaxBrightness_);
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
    err = CreateIntegerProperty("LED Enable State 6", led6Brightness_, false, pAct);
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
        if (!pHub)
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
        std::ostringstream os;

        //temp state from last set used as fallback
        uint8_t currentLEDState = ((static_cast<uint8_t>(led1State_) << 0) | (static_cast<uint8_t>(led2State_) << 1) | (static_cast<uint8_t>(led3State_) << 2)
            | (static_cast<uint8_t>(led4State_) << 3) | (static_cast<uint8_t>(led5State_) << 4) | (static_cast<uint8_t>(led6State_) << 5));

        //Get the current instances for hub and chrolis
        //In the event of error do not set the property. Set old value. Updated values will be pulled from getters if possible
        ChrolisHub* pHub = static_cast<ChrolisHub*>(GetParentHub());
        if (!pHub)
        {
            LogMessage("Hub not available");
            os << currentLEDState;
            OnPropertyChanged(pProp->GetName().c_str(), os.str().c_str());
            return ERR_HUB_NOT_AVAILABLE;
        }
        ThorlabsChrolisDeviceWrapper* wrapperInstance = static_cast<ThorlabsChrolisDeviceWrapper*>(pHub->GetChrolisDeviceInstance());
        if (!wrapperInstance->IsDeviceConnected())
        {
            LogMessage("CHROLIS not available");
            os << currentLEDState;
            OnPropertyChanged(pProp->GetName().c_str(), os.str().c_str());
            return ERR_CHROLIS_NOT_AVAIL;
        }


        long val; // incoming value from user
        pProp->Get(val);
        if (val >= pow(2, numPos_) || val < 0)
        {
            LogMessage("Requested state out of bounds");
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

                os << currentLEDState;
                OnPropertyChanged(pProp->GetName().c_str(), os.str().c_str());
            }

            return err;
        }

        os << val;
        OnPropertyChanged(pProp->GetName().c_str(), os.str().c_str());

        return DEVICE_OK;
    }
    return DEVICE_OK;
}

int ChrolisStateDevice::OnEnableStateChange(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    ViPBoolean ledBeingControlled;
    int numFromName = -1;
    std::string searchString = pProp->GetName();
    std::regex regexp(R"([-+]?([0-9]*\.[0-9]+|[0-9]+))");
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
        if (!pHub)
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
        std::ostringstream os;

        ChrolisHub* pHub = static_cast<ChrolisHub*>(GetParentHub());
        if (!pHub)
        {
            LogMessage("Hub not available");
            os << *ledBeingControlled;
            OnPropertyChanged(pProp->GetName().c_str(), os.str().c_str());
            return ERR_HUB_NOT_AVAILABLE;
        }
        ThorlabsChrolisDeviceWrapper* wrapperInstance = static_cast<ThorlabsChrolisDeviceWrapper*>(pHub->GetChrolisDeviceInstance());
        if (!wrapperInstance->IsDeviceConnected())
        {
            LogMessage("CHROLIS not available");
            os << *ledBeingControlled;
            OnPropertyChanged(pProp->GetName().c_str(), os.str().c_str());
            return ERR_CHROLIS_NOT_AVAIL;
        }

        int err = wrapperInstance->SetSingleLEDEnableState(numFromName-1, (ViBoolean)val);
        if (err != 0)
        {
            LogMessage("Error Setting LED state");
            wrapperInstance->GetSingleLEDEnableState(numFromName - 1, *ledBeingControlled);
            os << *ledBeingControlled;
            OnPropertyChanged(pProp->GetName().c_str(), os.str().c_str());
            return err;
        }

        *ledBeingControlled = (ViBoolean)val;
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
    std::regex regexp(R"([-+]?([0-9]*\.[0-9]+|[0-9]+))");    
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
        ledBeingControlled = &led1Brightness_;
        break;
    case 2:
        ledBeingControlled = &led2Brightness_;
        break;
    case 3:
        ledBeingControlled = &led3Brightness_;
        break;
    case 4:
        ledBeingControlled = &led4Brightness_;
        break;
    case 5:
        ledBeingControlled = &led5Brightness_;
        break;
    case 6:
        ledBeingControlled = &led6Brightness_;
        break;
    default:
        LogMessage("Error selecting LED state");
        return DEVICE_ERR;
        break;
    }

    if (eAct == MM::BeforeGet)
    {
        ChrolisHub* pHub = static_cast<ChrolisHub*>(GetParentHub());
        if (!pHub)
        {
            LogMessage("Hub not available");
        }
        ThorlabsChrolisDeviceWrapper* wrapperInstance = static_cast<ThorlabsChrolisDeviceWrapper*>(pHub->GetChrolisDeviceInstance());
        if (!wrapperInstance->IsDeviceConnected())
        {
            LogMessage("CHROLIS not available");
        }
        if (wrapperInstance->GetSingleLEDBrightnessState(numFromName - 1, *ledBeingControlled) != 0)
        {
            LogMessage("Error getting info from chrolis");
        }
        pProp->Set((long)*ledBeingControlled);
    }
    else if (eAct == MM::AfterSet)
    {
        double val;
        pProp->Get(val);
        std::ostringstream os;

        ChrolisHub* pHub = static_cast<ChrolisHub*>(GetParentHub());
        if (!pHub)
        {
            LogMessage("Hub not available");
            os << *ledBeingControlled;
            OnPropertyChanged(pProp->GetName().c_str(), os.str().c_str());
            return ERR_HUB_NOT_AVAILABLE;
        }
        ThorlabsChrolisDeviceWrapper* wrapperInstance = static_cast<ThorlabsChrolisDeviceWrapper*>(pHub->GetChrolisDeviceInstance());
        if (!wrapperInstance->IsDeviceConnected())
        {
            LogMessage("CHROLIS not available");
            os << *ledBeingControlled;
            OnPropertyChanged(pProp->GetName().c_str(), os.str().c_str());
            return ERR_CHROLIS_NOT_AVAIL;
        }

        int err = wrapperInstance->SetSingleLEDBrightnessState(numFromName - 1, (ViUInt16)val);
        if (err != 0)
        {
            LogMessage("Error Setting LED state");
            wrapperInstance->GetSingleLEDBrightnessState(numFromName - 1, *ledBeingControlled);
            os << *ledBeingControlled;
            OnPropertyChanged(pProp->GetName().c_str(), os.str().c_str());
            return err;
        }

        *ledBeingControlled = (ViUInt16)val;
        os << *ledBeingControlled;
        OnPropertyChanged(pProp->GetName().c_str(), os.str().c_str());
        return DEVICE_OK;
    }
    return DEVICE_OK;
}

