#include "ThorlabsChrolis.h"
#include "ModuleInterface.h"

#include <string>

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

static std::map<int, std::string> ErrorMessages()
{
    return {
        {ERR_HUB_NOT_AVAILABLE, "Hub is not available"},
        {ERR_CHROLIS_NOT_AVAIL, "CHROLIS Device is not available"},
        {ERR_IMPROPER_SET, "Error setting property value. Value will be reset"},
        {ERR_PARAM_NOT_VALID, "Value passed to property was out of bounds."},
        {ERR_NO_AVAIL_DEVICES, "No available devices were found on the system."},
        {ERR_INSUF_INFO, "Insufficient location information of the device or the resource is not present on the system"},
        {ERR_UNKOWN_HW_STATE, "Unknown Hardware State"},
        {ERR_VAL_OVERFLOW, "Parameter Value Overflow"},
        {INSTR_RUNTIME_ERROR, "CHROLIS Instrument Runtime Error"},
        {INSTR_REM_INTER_ERROR, "CHROLIS Instrument Internal Error"},
        {INSTR_AUTHENTICATION_ERROR, "CHROLIS Instrument Authentication Error"},
        {INSTR_PARAM_ERROR, "CHROLIS Invalid Parameter Error"},
        {INSTR_INTERNAL_TX_ERR, "CHROLIS Instrument Internal Command Sending Error"},
        {INSTR_INTERNAL_RX_ERR, "CHROLIS Instrument Internal Command Receiving Error"},
        {INSTR_INVAL_MODE_ERR, "CHROLIS Instrument Invalid Mode Error"},
        {INSTR_SERVICE_ERR, "CHROLIS Instrument Service Error"}
    };
}

static std::uint8_t EncodeLEDStatesInBits(std::array<ViBoolean, NUM_LEDS> states) {
    std::uint8_t ret = 0;
    for (auto i = 0; i < NUM_LEDS; i++)
    {
        bool state = states[i] != VI_FALSE; // Ensure 0 or 1, just in case
        ret |= static_cast<std::uint8_t>(static_cast<int>(state) << i);
    }
    return ret;
}

static std::array<ViBoolean, NUM_LEDS> DecodeLEDStatesFromBits(std::uint8_t bits) {
    std::array<ViBoolean, NUM_LEDS> ret{};
    for (auto i = 0; i < NUM_LEDS; i++)
    {
        ret[i] = ((bits & (1 << i)) != 0);
    }
    return ret;
}

static std::string DeviceStatusAsString(std::uint32_t status)
{
    if (status == 0)
    {
        return "No Error";
    }

    static const std::array<std::string, 7> bitToMsg{
        "Box is Open",
        "LLG not Connected",
        "Interlock is Open",
        "Using Default Adjustment",
        "Box Overheated",
        "LED Overheated",
        "Invalid Box Setup",
    };

    std::string message;
    for (int i = 0; i < bitToMsg.size(); i++)
    {
        if ((status & (1 << i)) != 0)
        {
            if (!message.empty())
            {
                message += ", ";
            }
            message += bitToMsg[i];
        }
    }
    if (!message.empty())
    {
        return message;
    }
    return "Unknown Status";
}

//Hub Methods
ChrolisHub::ChrolisHub()
{
    for (const auto& errMessage : ErrorMessages())
    {
        SetErrorText(errMessage.first, errMessage.second.c_str());
    }

    std::vector<std::string> serialNumbers;
    ChrolisDevice.GetAvailableSerialNumbers(serialNumbers);

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
    char buf[MM::MaxStrLength];
    (void)GetProperty("Serial Number", buf);

    int err = ChrolisDevice.InitializeDevice(buf);
    if (err != 0)
    {
        LogMessage("Error in CHROLIS Initialization");
        return err;
    }

    ViChar sNum[TL6WL_LONG_STRING_SIZE];
    ChrolisDevice.GetSerialNumber(sNum);
    err = CreateStringProperty("Device Serial Number", sNum, true);
    if (err != 0)
    {
        LogMessage("Error with property set in hub initialize");
        return DEVICE_ERR;
    }

    ViChar manfName[TL6WL_LONG_STRING_SIZE];
    ChrolisDevice.GetManufacturerName(manfName);
    err = CreateStringProperty("Manufacturer Name", manfName, true);
    if (err != 0)
    {
        LogMessage("Error with property set in hub initialize");
        return DEVICE_ERR;
    }

    std::string wavelengthList = "";
    std::array<ViUInt16, NUM_LEDS> wavelengths;
    err = ChrolisDevice.GetLEDWavelengths(wavelengths);
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

    err = CreateStringProperty("Device Status", deviceStatus_.c_str(), true,
        new CPropertyAction(this, &ChrolisHub::OnDeviceStatus));
    if (err != 0)
    {
        LogMessage("Error with property set in hub initialize");
        return DEVICE_ERR;
    }

    updateThread_ = std::thread(&ChrolisHub::StatusChangedPollingThread, this);
    return DEVICE_OK;
}

int ChrolisHub::Shutdown()
{
    {
        std::lock_guard<std::mutex> lock(pollingMutex_);
		pollingStopRequested_ = true;
    }

    if (updateThread_.joinable())
    {
        updateThread_.join();
    }

    if (ChrolisDevice.IsDeviceConnected())
    {
        auto err = ChrolisDevice.ShutdownDevice();
        if (err != 0)
        {
            LogMessage("Error shutting down device");
            return DEVICE_ERR;
        }
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

int ChrolisHub::OnDeviceStatus(MM::PropertyBase *pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        std::string status;
        {
            std::lock_guard<std::mutex> lock(pollingMutex_);
            status = deviceStatus_;
        }
        pProp->Set(status.c_str());
    }
    return DEVICE_OK;
}

void ChrolisHub::StatusChangedPollingThread()
{
    for (;;)
    {
        {
            std::lock_guard<std::mutex> lock(pollingMutex_);
            if (pollingStopRequested_)
            {
                return;
            }
        }

        if (ChrolisDevice.IsDeviceConnected())
        {
            ViUInt32 tempStatus = 0;
            auto err = ChrolisDevice.GetDeviceStatus(tempStatus);
            if (err != 0)
            {
                LogMessage("Error Getting Status");
                return;
            }
            const auto curStatus = currentDeviceStatusCode_;
            const bool statusChanged = curStatus != tempStatus;
            currentDeviceStatusCode_ = tempStatus;
            if (statusChanged)
            {
                if (curStatus != 0)
                {
                    std::array<ViBoolean, NUM_LEDS> tempEnableStates{};
                    ChrolisDevice.VerifyLEDEnableStatesWithLock();
                    if (ChrolisDevice.GetLEDEnableStates(tempEnableStates) != 0)
                    {
                        LogMessage("Error getting info from chrolis");
                    }
                    else
                    {
                        std::lock_guard<std::mutex> lock(pollingMutex_);
                        // Usually it is not a good idea to call arbitrary
                        // functions with a mutex held, but in this case we
                        // need to ensure that the callback remains valid.
                        if (stateBitsCallback_)
                        {
                            stateBitsCallback_(EncodeLEDStatesInBits(tempEnableStates));
                        }
                        if (stateCallback_)
                        {
                            for (int i = 0; i < NUM_LEDS; i++)
                            {
                                stateCallback_(i, tempEnableStates[i]);
                            }
                        }
                    }
                }
            }

            std::string message = DeviceStatusAsString(curStatus);
            {
                std::lock_guard<std::mutex> lock(pollingMutex_);
                deviceStatus_ = message;
            }
            OnPropertyChanged("Device Status", message.c_str());
        }
        Sleep(500);
    }
}

void ChrolisHub::SetStateBitsCallback(std::function<void(std::uint8_t)> function)
{
    std::lock_guard<std::mutex> lock(pollingMutex_);
    stateBitsCallback_ = function;
}

void ChrolisHub::SetStateCallback(std::function<void(int, ViBoolean)> function)
{
    std::lock_guard<std::mutex> lock(pollingMutex_);
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
    if (!pHub)
    {
        LogMessage("No Hub");
        return ERR_HUB_NOT_AVAILABLE;
    }

    if (pHub->ChrolisDevice.IsDeviceConnected())
    {
        auto err = pHub->ChrolisDevice.SetShutterState(false);
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
    if (!pHub)
    {
        LogMessage("Hub not available");
        return ERR_HUB_NOT_AVAILABLE;
    }
    if (!pHub->ChrolisDevice.IsDeviceConnected())
    {
        LogMessage("CHROLIS not available");
        return ERR_CHROLIS_NOT_AVAIL;
    }
    auto err = pHub->ChrolisDevice.SetShutterState(open);
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
    if (!pHub->ChrolisDevice.IsDeviceConnected())
    {
        LogMessage("CHROLIS not available");
        return ERR_CHROLIS_NOT_AVAIL;
    }
    pHub->ChrolisDevice.GetShutterState(open);
    return DEVICE_OK;
}


//Chrolis State Device Methods
ChrolisStateDevice::ChrolisStateDevice() 
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
    if (!pHub)
    {
        LogMessage("Hub not available");
        return ERR_HUB_NOT_AVAILABLE;
    }

    pHub->SetStateBitsCallback([this](std::uint8_t bits)
        {
            OnPropertyChanged(MM::g_Keyword_State, std::to_string(bits).c_str());
        });

    pHub->SetStateCallback([this](int ledIndex, ViBoolean state)
        {
            int ledNum = ledIndex + 1;
            OnPropertyChanged(("LED Enable State " + std::to_string(ledNum)).c_str(),
                std::to_string(state).c_str());
        });

    // create default positions and labels
    const int bufSize = 1024;
    char buf[bufSize];
    for (long i = 0; i < NUM_LEDS; i++)
    {
        snprintf(buf, bufSize, "-%ld", i);
        SetPositionLabel(i, buf);
    }
    int err;
    uint32_t tmpLedState = 0;
    if (pHub->ChrolisDevice.IsDeviceConnected())
    {
        err = pHub->ChrolisDevice.GetLEDEnableStates(ledStates_);
        err = pHub->ChrolisDevice.GetLEDBrightnessStates(ledBrightnesses_);
        tmpLedState = EncodeLEDStatesInBits(ledStates_);
    }

    //State Property
    CPropertyAction* pAct = new CPropertyAction(this, &ChrolisStateDevice::OnState);
    err = CreateIntegerProperty(MM::g_Keyword_State, tmpLedState, false, pAct);
    if (err != DEVICE_OK)
        return err;


    ////Properties for power control
    for (int i = 0; i < NUM_LEDS; i++)
    {
		auto* pActEx = new CPropertyActionEx(this, &ChrolisStateDevice::OnPowerChange, i);
        std::string propName = "LED " + std::to_string(i + 1) + " Power";
		err = CreateIntegerProperty(propName.c_str(), ledBrightnesses_[i], false, pActEx);
		if (err != 0)
		{
			LogMessage("Error with property set in power control");
			return DEVICE_ERR;
		}
		SetPropertyLimits(propName.c_str(), ledMinBrightness_, ledMaxBrightness_);
    }


    //Properties for state control
    for (int i = 0; i < NUM_LEDS; i++)
    {
        auto* pActEx = new CPropertyActionEx(this, &ChrolisStateDevice::OnEnableStateChange, i);
        std::string propName = "LED Enable State " + std::to_string(i + 1);
        err = CreateIntegerProperty(propName.c_str(), ledStates_[i], false, pActEx);
        if (err != 0)
        {
            LogMessage("Error with property set in state control");
            return DEVICE_ERR;
        }
        SetPropertyLimits(propName.c_str(), 0, 1);
    }

    return DEVICE_OK;
}

int ChrolisStateDevice::Shutdown()
{
    ChrolisHub* pHub = static_cast<ChrolisHub*>(GetParentHub());
    if (pHub)
    {
        pHub->SetStateCallback([](auto, auto) {});
        pHub->SetStateBitsCallback([](auto) {});
    }
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
            return ERR_HUB_NOT_AVAILABLE;
        }
        if (!pHub->ChrolisDevice.IsDeviceConnected())
        {
            LogMessage("CHROLIS not available");
        }
        if (pHub->ChrolisDevice.GetLEDEnableStates(ledStates_) != 0)
        {
            LogMessage("Error getting info from chrolis");
        }

        pProp->Set(static_cast<long>(EncodeLEDStatesInBits(ledStates_)));
    }
    else if (eAct == MM::AfterSet)
    {
        std::ostringstream os;

        //temp state from last set used as fallback
        uint8_t currentLEDState = EncodeLEDStatesInBits(ledStates_);

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
        if (!pHub->ChrolisDevice.IsDeviceConnected())
        {
            LogMessage("CHROLIS not available");
            os << currentLEDState;
            OnPropertyChanged(pProp->GetName().c_str(), os.str().c_str());
            return ERR_CHROLIS_NOT_AVAIL;
        }


        long val; // incoming value from user
        pProp->Get(val);
        if (val >= pow(2, NUM_LEDS) || val < 0)
        {
            LogMessage("Requested state out of bounds");
            os << currentLEDState;
            OnPropertyChanged(pProp->GetName().c_str(), os.str().c_str());
            return ERR_PARAM_NOT_VALID;
        }

        std::array<ViBoolean, NUM_LEDS> newStates =
            DecodeLEDStatesFromBits(static_cast<std::uint8_t>(val));
        int err = pHub->ChrolisDevice.SetLEDEnableStates(newStates);
        if (err != 0)
        {
            //Do not set the property in the case of this error. Let the property change handle it. 
            //This will cover error where LED failed to set but chrolis is still ok
            LogMessage("Error Setting LED state");
            if (err != ERR_CHROLIS_NOT_AVAIL)
            {
                if (pHub->ChrolisDevice.GetLEDEnableStates(ledStates_) != 0)
                {
                    LogMessage("Error getting info from chrolis");
                }
                currentLEDState = EncodeLEDStatesInBits(ledStates_);

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

int ChrolisStateDevice::OnEnableStateChange(MM::PropertyBase* pProp, MM::ActionType eAct, long ledIndex)
{
    ViPBoolean ledBeingControlled = &ledStates_[ledIndex];

    if (eAct == MM::BeforeGet)
    {
        ChrolisHub* pHub = static_cast<ChrolisHub*>(GetParentHub());
        if (!pHub)
        {
            LogMessage("Hub not available");
            return ERR_HUB_NOT_AVAILABLE;
        }
        if (!pHub->ChrolisDevice.IsDeviceConnected())
        {
            LogMessage("CHROLIS not available");
        }
        if (pHub->ChrolisDevice.GetSingleLEDEnableState(ledIndex, *ledBeingControlled) != 0)
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
        if (!pHub->ChrolisDevice.IsDeviceConnected())
        {
            LogMessage("CHROLIS not available");
            os << *ledBeingControlled;
            OnPropertyChanged(pProp->GetName().c_str(), os.str().c_str());
            return ERR_CHROLIS_NOT_AVAIL;
        }

        int err = pHub->ChrolisDevice.SetSingleLEDEnableState(ledIndex, (ViBoolean)val);
        if (err != 0)
        {
            LogMessage("Error Setting LED state");
            pHub->ChrolisDevice.GetSingleLEDEnableState(ledIndex, *ledBeingControlled);
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

int ChrolisStateDevice::OnPowerChange(MM::PropertyBase* pProp, MM::ActionType eAct, long ledIndex)
{
    ViPUInt16 ledBeingControlled = &ledBrightnesses_[ledIndex];

    if (eAct == MM::BeforeGet)
    {
        ChrolisHub* pHub = static_cast<ChrolisHub*>(GetParentHub());
        if (!pHub)
        {
            LogMessage("Hub not available");
            return ERR_HUB_NOT_AVAILABLE;
        }
        if (!pHub->ChrolisDevice.IsDeviceConnected())
        {
            LogMessage("CHROLIS not available");
        }
        if (pHub->ChrolisDevice.GetSingleLEDBrightnessState(ledIndex, *ledBeingControlled) != 0)
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
        if (!pHub->ChrolisDevice.IsDeviceConnected())
        {
            LogMessage("CHROLIS not available");
            os << *ledBeingControlled;
            OnPropertyChanged(pProp->GetName().c_str(), os.str().c_str());
            return ERR_CHROLIS_NOT_AVAIL;
        }

        int err = pHub->ChrolisDevice.SetSingleLEDBrightnessState(ledIndex, (ViUInt16)val);
        if (err != 0)
        {
            LogMessage("Error Setting LED state");
            pHub->ChrolisDevice.GetSingleLEDBrightnessState(ledIndex, *ledBeingControlled);
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

