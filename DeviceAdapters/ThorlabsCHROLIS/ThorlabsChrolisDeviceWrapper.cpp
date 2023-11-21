#include "ThorlabsChrolis.h"
#include <string>

ThorlabsChrolisDeviceWrapper::ThorlabsChrolisDeviceWrapper()
{}

ThorlabsChrolisDeviceWrapper::~ThorlabsChrolisDeviceWrapper()
{}

int ThorlabsChrolisDeviceWrapper::GetAvailableSerialNumbers(std::vector<std::string> &serialNumbers)
{
    int err = 0;
    ViUInt32 numDevices;
    CDeviceUtils::SleepMs(2000);
    err = TL6WL_findRsrc(NULL, &numDevices);
    if (err != 0)
    {
        return err;
    }
    if (numDevices == 0)
    {
        return ERR_NO_AVAIL_DEVICES;
    }
    for (ViUInt32 i = 0; i < numDevices; i++)
    {
        err = TL6WL_getRsrcInfo(NULL, i, deviceName_, serialNumber_, manufacturerName_, &deviceInUse_);
        if (err != 0)
        {
            return err;
        }
        serialNumbers.push_back(serialNumber_);
    }
    return DEVICE_OK;
}

int ThorlabsChrolisDeviceWrapper::InitializeDevice(std::string serialNumber)
{
    int err = 0;
    ViUInt32 numDevices;
    CDeviceUtils::SleepMs(2000);
    err = TL6WL_findRsrc(NULL, &numDevices);
    if (err != 0)
    {
        return err;
    }
    if (numDevices == 0)
    {
        return ERR_NO_AVAIL_DEVICES;
    }

    ViChar resource[512] = "";
    if(serialNumber.compare("") == 0 || serialNumber.compare("DEFAULT") == 0)
    {
        err = TL6WL_getRsrcName(NULL, 0, resource);
        if (err != 0)
        {
            return err;
        }
        err = TL6WL_getRsrcInfo(NULL, 0, deviceName_, serialNumber_, manufacturerName_, &deviceInUse_);
        if (err != 0)
        {
            return err;
        }
    }
    else
    {
        bool found = false;
        for (ViUInt32 i = 0; i < numDevices; i++)
        {
            err = TL6WL_getRsrcName(NULL, i, resource);
            if (err != 0)
            {
                return err;
            }
            err = TL6WL_getRsrcInfo(NULL, i, deviceName_, serialNumber_, manufacturerName_, &deviceInUse_);
            if (err != 0)
            {
                return err;
            }
            if (strcmp((char*)serialNumber_, serialNumber.c_str()))
            {
                found = true;
                break;
            }
        }
        if(!found)
        { 
            return err;
        }
    }
    err = TL6WL_init(resource, false, false, &deviceHandle_);
    if (err != 0)
    {
        return err;
    }
    deviceConnected_ = true;

    for (int i = 0; i < NUM_LEDS; i++)
    {
        err = TL6WL_readLED_HeadPeakWL(deviceHandle_, static_cast<ViUInt8>(i + 1), &ledWavelengths_[i]);
        if (err != 0)
        {
            return err;
        }
    }
    VerifyLEDStates();

	return DEVICE_OK;
}

int ThorlabsChrolisDeviceWrapper::ShutdownDevice()
{
    if (deviceHandle_ != ViSession(-1))
    {
        std::lock_guard<std::mutex> lock(instanceMutex_);
        auto err = TL6WL_close(deviceHandle_);
        if (err != 0)
        {
            return err;
        }
        deviceHandle_ = ViSession(-1);
    }
    deviceConnected_ = false;
	return DEVICE_OK;
}

bool ThorlabsChrolisDeviceWrapper::IsDeviceConnected()
{
    std::lock_guard<std::mutex> lock(instanceMutex_);
    return deviceConnected_;
}

int ThorlabsChrolisDeviceWrapper::GetSerialNumber(ViChar* serialNumber)
{
    std::lock_guard<std::mutex> lock(instanceMutex_);
    if (deviceConnected_)
    {
        strcpy(serialNumber, serialNumber_);
    }
    else
    {
        strcpy(serialNumber, "Not Initialized");
    }
    return DEVICE_OK;
}

int ThorlabsChrolisDeviceWrapper::GetManufacturerName(ViChar* manfName)
{
    std::lock_guard<std::mutex> lock(instanceMutex_);
    if (deviceConnected_)
    {
        strcpy(manfName, manufacturerName_);
    }
    else
    {
        strcpy(manfName, "Not Initialized");
    }
    return DEVICE_OK;
}

int ThorlabsChrolisDeviceWrapper::GetDeviceStatus(ViUInt32& status)
{
    std::lock_guard<std::mutex> lock(instanceMutex_);
    if (!deviceConnected_)
    {
        status = 0;
        return ERR_CHROLIS_NOT_AVAIL;
    }
    auto err = TL6WL_getBoxStatus(deviceHandle_, &status);
    if (err != 0)
    {
        return err;
    }
    return DEVICE_OK;
}

int ThorlabsChrolisDeviceWrapper::GetShutterState(bool& open)
{
    std::lock_guard<std::mutex> lock(instanceMutex_);

    if (!deviceConnected_)
    {
        open = false;
        return ERR_CHROLIS_NOT_AVAIL;
    }
    open = masterSwitchState_;
    return DEVICE_OK;
}

int ThorlabsChrolisDeviceWrapper::GetLEDWavelengths(std::array<ViUInt16, NUM_LEDS> &wavelengths)
{
    std::lock_guard<std::mutex> lock(instanceMutex_);

    if (!deviceConnected_)
    {
        wavelengths.fill(false);
        return ERR_CHROLIS_NOT_AVAIL;
    }
    wavelengths = ledWavelengths_;

    return DEVICE_OK;
}

int ThorlabsChrolisDeviceWrapper::GetSingleLEDEnableState(int led, ViBoolean& state)
{
    std::lock_guard<std::mutex> lock(instanceMutex_);

    if (led < NUM_LEDS && led >= 0)
    {
        if (!deviceConnected_)
        {
            return ERR_CHROLIS_NOT_AVAIL;
        }
        state = savedEnabledStates_[led];
    }
    else
    {
        return ERR_PARAM_NOT_VALID;
    }

    return DEVICE_OK;
}

int ThorlabsChrolisDeviceWrapper::GetLEDEnableStates(std::array<ViBoolean, NUM_LEDS> &states)
{
    std::lock_guard<std::mutex> lock(instanceMutex_);

    if (!deviceConnected_)
    {
        states.fill(false);
        return ERR_CHROLIS_NOT_AVAIL;
    }
	states = savedEnabledStates_;
    return DEVICE_OK;
}

int ThorlabsChrolisDeviceWrapper::GetSingleLEDBrightnessState(int led, ViUInt16& state)
{
    std::lock_guard<std::mutex> lock(instanceMutex_);

    if (led < NUM_LEDS && led >= 0)
    {
        if (!deviceConnected_)
        {
            return ERR_CHROLIS_NOT_AVAIL;
        }
        state = savedBrightnessStates_[led];
    }
    else
    {
        return ERR_PARAM_NOT_VALID;
    }

    return DEVICE_OK;
}

int ThorlabsChrolisDeviceWrapper::GetLEDBrightnessStates(std::array<ViUInt16, NUM_LEDS> &states)
{
    std::lock_guard<std::mutex> lock(instanceMutex_);

    if (!deviceConnected_)
    {
        states.fill(0);
        return ERR_CHROLIS_NOT_AVAIL;
    }
    states = savedBrightnessStates_;
    return DEVICE_OK;
}

int ThorlabsChrolisDeviceWrapper::SetShutterState(bool open)
{
    int err = DEVICE_OK;
    std::lock_guard<std::mutex> lock(instanceMutex_);
    if (!open)
    {
        std::array<ViBoolean, NUM_LEDS> tempEnableStates;
        err = TL6WL_setLED_HeadPowerStates(deviceHandle_, false, false, false, false, false, false);
        if (err != 0)
        {
            return err;
        }
        err = TL6WL_getLED_HeadPowerStates(deviceHandle_, &tempEnableStates[0], &tempEnableStates[1], &tempEnableStates[2], &tempEnableStates[3], &tempEnableStates[4], &tempEnableStates[5]);
        for (int i = 0; i < NUM_LEDS; i++)
        {
            if (tempEnableStates[i])
            {
                //led(s) failed to turn off handle error in mm device
                err = ERR_IMPROPER_SET;
                masterSwitchState_ = true;
                break;
            }
        }
        masterSwitchState_ = false;
    }
    else
    {
        err = TL6WL_setLED_HeadPowerStates(deviceHandle_, savedEnabledStates_[0], savedEnabledStates_[1], savedEnabledStates_[2], savedEnabledStates_[3], savedEnabledStates_[4], savedEnabledStates_[5]);
        if (err != 0)
        {
            return err;
        }
        masterSwitchState_ = true;
        if (!VerifyLEDEnableStates())
        {
            err = ERR_IMPROPER_SET;
        }

    }
    return err;
}

int ThorlabsChrolisDeviceWrapper::SetSingleLEDEnableState(int LED, ViBoolean state)
{
    if (LED < NUM_LEDS && LED >= 0)
    {
        std::lock_guard<std::mutex> lock(instanceMutex_);

        if (!deviceConnected_)
        {
            return ERR_CHROLIS_NOT_AVAIL;
        }
        int err = 0;
        savedEnabledStates_[LED] = state;

        if (masterSwitchState_)
        {
            err = TL6WL_setLED_HeadPowerStates(deviceHandle_, savedEnabledStates_[0], savedEnabledStates_[1], savedEnabledStates_[2],
                savedEnabledStates_[3], savedEnabledStates_[4], savedEnabledStates_[5]);
            if (err != 0)
            {
                VerifyLEDEnableStates(); // try to synch values if possible
                return err;
            }
            if (!VerifyLEDEnableStates())
            {
                err = ERR_IMPROPER_SET;
            }
        }
        return err;
    }
    else
    {
        return ERR_PARAM_NOT_VALID;
    }
}

int ThorlabsChrolisDeviceWrapper::SetLEDEnableStates(std::array<ViBoolean, NUM_LEDS> states)
{
    std::lock_guard<std::mutex> lock(instanceMutex_);

    if (!deviceConnected_)
    {
        return ERR_CHROLIS_NOT_AVAIL;
    }

    int err = 0;
    for (int i = 0; i < NUM_LEDS; i++)
    {
        savedEnabledStates_[i] = states[i];
    }
    if (masterSwitchState_)
    {
        err = TL6WL_setLED_HeadPowerStates(deviceHandle_, savedEnabledStates_[0], savedEnabledStates_[1], savedEnabledStates_[2],
            savedEnabledStates_[3], savedEnabledStates_[4], savedEnabledStates_[5]);
        if (err != 0)
        {
            VerifyLEDEnableStates();
            return err;
        }
        if (!VerifyLEDEnableStates())
        {
            err = ERR_IMPROPER_SET;
        }
    }

	return err;
}

int ThorlabsChrolisDeviceWrapper::SetSingleLEDBrightnessState(int LED, ViUInt16 state)
{
    if (LED < NUM_LEDS && LED >= 0)
    {
        std::lock_guard<std::mutex> lock(instanceMutex_);

        if (!deviceConnected_)
        {
            return ERR_CHROLIS_NOT_AVAIL;
        }

        int err = 0;
        savedBrightnessStates_[LED] = state;
        err = TL6WL_setLED_HeadBrightness(deviceHandle_, savedBrightnessStates_[0], savedBrightnessStates_[1], savedBrightnessStates_[2], 
            savedBrightnessStates_[3], savedBrightnessStates_[4], savedBrightnessStates_[5]);
        if (err != 0)
        {
            VerifyLEDPowerStates();
            return err;
        }
        if (!VerifyLEDPowerStates())
        {
            err = ERR_IMPROPER_SET;
        }
        return err;
    }
    else
    {
        return ERR_PARAM_NOT_VALID;
    }
}

int ThorlabsChrolisDeviceWrapper::SetLEDBrightnessStates(std::array<ViUInt16, NUM_LEDS> states)
{
    std::lock_guard<std::mutex> lock(instanceMutex_);

    if (!deviceConnected_)
    {
        return ERR_CHROLIS_NOT_AVAIL;
    }

    int i;
    for (i = 0; i < NUM_LEDS; i++)
    {
        savedBrightnessStates_[i] = states[i];
    }
    int err = 0;
    err = TL6WL_setLED_HeadBrightness(deviceHandle_, savedBrightnessStates_[0], savedBrightnessStates_[1], savedBrightnessStates_[2],
        savedBrightnessStates_[3], savedBrightnessStates_[4], savedBrightnessStates_[5]);
    if (err != 0)
    {
        VerifyLEDPowerStates();
        return err;
    }
    if (!VerifyLEDPowerStates())
    {
        err = ERR_IMPROPER_SET;
    }
    return err;

	return DEVICE_OK;
}

bool ThorlabsChrolisDeviceWrapper::VerifyLEDEnableStatesWithLock()
{
    std::lock_guard<std::mutex> lock(instanceMutex_);
    return VerifyLEDEnableStates();
}

bool ThorlabsChrolisDeviceWrapper::VerifyLEDStates()
{
    if (!deviceConnected_)
    {
        return false;
    }

    std::array<ViUInt16, NUM_LEDS> tempPowerStates{};
    std::array<ViBoolean , NUM_LEDS>tempEnableStates{};

    TL6WL_getLED_HeadPowerStates(deviceHandle_, &tempEnableStates[0], &tempEnableStates[1], &tempEnableStates[2], &tempEnableStates[3], &tempEnableStates[4], &tempEnableStates[5]);
    TL6WL_getLED_HeadBrightness(deviceHandle_, &tempPowerStates[0], &tempPowerStates[1], &tempPowerStates[2], &tempPowerStates[3], &tempPowerStates[4], &tempPowerStates[5]);

    bool stateCorrect = (tempEnableStates == savedEnabledStates_ &&
        tempPowerStates == savedBrightnessStates_);
    savedEnabledStates_ = tempEnableStates;
    savedBrightnessStates_ = tempPowerStates;
    return stateCorrect;
}

bool ThorlabsChrolisDeviceWrapper::VerifyLEDEnableStates()
{
    if (!deviceConnected_)
    {
        return false;
    }

    std::array<ViBoolean, NUM_LEDS> tempEnableStates{};

    TL6WL_getLED_HeadPowerStates(deviceHandle_, &tempEnableStates[0], &tempEnableStates[1], &tempEnableStates[2], &tempEnableStates[3], &tempEnableStates[4], &tempEnableStates[5]);

    bool stateCorrect = (tempEnableStates == savedEnabledStates_);
    savedEnabledStates_ = tempEnableStates;
    return stateCorrect;
}

bool ThorlabsChrolisDeviceWrapper::VerifyLEDPowerStates()
{
    if (!deviceConnected_)
    {
        return false;
    }

    std::array<ViUInt16, NUM_LEDS> tempPowerStates{};

    TL6WL_getLED_HeadBrightness(deviceHandle_, &tempPowerStates[0], &tempPowerStates[1], &tempPowerStates[2], &tempPowerStates[3], &tempPowerStates[4], &tempPowerStates[5]);

    bool stateCorrect = (tempPowerStates == savedBrightnessStates_);
    savedBrightnessStates_ = tempPowerStates;
    return stateCorrect;
}