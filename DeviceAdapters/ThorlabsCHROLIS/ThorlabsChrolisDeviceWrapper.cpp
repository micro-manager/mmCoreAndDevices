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
    for (int i = 0; i < numDevices; i++)
    {
        err = TL6WL_getRsrcInfo(NULL, i, deviceName_, serialNumber_, manufacturerName_, &deviceInUse_);
        if (err != 0)
        {
            return err;
        }
        serialNumbers.push_back(serialNumber_);
    }
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
        int i = 0;
        for (i = 0; i < numDevices; i++)
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

    for (int i = 0; i < numLEDs_; i++)
    {
        err = TL6WL_readLED_HeadPeakWL(deviceHandle_, i+1, &ledWavelengths_[i]);
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
    if (deviceHandle_ != -1)
    {
        std::lock_guard<std::mutex> lock(instanceMutex_);
        auto err = TL6WL_close(deviceHandle_);
        if (err != 0)
        {
            return err;
        }
        deviceHandle_ = -1;
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

int ThorlabsChrolisDeviceWrapper::GetLEDWavelengths(ViUInt16(&wavelengths)[6])
{
    std::lock_guard<std::mutex> lock(instanceMutex_);

    if (!deviceConnected_)
    {
        for (int i = 0; i < 6; i++)
        {
            wavelengths[i] = 0;
        }
        return ERR_CHROLIS_NOT_AVAIL;
    }
    wavelengths[0] = ledWavelengths_[0];
    wavelengths[1] = ledWavelengths_[1];
    wavelengths[2] = ledWavelengths_[2];
    wavelengths[3] = ledWavelengths_[3];
    wavelengths[4] = ledWavelengths_[4];
    wavelengths[5] = ledWavelengths_[5];

    return DEVICE_OK;
}

int ThorlabsChrolisDeviceWrapper::GetSingleLEDEnableState(int led, ViBoolean& state)
{
    std::lock_guard<std::mutex> lock(instanceMutex_);

    if (led < numLEDs_ && led >= 0)
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

int ThorlabsChrolisDeviceWrapper::GetLEDEnableStates(ViBoolean(&states)[6])
{
    std::lock_guard<std::mutex> lock(instanceMutex_);

    if (!deviceConnected_)
    {
        for (int i = 0; i < 6; i++)
        {
            states[i] = false;
        }
        return ERR_CHROLIS_NOT_AVAIL;
    }
    for (int i = 0; i < 6; i++)
    {
        states[i] = savedEnabledStates_[i];
    }
    return DEVICE_OK;
}

int ThorlabsChrolisDeviceWrapper::GetLEDEnableStates(ViBoolean& led1State, ViBoolean& led2State, ViBoolean& led3State, ViBoolean& led4State, ViBoolean& led5State, ViBoolean& led6State)
{
    std::lock_guard<std::mutex> lock(instanceMutex_);

    if (!deviceConnected_)
    {
        led1State = false;
        led2State = false;
        led3State = false;
        led4State = false;
        led5State = false;
        led6State = false;
        return ERR_CHROLIS_NOT_AVAIL;
    }
    led1State = savedEnabledStates_[0];
    led2State = savedEnabledStates_[1];
    led3State = savedEnabledStates_[2];
    led4State = savedEnabledStates_[3];
    led5State = savedEnabledStates_[4];
    led6State = savedEnabledStates_[5];

    return DEVICE_OK;
}

int ThorlabsChrolisDeviceWrapper::GetSingleLEDBrightnessState(int led, ViUInt16& state)
{
    std::lock_guard<std::mutex> lock(instanceMutex_);

    if (led < numLEDs_ && led >= 0)
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

int ThorlabsChrolisDeviceWrapper::GetLEDBrightnessStates(ViUInt16(&states)[6])
{
    std::lock_guard<std::mutex> lock(instanceMutex_);

    if (!deviceConnected_)
    {
        for (int i = 0; i < 6; i++)
        {
            states[i] = 0;
        }
        return ERR_CHROLIS_NOT_AVAIL;
    }
    for (int i = 0; i < 6; i++)
    {
        states[i] = savedBrightnessStates_[i];
    }
    return DEVICE_OK;
}

int ThorlabsChrolisDeviceWrapper::GetLEDBrightnessStates(ViUInt16& led1Brightness, ViUInt16& led2Brightness, ViUInt16& led3Brightness, ViUInt16& led4Brightness, ViUInt16& led5Brightness, ViUInt16& led6Brightness)
{
    std::lock_guard<std::mutex> lock(instanceMutex_);

    if (!deviceConnected_)
    {
        led1Brightness = 0;
        led2Brightness = 0;
        led3Brightness = 0;
        led4Brightness = 0;
        led5Brightness = 0;
        led6Brightness = 0;
        return ERR_CHROLIS_NOT_AVAIL;
    }
    led1Brightness = savedBrightnessStates_[0];
    led2Brightness = savedBrightnessStates_[1];
    led3Brightness = savedBrightnessStates_[2];
    led4Brightness = savedBrightnessStates_[2];
    led5Brightness = savedBrightnessStates_[4];
    led6Brightness = savedBrightnessStates_[5];

    return DEVICE_OK;
}

int ThorlabsChrolisDeviceWrapper::SetShutterState(bool open)
{
    int err = DEVICE_OK;
    std::lock_guard<std::mutex> lock(instanceMutex_);
    if (!open)
    {
        ViBoolean tempEnableStates[6];
        err = TL6WL_setLED_HeadPowerStates(deviceHandle_, false, false, false, false, false, false);
        if (err != 0)
        {
            return err;
        }
        err = TL6WL_getLED_HeadPowerStates(deviceHandle_, &tempEnableStates[0], &tempEnableStates[1], &tempEnableStates[2], &tempEnableStates[3], &tempEnableStates[4], &tempEnableStates[5]);
        for (int i = 0; i < numLEDs_; i++)
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
    if (LED < numLEDs_ && LED >= 0)
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

int ThorlabsChrolisDeviceWrapper::SetLEDEnableStates(ViBoolean states[6])
{
    std::lock_guard<std::mutex> lock(instanceMutex_);

    if (!deviceConnected_)
    {
        return ERR_CHROLIS_NOT_AVAIL;
    }

    int err = 0;
    for (int i = 0; i < numLEDs_; i++)
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
    if (LED < 6 && LED >= 0)
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

int ThorlabsChrolisDeviceWrapper::SetLEDBrightnessStates(ViUInt16 states[6])
{
    std::lock_guard<std::mutex> lock(instanceMutex_);

    if (!deviceConnected_)
    {
        return ERR_CHROLIS_NOT_AVAIL;
    }

    int i;
    for (i = 0; i < numLEDs_; i++)
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

    bool stateCorrect = true;
    ViUInt16 tempPowerStates[6] = {0,0,0,0,0,0};
    ViBoolean tempEnableStates[6] = { false, false, false, false, false, false };

    TL6WL_getLED_HeadPowerStates(deviceHandle_, &tempEnableStates[0], &tempEnableStates[1], &tempEnableStates[2], &tempEnableStates[3], &tempEnableStates[4], &tempEnableStates[5]);
    TL6WL_getLED_HeadBrightness(deviceHandle_, &tempPowerStates[0], &tempPowerStates[1], &tempPowerStates[2], &tempPowerStates[3], &tempPowerStates[4], &tempPowerStates[5]);

    for (int i = 0; i < numLEDs_; i++)
    {
        if (tempEnableStates[i] != savedEnabledStates_[i] || tempPowerStates[i] != savedBrightnessStates_[i])
        {
            stateCorrect = false;
            savedEnabledStates_[i] = tempEnableStates[i];
            savedBrightnessStates_[i] = tempPowerStates[i];
        }
    }

    return stateCorrect;
}

bool ThorlabsChrolisDeviceWrapper::VerifyLEDEnableStates()
{
    if (!deviceConnected_)
    {
        return false;
    }

    bool stateCorrect = true;
    ViBoolean tempEnableStates[6] = { false, false, false, false, false, false };

    TL6WL_getLED_HeadPowerStates(deviceHandle_, &tempEnableStates[0], &tempEnableStates[1], &tempEnableStates[2], &tempEnableStates[3], &tempEnableStates[4], &tempEnableStates[5]);

    for (int i = 0; i < numLEDs_; i++)
    {
        if (tempEnableStates[i] != savedEnabledStates_[i])
        {
            stateCorrect = false;
            savedEnabledStates_[i] = tempEnableStates[i];
        }
    }

    return stateCorrect;
}

bool ThorlabsChrolisDeviceWrapper::VerifyLEDPowerStates()
{
    if (!deviceConnected_)
    {
        return false;
    }

    bool stateCorrect = true;
    ViUInt16 tempPowerStates[6] = { 0,0,0,0,0,0 };

    TL6WL_getLED_HeadBrightness(deviceHandle_, &tempPowerStates[0], &tempPowerStates[1], &tempPowerStates[2], &tempPowerStates[3], &tempPowerStates[4], &tempPowerStates[5]);

    for (int i = 0; i < numLEDs_; i++)
    {
        if (tempPowerStates[i] != savedBrightnessStates_[i])
        {
            stateCorrect = false;
            savedBrightnessStates_[i] = tempPowerStates[i];
        }
    }
    
    return stateCorrect;
}