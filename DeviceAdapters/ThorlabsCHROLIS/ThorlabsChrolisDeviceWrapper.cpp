#include "ThorlabsChrolis.h"
#include <string>

ThorlabsChrolisDeviceWrapper::ThorlabsChrolisDeviceWrapper()
{
    deviceInUse_ = false;
    deviceConnected_ = false;
	masterSwitchState_ = false;
	deviceHandle_ = -1;
    numLEDs_ = 6;
}

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
    if(serialNumber.compare("") || serialNumber.compare("DEFAULT"))
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
        err = TL6WL_readLED_HeadPeakWL(deviceHandle_, i+1, &ledWavelengths[i]);
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
    return deviceConnected_;
}

int ThorlabsChrolisDeviceWrapper::GetSerialNumber(ViChar* serialNumber)
{
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

int ThorlabsChrolisDeviceWrapper::GetLEDWavelengths(ViUInt16(&wavelengths)[6])
{
    if (!deviceConnected_)
    {
        for (int i = 0; i < 6; i++)
        {
            wavelengths[i] = 0;
        }
        return ERR_CHROLIS_NOT_AVAIL;
    }
    wavelengths[0] = ledWavelengths[0];
    wavelengths[1] = ledWavelengths[1];
    wavelengths[2] = ledWavelengths[2];
    wavelengths[3] = ledWavelengths[3];
    wavelengths[4] = ledWavelengths[4];
    wavelengths[5] = ledWavelengths[5];

    return DEVICE_OK;
}

int ThorlabsChrolisDeviceWrapper::GetLEDEnableStates(ViBoolean(&states)[6])
{
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
        states[i] = savedEnabledStates[i];
    }
    return DEVICE_OK;
}

int ThorlabsChrolisDeviceWrapper::GetLEDEnableStates(ViBoolean& led1State, ViBoolean& led2State, ViBoolean& led3State, ViBoolean& led4State, ViBoolean& led5State, ViBoolean& led6State)
{
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
    led1State = savedEnabledStates[0];
    led2State = savedEnabledStates[1];
    led3State = savedEnabledStates[2];
    led4State = savedEnabledStates[2];
    led5State = savedEnabledStates[4];
    led6State = savedEnabledStates[5];

    return DEVICE_OK;
}

int ThorlabsChrolisDeviceWrapper::SetLEDEnableStates(ViBoolean states[6])
{
    if (!deviceConnected_)
    {
        return ERR_CHROLIS_NOT_AVAIL;
    }
    int i;
    for (i = 0; i < numLEDs_; i++)
    {
        savedEnabledStates[i] = states[i];
    }

    if (masterSwitchState_)
    {
        if (auto err = TL6WL_setLED_HeadPowerStates(deviceHandle_, savedEnabledStates[0], savedEnabledStates[1], savedEnabledStates[2],
            savedEnabledStates[3], savedEnabledStates[4], savedEnabledStates[5]) != 0)
        {
            return DEVICE_ERR;
        }
    }

	return DEVICE_OK;
}

int ThorlabsChrolisDeviceWrapper::GetLEDPowerStates(ViUInt16(&states)[6])
{
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
        states[i] = savedPowerStates[i];
    }
    return DEVICE_OK;
}

int ThorlabsChrolisDeviceWrapper::GetLEDPowerStates(ViUInt16& led1Power, ViUInt16& led2Power, ViUInt16& led3Power, ViUInt16& led4Power, ViUInt16& led5Power, ViUInt16& led6Power)
{
    if (!deviceConnected_)
    {
        led1Power = 0;
        led2Power = 0;
        led3Power = 0;
        led4Power = 0;
        led5Power = 0;
        led6Power = 0;
        return ERR_CHROLIS_NOT_AVAIL;
    }
    led1Power = savedPowerStates[0];
    led2Power = savedPowerStates[1];
    led3Power = savedPowerStates[2];
    led4Power = savedPowerStates[2];
    led5Power = savedPowerStates[4];
    led6Power = savedPowerStates[5];

    return DEVICE_OK;
}

int ThorlabsChrolisDeviceWrapper::SetLEDPowerStates(ViUInt16 states[6])
{
    if (!deviceConnected_)
    {
        return ERR_CHROLIS_NOT_AVAIL;
    }
    int i;
    for (i = 0; i < numLEDs_; i++)
    {
        savedPowerStates[i] = states[i];
    }

    if (auto err = TL6WL_setLED_HeadBrightness(deviceHandle_, savedPowerStates[0], savedPowerStates[1], savedPowerStates[2],
        savedPowerStates[3], savedPowerStates[4], savedPowerStates[5]) != 0)
    {
        return DEVICE_ERR;
    }

	return DEVICE_OK;
}

int ThorlabsChrolisDeviceWrapper::SetSingleLEDEnableState(int LED, ViBoolean state)
{
    if (LED < 6 && LED >= 0)
    {
        int err = 0;
        savedEnabledStates[LED] = state;
        err = SetLEDEnableStates(savedEnabledStates);
        if (err != 0)
        {
            //revert in case of error
            savedEnabledStates[LED] = !state;
            return err;
        }
        return DEVICE_OK;
    }
    else
    {
        return ERR_PARAM_NOT_VALID;
    }
}

int ThorlabsChrolisDeviceWrapper::SetSingleLEDPowerState(int LED, ViUInt16 state)
{
    if (LED < 6 && LED >= 0)
    {
        int err = 0;
        ViInt16 tmpPower = savedPowerStates[LED];
        savedPowerStates[LED] = state;
        err = SetLEDPowerStates(savedPowerStates);
        if (err != 0)
        {
            //revert in case of error
            savedPowerStates[LED] = tmpPower;
            return err;
        }
        return DEVICE_OK;
    }
    else
    {
        return ERR_PARAM_NOT_VALID;
    }
}

int ThorlabsChrolisDeviceWrapper::SetShutterState(bool open)
{
    int err;
    if (!open)
    {
        err = TL6WL_setLED_HeadPowerStates(deviceHandle_, false, false, false, false, false, false);
        if (err != 0)
        {
            return err;
        }
        masterSwitchState_ = false;
    }
    else
    {
        err = TL6WL_setLED_HeadPowerStates(deviceHandle_, savedEnabledStates[0], savedEnabledStates[1], savedEnabledStates[2], savedEnabledStates[3], savedEnabledStates[4], savedEnabledStates[5]);
        if (err != 0)
        {
            return err;
        }
        masterSwitchState_ = true;
    }
	return DEVICE_OK;
}

int ThorlabsChrolisDeviceWrapper::GetShutterState(bool& open)
{
    if (!deviceConnected_)
    {
        open = false;
        return ERR_CHROLIS_NOT_AVAIL;
    }
    open = masterSwitchState_;
	return DEVICE_OK;
}

int ThorlabsChrolisDeviceWrapper::GetDeviceStatus(ViUInt32& status)
{
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

int ThorlabsChrolisDeviceWrapper::RegisterStatusChangedHandler(void* handler)
{
    if (!deviceConnected_)
    {
        return ERR_CHROLIS_NOT_AVAIL;
    }
    auto err = TL6WL_registerBoxStatusChangedHandler(deviceHandle_, (Box6WL_StatusChangedHandler)handler);
    if (err != 0)
    {
        return err;
    }
    return DEVICE_OK;
}

//set or fix any issues with the stored led vals. Return if a correction needed to be made
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
        if (tempEnableStates[i] != savedEnabledStates[i] || tempPowerStates[i] != savedPowerStates[i])
        {
            stateCorrect = false;
            savedEnabledStates[i] = tempEnableStates[i];
            savedPowerStates[i] = tempPowerStates[i];
        }
    }

    return stateCorrect;
}