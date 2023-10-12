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

int ThorlabsChrolisDeviceWrapper::InitializeDevice(std::string serialNumber)
{
    int err = 0;
    ViUInt32 numDevices;
    CDeviceUtils::SleepMs(2000);
    err = TL6WL_findRsrc(NULL, &numDevices);
    if (err != 0)
    {
        return DEVICE_ERR;
    }
    if (numDevices == 0)
    {
        return DEVICE_ERR;
    }

    ViChar resource[512] = "";
    if(serialNumber.compare(""))
    {
        err = TL6WL_getRsrcName(NULL, 0, resource);
        if (err != 0)
        {
            return DEVICE_ERR;
        }
        err = TL6WL_getRsrcInfo(NULL, 0, deviceName_, serialNumber_, manufacturerName_, &deviceInUse_);
        if (err != 0)
        {
            return DEVICE_ERR;
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
                return DEVICE_ERR;
            }
            err = TL6WL_getRsrcInfo(NULL, i, deviceName_, serialNumber_, manufacturerName_, &deviceInUse_);
            if (err != 0)
            {
                return DEVICE_ERR;
            }
            if (strcmp((char*)serialNumber_, serialNumber.c_str()))
            {
                found = true;
                break;
            }
        }
        if(!found)
        { 
            return DEVICE_ERR;
        }
    }
    err = TL6WL_init(resource, false, false, &deviceHandle_);
    if (err != 0)
    {
        return DEVICE_ERR;
    }
    deviceConnected_ = true;

    for (int i = 0; i < numLEDs_; i++)
    {
        err = TL6WL_readLED_HeadCentroidWL(deviceHandle_, i, &ledWavelengths[i]);
        if (err != 0)
        {
            //Error reading head wavelengths
            return DEVICE_ERR;
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
            return DEVICE_ERR;
        }
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
        serialNumber = serialNumber_;
    }
    else
    {
        serialNumber = (ViChar*)"NOT INITIALIZED";
    }
    return DEVICE_OK;
}

int ThorlabsChrolisDeviceWrapper::GetManufacturerName(ViChar* manfName)
{
    if (deviceConnected_)
    {
        manfName = manufacturerName_;
    }
    else
    {
        manfName = (ViChar*)"NOT INITIALIZED";
    }
    return DEVICE_OK;
}

int ThorlabsChrolisDeviceWrapper::GetLEDWavelengths(ViUInt16(&wavelengths)[6])
{
    if (!deviceConnected_)
    {
        *wavelengths = NULL;
        return DEVICE_ERR;
    }
    *wavelengths = *ledWavelengths;
    return DEVICE_OK;
}

int ThorlabsChrolisDeviceWrapper::GetLEDEnableStates(ViBoolean(&states)[6])
{
    if (!deviceConnected_)
    {
        *states = NULL;
        return DEVICE_ERR;
    }
    *states = *savedEnabledStates;
    return DEVICE_OK;
}

int ThorlabsChrolisDeviceWrapper::GetLEDEnableStates(ViBoolean& led1State, ViBoolean& led2State, ViBoolean& led3State, ViBoolean& led4State, ViBoolean& led5State, ViBoolean& led6State)
{
    if (!deviceConnected_)
    {
        return DEVICE_ERR;
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
    if (states == NULL)
    {
        return DEVICE_ERR;
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
        *states = NULL;
        return DEVICE_ERR;
    }
    *states = *savedPowerStates;
    return DEVICE_OK;
}

int ThorlabsChrolisDeviceWrapper::GetLEDPowerStates(ViUInt16& led1Power, ViUInt16& led2Power, ViUInt16& led3Power, ViUInt16& led4Power, ViUInt16& led5Power, ViUInt16& led6Power)
{
    if (!deviceConnected_)
    {
        return DEVICE_ERR;
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
    if (states == NULL)
    {
        return DEVICE_ERR;
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
        savedEnabledStates[LED] = state;
        if (!SetLEDEnableStates(savedEnabledStates))
        {
            return DEVICE_OK;
        }
        else
        {
            //revert in case of error
            savedEnabledStates[LED] = !state;
        }
    }
    return DEVICE_ERR;
}

int ThorlabsChrolisDeviceWrapper::SetSingleLEDPowerState(int LED, ViUInt16 state)
{
    if (LED < 6 && LED >= 0)
    {
        ViInt16 tmpPower = savedPowerStates[LED];
        savedPowerStates[LED] = state;
        if (!SetLEDPowerStates(savedPowerStates))
        {
            return DEVICE_OK;
        }
        else
        {
            //revert in case of error
            savedPowerStates[LED] = tmpPower;
        }
    }
    return DEVICE_ERR;
}

int ThorlabsChrolisDeviceWrapper::SetShutterState(bool open)
{
    if (!open)
    {
        if (int err = TL6WL_setLED_HeadPowerStates(deviceHandle_, false, false, false, false, false, false) != 0)
        {
            return DEVICE_ERR;
        }
        masterSwitchState_ = false;
    }
    else
    {
        if (int err = TL6WL_setLED_HeadPowerStates(deviceHandle_, savedEnabledStates[0], savedEnabledStates[1], savedEnabledStates[2], savedEnabledStates[3], savedEnabledStates[4], savedEnabledStates[5]) != 0)
        {
            return DEVICE_ERR;
        }
        masterSwitchState_ = true;
    }
	return DEVICE_OK;
}

int ThorlabsChrolisDeviceWrapper::GetShutterState(bool& open)
{
    open = masterSwitchState_;
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

long GetIntegerStateValue()
{
    return 0;
}