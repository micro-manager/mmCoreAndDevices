#include "ThorlabsChrolis.h"
#include <string>

ThorlabsChrolisDeviceWrapper::ThorlabsChrolisDeviceWrapper()
{
	masterSwitchState_ = false;
	deviceHandle_ = -1;
}

ThorlabsChrolisDeviceWrapper::~ThorlabsChrolisDeviceWrapper()
{}

int ThorlabsChrolisDeviceWrapper::InitializeDevice(std::string serialNumber = "")
{
    int err = 0;
    ViUInt32 numDevices;
    CDeviceUtils::SleepMs(2000);
    err = TL6WL_findRsrc(NULL, &numDevices);
    if (err != 0)
    {
        LogMessage("Find Resource Failed: " + std::to_string(err));
        return DEVICE_ERR;
    }
    if (numDevices == 0)
    {
        LogMessage("Chrolis devices not found"); // to log file 
        return DEVICE_ERR;
    }

    ViChar resource[512] = "";
    err = TL6WL_getRsrcName(NULL, 0, resource);
    if (err != 0)
    {
        LogMessage("Get Resource Failed: " + std::to_string(err));
        return DEVICE_ERR;
    }

    err = TL6WL_init(resource, false, false, &deviceHandle_);
    if (err != 0)
    {
        LogMessage("Initialize Failed: " + std::to_string(err));
        return DEVICE_ERR;
    }
	return 0;
}

int ThorlabsChrolisDeviceWrapper::ShutdownDevice()
{
	return 0;
}

int ThorlabsChrolisDeviceWrapper::SetLEDEnableStates(ViBoolean states[6])
{
	return 0;
}

int ThorlabsChrolisDeviceWrapper::SetLEDPowerStates(ViUInt32 states[6])
{
	return 0;
}

int ThorlabsChrolisDeviceWrapper::SetShutterState(bool state)
{
	return 0;
}

int ThorlabsChrolisDeviceWrapper::GetShutterState(bool& state)
{
	state = masterSwitchState_;
	return 0;
}