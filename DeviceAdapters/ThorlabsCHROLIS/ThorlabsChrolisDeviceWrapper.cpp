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