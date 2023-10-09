#pragma once
#include <TL6WL.h>
#include "DeviceBase.h"
#define  CHROLIS_HUB_NAME  "CHROLIS_Hub"
#define  CHROLIS_SHUTTER_NAME  "CHROLIS_Shutter"
#define  CHROLIS_STATE_NAME  "CHROLIS_State_Device"
#define  CHROLIS_GENERIC_DEVICE_NAME "CHROLIS_Generic_Device"

class ChrolisHub : public HubBase <ChrolisHub>
{
public:
    ChrolisHub() :
        initialized_(false),
        busy_(false),
        deviceHandle_(-1)
    {
        CreateHubIDProperty();
    }
    ~ChrolisHub() {}

    int Initialize(); //TODO: Create property for serial number
    int Shutdown();
    void GetName(char* pszName) const;
    bool Busy();

    // HUB api
    int DetectInstalledDevices();

    int GetDeviceHandle(ViPSession deviceHandle);
    bool IsInitialized();

private:
    bool initialized_;
    bool busy_;
    ViSession deviceHandle_;
};

class ChrolisShutter : public CShutterBase <ChrolisShutter> //CRTP
{
    ViBoolean savedEnabledStates[6]{ false,false,false,false,false,false};
    bool masterShutterState = false;

public:
    ChrolisShutter()
    {
    }
    ~ChrolisShutter() {}

    int Initialize()
    {
        return DEVICE_OK;
    }

    int Shutdown()
    { 
        return DEVICE_OK; 
    }

    void GetName(char* name) const
    {
        CDeviceUtils::CopyLimitedString(name, CHROLIS_SHUTTER_NAME);
    }

    bool Busy()
    {
        return false;
    }

    // Shutter API
    int SetOpen(bool open = true);

    int GetOpen(bool& open);

    int Fire(double /*deltaT*/)
    {
        return DEVICE_UNSUPPORTED_COMMAND;
    }
};

class ChrolisStateDevice : public CStateDeviceBase<ChrolisStateDevice>
{
public:
    ChrolisStateDevice():
        numPos_(6)
    {}

    ~ChrolisStateDevice()
    {}

    int Initialize();
    int Shutdown();
    void GetName(char* pszName) const;
    bool Busy();

    unsigned long GetNumberOfPositions()const { return numPos_; }

    // action interface
    // ----------------
    int OnState(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnDelay(MM::PropertyBase* pProp, MM::ActionType eAct);

private:
    long numPos_;
};

class ChrolisPowerControl : public CGenericBase <ChrolisPowerControl>
{
public:
    ChrolisPowerControl()
    {}

    ~ChrolisPowerControl()
    {}

    int Initialize();
    int Shutdown();
    void GetName(char* pszName) const;
    bool Busy();
    
private:

};

class ThorlabsChrolisDeviceWrapper
{
public:
    ThorlabsChrolisDeviceWrapper();
    ~ThorlabsChrolisDeviceWrapper();

    int InitializeDevice(std::string serialNumber = "");
    int ShutdownDevice();
    int GetDeviceHandle();
    int SetLEDEnableStates(ViBoolean states[6]);
    int SetLEDPowerStates(ViUInt32 states[6]);
    int SetShutterState(bool state);
    int GetShutterState(bool& state);

private:
    ViSession deviceHandle_;
    bool masterSwitchState_;
    ViBoolean savedEnabledStates[6]{false,false,false,false,false,false};
    ViUInt32 savedPowerStates[6]{0,0,0,0,0,0};

};

