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
    ChrolisHub();
    ~ChrolisHub() {}

    int Initialize();
    int Shutdown();
    void GetName(char* pszName) const;
    bool Busy();

    // HUB api
    int DetectInstalledDevices();

    bool IsInitialized();
    void* GetChrolisDeviceInstance();

private:
    void* chrolisDeviceInstance_;
    bool initialized_;
    bool busy_;
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

    int Initialize();

    int Shutdown();

    void GetName(char* name)const;

    bool Busy();

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

//Wrapper for the basic functions used in this device adapter
class ThorlabsChrolisDeviceWrapper
{
public:
    ThorlabsChrolisDeviceWrapper();
    ~ThorlabsChrolisDeviceWrapper();

    int InitializeDevice(std::string serialNumber = "");
    int ShutdownDevice();
    bool IsDeviceConnected();
    int SetLEDEnableStates(ViBoolean states[6]);
    int SetLEDPowerStates(ViInt16 states[6]);
    int SetShutterState(bool open);
    int GetShutterState(bool& open);

private:
    int numLEDs_;
    bool deviceConnected_;
    ViSession deviceHandle_;
    ViBoolean deviceInUse_; //only used by the chrolis API
    ViChar deviceName_[256];
    ViChar serialNumber_[256];
    ViChar manufacturerName_[256];
    bool masterSwitchState_;
    ViBoolean savedEnabledStates[6]{false,false,false,false,false,false};
    ViInt16 savedPowerStates[6]{0,0,0,0,0,0};
};

