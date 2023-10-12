#pragma once
#include <TL6WL.h>
#include "DeviceBase.h"
#define  CHROLIS_HUB_NAME  "CHROLIS"
#define  CHROLIS_SHUTTER_NAME  "CHROLIS_Shutter"
#define  CHROLIS_STATE_NAME  "CHROLIS_LED_Control"
//#define  CHROLIS_GENERIC_DEVICE_NAME "CHROLIS_Power_Control"

//Custom Error Codes
#define ERR_UNKNOWN_MODE         102
#define ERR_UNKNOWN_LED_STATE    103
#define ERR_IN_SEQUENCE          104
#define ERR_SEQUENCE_INACTIVE    105
#define ERR_STAGE_MOVING         106
#define HUB_NOT_AVAILABLE        107

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
    ChrolisShutter();
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
    ChrolisStateDevice();

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

    //LED Control Methods
    int OnPowerChange(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnEnableStateChange(MM::PropertyBase* pProp, MM::ActionType eAct);

private:
    long curLedState_;
    long numPos_;

    ViInt16 led1Power_;
    ViInt16 led2Power_;
    ViInt16 led3Power_;
    ViInt16 led4Power_;
    ViInt16 led5Power_;
    ViInt16 led6Power_;

    ViBoolean led1State_;
    ViBoolean led2State_;
    ViBoolean led3State_;
    ViBoolean led4State_;
    ViBoolean led5State_;
    ViBoolean led6State_;

    ViInt16 ledMaxPower_;
    ViInt16 ledMinPower_;
};

//class ChrolisPowerControl : public CGenericBase <ChrolisPowerControl>
//{
//public:
//    ChrolisPowerControl();
//
//    ~ChrolisPowerControl()
//    {}
//
//    int Initialize();
//    int Shutdown();
//    void GetName(char* pszName) const;
//    bool Busy();
//
//    //Label Update
//    int OnPowerChange(MM::PropertyBase* pProp, MM::ActionType eAct);
//    
//private:
//    ViInt16 led1Power_;
//    ViInt16 led2Power_;
//    ViInt16 led3Power_;
//    ViInt16 led4Power_;
//    ViInt16 led5Power_;
//    ViInt16 led6Power_;
//
//    ViInt16 ledMaxPower_;
//    ViInt16 ledMinPower_;
//};

//Wrapper for the basic functions used in this device adapter
class ThorlabsChrolisDeviceWrapper
{
public:
    ThorlabsChrolisDeviceWrapper();
    ~ThorlabsChrolisDeviceWrapper();

    int InitializeDevice(std::string serialNumber = "");
    int ShutdownDevice();
    bool IsDeviceConnected();
    int GetLEDEnableStates(ViBoolean (&states)[6]);
    int SetLEDEnableStates(ViBoolean states[6]);
    int SetLEDPowerStates(ViInt16 states[6]);
    int SetSingleLEDEnableState(int LED, ViBoolean state);
    int SetSingleLEDPowerState(int LED, ViInt16 state);
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

