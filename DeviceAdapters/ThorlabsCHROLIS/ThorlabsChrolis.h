#pragma once
#include<thread>
#include <TL6WL.h>
#include <atomic>
#include <vector>
#include<mutex>
#include "DeviceBase.h"

#define  CHROLIS_HUB_NAME  "CHROLIS"
#define  CHROLIS_SHUTTER_NAME  "CHROLIS_Shutter"
#define  CHROLIS_STATE_NAME  "CHROLIS_LED_Control"

//Custom Error Codes
#define ERR_UNKNOWN_MODE         102 // not currently used
#define ERR_UNKNOWN_LED_STATE    103// don't think this is used
#define ERR_HUB_NOT_AVAILABLE    104
#define ERR_CHROLIS_NOT_AVAIL    105
#define ERR_CHROLIS_SET          106 //don't think this is used
#define ERR_CHROLIS_GET          107 // don't think this is used
#define ERR_PARAM_NOT_VALID      108
#define ERR_NO_AVAIL_DEVICES     109
#define ERR_IMPROPER_SET         110

//CHROLIS Specific Error Codes
#define ERR_HARDWARE_FAULT      -1074001669

//VISA Error Codes
#define ERR_INSUF_INFO          -1073807343
#define ERR_UNKOWN_HW_STATE     -1073676421
#define ERR_VAL_OVERFLOW        -1073481985

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
    void StatusChangedPollingThread();


private:
    void* chrolisDeviceInstance_;
    bool initialized_;
    bool busy_;
    std::atomic_bool threadRunning_;
    std::thread updateThread_;
    std::atomic_uint32_t currentDeviceStatusCode_;
    std::string deviceStatusMessage_;
};

class ChrolisShutter : public CShutterBase <ChrolisShutter> //CRTP
{
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
    long numPos_;

    ViUInt16 led1Power_;
    ViUInt16 led2Power_;
    ViUInt16 led3Power_;
    ViUInt16 led4Power_;
    ViUInt16 led5Power_;
    ViUInt16 led6Power_;

    ViBoolean led1State_;
    ViBoolean led2State_;
    ViBoolean led3State_;
    ViBoolean led4State_;
    ViBoolean led5State_;
    ViBoolean led6State_;

    ViInt16 ledMaxPower_;
    ViInt16 ledMinPower_;
};

//Wrapper for the basic functions used in this device adapter
class ThorlabsChrolisDeviceWrapper
{
public:
    ThorlabsChrolisDeviceWrapper();
    ~ThorlabsChrolisDeviceWrapper();

    int GetAvailableSerialNumbers(std::vector<std::string> &serialNumbers);
    int InitializeDevice(std::string serialNumber = "");
    int ShutdownDevice();
    bool IsDeviceConnected();
    int GetSerialNumber(ViChar* serialNumber);
    int GetManufacturerName(ViChar* manfName);
    int GetDeviceStatus(ViUInt32& status);
    int GetLEDWavelengths(ViUInt16(&wavelengths)[6]);
    int GetShutterState(bool& open);
    int GetSingleLEDEnableState(int led, ViBoolean& state);
    int GetLEDEnableStates(ViBoolean(&states)[6]);
    int GetLEDEnableStates(ViBoolean& led1State, ViBoolean& led2State, ViBoolean& led3State, ViBoolean& led4State, ViBoolean& led5State, ViBoolean& led6State);
    int GetSingleLEDPowerState(int led, ViUInt16& state);
    int GetLEDPowerStates(ViUInt16(&states)[6]);
    int GetLEDPowerStates(ViUInt16 &led1Power, ViUInt16&led2Power, ViUInt16&led3Power, ViUInt16&led4Power, ViUInt16&led5Power, ViUInt16&led6Power);

    int SetShutterState(bool open);
    int SetLEDEnableStates(ViBoolean states[6]);
    int SetSingleLEDEnableState(int LED, ViBoolean state);
    int SetLEDPowerStates(ViUInt16 states[6]);
    int SetSingleLEDPowerState(int LED, ViUInt16 state);

    bool SyncLEDEnableStates();

private:
    int numLEDs_;
    std::vector<std::string> serialNumberList_;
    std::mutex instanceMutex_;
    bool deviceConnected_;
    ViSession deviceHandle_;
    ViBoolean deviceInUse_; //only used by the chrolis API
    ViChar deviceName_[TL6WL_LONG_STRING_SIZE];
    ViChar serialNumber_[TL6WL_LONG_STRING_SIZE];
    ViChar manufacturerName_[TL6WL_LONG_STRING_SIZE];
    bool masterSwitchState_;
    ViBoolean savedEnabledStates[6]{false,false,false,false,false,false};
    ViUInt16 savedPowerStates[6]{0,0,0,0,0,0};
    ViUInt16 ledWavelengths[6]{0,0,0,0,0,0};

    bool VerifyLEDStates();
    bool VerifyLEDPowerStates();
    bool VerifyLEDEnableStates();
};

