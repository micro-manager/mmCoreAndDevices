#pragma once
#include<thread>
#include <TL6WL.h>
#include <atomic>
#include <vector>
#include<mutex>
#include "DeviceBase.h"
#include <functional>

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

static std::map<int, std::string> ErrorMessages()
{
    return {
        {ERR_HUB_NOT_AVAILABLE, "Hub is not available"},
        {ERR_CHROLIS_NOT_AVAIL, "CHROLIS Device is not available"},
        {ERR_IMPROPER_SET, "Error setting property value. Value will be reset"},
        {ERR_PARAM_NOT_VALID, "Value passed to property was out of bounds."},
        {ERR_NO_AVAIL_DEVICES, "No available devices were found on the system."},
        {ERR_INSUF_INFO, "Insufficient location information of the device or the resource is not present on the system"},
        {ERR_UNKOWN_HW_STATE, "Unknown Hardware State"},
        {ERR_VAL_OVERFLOW, "Parameter Value Overflow"},
        {INSTR_RUNTIME_ERROR, "CHROLIS Instrument Runtime Error"},
        {INSTR_REM_INTER_ERROR, "CHROLIS Instrument Internal Error"},
        {INSTR_AUTHENTICATION_ERROR, "CHROLIS Instrument Authentication Error"},
        {INSTR_PARAM_ERROR, "CHROLIS Invalid Parameter Error"},
        {INSTR_INTERNAL_TX_ERR, "CHROLIS Instrument Internal Command Sending Error"},
        {INSTR_INTERNAL_RX_ERR, "CHROLIS Instrument Internal Command Receiving Error"},
        {INSTR_INVAL_MODE_ERR, "CHROLIS Instrument Invalid Mode Error"},
        {INSTR_SERVICE_ERR, "CHROLIS Instrument Service Error"}
    };
}

// TODO: add mutex
class ThorlabsChrolisDeviceWrapper;

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

    ThorlabsChrolisDeviceWrapper* GetChrolisDeviceInstance();
    void StatusChangedPollingThread();
    void SetShutterCallback(std::function<void(int, int)>);
    void SetStateCallback(std::function<void(int, int)>);

private:
    ThorlabsChrolisDeviceWrapper* chrolisDeviceInstance_;
    std::atomic_bool threadRunning_;
    std::thread updateThread_;
    std::atomic_uint32_t currentDeviceStatusCode_;
    std::string deviceStatusMessage_ = "No Error";
    std::function<void(int, int)> shutterCallback_;
    std::function<void(int, int)> stateCallback_;
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
    long numPos_= 6;

    ViUInt16 led1Brightness_ = 0;
    ViUInt16 led2Brightness_ = 0;
    ViUInt16 led3Brightness_ = 0;
    ViUInt16 led4Brightness_ = 0;
    ViUInt16 led5Brightness_ = 0;
    ViUInt16 led6Brightness_ = 0;

    ViBoolean led1State_ = false;
    ViBoolean led2State_ = false;
    ViBoolean led3State_ = false;
    ViBoolean led4State_ = false;
    ViBoolean led5State_ = false;
    ViBoolean led6State_ = false;

    ViInt16 ledMaxBrightness_ = 1000;
    ViInt16 ledMinBrightness_ = 0;
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
    int GetSingleLEDBrightnessState(int led, ViUInt16& state);
    int GetLEDBrightnessStates(ViUInt16(&states)[6]);
    int GetLEDBrightnessStates(ViUInt16 &led1Brightness, ViUInt16&led2Brightness, ViUInt16&led3Brightness, ViUInt16&led4Brightness, ViUInt16&led5Brightness, ViUInt16&led6Brightness);

    int SetShutterState(bool open);
    int SetLEDEnableStates(ViBoolean states[6]);
    int SetSingleLEDEnableState(int LED, ViBoolean state);
    int SetLEDBrightnessStates(ViUInt16 states[6]);
    int SetSingleLEDBrightnessState(int LED, ViUInt16 state);

    bool VerifyLEDEnableStatesWithLock();

private:
    int numLEDs_ = 6;
    std::vector<std::string> serialNumberList_;
    std::mutex instanceMutex_;
    bool deviceConnected_ = false;
    ViSession deviceHandle_;
    ViBoolean deviceInUse_ = false; //only used by the chrolis API
    ViChar deviceName_[TL6WL_LONG_STRING_SIZE] = "";
    ViChar serialNumber_[TL6WL_LONG_STRING_SIZE] = "";
    ViChar manufacturerName_[TL6WL_LONG_STRING_SIZE] = "";
    bool masterSwitchState_ = false;
    ViBoolean savedEnabledStates_[6]{false,false,false,false,false,false};
    ViUInt16 savedBrightnessStates_[6]{0,0,0,0,0,0};
    ViUInt16 ledWavelengths_[6]{0,0,0,0,0,0};

    bool VerifyLEDStates();
    bool VerifyLEDPowerStates();
    bool VerifyLEDEnableStates();

};