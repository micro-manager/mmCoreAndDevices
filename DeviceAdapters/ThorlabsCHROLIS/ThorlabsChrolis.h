#pragma once

#include "DeviceBase.h"
#include <TL6WL.h>

#include <array>
#include <atomic>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#define  CHROLIS_HUB_NAME  "CHROLIS"
#define  CHROLIS_SHUTTER_NAME  "CHROLIS_Shutter"
#define  CHROLIS_STATE_NAME  "CHROLIS_LED_Control"

//Custom Error Codes
#define ERR_HUB_NOT_AVAILABLE    101
#define ERR_CHROLIS_NOT_AVAIL    102
#define ERR_PARAM_NOT_VALID      103
#define ERR_NO_AVAIL_DEVICES     104
#define ERR_IMPROPER_SET         105

//CHROLIS Specific Error Codes
#define ERR_HARDWARE_FAULT      -1074001669

//VISA Error Codes
#define ERR_INSUF_INFO          -1073807343
#define ERR_UNKOWN_HW_STATE     -1073676421
#define ERR_VAL_OVERFLOW        -1073481985

constexpr int NUM_LEDS = 6;

//Wrapper for the basic functions used in this device adapter
class ThorlabsChrolisDeviceWrapper
{
public:
    ThorlabsChrolisDeviceWrapper();
    ~ThorlabsChrolisDeviceWrapper();

    int GetAvailableSerialNumbers(std::vector<std::string>& serialNumbers);
    int InitializeDevice(std::string serialNumber = "");
    int ShutdownDevice();
    bool IsDeviceConnected();
    int GetSerialNumber(ViChar* serialNumber);
    int GetManufacturerName(ViChar* manfName);
    int GetDeviceStatus(ViUInt32& status);
    int GetLEDWavelengths(std::array<ViUInt16, NUM_LEDS> &wavelengths);
    int GetShutterState(bool& open);
    int GetSingleLEDEnableState(int led, ViBoolean& state);
    int GetLEDEnableStates(std::array<ViBoolean, NUM_LEDS> &states);
    int GetSingleLEDBrightnessState(int led, ViUInt16& state);
    int GetLEDBrightnessStates(std::array<ViUInt16, NUM_LEDS> &states);

    int SetShutterState(bool open);
    int SetLEDEnableStates(std::array<ViBoolean, NUM_LEDS> states);
    int SetSingleLEDEnableState(int LED, ViBoolean state);
    int SetLEDBrightnessStates(std::array<ViUInt16, NUM_LEDS> states);
    int SetSingleLEDBrightnessState(int LED, ViUInt16 state);

    bool VerifyLEDEnableStatesWithLock();

private:
    std::vector<std::string> serialNumberList_;
    std::mutex instanceMutex_;
    bool deviceConnected_ = false;
    ViSession deviceHandle_ = ViSession(-1);
    ViBoolean deviceInUse_ = false; //only used by the chrolis API
    ViChar deviceName_[TL6WL_LONG_STRING_SIZE] = "";
    ViChar serialNumber_[TL6WL_LONG_STRING_SIZE] = "";
    ViChar manufacturerName_[TL6WL_LONG_STRING_SIZE] = "";
    bool masterSwitchState_ = false;
    std::array<ViBoolean, NUM_LEDS> savedEnabledStates_{};
    std::array<ViUInt16, NUM_LEDS> savedBrightnessStates_{};
    std::array<ViUInt16, NUM_LEDS> ledWavelengths_{};

    bool VerifyLEDStates();
    bool VerifyLEDPowerStates();
    bool VerifyLEDEnableStates();

};

class ChrolisHub : public HubBase <ChrolisHub>
{
public:
    ChrolisHub();
    ~ChrolisHub() {}

    ThorlabsChrolisDeviceWrapper ChrolisDevice;

    int Initialize();
    int Shutdown();
    void GetName(char* pszName) const;
    bool Busy();

    // HUB api
    int DetectInstalledDevices();

    void StatusChangedPollingThread();
    void SetShutterCallback(std::function<void(int, int)>);
    void SetStateCallback(std::function<void(int, int)>);

private:
    std::atomic_bool threadRunning_;
    std::thread updateThread_;
    std::atomic_uint32_t currentDeviceStatusCode_;
    std::function<void(int, int)> shutterCallback_;
    std::function<void(int, ViBoolean)> stateCallback_;
};

class ChrolisShutter : public CShutterBase <ChrolisShutter>
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

    unsigned long GetNumberOfPositions()const { return NUM_LEDS; }

    // action interface
    // ----------------
    int OnState(MM::PropertyBase* pProp, MM::ActionType eAct);

    //LED Control Methods
    int OnPowerChange(MM::PropertyBase* pProp, MM::ActionType eAct, long ledIndex);
    int OnEnableStateChange(MM::PropertyBase* pProp, MM::ActionType eAct, long ledIndex);

private:
    std::array<ViUInt16, NUM_LEDS> ledBrightnesses_{};
    std::array<ViBoolean, NUM_LEDS> ledStates_{};

    ViInt16 ledMaxBrightness_ = 1000;
    ViInt16 ledMinBrightness_ = 0;
};