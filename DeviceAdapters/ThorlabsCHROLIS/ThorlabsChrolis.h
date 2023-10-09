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
    ViSession deviceHandle_ = -1;
    ViBoolean savedEnabledStates[6]{ false,false,false,false,false,false};
    ViUInt16 powerStates[6]{0,0,1000,0,0,0};
    bool connected = false;
    bool masterShutterState = false;

public:
    ChrolisShutter()
    {
    }
    ~ChrolisShutter() {}

    int Initialize()
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
        connected = true;

        err = TL6WL_getLED_HeadPowerStates(deviceHandle_, &savedEnabledStates[0], &savedEnabledStates[1], &savedEnabledStates[2], &savedEnabledStates[3], &savedEnabledStates[4], &savedEnabledStates[5]);
        if (err != 0)
        {
            LogMessage("Get Enable States Failed: " + std::to_string(err));
            return DEVICE_ERR;
        }

        err = TL6WL_getLED_HeadBrightness(deviceHandle_, &powerStates[0], &powerStates[1], &powerStates[2], &powerStates[3], &powerStates[4], &powerStates[5]);
        if (err != 0)
        {
            LogMessage("Get Power States Failed: " + std::to_string(err));
            return DEVICE_ERR;
        }

        err = TL6WL_setLED_HeadBrightness(deviceHandle_, powerStates[0], powerStates[1], powerStates[2], powerStates[3], powerStates[4], powerStates[5]);
        if (err != 0)
        {
            LogMessage("Set Power States Failed: " + std::to_string(err));
            return DEVICE_ERR;
        }

        return DEVICE_OK;
    }

    int Shutdown()
    { 
        if (connected)
        {
            auto err = TL6WL_close(deviceHandle_);
            if (err != 0)
            {
                LogMessage("Close Failed: " + std::to_string(err));
                return DEVICE_ERR;
            }
            connected = false;
            CDeviceUtils::SleepMs(1000);
        }
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
    int SetOpen(bool open = true)
    {
        if (!open)
        {
            if (int err = TL6WL_setLED_HeadPowerStates(deviceHandle_, false, false, false, false, false, false) != 0)
            {
                LogMessage("Get Enable States Failed: " + std::to_string(err));
                return DEVICE_ERR;
            }
            masterShutterState = false;
        }
        else
        {
            if (int err = TL6WL_setLED_HeadPowerStates(deviceHandle_, true, true, true, true, true, true) != 0)
            {
                LogMessage("Get Enable States Failed: " + std::to_string(err));
                return DEVICE_ERR;
            }
            masterShutterState = true;
        }
        return DEVICE_OK;
    }

    int GetOpen(bool& open)
    {
        open = masterShutterState;
        return DEVICE_OK;
    }

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

