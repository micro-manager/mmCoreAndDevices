#pragma once
#include "DeviceBase.h"
#include "MMDevice.h"
#include "Property.h"
#include "MMDeviceConstants.h"
#include "include/TLDC.h"

// MMDevice adapter for ThorLabs DC40 LED Driver
class DC40 : public CGenericBase<DC40>
{
public:
    DC40(const char* serialNr);
    ~DC40();

    // MMDevice API
    int Initialize();
    int Shutdown();
    void GetName(char* name) const;
    bool Busy();

    // Action handlers
    int OnOperatingMode(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnCurrent(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnState(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnCurrentLimit(MM::PropertyBase* pProp, MM::ActionType eAct);

private:
    // Utility functions
    int InitializeDevice();
    int HandleError(int error);

    bool initialized_;
    std::string name_;
    ViSession instrumentHandle_;
    std::string serialNr_;

    // Device state
    std::string operatingMode_;  // CW, TTL, or MOD
    double current_;             // LED current setpoint
    double currentLimit_;        // LED current limit
    bool state_;                 // On/Off state
};