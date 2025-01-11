#pragma once
#include "DeviceBase.h"
#include "MMDevice.h"
#include "Property.h"
#include "MMDeviceConstants.h"
#include "include/TLDC.h"

// MMDevice adapter for ThorLabs DC40 LED Driver
class DC40 : public CShutterBase<DC40>
{
public:
    DC40(const char* deviceName);
    ~DC40();

    // MMDevice API
    int Initialize();
    int Shutdown();
    void GetName(char* name) const;
    bool Busy();

    int SetOpen(bool open = true);
    int GetOpen(bool& open);
    /**
     * Opens the shutter for the given duration, then closes it again.
     * Currently not implemented in any shutter adapters
     */
    int Fire(double ) { return DEVICE_UNSUPPORTED_COMMAND; }

    // Action handlers
    int OnOperatingMode(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnCurrent(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnState(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnCurrentLimit(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnSerialNumber(MM::PropertyBase* pprop, MM::ActionType eAct);

private:
    // Utility functions
    int HandleError(int error);

    bool initialized_;
    ViSession instrumentHandle_;
    std::string serialNr_;
    std::string deviceName_;

    // Device state
    std::string operatingMode_;  // CW, TTL, or MOD
    double current_;             // LED current setpoint
    double currentLimit_;        // LED current limit
    bool state_;                 // On/Off state
};