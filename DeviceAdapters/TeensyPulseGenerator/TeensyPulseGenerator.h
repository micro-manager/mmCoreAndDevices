
#ifndef _TeensyPulseGenerator_H
#define _TeensyPulseGenerator_H_

#include "MMDevice.h"
#include "DeviceBase.h"
#include "DeviceUtils.h"
#include "ModuleInterface.h"
#include <cstdio>
#include <string>
#include <vector>

#define ERR_PORT_OPEN_FAILED 106
#define ERR_COMMUNICATION 107
#define ERR_NO_PORT_SET 108
#define ERR_VERSION_MISMATCH 109

class TeensyPulseGenerator : public CGenericBase<TeensyPulseGenerator>
{
public:
    TeensyPulseGenerator();
    ~TeensyPulseGenerator();

    // MMDevice API
    int Initialize();
    int Shutdown();
    void GetName(char* name) const;
    bool Busy();

    // action interface 
    int OnPort(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnInterval(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnPulseDuration(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnTriggerMode(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnState(MM::PropertyBase* pProp, MM::ActionType eAct);

private:
    bool initialized_;
    std::string port_;

    // Current configuration
    double interval_;     // Interval in microseconds
    double pulseDuration_; // Pulse duration in microseconds
    bool triggerMode_;    // Trigger mode enabled/disabled
    bool running_;        // Pulse generator running state

    // Helper methods for serial communication
    int SendCommand(uint8_t cmd, uint32_t param = 0);
    bool ReadResponse(std::string& response);
};

#endif
