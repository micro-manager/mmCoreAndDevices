
#ifndef _TeensyPulseGenerator_H
#define _TeensyPulseGenerator_H_

#include "DeviceBase.h"
#include "DeviceUtils.h"
#include "ModuleInterface.h"
#include "MMDevice.h"
#include "SingleThread.h"
#include <cstdio>
#include <string>
#include <vector>
#include <mutex>

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
    int OnStatus(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnRunUntilStopped(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnNrPulses(MM::PropertyBase* pProp, MM::ActionType eAct);

private:
    std::atomic<bool> initialized_;
    std::string port_;

    // Current configuration
    double interval_;     // Interval in milli-seconds
    double pulseDuration_; // Pulse duration in milli-seconds
    bool triggerMode_;    // Trigger mode enabled/disabled
    bool running_;        // Pulse generator running state
    bool runUntilStopped_;
    uint32_t version_;
    uint32_t nrPulses_;
    std::mutex mutex_;
    SingleThread singleThread_;

    // Helper methods for serial communication
    int SendCommand(uint8_t cmd, uint32_t param = 0);
    int Enquire(uint8_t cmd);
    int GetResponse(uint8_t cmd, uint32_t& param);
    void CheckStatus();
};

#endif
