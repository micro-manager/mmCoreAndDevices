
#ifndef _TeensyPulseGenerator_H
#define _TeensyPulseGenerator_H_

#include "DeviceBase.h"
#include "DeviceUtils.h"
#include "ModuleInterface.h"
#include "MMDevice.h"
#include "SingleThread.h"
#include "TeensyCom.h"
#include <cstdio>
#include <string>
#include <vector>
#include <mutex>

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
    int OnNrPulsesCounted(MM::PropertyBase* pProp, MM::ActionType eAct);

private:
    std::atomic<bool> initialized_;
    std::string port_;
    TeensyCom* teensyCom_;

    // Current configuration
    double interval_;     // Interval in milli-seconds
    double pulseDuration_; // Pulse duration in milli-seconds
    bool triggerMode_;    // Trigger mode enabled/disabled
    bool running_;        // Pulse generator running state
    bool runUntilStopped_;
    uint32_t version_;
    uint32_t nrPulses_; // nr Pulses we request
    uint32_t nrPulsesCounted_; // as returned by the Teensy
    std::mutex mutex_;
    SingleThread singleThread_;

    // Helper methods for serial communication
    void CheckStatus();
};

#endif
