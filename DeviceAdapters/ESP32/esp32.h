//////////////////////////////////////////////////////////////////////////////
// FILE:          ESP32.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Adapter for ESP32 board
//                Needs accompanying firmware to be installed on the board
// COPYRIGHT:     University of California, San Francisco, 2008
// LICENSE:       LGPL
//
// AUTHOR:        Nico Stuurman, nico@cmp.ucsf.edu, 11/09/2008
//                automatic device detection by Karl Hoover
//
//

#ifndef _ESP32_H_
#define _ESP32_H_

#include "MMDevice.h"
#include "DeviceBase.h"
#include <string>
#include <map>

//////////////////////////////////////////////////////////////////////////////
// Error codes
//
#define ERR_UNKNOWN_POSITION 101
#define ERR_INITIALIZE_FAILED 102
#define ERR_WRITE_FAILED 103
#define ERR_CLOSE_FAILED 104
#define ERR_BOARD_NOT_FOUND 105
#define ERR_PORT_OPEN_FAILED 106
#define ERR_COMMUNICATION 107
#define ERR_NO_PORT_SET 108
#define ERR_VERSION_MISMATCH 109
#define ERR_HUB_UNAVAILABLE 110
#define ERR_UNKNOWN_AXIS 111

class ESP32InputMonitorThread;

class CESP32Hub : public HubBase<CESP32Hub>
{
public:
    CESP32Hub();
    ~CESP32Hub();

    int Initialize();
    int Shutdown();
    void GetName(char* pszName) const;
    bool Busy();

    bool SupportsDeviceDetection(void);
    MM::DeviceDetectionStatus DetectDevice(void);
    int DetectInstalledDevices();

    // property handlers
    int OnPort(MM::PropertyBase* pPropt, MM::ActionType eAct);
    int OnLogic(MM::PropertyBase* pPropt, MM::ActionType eAct);
    int OnVersion(MM::PropertyBase* pPropt, MM::ActionType eAct);

    // custom interface for child devices
    bool IsPortAvailable() { return portAvailable_; }
    bool IsLogicInverted() { return invertedLogic_; }
    bool IsTimedOutputActive() { return timedOutputActive_; }
    void SetTimedOutput(bool active) { timedOutputActive_ = active; }

    int PurgeComPortH() { return PurgeComPort(port_.c_str()); }
    int WriteToComPortH(const unsigned char* command, unsigned len) { return WriteToComPort(port_.c_str(), command, len); }
    int ReadFromComPortH(unsigned char* answer, unsigned maxLen, unsigned long& bytesRead)
    {
        return ReadFromComPort(port_.c_str(), answer, maxLen, bytesRead);
    }


    static MMThreadLock& GetLock() { return lock_; }
    void SetShutterState(unsigned state) { shutterState_ = state; }
    void SetSwitchState(unsigned state) { switchState_ = state; }
    unsigned GetShutterState() { return shutterState_; }
    unsigned GetSwitchState() { return switchState_; }
    int GetAxisInfoH(int axis, double& travel) { return GetAxisInfo(axis, travel); }


private:
    int GetControllerVersion(int&);
    std::string port_;
    bool initialized_;
    bool portAvailable_;
    bool invertedLogic_;
    bool timedOutputActive_;
    ;
    int version_;
    static MMThreadLock lock_;
    unsigned switchState_;
    unsigned shutterState_;

    int GetControllerInfo();
    int GetAxisInfo(int, double&);
    bool hasZStage_;
    bool hasXYStage_;
};

class CESP32Shutter : public CShutterBase<CESP32Shutter>
{
public:
    CESP32Shutter();
    ~CESP32Shutter();

    // MMDevice API
    // ------------
    int Initialize();
    int Shutdown();

    void GetName(char* pszName) const;
    bool Busy();

    // Shutter API
    int SetOpen(bool open = true);
    int GetOpen(bool& open);
    int Fire(double deltaT);

    // action interface
    // ----------------
    int OnOnOff(MM::PropertyBase* pProp, MM::ActionType eAct);

private:
    int WriteToPort(long lnValue);
    MM::MMTime changedTime_;
    bool initialized_;
    std::string name_;
};

class CESP32Switch : public CStateDeviceBase<CESP32Switch>
{
public:
    CESP32Switch();
    ~CESP32Switch();

    // MMDevice API
    // ------------
    int Initialize();
    int Shutdown();

    void GetName(char* pszName) const;
    bool Busy() { return busy_; }

    unsigned long GetNumberOfPositions()const { return numPos_; }

    // action interface
    // ----------------
    int OnState(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnDelay(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnRepeatTimedPattern(MM::PropertyBase* pProp, MM::ActionType eAct);
    /*
    int OnSetPattern(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnGetPattern(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnPatternsUsed(MM::PropertyBase* pProp, MM::ActionType eAct);
    */
    int OnSkipTriggers(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnStartTrigger(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnStartTimedOutput(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnBlanking(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnBlankingTriggerDirection(MM::PropertyBase* pProp, MM::ActionType eAct);

    int OnSequence(MM::PropertyBase* pProp, MM::ActionType eAct);

private:
    static const unsigned int NUMPATTERNS = 12;

    int OpenPort(const char* pszName, long lnValue);
    int WriteToPort(long lnValue);
    int ClosePort();
    int LoadSequence(unsigned size, unsigned char* seq);

    unsigned pattern_[NUMPATTERNS];
    unsigned delay_[NUMPATTERNS];
    int nrPatternsUsed_;
    unsigned currentDelay_;
    bool sequenceOn_;
    bool blanking_;
    bool initialized_;
    long numPos_;
    bool busy_;
};

class CESP32DA : public CSignalIOBase<CESP32DA>
{
public:
    CESP32DA(int channel);
    ~CESP32DA();

    // MMDevice API
    // ------------
    int Initialize();
    int Shutdown();

    void GetName(char* pszName) const;
    bool Busy() { return busy_; }

    // DA API
    int SetGateOpen(bool open);
    int GetGateOpen(bool& open) { open = gateOpen_; return DEVICE_OK; };
    int SetSignal(double volts);
    int GetSignal(double& volts) { volts_ = volts; return DEVICE_UNSUPPORTED_COMMAND; }
    int GetLimits(double& minVolts, double& maxVolts) { minVolts = minV_; maxVolts = maxV_; return DEVICE_OK; }

    int IsDASequenceable(bool& isSequenceable) const { isSequenceable = false; return DEVICE_OK; }

    // action interface
    // ----------------
    int OnVolts(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnMaxVolt(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnChannel(MM::PropertyBase* pProp, MM::ActionType eAct);

private:
    int WriteToPort(double lnValue);
    int WriteSignal(double volts);

    bool initialized_;
    bool busy_;
    double minV_;
    double maxV_;
    double volts_;
    double gatedVolts_;
    unsigned channel_;
    unsigned maxChannel_;
    bool gateOpen_;
    std::string name_;
};

class CESP32Input : public CGenericBase<CESP32Input>
{
public:
    CESP32Input();
    ~CESP32Input();

    int Initialize();
    int Shutdown();
    void GetName(char* pszName) const;
    bool Busy();

    int OnDigitalInput(MM::PropertyBase* pPropt, MM::ActionType eAct);
    int OnAnalogInput(MM::PropertyBase* pProp, MM::ActionType eAct, long channel);

    int GetDigitalInput(long* state);
    int ReportStateChange(long newState);

private:
    //   int ReadNBytes(CESP32Hub* h, unsigned int n, unsigned char* answer);
    int SetPullUp(int pin, int state);

    MMThreadLock lock_;
    ESP32InputMonitorThread* mThread_;
    char pins_[MM::MaxStrLength];
    char pullUp_[MM::MaxStrLength];
    int pin_;
    bool initialized_;
    std::string name_;
};

class ESP32InputMonitorThread : public MMDeviceThreadBase
{
public:
    ESP32InputMonitorThread(CESP32Input& aInput);
    ~ESP32InputMonitorThread();
    int svc();
    int open(void*) { return 0; }
    int close(unsigned long) { return 0; }

    void Start();
    void Stop() { stop_ = true; }
    ESP32InputMonitorThread& operator=(const ESP32InputMonitorThread&)
    {
        return *this;
    }


private:
    long state_;
    CESP32Input& aInput_;
    bool stop_;
};

class CESP32Stage : public CStageBase<CESP32Stage>
{
public:
    CESP32Stage();
    ~CESP32Stage();

    bool Busy();
    void GetName(char* pszName) const;

    int Initialize();
    int Shutdown();

    // Stage API
    int SetPositionUm(double pos);
    int GetPositionUm(double& pos);
    double GetStepSize() { return stepSizeUm_; }

    int SetPositionSteps(long steps);
    int GetPositionSteps(long& steps);

    int SetOrigin() { return DEVICE_UNSUPPORTED_COMMAND; }

    int GetLimits(double& lower, double& upper)
    {
        lower = lowerLimit_;
        upper = upperLimit_;
        return DEVICE_OK;
    }

    int Move(double /*v*/) { return DEVICE_UNSUPPORTED_COMMAND; }

    bool IsContinuousFocusDrive() const { return false; }

    int OnStageMinPos(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnStageMaxPos(MM::PropertyBase* pProp, MM::ActionType eAct);

    int IsStageSequenceable(bool& isSequenceable) const
    {
        isSequenceable = false; return DEVICE_OK;
    }
    int GetStageSequenceMaxLength(long& nrEvents) const
    {
        nrEvents = 0; return DEVICE_OK;
    }

private:
    double stepSizeUm_;
    double pos_um_;
    bool busy_;
    bool initialized_;
    double lowerLimit_;
    double upperLimit_;

    int MoveZ(double pos);
};

class CESP32XYStage : public CXYStageBase<CESP32XYStage>
{
public:
    CESP32XYStage();
    ~CESP32XYStage();

    bool Busy();
    void GetName(char* pszName) const;

    int Initialize();
    int Shutdown();

    int GetPositionSteps(long& x, long& y);
    int SetPositionSteps(long x, long y);
    int SetRelativePositionSteps(long, long);

    virtual int Home() { return DEVICE_UNSUPPORTED_COMMAND; }
    virtual int Stop() { return DEVICE_UNSUPPORTED_COMMAND; }
    virtual int SetOrigin() { return DEVICE_UNSUPPORTED_COMMAND; }

    virtual int GetLimits(double& lower, double& upper)
    {
        lower = lowerLimitX_;
        upper = upperLimitY_;
        return DEVICE_OK;
    }

    virtual int GetLimitsUm(double& xMin, double& xMax, double& yMin, double& yMax)
    {
        xMin = lowerLimitX_; xMax = upperLimitX_;
        yMin = lowerLimitY_; yMax = upperLimitY_;
        return DEVICE_OK;
    }

    virtual int GetStepLimits(long& /*xMin*/, long& /*xMax*/, long& /*yMin*/, long& /*yMax*/)
    {
        return DEVICE_UNSUPPORTED_COMMAND;
    }

    double GetStepSizeXUm()
    {
        return stepSize_X_um_;
    }

    double GetStepSizeYUm()
    {
        return stepSize_Y_um_;
    }

    int Move(double /*vx*/, double /*vy*/) { return DEVICE_OK; }

    int IsXYStageSequenceable(bool& isSequenceable) const { isSequenceable = false; return DEVICE_OK; }

    int OnXStageMinPos(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnXStageMaxPos(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnYStageMinPos(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnYStageMaxPos(MM::PropertyBase* pProp, MM::ActionType eAct);

    int OnStepsPerSecond(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnMicrostepMultiplier(MM::PropertyBase* pProp, MM::ActionType eAct);

private:
    double stepSize_X_um_;
    double stepSize_Y_um_;
    double posX_um_;
    double posY_um_;
    unsigned int driveID_X_;
    unsigned int driveID_Y_;

    bool busy_;
    MM::TimeoutMs* timeOutTimer_;
    double velocity_;
    bool initialized_;
    double lowerLimitX_;
    double upperLimitX_;
    double lowerLimitY_;
    double upperLimitY_;

    int MoveX(double);
    int MoveY(double);
};

#endif //_ESP32_H_
