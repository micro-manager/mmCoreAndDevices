// Micro-Manager Device Adapter for Spectra-Physics Ultrafast Laser Systems
// Author: Gabriel Selzer
//
// Copyright 2026 Board of Regents of the University of Wisconsin System
//
// This file is distributed under the BSD license. License text is included
// with the source distribution.
//
// This file is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE.
//
// IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES.

#pragma once

#include "MMDevice.h"
#include "DeviceBase.h"

#include <chrono>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <thread>

//////////////////////////////////////////////////////////////////////////////
// Error codes
//
#define ERR_PORT_CHANGE_FORBIDDEN       10001
#define ERR_PUMP_LASER_NOT_WARM         10002
#define ERR_WAVELENGTH_CHANGING         10003
#define ERR_PUMP_LASER_TURNING_ON       10004
#define ERR_NO_HUB				        10005
#define ERR_INVALID_MODEL               10006
#define ERR_WATCHDOG_START_FAILED       10007

// Forward declaration
class SpectraPhysicsHub;

enum LaserModel {
	INSIGHT,
	MAITAI
};

class WatchdogThread
{
public:
    explicit WatchdogThread(SpectraPhysicsHub& device);
    ~WatchdogThread();
    int Start();
    void Stop();
private:
    void Run();
    SpectraPhysicsHub& device_;
    std::mutex stopMutex_;
    std::condition_variable timerCV_;
    bool stop_;
    std::chrono::steady_clock::duration interval_;
    std::thread thread_;
};

class SpectraPhysicsHub : public HubBase<SpectraPhysicsHub>
{
    friend class WatchdogThread;
    friend class SpectraPhysicsMain;
    friend class SpectraPhysicsInsight1040;
public:
	SpectraPhysicsHub();
	~SpectraPhysicsHub();

	// MMDevice API
	int Initialize();
	int Shutdown();
	void GetName(char* name) const;
	bool Busy();

	// Hub API
	int DetectInstalledDevices();

private:
	// Device state
	bool initialized_;
	LaserModel model_;
	std::string port_;
	std::string lastCommand_;
	std::string onClose_;
	std::chrono::steady_clock::time_point lastCommandTime_;
	std::mutex serialMutex_;
	WatchdogThread watchdogThread_;
	
	// Pre-Init Actions
	int OnPort(MM::PropertyBase * pProp, MM::ActionType eAct);
	// Actions
	int OnWatchdog(MM::PropertyBase * pProp, MM::ActionType eAct);
	int OnPumpLaser(MM::PropertyBase* pProp, MM::ActionType eAct);
	// Read-only Actions
	int OnWarmup(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnHumidity(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnDiode1Current(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnDiode2Current(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnDiode1Temp(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnDiode2Temp(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnDiode1Hours(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnDiode2Hours(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnPower(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnHistoryBuffer(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnLaserState(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnEmission(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnPulsing(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnServoOn(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnUserInterlock(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnKeyswitchInterlock(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnPowerSupplyInterlock(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnInternalInterlock(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnWarning(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnFault(MM::PropertyBase* pProp, MM::ActionType eAct);

	// Private Helpers
	int ExecuteCommand(const std::string& cmd);
	int ExecuteCommand(const std::string& cmd, std::string& answer);
	int SendCommand(const std::string& cmd);
	int StatusBit(unsigned int bitNumber, bool& bit);
	int LaserState(int& state);
};

class SpectraPhysicsMain : public CShutterBase<SpectraPhysicsMain>
{
public:
    SpectraPhysicsMain();
    ~SpectraPhysicsMain();

    // MMDevice API
    int Initialize();
    int Shutdown();
    void GetName(char* name) const;
    bool Busy();

	// Shutter API
	int SetOpen(bool open = true);
	int GetOpen(bool& open);
	int Fire(double deltaT);

	// Actions
	int OnTargetWavelength(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnActualWavelength(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnState(MM::PropertyBase* pProp, MM::ActionType eAct);
private:
	// Device state
	bool initialized_;
	SpectraPhysicsHub* parent_;
};

class SpectraPhysicsInsight1040 : public CShutterBase<SpectraPhysicsInsight1040>
{
public:
    SpectraPhysicsInsight1040();
    ~SpectraPhysicsInsight1040();

    // MMDevice API
    int Initialize();
    int Shutdown();
    void GetName(char* name) const;
    bool Busy();

	// Shutter API
	int SetOpen(bool open = true);
	int GetOpen(bool& open);
	int Fire(double deltaT);

	// Actions
	int OnState(MM::PropertyBase* pProp, MM::ActionType eAct);
private:
	// Device state
	bool initialized_;
	SpectraPhysicsHub* parent_;
};
