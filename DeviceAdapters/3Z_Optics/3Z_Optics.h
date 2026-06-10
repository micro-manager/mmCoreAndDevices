#pragma once
///////////////////////////////////////////////////////////////////////////////
// FILE:          3Z_Optics.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   3Z Optics Light Source driver
// COPYRIGHT:     3Z Optics
// LICENSE:       BSD license

#include "MMDevice.h"
#include "DeviceBase.h"
#include "DeviceThreads.h"
#include <string>
#include <vector>
#include <map>
#include <windows.h>
#include <cstdint>

//////////////////////////////////////////////////////////////////////////////
// Error codes
//
#define ERR_PORT_CHANGE_FORBIDDEN    13001
#define ERR_DEVICE_NOT_FOUND         13002
#define ERR_MODBUS_COMM_ERROR        13003
#define ERR_INIT                     13005

static const char* g_DeviceName = "3Z_Optics";
static const char* g_Prop_DeviceModel = "DeviceModel";
static const char* g_Prop_DeviceName = "DeviceName";

struct DeviceConfig
{
	std::string name;
	std::vector<std::string> channels;
	int brightnessMin;
	int brightnessMax;
};

class PollingThread;

class Controller : public CShutterBase<Controller>
{
public:
	Controller();
	~Controller();

	friend class PollingThread;

	// MMDevice API
	int Initialize();
	int Shutdown();

	void GetName(char* pszName) const;
	bool Busy();

	// Shutter API
	int SetOpen(bool open = true);
	int GetOpen(bool& open);
	int Fire(double /*interval*/) { return DEVICE_UNSUPPORTED_COMMAND; }

	// Action interface
	int OnPort(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnMode(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnRefresh(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnChannelSwitch(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnChannelIntensity(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnGlobalSwitch(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnGlobalIntensity(MM::PropertyBase* pProp, MM::ActionType eAct);

	static MMThreadLock& GetLock() { return lock_; }

private:
	bool initialized_;
	std::string port_;
	bool shutterState_;
	int deviceModelId_;
	DeviceConfig currentDevice_;
	std::vector<std::string> channels_;
	std::vector<bool> channelStates_;
	std::vector<int> channelIntensities_;
	std::map<std::string, int> channelSwitchLookup_;
	std::map<std::string, int> channelIntensityLookup_;
	int globalIntensity_;
	bool globalSwitch_;
	int currentMode_;
	double pollIntervalMs_;
	bool initializationComplete_;
	bool initializationInProgress_;

	// Update flags for properties
	std::vector<bool> channelSwitchUpdated_;
	std::vector<bool> channelIntensityUpdated_;
	bool globalSwitchUpdated_;
	bool globalIntensityUpdated_;
	bool modeUpdated_;

	static MMThreadLock lock_;
	MMThreadLock commLock_;
	PollingThread* mThread_;

	int ReadInputRegister(int addr, uint16_t& value);
	int WriteHoldingRegister(int addr, uint16_t value);
	int ReadHoldingRegister(int addr, uint16_t& value);
	int ReadMultipleHoldingRegisters(int startAddr, int count, std::vector<uint16_t>& values);
	int WriteSingleCoil(int addr, bool on);
	int ReadSingleCoil(int addr, bool& on);
	int ReadMultipleCoils(int startAddr, int count, std::vector<bool>& values);
	int ReadDeviceModel();
	bool LoadDeviceConfig(int modelId);
	int ApplyChannelStates();
	int TurnAllOff();
	int SetIntensity(int channel, int intensity);
	int ReadCurrentDeviceState();
	int PollDeviceStatus();
	int ReadAllChannelRegisters();
	int ReadDeviceStateByMode(int mode);
	void UpdatePropertiesFromDevice();
	uint16_t CalculateCRC(const uint8_t* data, size_t length);
	int SendModbusCommand(const std::vector<uint8_t>& request, std::vector<uint8_t>& response, int expectedResponseLength);
};

class PollingThread : public MMDeviceThreadBase
{
public:
	PollingThread(Controller& aController);
	~PollingThread();
	int svc();
	int open(void*) { return 0; }
	int close(unsigned long) { return 0; }

	void Start();
	void Stop() { stop_ = true; }

private:
	Controller& aController_;
	bool stop_;
};
