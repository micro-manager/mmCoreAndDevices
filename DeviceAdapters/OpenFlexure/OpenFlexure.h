///////////////////////////////////////////////////////////////////////////////
// FILE:          OpenFlexure.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Adapter for the OpenFlexure Microscope. This adapter is used on the v5 Sangaboard. 
//					- Tested with Sangaboard Firmware v1.0.1-dev || Sangaboard v0.5.x
//                
// AUTHOR:        Samdrea Hsu, samdreahsu@gmail.com, 09/23/2024
//
// COPYRIGHT:     Samdrea Hsu
//
// LICENSE:       This file is distributed under the BSD license.
//                License text is included with the source distribution.
//
//                This file is distributed in the hope that it will be useful,
//                but WITHOUT ANY WARRANTY; without even the implied warranty
//                of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
//
//                IN NO EVENT SHALL THE COPYRIGHT OWNER OR
//                CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
//                INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES.
//
//////////////////////////////////////////////////////////////////////////////

#ifndef _OpenFlexure_H_
#define _OpenFlexure_H_

#include "MMDevice.h"
#include "DeviceBase.h"
#include <string>
#include <map>

// Global keywords 
const char* g_XYStageDeviceName = "XY Stage";
const char* g_HubDeviceName = "SangaboardHub";
const char* g_ZStageDeviceName = "Z Stage";
const char* g_ShutterDeviceName = "LED illumination";
const char* g_Keyword_Response = "SerialResponse";
const char* g_Keyword_Command = "SerialCommand";
const char* g_Keyword_Brightness = "LED Brightness";
const char* g_Keyword_StepDelay = "Stage Step Delay (us)";
const char* g_Keyword_RampTime = "Stage Ramp Time (us)";
const char* g_Keyword_Extras = "Xtra Stage Commands";
const char* g_Keyword_None = "None";
const char* g_ExtraCommand_Stop = "stop";
const char* g_ExtraCommand_Zero = "zero";
const char* g_ExtraCommand_Release = "release";
const char* g_ExtraCommand_Version = "version";
const char* NoHubError = "Parent Hub not defined.";
const char* const g_Msg_DEVICE_STAGE_STILL_MOVING = "Stage is still moving. Current move aborted.";

// Custom Error texts
#define DEVICE_STAGE_STILL_MOVING     42

class SangaBoardHub : public HubBase<SangaBoardHub>
{
public:
	SangaBoardHub();
	~SangaBoardHub();

	// MMDevice API
	int Initialize();
	int Shutdown();
	void GetName(char* pszName) const;
	bool Busy();

	// Hub API
	int DetectInstalledDevices();

	// Action Handlers
	int OnPort(MM::PropertyBase* pPropt, MM::ActionType eAct);
	int OnManualCommand(MM::PropertyBase* pPropt, MM::ActionType eAct);
	int OnStepDelay(MM::PropertyBase* pPropt, MM::ActionType eAct);
	int OnRampTime(MM::PropertyBase* pPropt, MM::ActionType eAct);
	int OnExtraCommands(MM::PropertyBase* pPropt, MM::ActionType eAct);

	// Helper Functions
	void GetPort(std::string& port);
	int SendCommand(std::string cmd, std::string& res);
	int SyncState();
	long ExtractNumber(std::string serial_output);
	int SyncPeripherals();

private:
	bool initialized_;
	bool busy_;
	bool portAvailable_;
	long step_delay_;
	long ramp_time_;
	std::string port_;
	std::string _command;
	std::string _serial_answer;
	MMThreadLock serial_lock_;

	bool IsPortAvailable() { return portAvailable_; }
};

class XYStage : public  CXYStageBase<XYStage>
{
public:
	XYStage();
	~XYStage();

	// MMDevice API
	int Initialize();
	int Shutdown();

	// XYStage API
	int SetPositionSteps(long x, long y);
	int GetPositionSteps(long& x, long& y);
	int SetRelativePositionSteps(long x, long y);
	int GetPositionUm(double& posX, double& posY);
	int SetPositionUm(double posX, double posY);
	int SetRelativePositionUm(double posX, double posY);
	int SetOrigin();
	int SetAdapterOrigin();
	int Home();
	int Stop();
	double GetStepSizeXUm() { return stepSizeUm_; }
	double GetStepSizeYUm() { return stepSizeUm_; }
	int GetStepLimits(long& xMin, long& xMax, long& yMin, long& yMax);
	int GetLimitsUm(double& xMin, double& xMax, double& yMin, double& yMax);
	int IsXYStageSequenceable(bool& isSequenceable) const { isSequenceable = false; return DEVICE_OK; }

	bool Busy() { return false; }
	void GetName(char*) const;

	// Helper functions
	int SyncState();

private:
	long stepsX_;
	long stepsY_;
	bool initialized_;
	bool portAvailable_;
	double stepSizeUm_;
	std::string port_;
	std::string _serial_answer;
	SangaBoardHub* pHub;

	bool IsPortAvailable() { return portAvailable_; }
};

class ZStage : public CStageBase<ZStage>
{
public:
	ZStage();
	~ZStage();

	// MMDevice API
	int Initialize();
	int Shutdown();
	void GetName(char* name) const;

	// ZStage API
	int SetPositionUm(double pos) { return DEVICE_UNSUPPORTED_COMMAND;}
	int SetPositionSteps(long steps) { return DEVICE_UNSUPPORTED_COMMAND;}
	int SetRelativePositionUm(double d);
	int SetRelativePositionSteps(long z);
	int Stop() {return DEVICE_UNSUPPORTED_COMMAND;} 
	int GetPositionUm(double& pos);
	int GetPositionSteps(long& steps);
	int SetOrigin();
	int GetLimits(double& lower, double& upper) { return DEVICE_UNSUPPORTED_COMMAND;} // nah 
	int IsStageSequenceable(bool& isSequenceable) const { isSequenceable = false;  return DEVICE_OK;}
	bool IsContinuousFocusDrive() const  { return false; }
	bool Busy() { return false; }

	// Helper functions
	int SyncState();

private:
	long stepsZ_;
	bool initialized_;
	double stepSizeUm_;
	std::string _serial_answer;
	SangaBoardHub* pHub;
};

class LEDIllumination : public CShutterBase<LEDIllumination>
{
public:
	LEDIllumination() : state_(false), initialized_(false), changedTime_(0.0), brightness_(1.0), pHub(NULL){}
	~LEDIllumination();

	// MMDevice API
	int Initialize();
	int Shutdown();
	void GetName(char* name) const;


	// Shutter API
	int SetOpen(bool open);
	int GetOpen(bool& open);
	int Fire(double deltaT) { return DEVICE_UNSUPPORTED_COMMAND;}
	bool Busy() { return false; }

	// Action Handlers
	int OnState(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnBrightness(MM::PropertyBase* pPropt, MM::ActionType eAct);

	// Helper functions
	int SyncState();
	int SetBrightness();

private:
	bool state_;
	bool initialized_;
	double brightness_;
	MM::MMTime changedTime_;
	std::string _serial_answer;
	SangaBoardHub* pHub;
};

#endif 


