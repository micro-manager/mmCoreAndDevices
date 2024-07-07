//////////////////////////////////////////////////////////////////////////////
// FILE:          OpenFlexure.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Adapter for the OpenFlexure Microscope. This adapter is used on the v5 Sangaboard.
//
// COPYRIGHT:     Samdrea Hsu
//
// AUTHOR:        Samdrea Hsu, samdreahsu@gmail.com, 06/22/2024
//
//////////////////////////////////////////////////////////////////////////////

#include "OpenFlexure.h"
#include "ModuleInterface.h"
#include <sstream>
#include <cstdio>
#include <cstring>
#include <string>
#include <algorithm>
#include <math.h>

#ifdef WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

///////////////////////////////////////////////////////////////////////////////
// Exported MMDevice API
///////////////////////////////////////////////////////////////////////////////
MODULE_API void InitializeModuleData()
{
	RegisterDevice(g_XYStageDeviceName, MM::XYStageDevice, "XY Stage");
	RegisterDevice(g_HubDeviceName, MM::HubDevice, "Sangaboard Hub");
	RegisterDevice(g_ZStageDeviceName, MM::StageDevice, "Z Stage");
	RegisterDevice(g_ShutterDeviceName, MM::ShutterDevice, "LED Illumination");
}

MODULE_API MM::Device* CreateDevice(const char* deviceName)
{
	if (deviceName == 0)
		return 0;

	if (strcmp(deviceName, g_XYStageDeviceName) == 0)
		return new XYStage; // Create XY stage
	else if (strcmp(deviceName, g_HubDeviceName) == 0)
		return new SangaBoardHub; // Create hub
	else if (strcmp(deviceName, g_ZStageDeviceName) == 0)
		return new ZStage; // Create Z stage
	else if (strcmp(deviceName, g_ShutterDeviceName) == 0)
		return new LEDIllumination; // Create LED shutter device

	return 0; // Device name not recognized
}

MODULE_API void DeleteDevice(MM::Device* pDevice)
{
	delete pDevice;
}

///////////////////////////////////////////////////////////////////////////////
// Sangaboard hub implementation
///////////////////////////////////////////////////////////////////////////////

SangaBoardHub::SangaBoardHub() : initialized_(false), port_("Undefined"), portAvailable_(false), busy_(false)
{
	// Initialize default error messages
	InitializeDefaultErrorMessages();

	// Initialize custom error messages
	SetErrorText(DEVICE_STAGE_STILL_MOVING, g_Msg_DEVICE_STAGE_STILL_MOVING);

	// Pre-initialization property: port name
	CPropertyAction* pAct = new CPropertyAction(this, &SangaBoardHub::OnPort);
	CreateProperty(MM::g_Keyword_Port, "Undefined", MM::String, false, pAct, true);
}

SangaBoardHub::~SangaBoardHub()
{
	Shutdown();
}

int SangaBoardHub::Initialize()
{
	initialized_ = true;

	// Set up Manual Command Interface to send command directly to SangaBoard
	CPropertyAction* pCommand = new CPropertyAction(this, &SangaBoardHub::OnManualCommand);
	int ret = CreateProperty(g_Keyword_Command, "", MM::String, false, pCommand);
	assert(DEVICE_OK == ret);

	// Set up property to display most recent serial response (read-only)
	ret = CreateProperty(g_Keyword_Response, "", MM::String, false);
	assert(DEVICE_OK == ret);

	// Initialize the cache values to the current values stored on hardware
	this->SyncState();

	// Set up Step Delay property for changing velocity
	CPropertyAction* pStepDel = new CPropertyAction(this, &SangaBoardHub::OnStepDelay);
	ret = CreateIntegerProperty(g_Keyword_StepDelay, step_delay_, false, pStepDel);
	assert(DEVICE_OK == ret);
	AddAllowedValue(g_Keyword_StepDelay, "1000"); 
	AddAllowedValue(g_Keyword_StepDelay, "2000");
	AddAllowedValue(g_Keyword_StepDelay, "3000");
	AddAllowedValue(g_Keyword_StepDelay, "4000");
	AddAllowedValue(g_Keyword_StepDelay, "5000");

	// Set up Ramp Time property for changing acceleration
	CPropertyAction* pRampTime = new CPropertyAction(this, &SangaBoardHub::OnRampTime);
	ret = CreateIntegerProperty(g_Keyword_RampTime, ramp_time_, false, pRampTime);
	assert(DEVICE_OK == ret);
	AddAllowedValue(g_Keyword_RampTime, "0", 0);
	AddAllowedValue(g_Keyword_RampTime, "100000"); // Not really sure if these values really do much for acceleration...
	AddAllowedValue(g_Keyword_RampTime, "200000");
	AddAllowedValue(g_Keyword_RampTime, "300000");

	// Set up extra functions drop down menu
	CPropertyAction* pExtras= new CPropertyAction(this, &SangaBoardHub::OnExtraCommands);
	ret = CreateStringProperty(g_Keyword_Extras, g_Keyword_None, false, pExtras);
	assert(DEVICE_OK == ret);
	AddAllowedValue(g_Keyword_Extras, g_ExtraCommand_Stop);
	AddAllowedValue(g_Keyword_Extras, g_ExtraCommand_Zero);
	AddAllowedValue(g_Keyword_Extras, g_ExtraCommand_Release);
	AddAllowedValue(g_Keyword_Extras, g_ExtraCommand_Version);

	return DEVICE_OK;
}

int SangaBoardHub::Shutdown()
{
	initialized_ = false;
	return DEVICE_OK;
}

int SangaBoardHub::DetectInstalledDevices()
{
	ClearInstalledDevices();

	char hubName[MM::MaxStrLength];
	GetName(hubName); // this device name
	for (unsigned i = 0; i < GetNumberOfDevices(); i++)
	{
		char deviceName[MM::MaxStrLength];
		bool success = GetDeviceName(i, deviceName, MM::MaxStrLength);
		if (success && (strcmp(hubName, deviceName) != 0))
		{
			MM::Device* pDev = CreateDevice(deviceName);
			AddInstalledDevice(pDev);
		}
	}
	return DEVICE_OK;
}

void SangaBoardHub::GetName(char* name) const
{
	CDeviceUtils::CopyLimitedString(name, g_HubDeviceName);
}

/*
* Make sure no moves are in progress
*/
bool SangaBoardHub::Busy()
{
	//MM::MMTime timeout(0, 500000); // wait for 0.5 sec

	PurgeComPort(port_.c_str());

	// Send a query to check if stage is moving
	int ret = SendSerialCommand(port_.c_str(), "moving?", "\n");

	// Check response
	GetSerialAnswer(port_.c_str(), "\r", _serial_answer); // Should return "\ntrue" or "\nfalse"

	return _serial_answer.find("true") != std::string::npos;
}

///////////////////////////////////////////////////////////////////////////////
// SangaboardHub Action handlers
///////////////////////////////////////////////////////////////////////////////

int SangaBoardHub::OnPort(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set(port_.c_str());
	}
	else if (eAct == MM::AfterSet)
	{
		pProp->Get(port_);
		portAvailable_ = true;
	}

	return DEVICE_OK;
}

int SangaBoardHub::OnManualCommand(MM::PropertyBase* pProp, MM::ActionType pAct)
{
	if (pAct == MM::BeforeGet)
	{
		pProp->Set(_command.c_str());
	}
	else if (pAct == MM::AfterSet)
	{
		// Get and send the command typed into property
		pProp->Get(_command);
		PurgeComPort(port_.c_str());
		SendCommand(_command, _serial_answer);

		// Remember the response
		std::string ans = _serial_answer;

		// Sync the hub itself
		this->SyncState();

		// Sync the peripherals xy, z, and LED to possible changes made through serial call
		this->SyncPeripherals();

		// Display the response to the command
		SetProperty(g_Keyword_Response, ans.c_str());
	}

	// Search for error
	return _serial_answer.find("ERROR") != std::string::npos ? DEVICE_ERR : DEVICE_OK;
}

int SangaBoardHub::OnStepDelay(MM::PropertyBase* pPropt, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		pPropt->Set(step_delay_);
	}
	else if (eAct == MM::AfterSet)
	{
		pPropt->Get(step_delay_);
		std::string cmd = "dt " + std::to_string(step_delay_);
		this->SendCommand(cmd, _serial_answer);
	}

	return DEVICE_OK;
}

int SangaBoardHub::OnRampTime(MM::PropertyBase* pPropt, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		pPropt->Set(ramp_time_);
	}
	else if (eAct == MM::AfterSet)
	{
		pPropt->Get(ramp_time_);
		std::string cmd = "ramp_time " + std::to_string(ramp_time_);
		this->SendCommand(cmd, _serial_answer);
	}

	return DEVICE_OK;

}

int SangaBoardHub::OnExtraCommands(MM::PropertyBase* pPropt, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		pPropt->Set(g_Keyword_None);
	}
	else if (eAct == MM::AfterSet)
	{
		std::string cmd;
		pPropt->Get(cmd);
		this->SendCommand(cmd, _serial_answer);
		
		// Remember the response
		std::string ans = _serial_answer;

		// Sync the stages, because the extra commands mostly change the state of the stage
		this->SyncPeripherals();

		// Display the response to the command
		SetProperty(g_Keyword_Response, ans.c_str());
	}

	return DEVICE_OK;
}

///////////////////////////////////////////////////////////////////////////////
// SangaboardHub Helper Functions
///////////////////////////////////////////////////////////////////////////////
/**
* Return port name
*/
void SangaBoardHub::GetPort(std::string& port)
{
	port = this->port_;
}

/**
* Manages serial port calls by peripheral devices
* Locks thread to use serial port so only one serial call is done at a time
*/
int SangaBoardHub::SendCommand(std::string cmd, std::string& ans)
{
	// Lock the serial port with threadguard
	MMThreadGuard g(serial_lock_);

	// Allow previous move to finish before sending new command through serial port
	if (Busy() && cmd.find("mr") != -1) {
			return DEVICE_STAGE_STILL_MOVING;
	}

	// Send command and receive response
	int ret = SendSerialCommand(port_.c_str(), cmd.c_str(), "\n");

	if (ret != DEVICE_OK) {
		return ret;
	}

	ret = GetSerialAnswer(port_.c_str(), "\r", ans);

	if (ret != DEVICE_OK) {
		return ret;
	}

	// Reflect the command in the serial response property
	SetProperty(g_Keyword_Response, ans.c_str());

	return DEVICE_OK;

} // End of function automatically unlocks the lock

/**
* Prompt device to validate MM cached values for the hub 
*/
int SangaBoardHub::SyncState()
{
	// Update minimum step delay
	std::string cmd = "dt?";
	this->SendCommand(cmd, _serial_answer); // Output is something like "minimum step delay 1000"

	// Get the step delay from response 
	step_delay_ = ExtractNumber(_serial_answer);

	// Update ramp time
	cmd = "ramp_time?";
	this->SendCommand(cmd, _serial_answer); // Output is something like "ramp_time 0”

	// Get the ramp time from response
	ramp_time_ = ExtractNumber(_serial_answer);

	return DEVICE_OK;
}

/**
* Sync the peripherals if they exist
*/
int SangaBoardHub::SyncPeripherals()
{
		// Look for peripherals that need syncing
		for (unsigned i = 0; i < GetNumberOfDevices(); i++) {

			char deviceName[MM::MaxStrLength];
			bool success = GetDeviceName(i, deviceName, MM::MaxStrLength);

			// Sync xy stage
			if (success && (strcmp(g_XYStageDeviceName, deviceName) == 0))
			{
				XYStage* xyStage = dynamic_cast<XYStage*>(GetDevice(deviceName));
				xyStage->SyncState();	
			}

			// Sync z stage
			if (success && (strcmp(g_ZStageDeviceName, deviceName) == 0))
			{
				ZStage* zStage = dynamic_cast<ZStage*>(GetDevice(deviceName));
				zStage->SyncState();
			}

			// Sync illumination
			if (success && (strcmp(g_ShutterDeviceName, deviceName) == 0))
			{
				LEDIllumination* light = dynamic_cast<LEDIllumination*>(GetDevice(deviceName));
				light->SyncState();
			}
		}

		return DEVICE_OK;
}

/**
* Parse a sentence for numbers
*/
long SangaBoardHub::ExtractNumber(std::string str) {
	std::stringstream ss(str);
	std::string temp;
	long found;

	// Running loop till the end of the stream
	while (ss >> temp) {
		// Checking if the given word is a long or not
		if (std::stringstream(temp) >> found) {
			return found; // Return the first long found
		}
	}
	return 0; // Return 0 if no long is found
}

///////////////////////////////////////////////////////////////////////////////
// XYStage implementation
///////////////////////////////////////////////////////////////////////////////

XYStage::XYStage() : initialized_(false), portAvailable_(false), stepSizeUm_(0.07), stepsX_(0), stepsY_(0), pHub(NULL)
{
	// Parent ID display
	//CreateHubIDProperty();

	// Initialize default error messages
	InitializeDefaultErrorMessages();
}

XYStage::~XYStage()
{
	Shutdown();
}

int XYStage::Initialize()
{
	if (initialized_)
		return DEVICE_OK;

	// Get hub instance
	pHub = static_cast<SangaBoardHub*>(GetParentHub());
	if (pHub)
	{
		char hubLabel[MM::MaxStrLength];
		pHub->GetLabel(hubLabel);
		SetParentID(hubLabel); // for backward comp.
	}
	else
		LogMessage(NoHubError);

	// Check status to see if device is ready to start
	int status = UpdateStatus();
	if (status != DEVICE_OK) {
		return status;
	}

	// Use non-blocking moves
	std::string cmd = "blocking_moves false";
	pHub->SendCommand(cmd, _serial_answer);
	if (_serial_answer.find("done") == -1) {
		return DEVICE_ERR;
	}

	// Set the current stage position
	SyncState();

	// Device is now initialized
	initialized_ = true;

	return DEVICE_OK; 
}

int XYStage::Shutdown()
{
	//De-energize the motors
	std::string cmd = "release";
	pHub->SendCommand(cmd, _serial_answer);
	initialized_ = false;
	return DEVICE_OK;
}

int XYStage::SetPositionSteps(long x, long y)
{
	// Probably best to call SetRelativePosition() to complete this function
	return DEVICE_OK;
}

int XYStage::SetPositionUm(double posX, double posY)
{
	// Manual changing position
	return DEVICE_OK;
}

int XYStage::GetPositionUm(double& posX, double& posY)
{
	posX = stepsX_ * stepSizeUm_;
	posY = stepsY_ * stepSizeUm_;

	return DEVICE_OK;
}

/**
* Should be called by GetPositionUm(), but probably redundant, because I'm keeping stepsX_ and stepsY_ global variables
*/
int XYStage::GetPositionSteps(long& x, long& y)
{
	x = stepsX_;
	y = stepsY_;

	return DEVICE_OK;
}

int XYStage::SetRelativePositionUm(double dx, double dy)
{
	long dxSteps = nint(dx / stepSizeUm_);
	long dySteps = nint(dy / stepSizeUm_);
	int ret = SetRelativePositionSteps(dxSteps, dySteps); // Stage starts moving after this step

	if (ret == DEVICE_OK) {
		stepsX_ += dxSteps;
		stepsY_ += dySteps;
		this->OnXYStagePositionChanged(stepsX_ * stepSizeUm_, stepsY_ * stepSizeUm_);
	}

	return DEVICE_OK;
}

int XYStage::SetRelativePositionSteps(long x, long y)
{

	// Sending two commands sequentially
	std::ostringstream cmd;
	cmd << "mrx " << x << "\nmry " << y; // move in x first then y (arbitrary choice)

	int ret = pHub->SendCommand(cmd.str(), _serial_answer);

	return ret;
}

int XYStage::SetOrigin()
{

	// Set current position as origin (all motor positions set to 0)
	std::string cmd = "zero";
	int ret = pHub->SendCommand(cmd, _serial_answer);

	return ret;
}

int XYStage::SetAdapterOrigin()
{
	//Could be the function to sync adapter to the stage's actual positions...
	return DEVICE_OK;
}

/**
* Return the device to the origin 
*/
int XYStage::Home()
{
	//TODO: Query the position steps and set the number of steps to opposite that
	return DEVICE_OK;
}

int XYStage::Stop()
{
	// send the stop command to the stage
	std::string cmd = "stop";
	pHub->SendCommand(cmd, _serial_answer);

	// Make sure current position is synched
	SyncState();

	return DEVICE_OK;

}

int XYStage::GetStepLimits(long& xMin, long& xMax, long& yMin, long& yMax)
{
	return DEVICE_OK;
}

int XYStage::GetLimitsUm(double& xMin, double& xMax, double& yMin, double& yMax)
{
	return DEVICE_OK;
}

void XYStage::GetName(char* name) const
{
	CDeviceUtils::CopyLimitedString(name, g_XYStageDeviceName);
}

///////////////////////////////////////////////////////////////////////////////
// XYStage Helper Functions
///////////////////////////////////////////////////////////////////////////////

/**
* Sync the starting position of the stage to the cached values in the adapter
*/
int XYStage::SyncState()
{
	// Query for the current position [x y z] of the stage
	std::string cmd = "p";
	pHub->SendCommand(cmd, _serial_answer);

	// Parse the position of the stage into the x and y componennts
	std::istringstream iss(_serial_answer);

	iss >> stepsX_;

	iss >> stepsY_;

	// Reflect the synch-ed state in display
	int ret = OnXYStagePositionChanged(stepsX_ * stepSizeUm_, stepsY_ * stepSizeUm_);

	if (ret != DEVICE_OK) {
		return ret;
	}

	return DEVICE_OK;
}

///////////////////////////////////////////////////////////////////////////////
// ZsStage implementation
///////////////////////////////////////////////////////////////////////////////

ZStage::ZStage() : initialized_(false), stepSizeUm_(0.05), stepsZ_(0), pHub(NULL)
{
	// Parent ID display
	//CreateHubIDProperty();

	// Initialize default error messages
	InitializeDefaultErrorMessages();
}

ZStage::~ZStage()
{
	Shutdown();
}

int ZStage::Initialize()
{
	if (initialized_) 
		return DEVICE_OK;

	// Create pointer to parent hub (sangaboard)
	pHub = static_cast<SangaBoardHub*>(GetParentHub());
	if (pHub)
	{
		char hubLabel[MM::MaxStrLength];
		pHub->GetLabel(hubLabel);
		SetParentID(hubLabel); // for backward comp.
	}
	else
		LogMessage(NoHubError);

	// Check status to see if device is ready to start
	int status = UpdateStatus();
	if (status != DEVICE_OK) {
		return status;
	}

	// Use non-blocking moves
	std::string cmd = "blocking_moves false";
	pHub->SendCommand(cmd, _serial_answer);
	if (_serial_answer.find("done") == -1) {
		return DEVICE_ERR;
	}

	// Set the current stagePosition
	SyncState();

	// Device is now initialized
	initialized_ = true;

	return DEVICE_OK;
}

int ZStage::Shutdown()
{ 
	// De-energize the motors
	std::string cmd = "release";
	pHub->SendCommand(cmd, _serial_answer);
	initialized_ = false; 
	return DEVICE_OK; 
}

int ZStage::GetPositionUm(double& pos)
{
	pos = stepsZ_ * stepSizeUm_;
	return DEVICE_OK;
}

int ZStage::GetPositionSteps(long& steps)
{
	steps = stepsZ_;
	return DEVICE_OK;
}

int ZStage::SetRelativePositionUm(double d)
{
	long dSteps = nint(d / stepSizeUm_);
	int ret = SetRelativePositionSteps(dSteps); // Stage starts moving after this step

	if (ret == DEVICE_OK) {
		stepsZ_ += dSteps;
		this->OnStagePositionChanged(stepsZ_ * stepSizeUm_);
	}

	return DEVICE_OK;
}


int ZStage::SetRelativePositionSteps(long z)
{
	// Concatenate the command and number of steps
	std::ostringstream cmd;
	cmd << "mrz " << z;

	// Send command through hub
	int ret = pHub->SendCommand(cmd.str(), _serial_answer);

	return ret;
}


int ZStage::SetOrigin()
{
	// Set current position as origin (all motor positions set to 0)
	std::string cmd = "zero";
	int ret = pHub->SendCommand(cmd, _serial_answer);

	return ret;
}

void ZStage::GetName(char* name) const
{
	CDeviceUtils::CopyLimitedString(name, g_ZStageDeviceName);
}

///////////////////////////////////////////////////////////////////////////////
// ZStage Helper functions
///////////////////////////////////////////////////////////////////////////////

/**
* Sync the starting position of the stage to the cached values in the adapter
*/
int ZStage::SyncState()
{
	// Query for the current position [x y z] of the stage
	std::string cmd = "p";
	pHub->SendCommand(cmd, _serial_answer);

	// Parse the position of the stage to just get the z position
	std::istringstream iss(_serial_answer);

	iss >> stepsZ_ >> stepsZ_ >> stepsZ_;

	// Reflect the synch-ed state in display
	int ret = OnStagePositionChanged(stepsZ_ * stepSizeUm_);

	if (ret != DEVICE_OK) {
		return ret;
	}

	return DEVICE_OK;
}

///////////////////////////////////////////////////////////////////////////////
// LED Illumination implementation
///////////////////////////////////////////////////////////////////////////////

LEDIllumination::~LEDIllumination()
{
	Shutdown();
}

int LEDIllumination::Initialize()
{
	pHub = static_cast<SangaBoardHub*>(GetParentHub());
	if (pHub)
	{
		char hubLabel[MM::MaxStrLength];
		pHub->GetLabel(hubLabel);
		SetParentID(hubLabel); // for backward comp.
	}
	else
		LogMessage(NoHubError);

	if (initialized_)
		return DEVICE_OK;

	// Sync current device values to cached values
	SyncState();

	// Set property list
	// ------------------
	// state
	CPropertyAction* pAct = new CPropertyAction(this, &LEDIllumination::OnState);
	int ret = CreateIntegerProperty(MM::g_Keyword_State, state_, false, pAct);
	if (ret != DEVICE_OK)
		return ret;

	AddAllowedValue(MM::g_Keyword_State, "0"); // Closed
	AddAllowedValue(MM::g_Keyword_State, "1"); // Open

	// Brightness property is a slider
	CPropertyAction* pActbr = new CPropertyAction(this, &LEDIllumination::OnBrightness);
	CreateProperty(g_Keyword_Brightness, std::to_string(brightness_).c_str(), MM::Float, false, pActbr);
	SetPropertyLimits(g_Keyword_Brightness, 0, 1.0);

	ret = UpdateStatus();
	if (ret != DEVICE_OK)
		return ret;

	initialized_ = true;

	return DEVICE_OK;
}

int LEDIllumination::Shutdown()
{ 
	// Turn LED off
	SetOpen(false);
	initialized_ = false; 
	return DEVICE_OK; 
}

int LEDIllumination::SetOpen(bool open)//bool open = true)
{
	state_ = open;
	changedTime_ = GetCurrentMMTime();

	if (state_ == true) {
		SetBrightness();
	}
	else {
		std::string cmd = "led_cc 0";
		pHub->SendCommand(cmd, _serial_answer);
	}

	return DEVICE_OK;
}

int LEDIllumination::GetOpen(bool& open)
{
	open = state_;
	return DEVICE_OK;
}

void LEDIllumination::GetName(char* name) const
{
	CDeviceUtils::CopyLimitedString(name, g_ShutterDeviceName);
}

///////////////////////////////////////////////////////////////////////////////
// LED Illumination Action handlers
///////////////////////////////////////////////////////////////////////////////

int LEDIllumination::OnState(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		if (state_)
			pProp->Set(1L);
		else
			pProp->Set(0L);
	}
	else if (eAct == MM::AfterSet)
	{
		// Set timer for the Busy signal
		changedTime_ = GetCurrentMMTime();

		long pos;
		pProp->Get(pos);

		// apply the value
		state_ = pos == 0 ? false : true;
		SetOpen(state_);
	}

	return DEVICE_OK;
}

int LEDIllumination::OnBrightness(MM::PropertyBase* pProp, MM::ActionType eAct) 
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set(brightness_); // set the property display in mm to the brightness variable stored on cache
	}
	else if (eAct == MM::AfterSet)
	{
		// get value from mm property display and store as brightness variable on cache
		pProp->Get(brightness_);  
		
		if (state_) {
			// actually send command to set brightness of the LedArray
			SetBrightness();
		}
	}
	return DEVICE_OK;
}

///////////////////////////////////////////////////////////////////////////////
// LED Illumination Helper Functions
///////////////////////////////////////////////////////////////////////////////

/**
* Queries actual hardware for current values to sync the MM cached values
*/
int LEDIllumination::SyncState()
{
	if (!initialized_) {
		return DEVICE_OK; // Keep device at default brightness
	}

	// Query for the brightness of the LED
	std::string cmd = "led_cc?";
	pHub->SendCommand(cmd, _serial_answer); // Output is something like CC LED:1.00, so i can't use hub's ExtractNumber()

	// Find the position of the numeric value in the response
	size_t colonPos = _serial_answer.find(':');
	if (colonPos != std::string::npos) {
		std::string valueStr = _serial_answer.substr(colonPos + 1); // Extract the substring after the colon
		
		double LEDVal = std::stod(valueStr);

		// If only sync if light was manually turned on to a certain brightness
		if (LEDVal != 0.0) {
			state_ = true;
			brightness_ = LEDVal;
		}
		else {
			state_ = false;
			// Note: brightness_ is not updated when LEDVal is 0.0, so it retains its last value.
		}
	}
	else {
		return DEVICE_ERR;
	}
	return DEVICE_OK;
}

/**
* Turn LED on to a certain brightness-
*/
int LEDIllumination::SetBrightness()
{
	// actually send command to set brightness of the LedArray
	std::string cmd = "led_cc " + std::to_string(brightness_);
	pHub->SendCommand(cmd, _serial_answer);

	return DEVICE_OK;
}