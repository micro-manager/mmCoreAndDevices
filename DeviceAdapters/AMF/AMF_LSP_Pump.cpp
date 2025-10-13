///////////////////////////////////////////////////////////////////////////////
// FILE:          AMF_LSP_Pump.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Device adapter for AMP LSP Pumps
//                 
// AUTHOR:        Lars Kool, Institut Pierre-Gilles de Gennes, Paris, France
//
// YEAR:          2025
//                
// VERSION:       0.1
//
// LICENSE:       This file is distributed under the BSD license.
//                License text is included with the source distribution.
//
//                This file is distributed in the hope that it will be useful,
//                but WITHOUT ANY WARRANTY; without even the implied warranty
//                of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
//
//                IN NO EVENT SHALL THE COPYRIGHT OWNER OR
//                CONTRIBUTORS BE   LIABLE FOR ANY DIRECT, INDIRECT,
//                INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES.
//
//LAST UPDATE:    04.04.2025 LK


#include "AMF_Commands.h"
#include "AMF_LSP_Hub.h"
#include "AMF_LSP_Pump.h"

#include "DeviceBase.h"
#include "DeviceThreads.h"
#include "ModuleInterface.h"
#include "DeviceUtils.h"

#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

double AMF_LSP_SPEED_STEP_STANDARD = 0.00745;
double AMF_LSP_SPEED_STEP_HD = 0.000552;
int AMF_LSP_SPEED_MAX_STANDARD = 214750;
int AMF_LSP_SPEED_MAX_HD = 905970;
int AMF_LSP_SPEED_MIN_STANDARD = 1;
int AMF_LSP_SPEED_MIN_HD = 13;
int AMF_LSP_ACCELERATION_MIN = 100;
int AMF_LSP_ACCELERATION_MAX = 59590;
int AMF_LSP_N_STEPS = 24000;

///////////////////////////////////////////////////////////////////////////////
//  AMF_LSP_Pump class
//  Device adapter for AMF LSP pumps.
///////////////////////////////////////////////////////////////////////////////

AMF_LSP_Pump::AMF_LSP_Pump() :
    initialized_(false)
{
    // parent ID display
    CreateHubIDProperty();

    // Get pump type
    std::vector<std::string> allowedTypes = { "Standard", "HD" };
    CPropertyAction* pAct = new CPropertyAction(this, &AMF_LSP_Pump::OnPumpType);
    CreateStringProperty("Pump Type", "Standard", false, pAct, true);
    SetAllowedValues("Pump Type", allowedTypes);

    // Get maximum volume
    std::vector<std::string> allowedMaxVolumes = { "25", "50", "100", "250", "500", "1000", "2500", "5000" };
    pAct = new CPropertyAction(this, &AMF_LSP_Pump::OnMaxVolume);
    CreateIntegerProperty(MM::g_Keyword_Max_Volume, (long)maxVolumeUl_, false, pAct, true);
    SetAllowedValues(MM::g_Keyword_Max_Volume, allowedMaxVolumes);
}

AMF_LSP_Pump::~AMF_LSP_Pump() {
    Shutdown();
};

///////////////////////////////////////////////////////////////////////////////
//  AMF_LSP_Pump class
//  MMDevice API
///////////////////////////////////////////////////////////////////////////////

int AMF_LSP_Pump::Initialize()
{
    if (initialized_) { return DEVICE_OK; }

    // Link to parent Hub
    AMF_LSP_Hub* pHub = static_cast<AMF_LSP_Hub*>(GetParentHub());
    if (pHub)
    {
        char hubLabel[MM::MaxStrLength];
        pHub->GetLabel(hubLabel);
        SetParentID(hubLabel); // for backward comp.
    }
    else {
        LogMessage("Parent Hub not defined.");
    }
    pHub->GetPort(port_); // Get COM-port from Hub
    
    // Get device address
    int ret = GetAddress(address_);

    // Name
    name_ = AMF_LSP_Pump_Name + std::string(" ") + port_;
    ret = CreateStringProperty(MM::g_Keyword_Name, name_.c_str(), true);
    if (DEVICE_OK != ret) {
        LogMessage("Could not create " + std::string(MM::g_Keyword_Name) + " property.");
        return ret;
    }

    // Description
    ret = CreateStringProperty(MM::g_Keyword_Description, "AMF LSP Pump", true);
    if (DEVICE_OK != ret) {
        LogMessage("Could not create " + std::string(MM::g_Keyword_Description) + " property.");
        return ret;
    }
    
    // Home
    CPropertyAction* pAct;
    std::vector<std::string> allowedHomeValues = { "0", "1" };
    pAct = new CPropertyAction(this, &AMF_LSP_Pump::OnHome);
    ret = CreateIntegerProperty("Home pump", 0, false, pAct);
    SetAllowedValues("Home pump", allowedHomeValues);

    if (!IsHomed()) {
		Home();
        while (IsPumping()) { CDeviceUtils::SleepMs(100); }
    }

    // Step size
    SetNSteps(AMF_LSP_N_STEPS);
    GetNSteps(nSteps_);
    stepVolumeUl_ = maxVolumeUl_ / AMF_LSP_N_STEPS;

    // Current volume
    pAct = new CPropertyAction(this, &AMF_LSP_Pump::OnCurrentVolume);
    ret = CreateFloatProperty(MM::g_Keyword_Current_Volume, volumeUl_, true, pAct);

    // Flow rate
    if ("Standard" == pumpType_) {
        maxFlowrate_ = PulsesToFlowrate(AMF_LSP_SPEED_MAX_STANDARD);
    }
    else {
        maxFlowrate_ = PulsesToFlowrate(AMF_LSP_SPEED_MAX_HD);
    }
    pAct = new CPropertyAction(this, &AMF_LSP_Pump::OnFlowrate);
    ret = CreateFloatProperty(MM::g_Keyword_Flowrate, flowrateUlperSecond_, false, pAct);
    SetPropertyLimits(MM::g_Keyword_Flowrate, -maxFlowrate_, maxFlowrate_);

    // Acceleration
    ret = GetAcceleration(acceleration_);
    if (DEVICE_OK != ret) {
        LogMessage("Could not get acceleration during initialization.");
        return ret;
    }
    pAct = new CPropertyAction(this, &AMF_LSP_Pump::OnAcceleration);
    ret = CreateIntegerProperty("Pump Acceleration", acceleration_, false, pAct);
    SetPropertyLimits("Pump Acceleration", AMF_LSP_ACCELERATION_MIN, AMF_LSP_ACCELERATION_MAX);

    // Deceleration
    ret = GetDeceleration(deceleration_);
    if (DEVICE_OK != ret) {
        LogMessage("Could not get deceleration during initialization.");
        return ret;
    }
    pAct = new CPropertyAction(this, &AMF_LSP_Pump::OnDeceleration);
    ret = CreateIntegerProperty("Pump Deceleration", deceleration_, false, pAct);
    SetPropertyLimits("Pump Deceleration", AMF_LSP_ACCELERATION_MIN, AMF_LSP_ACCELERATION_MAX);

    // Run
    std::vector<std::string> allowedRunValues = { "0", "1" };
    pAct = new CPropertyAction(this, &AMF_LSP_Pump::OnRun);
    ret = CreateIntegerProperty("Run", 0, false, pAct);
    SetAllowedValues("Run", allowedRunValues);

    return ret;
}

int AMF_LSP_Pump::Shutdown()
{
    int ret = DEVICE_OK;
    if (Busy()) {
        ret = Stop();
    }
    return ret;
}

void AMF_LSP_Pump::GetName(char* pName) const
{
    CDeviceUtils::CopyLimitedString(pName, AMF_LSP_Pump_Name);
}

bool AMF_LSP_Pump::Busy()
{
    return busy_;
}


///////////////////////////////////////////////////////////////////////////////
//  AMF_LSP_Pump class
//  MMVolumetricPump API
///////////////////////////////////////////////////////////////////////////////

int AMF_LSP_Pump::Home()
{
    long value = 0;
    int ret = SendRecv(AMF_Command::Initialize, value);
    if (ret != DEVICE_OK) {
        LogMessage("Could not home pump. Please check the manual for further instructions.");
        return ret;
    }
    isHomed_ = true;
    return ret;
}

int AMF_LSP_Pump::Stop()
{
    long value = 0;
    int ret = SendRecv(AMF_Command::Stop, value);
    if (ret != DEVICE_OK) {
        LogMessage("Could not stop pump. Please check the manual for further instructions.");
        return ret;
    }
    return ret;
}

int AMF_LSP_Pump::GetMaxVolumeUl(double& volUl)
{
    volUl = maxVolumeUl_;
    return DEVICE_OK;
}

int AMF_LSP_Pump::SetMaxVolumeUl(double /* volUl */)
{
    LogMessage(std::string("AMF Pumps do not support the manual change of maximum volume") + 
        ", as it is pump specific and not user modifiable.");
    return DEVICE_UNSUPPORTED_COMMAND;
}

int AMF_LSP_Pump::GetVolumeUl(double& volUl)
{
    long value = 0;
    int ret = SendRecv(AMF_Command::Get_plunger_position, value);
    ret = CheckStatus();
    if (ret != DEVICE_OK) {
        LogMessage("Could not get the current volume.");
        return ret;
    }
    volUl = StepsToVolume(value);
    return ret;
}

int AMF_LSP_Pump::SetVolumeUl(double /* volUl */)
{
    LogMessage(std::string("AMF pumps do not support the manual change of volume in the pump.") +
        " The volume can only be changed by withdrawing or dispensing liquid.");
    return DEVICE_UNSUPPORTED_COMMAND;
}

int AMF_LSP_Pump::IsDirectionInverted(bool& invert)
{
    invert = false;
    return DEVICE_OK;
}

int AMF_LSP_Pump::InvertDirection(bool /* invert */)
{
    return DEVICE_UNSUPPORTED_COMMAND;
}

int AMF_LSP_Pump::GetFlowrateUlPerSecond(double& flowrate)
{
    flowrate =  flowrateUlperSecond_;
    return DEVICE_OK;
}

int AMF_LSP_Pump::SetFlowrateUlPerSecond(double flowrate)
{
    // Set flowrate
    // Withdraw/dispense is controlled by different commands. Flowrate stays positive.
    long value = FlowrateToPulses((abs(flowrate)));
    LogMessage(std::string("Setting flowrate to: " + std::to_string(value)));
    int ret = SendRecv(AMF_Command::Set_flowrate, value);
    if (ret != DEVICE_OK) {
        LogMessage("Could not set flowrate.");
        return ret;
    }

    // Get actual flowrate after set
    ret = SendRecv(AMF_Command::Get_flowrate, value);
    if (ret != DEVICE_OK) {
        LogMessage("Could not get flowrate.");
        return ret;
    }
    // Make sure flowrate has correct sign.
    flowrateUlperSecond_ = (flowrate > 0) ?
        PulsesToFlowrate(value) : -PulsesToFlowrate(value);
    LogMessage("Actual flowrate: " + std::to_string(flowrateUlperSecond_));
    return ret;
}

int AMF_LSP_Pump::Start()
{
    if (IsPumping()) { return DEVICE_PUMP_IS_RUNNING; }
    long position = (flowrateUlperSecond_ < 0) ? AMF_LSP_N_STEPS : 0;
    return MovePlunger(position);
}

int AMF_LSP_Pump::DispenseVolumeUl(double volUl)
{
    if (IsPumping()) { return DEVICE_PUMP_IS_RUNNING; }
    long position = (flowrateUlperSecond_ < 0) ? currStep_ + VolumeToSteps(volUl) : currStep_ - VolumeToSteps(volUl);
    if (position < 0) {
        LogMessage(std::string("Dispensing ") + std::to_string(volUl) +
            " is not possible with only " + std::to_string(StepsToVolume(currStep_)) + " remaining."
        );
        return DEVICE_INVALID_INPUT_PARAM;
    }
    if (position > nSteps_) {
        LogMessage(std::string("Withdrawing ") + std::to_string(volUl) + " would exceed the maximum capacity of the pump, " +
            " and is therefore not possible.");
        return DEVICE_INVALID_INPUT_PARAM;
    }
    return MovePlunger(position);
}

int AMF_LSP_Pump::DispenseDurationSeconds(double seconds)
{
    int displacement = (int)std::round(VolumeToSteps(flowrateUlperSecond_) * seconds);
    long position = (flowrateUlperSecond_ < 0) ? currStep_ + displacement : currStep_ - displacement;
    if (position < 0) {
        LogMessage(std::string("Dispensing for ") + std::to_string(seconds) + " at " +
            std::to_string(flowrateUlperSecond_) + "uL/s is not possible, as only " +
            std::to_string(StepsToVolume(currStep_)) + " uL is remaining."
        );
        return DEVICE_INVALID_INPUT_PARAM;
    }
    if (position > nSteps_) {
        LogMessage(std::string("Withdrawing for ") + std::to_string(seconds) + " at " +
            std::to_string(-flowrateUlperSecond_) + " uL/s is not possible, as only " +
            std::to_string(StepsToVolume(nSteps_ - currStep_)) + " uL can be withdrawn untill " +
            "the maximum volume is reached."
        );
        return DEVICE_INVALID_INPUT_PARAM;
    }
    return MovePlunger(position);
}

///////////////////////////////////////////////////////////////////////////////
//  AMF_LSP_Pump class
//  Action Handlers
///////////////////////////////////////////////////////////////////////////////

int AMF_LSP_Pump::OnHome(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    int ret = DEVICE_OK;
    switch (eAct) {
    case MM::AfterSet: {
        if (IsPumping()) { return DEVICE_PUMP_IS_RUNNING; }

        long value = 0;
        pProp->Get(value);
        if (value == 1) {
            Home();
        }
        break;
    }
    case MM::BeforeGet: {
        pProp->Set((long)0);
        break;
    }
    }
    return ret;
}

int AMF_LSP_Pump::OnMaxVolume(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    switch (eAct) {
    case MM::AfterSet: {
        long value = 0;
        pProp->Get(value);
        maxVolumeUl_ = (double)value;
        break;
    }
    case MM::BeforeGet: {
        long value = (long)maxVolumeUl_;
        pProp->Set(value);
        break;
    }
    }
    return DEVICE_OK;
}

int AMF_LSP_Pump::OnCurrentVolume(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    int ret = DEVICE_OK;
    switch (eAct) {
    case MM::AfterSet:
        // This property is read-only
        LogMessage("Should not be able to set the current volume. Please report this bug.");
        break;
    case MM::BeforeGet:
        ret = GetVolumeUl(volumeUl_);
        if (DEVICE_OK != ret) { return ret; }
        pProp->Set(volumeUl_);
    }
    return ret;
}

int AMF_LSP_Pump::OnFlowrate(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    int ret = DEVICE_OK;
    switch (eAct) {
    case MM::AfterSet: {
        double flowrate = 0;
        pProp->Get(flowrate);

        ret = SetFlowrateUlPerSecond(flowrate);
        if (DEVICE_OK != ret) { return DEVICE_OK; }
        break;
    }
    case MM::BeforeGet:
        pProp->Set(flowrateUlperSecond_);
        break;
    }
    return ret;
}

int AMF_LSP_Pump::OnRun(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    int ret = DEVICE_OK;
    switch (eAct) {
    case MM::AfterSet: {
        long value = 0;
        pProp->Get(value);

        if (value == 1 and run_ == 0) {
            ret = Start();
        }
        else if (value == 0 and run_ == 1) {
            ret = Stop();
        }

        if (DEVICE_OK == ret) {
			run_ = value;
        }
        break;
    }
    case MM::BeforeGet: {
        pProp->Set(run_);
        break;
    }
    }
    return ret;
}

int AMF_LSP_Pump::OnPumpType(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    switch (eAct) {
    case MM::AfterSet:
        pProp->Get(pumpType_);
        break;
    case MM::BeforeGet:
        pProp->Set(pumpType_.c_str());
        break;
    }
    return DEVICE_OK;
}

int AMF_LSP_Pump::OnAcceleration(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    int ret = DEVICE_OK;
    switch (eAct) {
    case MM::AfterSet: {
        long value = 0;
        pProp->Get(value);
        ret = SetAcceleration(value);
        break;
    }
    case MM::BeforeGet: {
        long value = 0;
        ret = GetAcceleration(value);
        pProp->Set(value);
        break;
    }
    }
    return ret;
}

int AMF_LSP_Pump::OnDeceleration(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    int ret = DEVICE_OK;
    switch (eAct) {
    case MM::AfterSet: {
        long value = 0;
        pProp->Get(value);
        ret = SetDeceleration(value);
        break;
    }
    case MM::BeforeGet: {
        long value = 0;
        ret = GetDeceleration(value);
        pProp->Set(value);
        break;
    }
    }
    return ret;
}


///////////////////////////////////////////////////////////////////////////////
//  AMF_LSP_Pump class
//  Utility methods
///////////////////////////////////////////////////////////////////////////////

int AMF_LSP_Pump::SendRecv(AMF_Command cmd, long& value)
{
    std::string cmd_string = AMF_get_command_string(address_, cmd, value);
    int ret = SendSerialCommand(port_.c_str(), cmd_string.c_str(), AMF_TERM);
    if (ret != DEVICE_OK) {
        LogMessage("Could not send serial command.");
        return DEVICE_SERIAL_COMMAND_FAILED;
    }

    std::string answer = "";
    ret = GetSerialAnswer(port_.c_str(), AMF_EOL, answer);
    if (ret != DEVICE_OK) {
        LogMessage("Could not receive response to serial command.");
        return DEVICE_SERIAL_INVALID_RESPONSE;
    }

    if (answer.size() == 5) {
        value = 0;
    }
    else {
		value = std::stol(answer.substr(3, answer.size() - 5));
    }
    return DEVICE_OK;
}

int AMF_LSP_Pump::CheckStatus()
{
    // Check for error
    long value = 0;
    std::string cmd_string = AMF_get_command_string(address_, AMF_Command::Get_pump_status, value);
    int ret = SendSerialCommand(port_.c_str(), cmd_string.c_str(), AMF_TERM);
    if (DEVICE_OK != ret) {
        LogMessage("Could not send status check of pump.");
        return DEVICE_SERIAL_COMMAND_FAILED;
    }

    std::string status = "";
    ret = GetSerialAnswer(port_.c_str(), AMF_EOL, status);
    if (DEVICE_OK != ret) {
        LogMessage("Could not receive status of pump.");
        return DEVICE_SERIAL_INVALID_RESPONSE;
    }

    int status_val = std::stol(status.substr(3, status.size() - 5));
    if (255 == status_val) { isPumping_ = true; }
    else if (0 == status_val) { isPumping_ = false; }
    else { return DEVICE_ERR + 10000; }
	return DEVICE_OK;
}

double AMF_LSP_Pump::StepsToVolume(int steps)
{
    return steps * stepVolumeUl_;
}

int AMF_LSP_Pump::VolumeToSteps(double volume)
{
    return (int)(volume / stepVolumeUl_);
}

double AMF_LSP_Pump::PulsesToFlowrate(int pulses)
{
    if ("Standard" == pumpType_) {
        return pulses * AMF_LSP_SPEED_STEP_STANDARD * maxVolumeUl_ / 3000;
    }
	return pulses * AMF_LSP_SPEED_STEP_HD * maxVolumeUl_ / 3000;
}

int AMF_LSP_Pump::FlowrateToPulses(double flowrate)
{
    if ("Standard" == pumpType_) {
        return (int)(3000.0 * flowrate / (AMF_LSP_SPEED_STEP_STANDARD * maxVolumeUl_));
    }
    return (int)(3000.0 * flowrate / (AMF_LSP_SPEED_STEP_HD * maxVolumeUl_));
}

///////////////////////////////////////////////////////////////////////////////
//  AMF_LSP_Pump class
//  Utility methods
///////////////////////////////////////////////////////////////////////////////


bool AMF_LSP_Pump::IsHomed()
{
    long value = 0;
    int ret = SendRecv(AMF_Command::Is_initialized, value);
    ret = CheckStatus();
    if (DEVICE_OK != ret) {
        LogMessage("Could not determine whether pump is homed.");
        return ret;
    }
    return (value == 0) ? false : true;
}


bool AMF_LSP_Pump::IsPumping()
{
    long value = 0;
    int ret = SendRecv(AMF_Command::Get_pump_status, value);
    ret = CheckStatus();
    if (ret != DEVICE_OK) {
        LogMessage("Could not get pump info.");
        return ret;
    }
    return (value == 255);
}

int AMF_LSP_Pump::GetAddress(long& address)
{
    long value = 0;
    int ret = SendRecv(AMF_Command::Get_address, value);
    address = value;
    ret = CheckStatus();
    if (ret != DEVICE_OK) {
        LogMessage("Could not obtain the device address.");
        return ret;
    }  
    return ret;
}

int AMF_LSP_Pump::MovePlunger(long position)
{
    LogMessage("Set plunger position to: " + std::to_string(position));
    if (position < 0 || position > nSteps_) {
        LogMessage("Plunger position outside of range. Please check remaining volume in pump.");
        return DEVICE_INVALID_INPUT_PARAM;
    }
    int ret = SendRecv(AMF_Command::Move_plunger_absolute, position);
    ret = CheckStatus();
    if (ret != DEVICE_OK) {
        LogMessage("Could not start the pump.");
    }
    return ret;
}

int AMF_LSP_Pump::GetNSteps(long& nSteps)
{
    long value = 0;
    int ret = SendRecv(AMF_Command::Get_n_steps, value);
    ret = CheckStatus();
    if (DEVICE_OK != ret) {
        LogMessage("Could not get number of steps");
        return ret;
    }
    nSteps = (value == 0) ? 3000 : AMF_LSP_N_STEPS;
    return ret;
}

int AMF_LSP_Pump::GetAcceleration(long& acc)
{
    long value = 0;
    int ret = SendRecv(AMF_Command::Get_acceleration, value);
    ret = CheckStatus();
    if (DEVICE_OK != ret) {
        LogMessage("Could not get the acceleration.");
        return ret;
    }
    acc = value;
    return ret;
}

int AMF_LSP_Pump::GetDeceleration(long& decel)
{
    long value = 0;
    int ret = SendRecv(AMF_Command::Get_deceleration, value);
    ret = CheckStatus();
    if (DEVICE_OK != ret) {
        LogMessage("Could not get the deceleration.");
        return ret;
    }
    decel = value;
    return ret;
}

int AMF_LSP_Pump::SetNSteps(long nSteps)
{
    long value = (nSteps == 3000) ? 0 : 1;
	int ret = SendRecv(AMF_Command::Set_n_steps, value);
    ret = CheckStatus();
	if (ret != DEVICE_OK) {
		LogMessage("Could not change stepsize");
	}
    return ret;
}

int AMF_LSP_Pump::SetAcceleration(long acc)
{
    int ret = SendRecv(AMF_Command::Set_acceleration, acc);
    ret = CheckStatus();
    if (DEVICE_OK != ret) {
        LogMessage("Could not set acceleration.");
        return ret;
    }
    acceleration_ = acc;
    return ret;
}

int AMF_LSP_Pump::SetDeceleration(long decel)
{
    int ret = SendRecv(AMF_Command::Set_deceleration, decel);
    ret = CheckStatus();
    if (DEVICE_OK != ret) {
        LogMessage("Could not set the deceleration.");
        return ret;
    }
    deceleration_ = decel;
    return ret;
}