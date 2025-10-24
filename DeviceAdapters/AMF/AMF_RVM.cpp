///////////////////////////////////////////////////////////////////////////////
// FILE:          AMF_RVM.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Device adapter for Advanced MicroFluidics (AMF) Rotary
//				  Valve Modules (RVM).
//                
// AUTHOR:        Lars Kool, Institut Pierre-Gilles de Gennes, Paris, France
//
// YEAR:          2025
//                
// VERSION:       1.0
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
//LAST UPDATE:    15.10.2025 LK

#include "AMF_RVM.h" // Should be first, otherwise winsock.h is loaded before boost asio tries to load winsock.h
#include "AMF_Commands.h"
#include "AMF_LSP_Hub.h"
#include "ModuleInterface.h"
#include "DeviceUtils.h"

#include <cstdio>
#include <string>
#include <math.h>
#include <sstream>
#include <algorithm>
#include <iostream>
#include <vector>
#include <future>


const char* AMF_Rotation_Property = "Rotation Direction";

using namespace std;

///////////////////////////////////////////////////////////////////////////////
// AMF_RVM implementation
///////////////////////////////////////////////////////////////////////////////

AMF_RVM::AMF_RVM() :
    nPos_(0),
    initialized_(false),
    changedTime_(0.0),
    busy_(false),
    position_(0)
{
    // Number of positions
    CPropertyAction* pAct = new CPropertyAction(this, &AMF_RVM::OnNumberOfStates);
    CreateIntegerProperty("Number of ports", 6, false, pAct, true);
    std::vector<std::string> values = { "2", "4", "6", "8", "10", "12" };
    SetAllowedValues("Number of ports", values);
}

AMF_RVM::~AMF_RVM()
{
    Shutdown();
}

void AMF_RVM::GetName(char* Name) const
{
    CDeviceUtils::CopyLimitedString(Name, AMF_RVM_Name);
}


int AMF_RVM::Initialize()
{
    if (initialized_)
        return DEVICE_OK;

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

    // If RVM is part of LSP pump, the device might already be
    // homing. Just wait for the valve to be available.
    while (Busy()) {
        CDeviceUtils::SleepMs(100);
    }
    LogMessage("Passed busy wait step.");

    // Initialize valve
    if (!IsHomed()) {
        Home();
        while (Busy()) { CDeviceUtils::SleepMs(100); }
        LogMessage("Finished homing.");
    }
    LogMessage("Homing step passed.");
    // Name
    int ret = CreateStringProperty(MM::g_Keyword_Name, AMF_RVM_Name, true);
    if (DEVICE_OK != ret) { return ret; }

    // Description
    ret = CreateStringProperty(MM::g_Keyword_Description, "Driver for AMF Rotary Valve Modules", true);
    if (DEVICE_OK != ret) { return ret; }

    // Number of positions
    long temp = 0;
    ret = GetNValves(temp);
    if (DEVICE_OK != ret) { return ret; }
    LogMessage(("Number of positions: " + to_string(nPos_)).c_str());
    if (nPos_ != temp) {
        LogMessage("Provided number of ports (" + std::to_string(nPos_) + ") does not correspond to the number of ports\n" +
            std::string("by the pump (" + std::to_string(temp) + "). Please enter the correct number of ports."));
        return DEVICE_INVALID_PROPERTY_VALUE;
    }

    // Rotation direction
    CPropertyAction* pAct = new CPropertyAction(this, &AMF_RVM::OnRotationDirection);
    vector<string> allowedDirections = {
        AMF_Rotation_Shortest,
        AMF_Rotation_Clockwise,
        AMF_Rotation_CounterClockwise
    };
    ret = CreateStringProperty(AMF_Rotation_Property, AMF_Rotation_Shortest, false, pAct);
    SetAllowedValues(AMF_Rotation_Property, allowedDirections);

    // create default positions and labels
    const int bufSize = 1024;
    char buf[bufSize];
    for (long i = 0; i < nPos_; i++)
    {
        snprintf(buf, bufSize, "State-%ld", i);
        SetPositionLabel(i, buf);
        snprintf(buf, bufSize, "%ld", i);
        AddAllowedValue(MM::g_Keyword_Closed_Position, buf);
    }

    // State
    // -----
    pAct = new CPropertyAction(this, &AMF_RVM::OnState);
    ret = CreateIntegerProperty(MM::g_Keyword_State, 0, false, pAct);
    if (DEVICE_OK != ret) { return ret; }

    // Label
    // -----
    pAct = new CPropertyAction(this, &CStateBase::OnLabel);
    ret = CreateStringProperty(MM::g_Keyword_Label, "", false, pAct);
    if (DEVICE_OK != ret) { return ret; }


    ret = UpdateStatus();
    if (DEVICE_OK != ret) { return ret; }

    initialized_ = true;
    return DEVICE_OK;
}

bool AMF_RVM::Busy()
{
    MM::MMTime interval = GetCurrentMMTime() - changedTime_;
    MM::MMTime delay(GetDelayMs() * 1000.0);
    if (interval < delay)
        return true;
    else
        return false;
}

int AMF_RVM::Shutdown()
{
    if (initialized_)
    {
        initialized_ = false;
    }
    return DEVICE_OK;
}


///////////////////////////////////////////////////////////////////////////////
// Action handlers
///////////////////////////////////////////////////////////////////////////////

int AMF_RVM::OnPort(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        pProp->Set(port_.c_str());
    }
    else if (eAct == MM::AfterSet)
    {
        if (!initialized_)
            pProp->Get(port_);
    }

    return DEVICE_OK;
}

int AMF_RVM::OnState(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        pProp->Set(position_);
        // nothing to do, let the caller use cached property
    }
    else if (eAct == MM::AfterSet)
    {
        // Set timer for the Busy signal
        changedTime_ = GetCurrentMMTime();

        long pos;
        pProp->Get(pos);
        int ret = SetValvePosition(position_);
        if (DEVICE_OK != ret) { return ret; }
        position_ = pos;
        return ret;
    }
    return DEVICE_OK;
}

int AMF_RVM::OnNumberOfStates(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        pProp->Set(nPos_);
    }
    else if (eAct == MM::AfterSet)
    {
        if (!initialized_)
            pProp->Get(nPos_);
    }

    return DEVICE_OK;
}

int AMF_RVM::OnRotationDirection(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        pProp->Set(rotationDirection_.c_str());
    }
    else if (eAct == MM::AfterSet)
    {
        pProp->Get(rotationDirection_);
    }

    return DEVICE_OK;
}

///////////////////////////////////////////////////////////////////////////////
// Utility members
///////////////////////////////////////////////////////////////////////////////

int AMF_RVM::SendRecv(AMF_Command cmd, long& value)
{
    std::string cmd_string = AMF_get_command_string(address_, cmd, value);
    int ret = SendSerialCommand(port_.c_str(), cmd_string.c_str(), AMF_TERM);
    if (DEVICE_OK != ret) {
        LogMessage("Could not send serial command.");
        return DEVICE_SERIAL_COMMAND_FAILED;
    }

    std::string answer = "";
    ret = GetSerialAnswer(port_.c_str(), AMF_EOL, answer);
    if (DEVICE_OK != ret) {
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

int AMF_RVM::CheckStatus()
{
    // Check for error
    long value = 0;
    std::string cmd_string = AMF_get_command_string(address_, AMF_Command::Get_valve_status, value);
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
    if (255 == status_val) { busy_ = true; }
    else if (0 == status_val) { busy_ = false; }
    else { return DEVICE_ERR + 10000; }
	return DEVICE_OK;
}


int AMF_RVM::Home()
{
    long value = 0;
    int ret = SendRecv(AMF_Command::Initialize, value);
    ret = CheckStatus();
    if (DEVICE_OK != ret) {
        LogMessage("Could not home the valve.");
        return ret;
    }
    return DEVICE_OK;
}

bool AMF_RVM::IsHomed()
{
    long value = 0;
    SendRecv(AMF_Command::Is_initialized, value);
    return (value == 1);
}

int AMF_RVM::GetValvePosition(long& pos) {
    long value = 0;
    int ret = SendRecv(AMF_Command::Get_valve_position, value);
    ret = CheckStatus();
    if (DEVICE_OK != ret) { return ret; }
    pos = value;
    return DEVICE_OK;
}

int AMF_RVM::SetValvePosition(long pos)
{
    if (pos < 0 || pos >= nPos_) {
        LogMessage("Position outside of valid range.");
        return DEVICE_INVALID_PROPERTY_VALUE;
    }
    int ret = DEVICE_OK;
    if (AMF_Rotation_Shortest == rotationDirection_) {
        ret = SendRecv(AMF_Command::Move_valve_shortest, pos);
    }
    else if (AMF_Rotation_Clockwise == rotationDirection_) {
        ret = SendRecv(AMF_Command::Move_valve_cw, pos);
    }
    else if (AMF_Rotation_CounterClockwise == rotationDirection_) {
        ret = SendRecv(AMF_Command::Move_valve_ccw, pos);
    }
    else {
        LogMessage("Invalid rotation direction, defaulted to shortest");
        rotationDirection_ = AMF_Rotation_Shortest;
        ret = SendRecv(AMF_Command::Move_valve_shortest, pos);
    }
    ret = CheckStatus();
    if (DEVICE_OK != ret) {
        LogMessage("Could not set valve position.");
        return ret;
    }
    LogMessage("Set valve to position: " + to_string(pos) + ".");
    return ret;
}

int AMF_RVM::GetNValves(long& nPos)
{
    int ret = SendRecv(AMF_Command::Get_n_valves, nPos);
    ret = CheckStatus();
    if (DEVICE_OK != ret) {
        LogMessage("Could not get number of valves.");
    }
    return ret;
}

int AMF_RVM::GetAddress(long& address)
{
    int ret = SendRecv(AMF_Command::Get_address, address);
    ret = CheckStatus();
    if (DEVICE_OK != ret) {
        LogMessage("Could not get address.");
    }
    return ret;
}