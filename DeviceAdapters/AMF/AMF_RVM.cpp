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
// YEAR:          2024
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
//LAST UPDATE:    26.02.2024 LK

#include "AMF_RVM.h" // Should be first, otherwise winsock.h is loaded before boost asio tries to load winsock.h
#include "AMF_Commands.h"
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
    AMF_Initialize_Error_Messages();
    SetErrorText(ERR_UNKNOWN_POSITION, "Requested position not available in this device");
    EnableDelay(); // signals that the dealy setting will be used

    // COM Port
    if (!GetParentHub()) {
		CPropertyAction* pAct = new CPropertyAction(this, &AMF_RVM::OnPort);
		CreateStringProperty(MM::g_Keyword_Port, "COM1", false, pAct, true);
    }
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

    // Initialize valve
    long value = 0;
    int ret = SendRecv(AMF_Command::Initialize, value);
    if (DEVICE_OK != ret) { return ret; }

    // Name
    ret = CreateStringProperty(MM::g_Keyword_Name, AMF_RVM_Name, true);
    if (DEVICE_OK != ret) { return ret; }

    // Description
    ret = CreateStringProperty(MM::g_Keyword_Description, "Driver for AMF Rotary Valve Modules", true);
    if (DEVICE_OK != ret) { return ret; }

    // Number of positions
    ret = GetNValves(nPos_);
    if (ret != DEVICE_OK) { return ret; }
    LogMessage(("Number of positions: " + to_string(nPos_)).c_str());
    CreateIntegerProperty("Number of positions", nPos_, true);

    // Rotation direction
    CPropertyAction* pAct = new CPropertyAction(this, &AMF_RVM::OnRotationDirection);
    vector<string> allowedDirections = { "Shortest rotation direction", "Clockwise", "Counterclockwise" };
    ret = CreateStringProperty(AMF_Rotation_Property, "Shortest rotation direction", false, pAct);
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
    if (ret != DEVICE_OK) { return ret; }

    // Label
    // -----
    pAct = new CPropertyAction(this, &CStateBase::OnLabel);
    ret = CreateStringProperty(MM::g_Keyword_Label, "", false, pAct);
    if (ret != DEVICE_OK) { return ret; }


    ret = UpdateStatus();
    if (ret != DEVICE_OK) { return ret; }

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
        pProp->Set(RotationDirectionToString(rotationDirection_).c_str());
    }
    else if (eAct == MM::AfterSet)
    {
        string temp;
        pProp->Get(temp);
        rotationDirection_ = RotationDirectionFromString(temp);
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
    if (ret != DEVICE_OK) {
        LogMessage("Could not send serial command.");
        return DEVICE_SERIAL_COMMAND_FAILED;
    }

    std::string answer = "";
    ret = GetSerialAnswer(port_.c_str(), AMF_TERM, answer);
    if (ret != DEVICE_OK) {
        LogMessage("Could not receive response to serial command.");
        return DEVICE_SERIAL_INVALID_RESPONSE;
    }

    ret = AMF_Parse_Status(answer[3]);
    if (ret != DEVICE_OK) { return ret; }

    if (answer.size() == 6) { return ret; }
    value = std::stol(answer.substr(4, answer.size() - 3));
    return DEVICE_OK;
}

int AMF_RVM::GetValvePosition(long& pos) {
    long value = 0;
    int ret = SendRecv(AMF_Command::Get_valve_position, value);
    if (ret != DEVICE_OK) { return ret; }
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
    switch (rotationDirection_)
    {
    case SHORTEST:
        ret = SendRecv(AMF_Command::Move_valve_shortest, pos);
        break;
    case CLOCKWISE:
        ret = SendRecv(AMF_Command::Move_valve_cw, pos);
        break;
    case COUNTERCLOCKWISE:
        ret = SendRecv(AMF_Command::Move_valve_ccw, pos);
        break;
    }
    LogMessage("Set valve to position: " + to_string(pos) + ".");
    return ret;
}

std::string AMF_RVM::RotationDirectionToString(RotationDirection rd)
{
    switch (rd)
    {
    case CLOCKWISE:
        return "Clockwise";
    case COUNTERCLOCKWISE:
        return "Counterclockwise";
    case SHORTEST:
        return "Shortest rotation direction";
    }
    return "";
}

RotationDirection AMF_RVM::RotationDirectionFromString(std::string& s) {
    if (s == "Clockwise")
        return CLOCKWISE;
    else if (s == "Counterclockwise")
        return COUNTERCLOCKWISE;
    else
        return SHORTEST;
}

int AMF_RVM::GetNValves(long& nPos)
{
    int ret = SendRecv(AMF_Command::Get_n_valves, nPos);
    if (DEVICE_OK != ret) {
        LogMessage("Could not get number of valves.");
    }
    return ret;
}

int AMF_RVM::GetAddress(long& address)
{
    int ret = SendRecv(AMF_Command::Get_address, address);
    if (DEVICE_OK != ret) {
        LogMessage("Could not get address.");
    }
    return ret;
}