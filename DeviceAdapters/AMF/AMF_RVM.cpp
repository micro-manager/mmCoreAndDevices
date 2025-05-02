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
#include <cstdio>
#include <string>
#include <math.h>
#include "ModuleInterface.h"
#include <sstream>
#include <algorithm>
#include <iostream>
#include <vector>
#include <future>
#include "AMF_Commands.h"
#include "DeviceUtils.h"


extern const char* g_AMF_RVM_Name;
const char* g_Rotation_Property = "Rotation Direction";
const char* NoHubError = "Parent Hub not defined.";

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
    InitializeDefaultErrorMessages();
    SetErrorText(ERR_UNKNOWN_POSITION, "Requested position not available in this device");
    EnableDelay(); // signals that the dealy setting will be used

    // COM Port
    CPropertyAction* pAct = new CPropertyAction(this, &AMF_RVM::OnPort);
    CreateStringProperty(MM::g_Keyword_Port, "COM1", false, pAct, true);

    // Address
    pAct = new CPropertyAction(this, &AMF_RVM::OnChangeAddress);
    CreateIntegerProperty("AMF Address", 1, false, pAct, true);
    SetPropertyLimits("AMF Address", 1, 16);
}

AMF_RVM::~AMF_RVM()
{
    Shutdown();
}

void AMF_RVM::GetName(char* Name) const
{
    CDeviceUtils::CopyLimitedString(Name, g_AMF_RVM_Name);
}


int AMF_RVM::Initialize()
{
    if (initialized_)
        return DEVICE_OK;

    // Initialize valve
    InitializationCommand init(address_);
    int ret = SendRecv(init);
    if (ret != DEVICE_OK)
        return ret;

    // Name
    ret = CreateStringProperty(MM::g_Keyword_Name, g_AMF_RVM_Name, true);
    if (DEVICE_OK != ret)
        return ret;

    // Description
    ret = CreateStringProperty(MM::g_Keyword_Description, "Driver for AMF Rotary Valve Modules", true);
    if (DEVICE_OK != ret)
        return ret;

    // Version
    FirmwareVersionRequest firmwareVersion(address_);
    ret = SendRecv(firmwareVersion);
    if (ret != DEVICE_OK)
        return ret;
    version_ = firmwareVersion.GetFirmwareVersion();
    LogMessage("Firmware version: " + version_);
    ret = CreateStringProperty("Firmware version", version_.c_str(), true);
    


    // Number of positions
    ValveMaxPositionsRequest maxPosRequest(address_);
    ret = SendRecv(maxPosRequest);
    if (ret != DEVICE_OK) { return ret; }
    nPos_ = maxPosRequest.GetMaxPositions();
    LogMessage(("Number of positions: " + to_string(nPos_)).c_str());
    CreateIntegerProperty("Number of positions", nPos_, true);

    // Rotation direction
    CPropertyAction* pAct = new CPropertyAction(this, &AMF_RVM::OnRotationDirection);
    vector<string> allowedDirections = { "Shortest rotation direction", "Clockwise", "Counterclockwise" };
    ret = CreateStringProperty(g_Rotation_Property, "Shortest rotation direction", false, pAct);
    SetAllowedValues(g_Rotation_Property, allowedDirections);

    // Set timer for the Busy signal, or we'll get a time-out the first time we check the state of the shutter, for good measure, go back 'delay' time into the past
    changedTime_ = GetCurrentMMTime();

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
    if (ret != DEVICE_OK)
        return ret;

    // Label
    // -----
    pAct = new CPropertyAction(this, &CStateBase::OnLabel);
    ret = CreateStringProperty(MM::g_Keyword_Label, "", false, pAct);
    if (ret != DEVICE_OK)
        return ret;


    ret = UpdateStatus();
    if (ret != DEVICE_OK)
        return ret;

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

int AMF_RVM::OnChangeAddress(MM::PropertyBase* pProp, MM::ActionType eAct) {
    if (eAct == MM::BeforeGet) {
        pProp->Set((long)address_);
    }
    else if (eAct == MM::AfterSet) {
        long addr;
        pProp->Get(addr);
        address_ = addr;
    }
    return DEVICE_OK;
}

int AMF_RVM::OnState(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        pProp->Set(position_);
        // nothing to do, let the caller to use cached property
    }
    else if (eAct == MM::AfterSet)
    {
        // Set timer for the Busy signal
        changedTime_ = GetCurrentMMTime();

        long pos;
        pProp->Get(pos);
        if (pos >= nPos_ || pos < 0)
        {
            pProp->Set(position_); // revert
            return ERR_UNKNOWN_POSITION;
        }
        position_ = pos;
        return SetValvePosition(position_);
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

int AMF_RVM::SendRecv(AMFCommand& cmd)
{
    int err = DEVICE_OK;
    LogMessage("Request: " + cmd.Get());
    err = SendSerialCommand(port_.c_str(), cmd.Get().c_str(), AMF_TERM);
    if (err != DEVICE_OK)
        return err;

    std::string answer;
    err = GetSerialAnswer(port_.c_str(), AMF_TERM, answer);
    if (err != DEVICE_OK)
        return err;

    LogMessage("Response: " + answer);
    err = cmd.ParseResponse(answer);
    if (err != DEVICE_OK)
        return err;
    return DEVICE_OK;
}

int AMF_RVM::GetValvePosition(int& pos) {
    ValvePositionRequest req(address_);
    int err = SendRecv(req);
    if (err != DEVICE_OK)
        return err;
    pos = req.GetPosition();
    return DEVICE_OK;
}

int AMF_RVM::SetValvePosition(int pos) {
    int currPos = 0;
    GetValvePosition(currPos);
    if (currPos == pos)
        return DEVICE_OK;

    LogMessage("Current Position: " + to_string(currPos) + ". Next position: " + to_string(pos) + ".");

    char direction = 'B';
    switch (rotationDirection_)
    {
    case SHORTEST:
        direction = 'B';
        break;
    case CLOCKWISE:
        direction = 'I';
        break;
    case COUNTERCLOCKWISE:
        direction = 'O';
        break;
    }

    ValvePositionCommand req(address_, pos, direction);
    int err = SendRecv(req);
    return err;
}

std::string AMF_RVM::RotationDirectionToString(RotationDirection rd) {
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