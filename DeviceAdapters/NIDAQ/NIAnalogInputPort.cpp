// DESCRIPTION:   Drive multiple analog and digital outputs on NI DAQ
// AUTHOR:        Mark Tsuchida, 2015, Nico Stuurman 2022
// COPYRIGHT:     2015-2016, Open Imaging, Inc., 2022 Altos Labs
// LICENSE:       This library is free software; you can redistribute it and/or
//                modify it under the terms of the GNU Lesser General Public
//                License as published by the Free Software Foundation; either
//                version 2.1 of the License, or (at your option) any later
//                version.
//
//                This library is distributed in the hope that it will be
//                useful, but WITHOUT ANY WARRANTY; without even the implied
//                warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
//                PURPOSE.  See the GNU Lesser General Public License for more
//                details.
//
//                You should have received a copy of the GNU Lesser General
//                Public License along with this library; if not, write to the
//                Free Software Foundation, Inc., 51 Franklin Street, Fifth
//                Floor, Boston, MA  02110-1301  USA

#include "NIDAQ.h"

#include "ModuleInterface.h"

//
// MultiAnalogInPort
//

NIAnalogInputPort::NIAnalogInputPort(const std::string& port) :
    ErrorTranslator(22000, 22999, &NIAnalogInputPort::SetErrorText),
    niPort_(port),
    initialized_(false),
    running_(false),
    state_(0.0)
{
    InitializeDefaultErrorMessages();
}


NIAnalogInputPort::~NIAnalogInputPort()
{
    Shutdown();
}


int NIAnalogInputPort::Initialize()
{
    if (initialized_)
        return DEVICE_OK;

    CPropertyAction* pAct = new CPropertyAction(this, &NIAnalogInputPort::OnVoltage);
    int err = CreateFloatProperty("Voltage", state_, true, pAct);
    if (err != DEVICE_OK)
        return err;

    pAct = new CPropertyAction(this, &NIAnalogInputPort::OnMeasuring);
    err = CreateStringProperty("Measuring", "false", false, pAct);
    if (err != DEVICE_OK)
        return err;
    AddAllowedValue("Measuring", "false");
    AddAllowedValue("Measuring", "true");

    initialized_ = true;
    return DEVICE_OK;
}


int NIAnalogInputPort::Shutdown()
{
    if (!initialized_)
        return DEVICE_OK;

    running_ = false;
    initialized_ = false;
    return DEVICE_OK;
}


void NIAnalogInputPort::GetName(char* name) const
{
    CDeviceUtils::CopyLimitedString(name,
        (g_DeviceNameNIDAQAIPortPrefix + niPort_).c_str());
}


int NIAnalogInputPort::GetSignal(double& volts)
{
    volts = state_;
    return DEVICE_OK;
}


int NIAnalogInputPort::GetLimits(double& minVolts, double& maxVolts)
{
    float64 ranges[1024] = {0};
    char hub_name[1024];
    GetHub()->GetName(hub_name);

    int err = DAQmxGetDevAIVoltageRngs(hub_name, ranges, 1024);
    if (err != DEVICE_OK)
        return err;

    maxVolts = *std::max_element(ranges, ranges + 1024);
    minVolts = *std::min_element(ranges, ranges + 1024);

    return DEVICE_OK;
}


int NIAnalogInputPort::SetRunning(bool open)
{
    if (open && !running_)
    {
        int err = GetHub()->StartAIMeasuringForPort(this);
        if (err != DEVICE_OK)
            return err;
    }
    else if (!open && running_)
    {
        int err = GetHub()->StopAIMeasuringForPort(this);
        if (err != DEVICE_OK)
            return err;
    }

    running_ = open;
    return DEVICE_OK;
}


int NIAnalogInputPort::GetRunning(bool& open)
{
    open = running_;
    return DEVICE_OK;
}


int NIAnalogInputPort::OnVoltage(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        pProp->Set(state_);
    }
    return DEVICE_OK;
}

int NIAnalogInputPort::OnMeasuring(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        pProp->Set(running_? "true": "false");
    }
    else if (eAct == MM::AfterSet)
    {
        std::string running;
        pProp->Get(running);
        int err = SetRunning(running.compare("true") == 0);
        if (err != DEVICE_OK)
            return err;
    }
    return DEVICE_OK;
}

int NIAnalogInputPort::UpdateState(float value)
{
    state_ = value;
    OnPropertiesChanged();
    return DEVICE_OK;
}


int NIAnalogInputPort::TranslateHubError(int err)
{
    if (err == DEVICE_OK)
        return DEVICE_OK;
    char buf[MM::MaxStrLength];
    if (GetHub()->GetErrorText(err, buf))
        return NewErrorCode(buf);
    return NewErrorCode("Unknown hub error");
}