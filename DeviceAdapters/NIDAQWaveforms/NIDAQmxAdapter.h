///////////////////////////////////////////////////////////////////////////////
// FILE:          NIDAQmxAdapter.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   NIDAQmx implementation of IDAQDevice.
//
// AUTHOR:        Kyle M. Douglass, https://kylemdouglass.com
//
// VERSION:       0.0.0
//
// COPYRIGHT:     ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland
//                Laboratory of Experimental Biophysics (LEB), 2026
//

#pragma once

#include "IDAQDevice.h"

#include <NIDAQmx.h>

/// NIDAQmx-based DAQ device for real hardware.
class NIDAQmxAdapter : public IDAQDevice
{
public:
    void addAnalogOutputChannel(
        const std::string& channel,
        double minVoltage,
        double maxVoltage
    ) override {}

    void configureTiming(
        double sampleRateHz,
        size_t samplesPerChannel,
        const std::string& triggerSource
    ) override {}

    void writeAnalogOutput(
        const std::vector<double>& data,
        size_t numChannels,
        size_t samplesPerChannel
    ) override {}

    void start() override {}
    void stop() override {}
    void clearTasks() override {}

    std::vector<std::string> getDeviceNames() const override
    {
        // TODO: Implement using DAQmxGetSysDevNames()
        return {};
    }

    std::vector<std::string> getAnalogOutputChannels(
        const std::string& deviceName
    ) const override
    {
        // TODO: Implement using DAQmxGetDevAOPhysicalChans()
        (void)deviceName;
        return {};
    }
};
