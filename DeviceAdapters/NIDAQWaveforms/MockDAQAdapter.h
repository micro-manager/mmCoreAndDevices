///////////////////////////////////////////////////////////////////////////////
// FILE:          MockDAQAdapter.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Mock implementation of IDAQDevice for testing.
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

/// Mock DAQ device for testing without hardware.
class MockDAQAdapter : public IDAQDevice
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
        return {"MockDev1", "MockDev2"};
    }

    std::vector<std::string> getAnalogOutputChannels(
        const std::string& deviceName
    ) const override
    {
        if (deviceName == "MockDev1")
            return {"MockDev1/ao0", "MockDev1/ao1", "MockDev1/ao2", "MockDev1/ao3"};
        else if (deviceName == "MockDev2")
            return {"MockDev2/ao0", "MockDev2/ao1"};
        return {};
    }
};
