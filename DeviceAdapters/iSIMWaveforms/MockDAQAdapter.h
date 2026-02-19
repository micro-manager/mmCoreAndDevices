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

#include <algorithm>
#include <functional>
#include <sstream>

/// Mock DAQ device for testing without hardware.
class MockDAQAdapter : public IDAQDevice
{
public:
    using LogCallback = std::function<void(const std::string&)>;

    void setLogger(LogCallback callback)
    {
        logger_ = std::move(callback);
    }

    void addAnalogOutputChannel(
        const std::string& channel,
        double minVoltage,
        double maxVoltage
    ) override
    {
        ++channelCount_;
        if (logger_)
        {
            std::ostringstream oss;
            oss << "[MockDAQ] addAnalogOutputChannel: channel=" << channel
                << ", minV=" << minVoltage << ", maxV=" << maxVoltage
                << " (total channels: " << channelCount_ << ")";
            logger_(oss.str());
        }
    }

    void configureTiming(
        double sampleRateHz,
        size_t samplesPerChannel,
        const std::string& triggerSource,
        const std::string& counterChannel,
        const std::string& clockSource,
        size_t counterSamplesPerTrigger = 0
    ) override
    {
        if (logger_)
        {
            std::ostringstream oss;
            oss << "[MockDAQ] configureTiming: sampleRate=" << sampleRateHz
                << " Hz, samplesPerChannel=" << samplesPerChannel
                << ", trigger=" << triggerSource
                << ", counter=" << counterChannel
                << ", clockSource=" << clockSource
                << ", counterSamplesPerTrigger=" << counterSamplesPerTrigger;
            logger_(oss.str());
        }
    }

    void writeAnalogOutput(
        const std::vector<double>& data,
        size_t numChannels,
        size_t samplesPerChannel
    ) override
    {
        if (logger_)
        {
            double minVal = data.empty() ? 0.0 : *std::min_element(data.begin(), data.end());
            double maxVal = data.empty() ? 0.0 : *std::max_element(data.begin(), data.end());
            std::ostringstream oss;
            oss << "[MockDAQ] writeAnalogOutput: dataSize=" << data.size()
                << ", numChannels=" << numChannels
                << ", samplesPerChannel=" << samplesPerChannel
                << ", valueRange=[" << minVal << ", " << maxVal << "]";
            logger_(oss.str());
        }
    }

    void start() override
    {
        if (logger_)
            logger_("[MockDAQ] start()");
    }

    void stop() override
    {
        if (logger_)
            logger_("[MockDAQ] stop()");
    }

    void clearTasks() override
    {
        channelCount_ = 0;
        if (logger_)
            logger_("[MockDAQ] clearTasks()");
    }

    std::vector<std::string> getDeviceNames() const override
    {
        return {"MockDev1", "MockDev2"};
    }

    std::vector<std::string> getAnalogOutputChannels(
        const std::string& deviceName
    ) const override
    {
        if (deviceName == "MockDev1")
            return {
                "MockDev1/ao0",
                "MockDev1/ao1",
                "MockDev1/ao2",
                "MockDev1/ao3",
                "MockDev1/ao4"
            };
        else if (deviceName == "MockDev2")
            return {"MockDev2/ao0", "MockDev2/ao1"};
        return {};
    }

    std::vector<std::string> getTriggerChannels(
        const std::string& deviceName
    ) const override
    {
        if (deviceName == "MockDev1")
            return {
                "/MockDev1/PFI0", "/MockDev1/PFI1",
                "/MockDev1/PFI2", "/MockDev1/PFI3",
                "/MockDev1/PFI4", "/MockDev1/PFI5",
                "/MockDev1/PFI6", "/MockDev1/PFI7"
            };
        else if (deviceName == "MockDev2")
            return {"/MockDev2/PFI0", "/MockDev2/PFI1"};
        return {};
    }

private:
    LogCallback logger_;
    size_t channelCount_ = 0;
};
