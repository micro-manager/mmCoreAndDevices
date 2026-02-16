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

#include <algorithm>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

/// NIDAQmx-based DAQ device for real hardware.
///
/// Manages two NIDAQmx tasks: a counter task that produces a retriggerable
/// pulse train (gated by an external trigger), and an AO task that uses the
/// counter's internal output as its sample clock for continuous waveform
/// regeneration.
class NIDAQmxAdapter : public IDAQDevice
{
public:
    using LogCallback = std::function<void(const std::string&)>;

    NIDAQmxAdapter() = default;

    ~NIDAQmxAdapter()
    {
        safeClear(counterTask_);
        safeClear(aoTask_);
    }

    // Non-copyable: this class owns raw TaskHandle resources that cannot
    // be safely shared between instances.
    NIDAQmxAdapter(const NIDAQmxAdapter&) = delete;
    NIDAQmxAdapter& operator=(const NIDAQmxAdapter&) = delete;

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
        if (aoTask_ == 0)
        {
            checkError(DAQmxCreateTask("AOTask", &aoTask_));
        }

        checkError(DAQmxCreateAOVoltageChan(
            aoTask_,
            channel.c_str(),
            nullptr,
            minVoltage,
            maxVoltage,
            DAQmx_Val_Volts,
            nullptr
        ));

        ++channelCount_;
        if (logger_)
        {
            std::ostringstream oss;
            oss << "[NIDAQmx] addAnalogOutputChannel: channel=" << channel
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
            oss << "[NIDAQmx] configureTiming: sampleRate=" << sampleRateHz
                << " Hz, samplesPerChannel=" << samplesPerChannel
                << ", trigger=" << triggerSource
                << ", counter=" << counterChannel
                << ", clockSource=" << clockSource
                << ", counterSamplesPerTrigger=" << counterSamplesPerTrigger;
            logger_(oss.str());
        }

        // --- Counter task setup ---
        safeClear(counterTask_);
        checkError(DAQmxCreateTask("CounterTask", &counterTask_));

        // Create counter output pulse channel at the desired frequency
        checkError(DAQmxCreateCOPulseChanFreq(
            counterTask_,
            counterChannel.c_str(),
            nullptr,
            DAQmx_Val_Hz,
            DAQmx_Val_Low,
            0.0,
            sampleRateHz,
            0.5
        ));

        // Counter produces a burst of clock pulses per trigger.
        // Default (counterSamplesPerTrigger == 0): one frame per trigger
        // (samplesPerChannel / 2 for a 2-frame buffer).
        size_t counterSamples = (counterSamplesPerTrigger > 0)
            ? counterSamplesPerTrigger
            : samplesPerChannel / 2;

        if (logger_)
        {
            logger_("[NIDAQmx] counterSamples=" + std::to_string(counterSamples));
        }

        checkError(DAQmxCfgImplicitTiming(
            counterTask_,
            DAQmx_Val_FiniteSamps,
            static_cast<uInt64>(counterSamples)
        ));

        // Trigger counter on rising edge of external trigger
        checkError(DAQmxCfgDigEdgeStartTrig(
            counterTask_,
            triggerSource.c_str(),
            DAQmx_Val_Rising
        ));

        // Re-arm counter after each burst
        checkError(DAQmxSetStartTrigRetriggerable(counterTask_, TRUE));

        if (logger_)
        {
            logger_("[NIDAQmx] Counter task configured: retriggerable on "
                    + triggerSource);
        }

        // --- AO task timing setup ---
        // Use counter internal output as sample clock, continuous mode
        checkError(DAQmxCfgSampClkTiming(
            aoTask_,
            clockSource.c_str(),
            sampleRateHz,
            DAQmx_Val_Rising,
            DAQmx_Val_ContSamps,
            static_cast<uInt64>(samplesPerChannel)
        ));

        // Enable waveform regeneration: AO buffer loops continuously
        checkError(DAQmxSetWriteRegenMode(aoTask_, DAQmx_Val_AllowRegen));

        if (logger_)
        {
            logger_("[NIDAQmx] AO task timing configured: clockSource="
                    + clockSource + ", regen enabled");
        }
    }

    void writeAnalogOutput(
        const std::vector<double>& data,
        size_t numChannels,
        size_t samplesPerChannel
    ) override
    {
        (void)numChannels;
        int32 sampsWritten = 0;
        checkError(DAQmxWriteAnalogF64(
            aoTask_,
            static_cast<int32>(samplesPerChannel),
            FALSE,
            DAQmx_Val_WaitInfinitely,
            DAQmx_Val_GroupByChannel,
            data.data(),
            &sampsWritten,
            nullptr
        ));

        if (static_cast<size_t>(sampsWritten) != samplesPerChannel)
        {
            throw std::runtime_error(
                "NIDAQmx: wrote " + std::to_string(sampsWritten) +
                " samples but expected " + std::to_string(samplesPerChannel));
        }

        if (logger_)
        {
            double minVal = data.empty() ? 0.0 : *std::min_element(data.begin(), data.end());
            double maxVal = data.empty() ? 0.0 : *std::max_element(data.begin(), data.end());
            std::ostringstream oss;
            oss << "[NIDAQmx] writeAnalogOutput: dataSize=" << data.size()
                << ", numChannels=" << numChannels
                << ", samplesPerChannel=" << samplesPerChannel
                << ", written=" << sampsWritten
                << ", valueRange=[" << minVal << ", " << maxVal << "]";
            logger_(oss.str());
        }
    }

    void start() override
    {
        // AO must be armed before the counter starts producing clock edges
        if (aoTask_ != 0)
        {
            checkError(DAQmxStartTask(aoTask_));
            if (logger_)
                logger_("[NIDAQmx] Started AO task");
        }
        if (counterTask_ != 0)
        {
            checkError(DAQmxStartTask(counterTask_));
            if (logger_)
                logger_("[NIDAQmx] Started counter task");
        }
    }

    void stop() override
    {
        if (logger_)
            logger_("[NIDAQmx] stop()");

        // Stop counter first (stop clock), then AO.
        // Attempt both even if the first fails.
        int32 counterErr = 0;
        if (counterTask_ != 0)
            counterErr = DAQmxStopTask(counterTask_);
        if (aoTask_ != 0)
            checkError(DAQmxStopTask(aoTask_));
        if (counterErr < 0)
            checkError(counterErr);
    }

    void clearTasks() override
    {
        if (logger_)
            logger_("[NIDAQmx] clearTasks()");

        safeClear(counterTask_);
        safeClear(aoTask_);
        channelCount_ = 0;
    }

    std::vector<std::string> getDeviceNames() const override
    {
        int32 bufSize = DAQmxGetSysDevNames(nullptr, 0);
        if (bufSize <= 0)
            return {};

        std::vector<char> buf(static_cast<size_t>(bufSize));
        int32 err = DAQmxGetSysDevNames(buf.data(), bufSize);
        if (err != 0)
            return {};

        return splitCommaList(buf.data());
    }

    std::vector<std::string> getAnalogOutputChannels(
        const std::string& deviceName
    ) const override
    {
        char buf[4096] = {};
        int32 err = DAQmxGetDevAOPhysicalChans(deviceName.c_str(), buf, sizeof(buf));
        if (err != 0)
            return {};

        return splitCommaList(buf);
    }

private:
    TaskHandle aoTask_ = 0;
    TaskHandle counterTask_ = 0;
    LogCallback logger_;
    size_t channelCount_ = 0;

    /// Throw std::runtime_error if a NIDAQmx call returned an error.
    /// Positive return values are warnings and are logged.
    void checkError(int32 error)
    {
        if (error > 0)
        {
            // Positive values are DAQmx warnings
            if (logger_)
            {
                char warnBuf[2048] = {};
                DAQmxGetErrorString(error, warnBuf, sizeof(warnBuf));
                logger_(std::string("[NIDAQmx] WARNING (") +
                        std::to_string(error) + "): " + warnBuf);
            }
        }
        else if (error < 0)
        {
            char errBuf[2048] = {};
            DAQmxGetErrorString(error, errBuf, sizeof(errBuf));

            char extBuf[2048] = {};
            DAQmxGetExtendedErrorInfo(extBuf, sizeof(extBuf));

            std::string msg = "NIDAQmx error: ";
            msg += errBuf;
            if (extBuf[0] != '\0')
            {
                msg += " | Details: ";
                msg += extBuf;
            }
            throw std::runtime_error(msg);
        }
    }

    /// Stop and clear a task handle, setting it to 0.
    static void safeClear(TaskHandle& task)
    {
        if (task != 0)
        {
            DAQmxStopTask(task);
            DAQmxClearTask(task);
            task = 0;
        }
    }

    /// Parse a comma-separated string (as returned by DAQmx query functions)
    /// into a vector of trimmed strings.
    static std::vector<std::string> splitCommaList(const char* buffer)
    {
        std::vector<std::string> result;
        std::string input(buffer);
        size_t pos = 0;
        while (pos < input.size())
        {
            while (pos < input.size() && (input[pos] == ',' || input[pos] == ' '))
                ++pos;
            if (pos >= input.size())
                break;
            size_t end = input.find(',', pos);
            if (end == std::string::npos)
                end = input.size();
            size_t trimEnd = end;
            while (trimEnd > pos && input[trimEnd - 1] == ' ')
                --trimEnd;
            if (trimEnd > pos)
                result.push_back(input.substr(pos, trimEnd - pos));
            pos = end + 1;
        }
        return result;
    }
};
