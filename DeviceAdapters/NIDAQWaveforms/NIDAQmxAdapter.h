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
    }

    void configureTiming(
        double sampleRateHz,
        size_t samplesPerChannel,
        const std::string& triggerSource,
        const std::string& counterChannel,
        const std::string& clockSource
    ) override
    {
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

        // Counter produces half the waveform samples per trigger (one frame).
        // Each external trigger advances the AO output by one frame; two
        // triggers complete one full galvo period (the 2-frame waveform).
        checkError(DAQmxCfgImplicitTiming(
            counterTask_,
            DAQmx_Val_FiniteSamps,
            static_cast<uInt64>(samplesPerChannel / 2)
        ));

        // Trigger counter on rising edge of external trigger
        checkError(DAQmxCfgDigEdgeStartTrig(
            counterTask_,
            triggerSource.c_str(),
            DAQmx_Val_Rising
        ));

        // Re-arm counter after each burst
        checkError(DAQmxSetStartTrigRetriggerable(counterTask_, TRUE));

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
    }

    void start() override
    {
        // AO must be armed before the counter starts producing clock edges
        if (aoTask_ != 0)
            checkError(DAQmxStartTask(aoTask_));
        if (counterTask_ != 0)
            checkError(DAQmxStartTask(counterTask_));
    }

    void stop() override
    {
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
        safeClear(counterTask_);
        safeClear(aoTask_);
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

    /// Throw std::runtime_error if a NIDAQmx call returned an error.
    /// Positive return values are warnings and are not treated as errors.
    static void checkError(int32 error)
    {
        if (error < 0)
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
