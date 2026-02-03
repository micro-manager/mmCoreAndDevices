///////////////////////////////////////////////////////////////////////////////
// FILE:          IDAQDevice.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Interface for DAQ waveform generation devices.
//
// AUTHOR:        Kyle M. Douglass, https://kylemdouglass.com
//
// VERSION:       0.0.0
//
// COPYRIGHT:     ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland
//                Laboratory of Experimental Biophysics (LEB), 2026
//

#pragma once

#include <string>
#include <vector>

/// Minimal interface for DAQ waveform generation.
///
/// This interface abstracts the configuration and execution of retriggerable
/// analog waveform output. Implementation details such as counter-based clocking
/// (NIDAQmx) are hidden from the caller.
class IDAQDevice
{
public:
    virtual ~IDAQDevice() = default;

    /// Add an analog output channel.
    ///
    /// @param channel Physical channel name (e.g., "Dev1/ao0")
    /// @param minVoltage Minimum output voltage
    /// @param maxVoltage Maximum output voltage
    virtual void addAnalogOutputChannel(
        const std::string& channel,
        double minVoltage,
        double maxVoltage
    ) = 0;

    /// Configure timing and trigger settings.
    ///
    /// Sets up the sample clock rate and external trigger for retriggerable
    /// waveform output. The implementation handles internal clocking details.
    ///
    /// @param sampleRateHz Output sample rate in Hz
    /// @param samplesPerChannel Number of samples per channel per trigger
    /// @param triggerSource External trigger terminal (e.g., "/Dev1/PFI3")
    virtual void configureTiming(
        double sampleRateHz,
        size_t samplesPerChannel,
        const std::string& triggerSource
    ) = 0;

    /// Write waveform data to all configured channels.
    ///
    /// Data is in row-major order: channel 0 samples, then channel 1, etc.
    /// The total size must equal numChannels * samplesPerChannel.
    ///
    /// @param data Waveform samples in row-major layout
    /// @param numChannels Number of analog output channels
    /// @param samplesPerChannel Samples per channel
    virtual void writeAnalogOutput(
        const std::vector<double>& data,
        size_t numChannels,
        size_t samplesPerChannel
    ) = 0;

    /// Start waveform generation.
    ///
    /// The waveform output begins when the configured trigger is received.
    virtual void start() = 0;

    /// Stop waveform generation.
    virtual void stop() = 0;

    /// Clear all task configurations.
    ///
    /// Resets the device state to allow reconfiguration. Call this before
    /// setting up a new waveform configuration.
    virtual void clearTasks() = 0;

    // =========================================================================
    // Discovery methods
    // =========================================================================

    /// Get list of available DAQ devices in the system.
    ///
    /// @return Vector of device names (e.g., "Dev1", "Dev2")
    virtual std::vector<std::string> getDeviceNames() const = 0;

    /// Get list of analog output channels for a device.
    ///
    /// @param deviceName The device to query (e.g., "Dev1")
    /// @return Vector of channel names (e.g., "Dev1/ao0", "Dev1/ao1")
    virtual std::vector<std::string> getAnalogOutputChannels(
        const std::string& deviceName
    ) const = 0;
};
