///////////////////////////////////////////////////////////////////////////////
// FILE:          EvidentModel.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Evident IX85 microscope state model
//
// COPYRIGHT:     University of California, San Francisco, 2025
//
// LICENSE:       This file is distributed under the BSD license.
//                License text is included with the source distribution.
//
//                This file is distributed in the hope that it will be useful,
//                but WITHOUT ANY WARRANTY; without even the implied warranty
//                of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
//
//                IN NO EVENT SHALL THE COPYRIGHT OWNER OR
//                CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
//                INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES.
//
// AUTHOR:        Nico Stuurman, 2025

#pragma once

#include "MMDevice.h"
#include <mutex>
#include <string>
#include <map>

namespace EvidentIX85 {

// Device type enumeration
enum DeviceType
{
    DeviceType_Unknown = 0,
    DeviceType_Focus,
    DeviceType_Nosepiece,
    DeviceType_Magnification,
    DeviceType_LightPath,
    DeviceType_CondenserTurret,
    DeviceType_Condenser,
    DeviceType_DIAAperture,
    DeviceType_DIAShutter,
    DeviceType_Polarizer,
    DeviceType_DICPrism,
    DeviceType_DICRetardation,
    DeviceType_EPIShutter1,
    DeviceType_EPIShutter2,
    DeviceType_MirrorUnit1,
    DeviceType_MirrorUnit2,
    DeviceType_EPIND,
    DeviceType_RightPort,
    DeviceType_CorrectionCollar,
    DeviceType_Autofocus,
    DeviceType_OffsetLens
};

// Device state structure
struct DeviceState
{
    DeviceType type;
    bool present;
    bool busy;
    long currentPos;
    long targetPos;
    long minPos;
    long maxPos;
    int numPositions;  // For state devices
    MM::MMTime lastUpdateTime;
    MM::MMTime lastRequestTime;

    DeviceState() :
        type(DeviceType_Unknown),
        present(false),
        busy(false),
        currentPos(0),
        targetPos(0),
        minPos(0),
        maxPos(0),
        numPositions(0),
        lastUpdateTime(0.0),
        lastRequestTime(0.0)
    {}
};

// Microscope model - centralized state for all devices
class MicroscopeModel
{
public:
    MicroscopeModel();
    ~MicroscopeModel();

    // Device presence
    void SetDevicePresent(DeviceType type, bool present);
    bool IsDevicePresent(DeviceType type) const;

    // Position access (thread-safe)
    void SetPosition(DeviceType type, long position);
    long GetPosition(DeviceType type) const;
    bool IsPositionUnknown(DeviceType type) const;

    // Target position
    void SetTargetPosition(DeviceType type, long position);
    long GetTargetPosition(DeviceType type) const;

    // Busy state
    void SetBusy(DeviceType type, bool busy);
    bool IsBusy(DeviceType type) const;

    // Position limits
    void SetLimits(DeviceType type, long minPos, long maxPos);
    void GetLimits(DeviceType type, long& minPos, long& maxPos) const;

    // Number of positions (for state devices)
    void SetNumPositions(DeviceType type, int numPos);
    int GetNumPositions(DeviceType type) const;

    // Timestamps
    void SetLastUpdateTime(DeviceType type, MM::MMTime time);
    MM::MMTime GetLastUpdateTime(DeviceType type) const;

    void SetLastRequestTime(DeviceType type, MM::MMTime time);
    MM::MMTime GetLastRequestTime(DeviceType type) const;

    // Full state access (for initialization)
    DeviceState GetDeviceState(DeviceType type) const;
    void SetDeviceState(DeviceType type, const DeviceState& state);

    // Clear all state
    void Clear();

private:
    mutable std::mutex mutex_;
    std::map<DeviceType, DeviceState> devices_;

    // Helper to get or create device state
    DeviceState& GetOrCreateState(DeviceType type);
    const DeviceState& GetStateConst(DeviceType type) const;
};

} // namespace EvidentIX85
