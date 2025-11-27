///////////////////////////////////////////////////////////////////////////////
// FILE:          EvidentModel.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Evident IX85 microscope state model implementation
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

#include "EvidentModelWin.h"
#include <mutex>

namespace EvidentIX85Win {

MicroscopeModel::MicroscopeModel() :
    measuredZOffset_(0),
    measuredZOffsetValid_(false)
{
}

MicroscopeModel::~MicroscopeModel()
{
}

long long MicroscopeModel::SteadyMicroseconds()
{
   using namespace std::chrono;
   auto now = steady_clock::now().time_since_epoch();
   auto usec = duration_cast<microseconds>(now);
   return usec.count();
}

void MicroscopeModel::SetDevicePresent(DeviceType type, bool present)
{
    std::lock_guard<std::mutex> lock(mutex_);
    GetOrCreateState(type).present = present;
}

bool MicroscopeModel::IsDevicePresent(DeviceType type) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return GetStateConst(type).present;
}

void MicroscopeModel::SetPosition(DeviceType type, long position)
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto& state = GetOrCreateState(type);
    state.currentPos = position;
    state.lastUpdateTime = MM::MMTime::fromUs(SteadyMicroseconds());
}

long MicroscopeModel::GetPosition(DeviceType type) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return GetStateConst(type).currentPos;
}

bool MicroscopeModel::IsPositionUnknown(DeviceType type) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return GetStateConst(type).currentPos == -1;
}

void MicroscopeModel::SetTargetPosition(DeviceType type, long position)
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto& state = GetOrCreateState(type);
    state.targetPos = position;
    state.lastRequestTime = MM::MMTime::fromUs(SteadyMicroseconds());
}

long MicroscopeModel::GetTargetPosition(DeviceType type) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return GetStateConst(type).targetPos;
}

void MicroscopeModel::SetBusy(DeviceType type, bool busy)
{
    std::lock_guard<std::mutex> lock(mutex_);
    GetOrCreateState(type).busy = busy;
}

bool MicroscopeModel::IsBusy(DeviceType type) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return GetStateConst(type).busy;
}

void MicroscopeModel::SetLimits(DeviceType type, long minPos, long maxPos)
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto& state = GetOrCreateState(type);
    state.minPos = minPos;
    state.maxPos = maxPos;
}

void MicroscopeModel::GetLimits(DeviceType type, long& minPos, long& maxPos) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    const auto& state = GetStateConst(type);
    minPos = state.minPos;
    maxPos = state.maxPos;
}

void MicroscopeModel::SetNumPositions(DeviceType type, int numPos)
{
    std::lock_guard<std::mutex> lock(mutex_);
    GetOrCreateState(type).numPositions = numPos;
}

int MicroscopeModel::GetNumPositions(DeviceType type) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return GetStateConst(type).numPositions;
}

void MicroscopeModel::SetLastUpdateTime(DeviceType type, MM::MMTime time)
{
    std::lock_guard<std::mutex> lock(mutex_);
    GetOrCreateState(type).lastUpdateTime = time;
}

MM::MMTime MicroscopeModel::GetLastUpdateTime(DeviceType type) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return GetStateConst(type).lastUpdateTime;
}

void MicroscopeModel::SetLastRequestTime(DeviceType type, MM::MMTime time)
{
    std::lock_guard<std::mutex> lock(mutex_);
    GetOrCreateState(type).lastRequestTime = time;
}

MM::MMTime MicroscopeModel::GetLastRequestTime(DeviceType type) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return GetStateConst(type).lastRequestTime;
}

DeviceState MicroscopeModel::GetDeviceState(DeviceType type) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return GetStateConst(type);
}

void MicroscopeModel::SetDeviceState(DeviceType type, const DeviceState& state)
{
    std::lock_guard<std::mutex> lock(mutex_);
    devices_[type] = state;
}

void MicroscopeModel::SetDeviceVersion(DeviceType type, const std::string& version)
{
    std::lock_guard<std::mutex> lock(mutex_);
    GetOrCreateState(type).version = version;
}

std::string MicroscopeModel::GetDeviceVersion(DeviceType type) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return GetStateConst(type).version;
}

void MicroscopeModel::Clear()
{
    std::lock_guard<std::mutex> lock(mutex_);
    devices_.clear();
    measuredZOffset_ = 0;
    measuredZOffsetValid_ = false;
}

DeviceState& MicroscopeModel::GetOrCreateState(DeviceType type)
{
    auto it = devices_.find(type);
    if (it == devices_.end())
    {
        DeviceState newState;
        newState.type = type;
        devices_[type] = newState;
        return devices_[type];
    }
    return it->second;
}

const DeviceState& MicroscopeModel::GetStateConst(DeviceType type) const
{
    static DeviceState emptyState;
    auto it = devices_.find(type);
    if (it == devices_.end())
        return emptyState;
    return it->second;
}

void MicroscopeModel::SetMeasuredZOffset(long offset)
{
    std::lock_guard<std::mutex> lock(mutex_);
    measuredZOffset_ = offset;
    measuredZOffsetValid_ = true;
}

long MicroscopeModel::GetMeasuredZOffset() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return measuredZOffset_;
}

bool MicroscopeModel::IsMeasuredZOffsetValid() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return measuredZOffsetValid_;
}

} // namespace EvidentIX85Win
