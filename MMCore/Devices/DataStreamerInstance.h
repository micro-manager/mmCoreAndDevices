// PROJECT:       Micro-Manager
// SUBSYSTEM:     MMCore
//
// COPYRIGHT:     University of California, San Francisco, 2014,
//                All Rights reserved
//
// LICENSE:       This file is distributed under the "Lesser GPL" (LGPL) license.
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
// AUTHOR:        Mark Tsuchida

#pragma once

#include "DeviceInstanceBase.h"


class DataStreamerInstance : public DeviceInstanceBase<MM::DataStreamer>
{
public:
    DataStreamerInstance(CMMCore* core,
         std::shared_ptr<LoadedDeviceAdapter> adapter,
         const std::string& name,
         MM::Device* pDevice,
         DeleteDeviceFunction deleteFunction,
         const std::string& label,
         mm::logging::Logger deviceLogger,
         mm::logging::Logger coreLogger) :
      DeviceInstanceBase<MM::DataStreamer>(core, adapter, name, pDevice, deleteFunction, label, deviceLogger, coreLogger)
   {}

    int GetBufferSize(unsigned& dataBufferSiize);
    std::unique_ptr<char[]> GetBuffer(unsigned expectedDataBufferSize, unsigned& actualDataBufferSize, int& exitStatus);
    int ProcessBuffer(std::unique_ptr<char[]>& pDataBuffer, unsigned dataSize);
    int StartStream();
    int StopStream();
    bool GetOverflowStatus();
    int ResetOverflowStatus();
    int GetAcquisitionExitStatus();
    int GetProcessingExitStatus();
    int SetAcquisitionPause(bool pause);
    bool GetAcquisitionPause();
    bool IsStreaming();
    int SetStreamParameters(bool stopOnOverflow, bool pauseAcquisitionBeforeOverflow, int numberOfBuffers, int durationUs, int updatePeriodUs);
    int GetStreamParameters(bool& stopOnOverflow, bool& pauseAcquisitionBeforeOverflow, int& numberOfBuffers, int& durationUs, int& updatePeriodUs);
    int SetCircularAcquisitionBufferCapacity(int capacity);
    int GetCircularAcquisitionBufferCapacity();
};
