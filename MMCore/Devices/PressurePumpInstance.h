// PROJECT:       Micro-Manager
// SUBSYSTEM:     MMCore
//
// DESCRIPTION:   Pump device instance wrapper
//
// COPYRIGHT:     Institut Pierre-Gilles de Gennes, Paris, 2024,
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
// AUTHOR:        Lars Kool, Institut Pierre-Gilles de Gennes

#include "DeviceInstanceBase.h"
#include "../../MMDevice/MMDeviceConstants.h"

class PressurePumpInstance : public DeviceInstanceBase<MM::PressurePump>
{
public:
    PressurePumpInstance(CMMCore* core,
        std::shared_ptr<LoadedDeviceAdapter> adapter,
        const std::string& name,
        MM::Device* pDevice,
        DeleteDeviceFunction deleteFunction,
        const std::string& label,
        mm::logging::Logger deviceLogger,
        mm::logging::Logger coreLogger) :
        DeviceInstanceBase<MM::PressurePump>(core, adapter, name, pDevice, deleteFunction, label, deviceLogger, coreLogger)
    {}


    int Calibrate();
    int Stop();
    bool RequiresCalibration();
    int SetPressureKPa(double pressure);
    int GetPressureKPa(double& pressure);
};
