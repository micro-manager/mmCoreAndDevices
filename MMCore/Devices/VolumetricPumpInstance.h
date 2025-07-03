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

class VolumetricPumpInstance : public DeviceInstanceBase<MM::VolumetricPump>
{
public:
    VolumetricPumpInstance(CMMCore* core,
        std::shared_ptr<LoadedDeviceAdapter> adapter,
        const std::string& name,
        MM::Device* pDevice,
        DeleteDeviceFunction deleteFunction,
        const std::string& label,
        mm::logging::Logger deviceLogger,
        mm::logging::Logger coreLogger) :
        DeviceInstanceBase<MM::VolumetricPump>(core, adapter, name, pDevice, deleteFunction, label, deviceLogger, coreLogger)
    {}

    int Home();
    int Stop();
    bool RequiresHoming();
    int InvertDirection(bool state);
    int IsDirectionInverted(bool& state);
    int SetVolumeUl(double volUl);
    int GetVolumeUl(double& volUl);
    int SetMaxVolumeUl(double volUl);
    int GetMaxVolumeUl(double& volUl);
    int SetFlowrateUlPerSecond(double flowrate);
    int GetFlowrateUlPerSecond(double& flowrate);
    int Start();
    int DispenseDurationSeconds(double durSec);
    int DispenseVolumeUl(double volUl);
};
