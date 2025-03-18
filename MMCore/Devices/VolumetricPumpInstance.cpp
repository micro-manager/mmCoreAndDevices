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

#include "VolumetricPumpInstance.h"
#include "../../MMDevice/MMDeviceConstants.h"

// Volume controlled pump functions
int VolumetricPumpInstance::Home() { return GetImpl()->Home(); }
int VolumetricPumpInstance::Stop() { return GetImpl()->Stop(); }
bool VolumetricPumpInstance::RequiresHoming() { return GetImpl()->RequiresHoming(); }
int VolumetricPumpInstance::InvertDirection(bool state) { return GetImpl()->InvertDirection(state); }
int VolumetricPumpInstance::IsDirectionInverted(bool& state) { return GetImpl()->IsDirectionInverted(state); }
int VolumetricPumpInstance::SetVolumeUl(double volume) { return GetImpl()->SetVolumeUl(volume); }
int VolumetricPumpInstance::GetVolumeUl(double& volume) { return GetImpl()->GetVolumeUl(volume); }
int VolumetricPumpInstance::SetMaxVolumeUl(double volume) { return GetImpl()->SetMaxVolumeUl(volume); }
int VolumetricPumpInstance::GetMaxVolumeUl(double& volume) { return GetImpl()->GetMaxVolumeUl(volume); }
int VolumetricPumpInstance::SetFlowrateUlPerSec(double flowrate) { return GetImpl()->SetFlowrateUlPerSecond(flowrate); }
int VolumetricPumpInstance::GetFlowrateUlPerSec(double& flowrate) { return GetImpl()->GetFlowrateUlPerSecond(flowrate); }
int VolumetricPumpInstance::Start() { return GetImpl()->Start(); }
int VolumetricPumpInstance::DispenseDuration(double durSec) { return GetImpl()->DispenseDuration(durSec); }
int VolumetricPumpInstance::DispenseVolume(double volUl) { return GetImpl()->DispenseVolume(volUl); }
