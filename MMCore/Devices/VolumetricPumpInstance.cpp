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
bool VolumetricPumpInstance::requiresHoming() { return GetImpl()->RequiresHoming(); }
int VolumetricPumpInstance::invertDirection(bool state) { return GetImpl()->InvertDirection(state); }
int VolumetricPumpInstance::isDirectionInverted(bool& state) { return GetImpl()->IsDirectionInverted(state); }
int VolumetricPumpInstance::setVolumeUl(double volume) { return GetImpl()->SetVolumeUl(volume); }
int VolumetricPumpInstance::getVolumeUl(double& volume) { return GetImpl()->GetVolumeUl(volume); }
int VolumetricPumpInstance::setMaxVolumeUl(double volume) { return GetImpl()->SetMaxVolumeUl(volume); }
int VolumetricPumpInstance::getMaxVolumeUl(double& volume) { return GetImpl()->GetMaxVolumeUl(volume); }
int VolumetricPumpInstance::setFlowrateUlPerSec(double flowrate) { return GetImpl()->SetFlowrateUlPerSecond(flowrate); }
int VolumetricPumpInstance::getFlowrateUlPerSec(double& flowrate) { return GetImpl()->GetFlowrateUlPerSecond(flowrate); }
int VolumetricPumpInstance::Start() { return GetImpl()->Start(); }
int VolumetricPumpInstance::DispenseDuration(double durSec) { return GetImpl()->DispenseDuration(durSec); }
int VolumetricPumpInstance::DispenseVolume(double volUl) { return GetImpl()->DispenseVolume(volUl); }
