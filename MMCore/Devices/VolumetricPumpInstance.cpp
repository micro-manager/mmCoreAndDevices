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
int VolumetricPumpInstance::Home() { RequireInitialized(__func__); return GetImpl()->Home(); }
int VolumetricPumpInstance::Stop() { RequireInitialized(__func__); return GetImpl()->Stop(); }
bool VolumetricPumpInstance::RequiresHoming() { RequireInitialized(__func__); return GetImpl()->RequiresHoming(); }
int VolumetricPumpInstance::InvertDirection(bool state) { RequireInitialized(__func__); return GetImpl()->InvertDirection(state); }
int VolumetricPumpInstance::IsDirectionInverted(bool& state) { RequireInitialized(__func__); return GetImpl()->IsDirectionInverted(state); }
int VolumetricPumpInstance::SetVolumeUl(double volume) { RequireInitialized(__func__); return GetImpl()->SetVolumeUl(volume); }
int VolumetricPumpInstance::GetVolumeUl(double& volume) { RequireInitialized(__func__); return GetImpl()->GetVolumeUl(volume); }
int VolumetricPumpInstance::SetMaxVolumeUl(double volume) { RequireInitialized(__func__); return GetImpl()->SetMaxVolumeUl(volume); }
int VolumetricPumpInstance::GetMaxVolumeUl(double& volume) { RequireInitialized(__func__); return GetImpl()->GetMaxVolumeUl(volume); }
int VolumetricPumpInstance::SetFlowrateUlPerSecond(double flowrate) { RequireInitialized(__func__); return GetImpl()->SetFlowrateUlPerSecond(flowrate); }
int VolumetricPumpInstance::GetFlowrateUlPerSecond(double& flowrate) { RequireInitialized(__func__); return GetImpl()->GetFlowrateUlPerSecond(flowrate); }
int VolumetricPumpInstance::Start() { RequireInitialized(__func__); return GetImpl()->Start(); }
int VolumetricPumpInstance::DispenseDurationSeconds(double durSec) { RequireInitialized(__func__); return GetImpl()->DispenseDurationSeconds(durSec); }
int VolumetricPumpInstance::DispenseVolumeUl(double volUl) { RequireInitialized(__func__); return GetImpl()->DispenseVolumeUl(volUl); }
