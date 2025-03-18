// PROJECT:       Micro-Manager
// SUBSYSTEM:     MMCore
//
// DESCRIPTION:   PressurePump device instance wrapper
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

#include "PressurePumpInstance.h"
#include "../../MMDevice/MMDeviceConstants.h"

// General pump functions
int PressurePumpInstance::Stop() { RequireInitialized(__func__); return GetImpl()->Stop(); }
int PressurePumpInstance::Calibrate() { RequireInitialized(__func__); return GetImpl()->Calibrate(); }
bool PressurePumpInstance::RequiresCalibration() { RequireInitialized(__func__); return GetImpl()->RequiresCalibration(); }
int PressurePumpInstance::SetPressureKPa(double pressure) { RequireInitialized(__func__); return GetImpl()->SetPressureKPa(pressure); }
int PressurePumpInstance::GetPressureKPa(double& pressure) { RequireInitialized(__func__); return GetImpl()->GetPressureKPa(pressure); }