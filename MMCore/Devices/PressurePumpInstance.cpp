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
int PressurePumpInstance::Stop() { return GetImpl()->Stop(); }
int PressurePumpInstance::Calibrate() { return GetImpl()->Calibrate(); }
bool PressurePumpInstance::requiresCalibration() { return GetImpl()->RequiresCalibration(); }
int PressurePumpInstance::setPressure(double pressure) { return GetImpl()->SetPressure(pressure); }
int PressurePumpInstance::getPressure(double& pressure) { return GetImpl()->GetPressure(pressure); }