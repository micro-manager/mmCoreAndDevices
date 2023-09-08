// PROJECT:       Micro-Manager
// SUBSYSTEM:     MMCore
//
// DESCRIPTION:   XY Stage device instance wrapper
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

#include "XYStageInstance.h"


int XYStageInstance::SetPositionUm(double x, double y) { RequireInitialized(); return GetImpl()->SetPositionUm(x, y); }
int XYStageInstance::SetRelativePositionUm(double dx, double dy) { RequireInitialized(); return GetImpl()->SetRelativePositionUm(dx, dy); }
int XYStageInstance::SetAdapterOriginUm(double x, double y) { RequireInitialized(); return GetImpl()->SetAdapterOriginUm(x, y); }
int XYStageInstance::GetPositionUm(double& x, double& y) { RequireInitialized(); return GetImpl()->GetPositionUm(x, y); }
int XYStageInstance::GetLimitsUm(double& xMin, double& xMax, double& yMin, double& yMax) { RequireInitialized(); return GetImpl()->GetLimitsUm(xMin, xMax, yMin, yMax); }
int XYStageInstance::Move(double vx, double vy) { RequireInitialized(); return GetImpl()->Move(vx, vy); }
int XYStageInstance::SetPositionSteps(long x, long y) { RequireInitialized(); return GetImpl()->SetPositionSteps(x, y); }
int XYStageInstance::GetPositionSteps(long& x, long& y) { RequireInitialized(); return GetImpl()->GetPositionSteps(x, y); }
int XYStageInstance::SetRelativePositionSteps(long x, long y) { RequireInitialized(); return GetImpl()->SetRelativePositionSteps(x, y); }
int XYStageInstance::Home() { RequireInitialized(); return GetImpl()->Home(); }
int XYStageInstance::Stop() { RequireInitialized(); return GetImpl()->Stop(); }
int XYStageInstance::SetOrigin() { RequireInitialized(); return GetImpl()->SetOrigin(); }
int XYStageInstance::SetXOrigin() { RequireInitialized(); return GetImpl()->SetXOrigin(); }
int XYStageInstance::SetYOrigin() { RequireInitialized(); return GetImpl()->SetYOrigin(); }
int XYStageInstance::GetStepLimits(long& xMin, long& xMax, long& yMin, long& yMax) { RequireInitialized(); return GetImpl()->GetStepLimits(xMin, xMax, yMin, yMax); }
double XYStageInstance::GetStepSizeXUm() { RequireInitialized(); return GetImpl()->GetStepSizeXUm(); }
double XYStageInstance::GetStepSizeYUm() { RequireInitialized(); return GetImpl()->GetStepSizeYUm(); }
int XYStageInstance::IsXYStageSequenceable(bool& isSequenceable) const { RequireInitialized(); return GetImpl()->IsXYStageSequenceable(isSequenceable); }
int XYStageInstance::GetXYStageSequenceMaxLength(long& nrEvents) const { RequireInitialized(); return GetImpl()->GetXYStageSequenceMaxLength(nrEvents); }
int XYStageInstance::StartXYStageSequence() { RequireInitialized(); return GetImpl()->StartXYStageSequence(); }
int XYStageInstance::StopXYStageSequence() { RequireInitialized(); return GetImpl()->StopXYStageSequence(); }
int XYStageInstance::ClearXYStageSequence() { RequireInitialized(); return GetImpl()->ClearXYStageSequence(); }
int XYStageInstance::AddToXYStageSequence(double positionX, double positionY) { RequireInitialized(); return GetImpl()->AddToXYStageSequence(positionX, positionY); }
int XYStageInstance::SendXYStageSequence() { RequireInitialized(); return GetImpl()->SendXYStageSequence(); }
