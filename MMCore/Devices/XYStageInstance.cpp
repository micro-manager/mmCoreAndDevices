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


int XYStageInstance::SetPositionUm(double x, double y) { RequireInitialized(__func__); return GetImpl()->SetPositionUm(x, y); }
int XYStageInstance::SetRelativePositionUm(double dx, double dy) { RequireInitialized(__func__); return GetImpl()->SetRelativePositionUm(dx, dy); }
int XYStageInstance::SetAdapterOriginUm(double x, double y) { RequireInitialized(__func__); return GetImpl()->SetAdapterOriginUm(x, y); }
int XYStageInstance::GetPositionUm(double& x, double& y) { RequireInitialized(__func__); return GetImpl()->GetPositionUm(x, y); }
int XYStageInstance::GetLimitsUm(double& xMin, double& xMax, double& yMin, double& yMax) { RequireInitialized(__func__); return GetImpl()->GetLimitsUm(xMin, xMax, yMin, yMax); }
int XYStageInstance::Move(double vx, double vy) { RequireInitialized(__func__); return GetImpl()->Move(vx, vy); }
int XYStageInstance::SetPositionSteps(long x, long y) { RequireInitialized(__func__); return GetImpl()->SetPositionSteps(x, y); }
int XYStageInstance::GetPositionSteps(long& x, long& y) { RequireInitialized(__func__); return GetImpl()->GetPositionSteps(x, y); }
int XYStageInstance::SetRelativePositionSteps(long x, long y) { RequireInitialized(__func__); return GetImpl()->SetRelativePositionSteps(x, y); }
int XYStageInstance::Home() { RequireInitialized(__func__); return GetImpl()->Home(); }
int XYStageInstance::Stop() { RequireInitialized(__func__); return GetImpl()->Stop(); }
int XYStageInstance::SetOrigin() { RequireInitialized(__func__); return GetImpl()->SetOrigin(); }
int XYStageInstance::SetXOrigin() { RequireInitialized(__func__); return GetImpl()->SetXOrigin(); }
int XYStageInstance::SetYOrigin() { RequireInitialized(__func__); return GetImpl()->SetYOrigin(); }
int XYStageInstance::GetStepLimits(long& xMin, long& xMax, long& yMin, long& yMax) { RequireInitialized(__func__); return GetImpl()->GetStepLimits(xMin, xMax, yMin, yMax); }
double XYStageInstance::GetStepSizeXUm() { RequireInitialized(__func__); return GetImpl()->GetStepSizeXUm(); }
double XYStageInstance::GetStepSizeYUm() { RequireInitialized(__func__); return GetImpl()->GetStepSizeYUm(); }
int XYStageInstance::IsXYStageSequenceable(bool& isSequenceable) const { RequireInitialized(__func__); return GetImpl()->IsXYStageSequenceable(isSequenceable); }
int XYStageInstance::GetXYStageSequenceMaxLength(long& nrEvents) const { RequireInitialized(__func__); return GetImpl()->GetXYStageSequenceMaxLength(nrEvents); }
int XYStageInstance::StartXYStageSequence() { RequireInitialized(__func__); return GetImpl()->StartXYStageSequence(); }
int XYStageInstance::StopXYStageSequence() { RequireInitialized(__func__); return GetImpl()->StopXYStageSequence(); }
int XYStageInstance::ClearXYStageSequence() { RequireInitialized(__func__); return GetImpl()->ClearXYStageSequence(); }
int XYStageInstance::AddToXYStageSequence(double positionX, double positionY) { RequireInitialized(__func__); return GetImpl()->AddToXYStageSequence(positionX, positionY); }
int XYStageInstance::SendXYStageSequence() { RequireInitialized(__func__); return GetImpl()->SendXYStageSequence(); }
