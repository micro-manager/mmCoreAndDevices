// PROJECT:       Micro-Manager
// SUBSYSTEM:     MMCore
//
// DESCRIPTION:   Stage device instance wrapper
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

#include "StageInstance.h"


int StageInstance::SetPositionUm(double pos) { RequireInitialized(__func__); return GetImpl()->SetPositionUm(pos); }
int StageInstance::SetRelativePositionUm(double d) { RequireInitialized(__func__); return GetImpl()->SetRelativePositionUm(d); }
int StageInstance::Move(double velocity) { RequireInitialized(__func__); return GetImpl()->Move(velocity); }
int StageInstance::Stop() { RequireInitialized(__func__); return GetImpl()->Stop(); }
int StageInstance::Home() { RequireInitialized(__func__); return GetImpl()->Home(); }
int StageInstance::SetAdapterOriginUm(double d) { RequireInitialized(__func__); return GetImpl()->SetAdapterOriginUm(d); }
int StageInstance::GetPositionUm(double& pos) { RequireInitialized(__func__); return GetImpl()->GetPositionUm(pos); }
int StageInstance::SetPositionSteps(long steps) { RequireInitialized(__func__); return GetImpl()->SetPositionSteps(steps); }
int StageInstance::GetPositionSteps(long& steps) { RequireInitialized(__func__); return GetImpl()->GetPositionSteps(steps); }
int StageInstance::SetOrigin() { RequireInitialized(__func__); return GetImpl()->SetOrigin(); }
int StageInstance::GetLimits(double& lower, double& upper) { RequireInitialized(__func__); return GetImpl()->GetLimits(lower, upper); }

MM::FocusDirection
StageInstance::GetFocusDirection()
{
   // Default to what the device adapter says.
   if (!focusDirectionHasBeenSet_)
   {
      MM::FocusDirection direction;
      int err = GetImpl()->GetFocusDirection(direction);
      ThrowIfError(err, "Cannot get focus direction");

      focusDirection_ = direction;
      focusDirectionHasBeenSet_ = true;
   }
   return focusDirection_;
}

void
StageInstance::SetFocusDirection(MM::FocusDirection direction)
{
   focusDirection_ = direction;
   focusDirectionHasBeenSet_ = true;
}

int StageInstance::IsStageSequenceable(bool& isSequenceable) const { RequireInitialized(__func__); return GetImpl()->IsStageSequenceable(isSequenceable); }
int StageInstance::IsStageLinearSequenceable(bool& isSequenceable) const { RequireInitialized(__func__); return GetImpl()->IsStageLinearSequenceable(isSequenceable); }
bool StageInstance::IsContinuousFocusDrive() const { RequireInitialized(__func__); return GetImpl()->IsContinuousFocusDrive(); }
int StageInstance::GetStageSequenceMaxLength(long& nrEvents) const { RequireInitialized(__func__); return GetImpl()->GetStageSequenceMaxLength(nrEvents); }
int StageInstance::StartStageSequence() { RequireInitialized(__func__); return GetImpl()->StartStageSequence(); }
int StageInstance::StopStageSequence() { RequireInitialized(__func__); return GetImpl()->StopStageSequence(); }
int StageInstance::ClearStageSequence() { RequireInitialized(__func__); return GetImpl()->ClearStageSequence(); }
int StageInstance::AddToStageSequence(double position) { RequireInitialized(__func__); return GetImpl()->AddToStageSequence(position); }
int StageInstance::SendStageSequence() { RequireInitialized(__func__); return GetImpl()->SendStageSequence(); }
int StageInstance::SetStageLinearSequence(double dZ_um, long nSlices)
{ RequireInitialized(__func__); return GetImpl()->SetStageLinearSequence(dZ_um, nSlices); }